# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import ctypes
import json
import os
import statistics
import time
from pathlib import Path

import acl
import torch
import torch.distributed as dist

import shmem as ash
import shmem.core as core


ACL_MEMCPY_DEVICE_TO_HOST = 2
DEFAULT_HEAP_SIZE = 1024 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Python core.put against a C++ aclshmemx_putmem_on_stream "
            "loop using the same buffers, PE route, shape, and ACL stream."
        )
    )
    parser.add_argument("--bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--threshold-percent", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cpp-lib",
        type=Path,
        default=Path(__file__).resolve().parent / "libputmem_on_stream_cpp_ref.so",
    )
    return parser.parse_args()


def check_acl(ret, operation):
    if ret != 0:
        raise RuntimeError(f"{operation} failed, ret={ret}")


def read_device_byte(addr):
    value = ctypes.c_uint8(0)
    ret = acl.rt.memcpy(
        ctypes.addressof(value),
        ctypes.sizeof(value),
        addr,
        ctypes.sizeof(value),
        ACL_MEMCPY_DEVICE_TO_HOST,
    )
    check_acl(ret, "acl.rt.memcpy")
    return value.value


def load_cpp_reference(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"C++ reference helper not found: {path}. Run build_cpp_ref.sh first."
        )

    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    function = library.shmem_perf_putmem_on_stream_cpp
    function.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = ctypes.c_int
    return library, function


def run_cpp(reference, dst, src, size, pe, stream, iterations):
    elapsed_us = ctypes.c_double(0.0)
    ret = reference(
        dst,
        src,
        size,
        pe,
        stream,
        iterations,
        ctypes.byref(elapsed_us),
    )
    if ret != 0:
        raise RuntimeError(f"C++ putmem_on_stream reference failed, ret={ret}")
    return elapsed_us.value


def run_python(dst, src, pe, stream, iterations):
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        core.put(dst, src, pe, stream)
    check_acl(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")
    return (time.perf_counter_ns() - start_ns) / 1000.0


def initialize_shmem(rank, world_size, heap_size):
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise RuntimeError(f"Disable config-store TLS failed, ret={ret}")

    unique_id = core.get_unique_id() if rank == 0 else None
    uid_list = [unique_id]
    dist.broadcast_object_list(uid_list, src=0)
    core.init(
        rank=rank,
        nranks=world_size,
        mem_size=heap_size,
        uid=uid_list[0],
        initializer_method="uid",
    )


def aggregate_results(args, world_size, local_result):
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_result)

    if dist.get_rank() != 0:
        return None

    cpp_job_rounds = [
        max(rank_result["cpp_us"][round_idx] for rank_result in gathered)
        for round_idx in range(args.rounds)
    ]
    python_job_rounds = [
        max(rank_result["python_us"][round_idx] for rank_result in gathered)
        for round_idx in range(args.rounds)
    ]

    cpp_us_per_op = statistics.median(cpp_job_rounds) / args.iterations
    python_us_per_op = statistics.median(python_job_rounds) / args.iterations
    overhead_percent = ((python_us_per_op / cpp_us_per_op) - 1.0) * 100.0

    per_rank = []
    for rank_result in gathered:
        cpp_rank = statistics.median(rank_result["cpp_us"]) / args.iterations
        python_rank = statistics.median(rank_result["python_us"]) / args.iterations
        per_rank.append(
            {
                "rank": rank_result["rank"],
                "cpp_us_per_op": cpp_rank,
                "python_us_per_op": python_rank,
                "overhead_percent": ((python_rank / cpp_rank) - 1.0) * 100.0,
            }
        )

    return {
        "world_size": world_size,
        "bytes": args.bytes,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "stream": "same explicit ACL stream",
        "route": "ring PE -> (PE + 1) % world_size",
        "cpp_us_per_op": cpp_us_per_op,
        "python_us_per_op": python_us_per_op,
        "overhead_percent": overhead_percent,
        "threshold_percent": args.threshold_percent,
        "passed": overhead_percent <= args.threshold_percent,
        "correctness": "bit-exact sampled first/last byte",
        "per_rank": per_rank,
    }


def main():
    args = parse_args()
    if args.bytes <= 0 or args.iterations <= 0 or args.warmup <= 0 or args.rounds <= 0:
        raise ValueError("bytes, iterations, warmup, and rounds must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    next_pe = (rank + 1) % world_size
    previous_pe = (rank - 1 + world_size) % world_size

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="gloo", init_method="env://")
    initialize_shmem(rank, world_size, max(DEFAULT_HEAP_SIZE, args.bytes * 4))

    send_buffer = core.buffer(args.bytes)
    receive_buffer = core.buffer(args.bytes)
    check_acl(
        acl.rt.memset(send_buffer.addr, args.bytes, rank + 1, args.bytes),
        "initialize send buffer",
    )
    check_acl(
        acl.rt.memset(receive_buffer.addr, args.bytes, 0, args.bytes),
        "initialize receive buffer",
    )

    stream, ret = acl.rt.create_stream()
    check_acl(ret, "acl.rt.create_stream")
    try:
        cpp_library, cpp_reference = load_cpp_reference(args.cpp_lib.resolve())
    except (OSError, AttributeError) as exc:
        raise RuntimeError(f"Failed to load C++ reference helper: {exc}") from exc
    if cpp_library is None or cpp_reference is None:
        raise RuntimeError("C++ reference helper returned an invalid library or function")

    dist.barrier()
    for _ in range(args.warmup):
        core.put(receive_buffer, send_buffer, next_pe, stream)
    check_acl(acl.rt.synchronize_stream(stream), "Python warmup stream sync")
    dist.barrier()
    cpp_warmup_us = run_cpp(
        cpp_reference,
        receive_buffer.addr,
        send_buffer.addr,
        args.bytes,
        next_pe,
        stream,
        args.warmup,
    )
    if cpp_warmup_us < 0:
        raise RuntimeError(f"C++ warmup returned an invalid elapsed time: {cpp_warmup_us}")
    dist.barrier()

    cpp_times = []
    python_times = []
    for round_idx in range(args.rounds):
        # Alternate order to reduce systematic first/second-run bias.
        paths = ("cpp", "python") if round_idx % 2 == 0 else ("python", "cpp")
        for path in paths:
            dist.barrier()
            if path == "cpp":
                cpp_times.append(
                    run_cpp(
                        cpp_reference,
                        receive_buffer.addr,
                        send_buffer.addr,
                        args.bytes,
                        next_pe,
                        stream,
                        args.iterations,
                    )
                )
            else:
                python_times.append(
                    run_python(
                        receive_buffer,
                        send_buffer,
                        next_pe,
                        stream,
                        args.iterations,
                    )
                )
            dist.barrier()

    expected = previous_pe + 1
    actual_first = read_device_byte(receive_buffer.addr)
    actual_last = read_device_byte(receive_buffer.addr + args.bytes - 1)
    if actual_first != expected or actual_last != expected:
        raise AssertionError(
            f"Rank {rank} correctness failure: expected byte {expected}, "
            f"got first={actual_first}, last={actual_last}"
        )

    result = aggregate_results(
        args,
        world_size,
        {
            "rank": rank,
            "cpp_us": cpp_times,
            "python_us": python_times,
        },
    )

    passed = None
    if rank == 0:
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        print(f"PERF_RESULT_JSON={json.dumps(result, sort_keys=True)}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Saved result: {args.output}")
        passed = result["passed"]

    passed_list = [passed]
    dist.broadcast_object_list(passed_list, src=0)

    check_acl(acl.rt.destroy_stream(stream), "acl.rt.destroy_stream")
    core.free(receive_buffer)
    core.free(send_buffer)
    core.finalize()
    dist.destroy_process_group()

    if not passed_list[0]:
        raise AssertionError(
            f"Python overhead exceeded {args.threshold_percent:.2f}% threshold"
        )


if __name__ == "__main__":
    main()
