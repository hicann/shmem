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
from pathlib import Path

import numpy as np

TP_SIZE = 2


def get_dtype(name):
    if name in ("int", "int32_t"):
        return np.int32
    if name == "float16_t":
        return np.float16
    raise ValueError(f"unsupported dtype: {name}")


def get_case_id(pes, elements, tp_size, dtype):
    return f"shape_{elements}_{pes}_{tp_size}_{dtype}"


def add_common_arguments(parser):
    parser.add_argument("--pes", type=int, required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=TP_SIZE)
    parser.add_argument("--dtype", type=str, default="float16_t")


def validate_common_arguments(args):
    if args.tp_size != TP_SIZE:
        raise ValueError(f"tp_size must be {TP_SIZE}")
    if args.pes <= 0 or args.pes % args.tp_size != 0:
        raise ValueError("pes must be positive and divisible by tp_size")
    if args.elements <= 0 or args.elements % args.tp_size != 0:
        raise ValueError("elements must be positive and divisible by tp_size")


def generate_input(rng, elements, dtype):
    if np.issubdtype(dtype, np.integer):
        return rng.integers(-16, 16, size=elements, endpoint=False, dtype=dtype)
    return rng.uniform(low=-1.0, high=1.0, size=elements).astype(dtype)


def build_golden(inputs, pes, tp_size, dtype):
    golden = []
    for rank in range(pes):
        group_base = rank // tp_size * tp_size
        group_inputs = inputs[group_base : group_base + tp_size]
        if np.issubdtype(dtype, np.integer):
            reduced = np.zeros_like(group_inputs[0], dtype=np.int64)
            for item in group_inputs:
                reduced += item.astype(np.int64)
            golden.append(reduced.astype(dtype))
        else:
            reduced = np.zeros_like(group_inputs[0], dtype=np.float32)
            for item in group_inputs:
                reduced += item.astype(np.float32)
            golden.append(reduced.astype(dtype))
    return golden


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--golden-root", type=Path, default=Path("golden"))
    args = parser.parse_args()

    validate_common_arguments(args)
    dtype = get_dtype(args.dtype)
    case_id = get_case_id(args.pes, args.elements, args.tp_size, args.dtype)
    case_dir = args.golden_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    inputs = [generate_input(rng, args.elements, dtype) for _ in range(args.pes)]
    golden = build_golden(inputs, args.pes, args.tp_size, dtype)

    for rank in range(args.pes):
        rank_dir = case_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        inputs[rank].tofile(rank_dir / "input_gm.bin")
        golden[rank].tofile(rank_dir / "golden.bin")

    print(
        f"generated tp_allreduce_udma data: case_dir={case_dir}, pes={args.pes}, "
        f"elements={args.elements}, tp_size={args.tp_size}, dtype={args.dtype}"
    )


if __name__ == "__main__":
    main()
