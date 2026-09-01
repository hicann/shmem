#!/usr/bin/env python3
import argparse
import os

import numpy as np


def dtype_from_name(name):
    if name in ("int", "int32_t"):
        return np.int32
    if name in ("float16_t", "fp16"):
        return np.float16
    if name in ("float32_t", "float", "fp32"):
        return np.float32
    raise ValueError(f"unsupported dtype: {name}")


def gen_input(rng, shape, dtype):
    if dtype == np.int32:
        return rng.integers(-5, 6, size=shape, dtype=np.int32)
    return rng.uniform(-1.0, 1.0, size=shape).astype(dtype)


def bnsd_to_bsnd_alltoall_b_axis(rank, rank_size, per_rank_inputs):
    R = rank_size
    B = per_rank_inputs[0].shape[0]
    chunk_size = B // R

    alltoall_result = np.empty_like(per_rank_inputs[0])
    for i in range(R):
        chunk = per_rank_inputs[i][rank * chunk_size : (rank + 1) * chunk_size]
        alltoall_result[i * chunk_size : (i + 1) * chunk_size] = chunk

    bsnd = np.transpose(alltoall_result, (0, 2, 1, 3))
    return np.ascontiguousarray(bsnd)


def write_bin(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankSize', type=int, required=True)
    parser.add_argument('--B', type=int, required=True, help='per-rank batch size')
    parser.add_argument('--N', type=int, required=True, help='number of experts/heads')
    parser.add_argument('--S', type=int, required=True, help='sequence length')
    parser.add_argument('--D', type=int, required=True, help='head dimension')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dtype', type=str, default='float16_t',
                        help='float16_t | float32_t | int32_t')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.rankSize <= 0:
        parser.error("rankSize must be a positive integer")

    R = args.rankSize
    B = args.B
    N = args.N
    S = args.S
    D = args.D

    assert B % R == 0, f"B={B} must be divisible by rankSize={R}"

    dtype = dtype_from_name(args.dtype)
    os.makedirs(args.data_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    per_rank_inputs = [gen_input(rng, (B, N, S, D), dtype) for _ in range(R)]
    for r in range(R):
        write_bin(os.path.join(args.data_dir, f"rank_{r}_input.bin"), per_rank_inputs[r])

    for r in range(R):
        golden = bnsd_to_bsnd_alltoall_b_axis(r, R, per_rank_inputs)
        assert golden.shape == (B, S, N, D), \
            f"golden shape {golden.shape} != ({B}, {S}, {N}, {D})"
        write_bin(os.path.join(args.data_dir, f"rank_{r}_golden.bin"), golden)

    print(f"[OK] generated inputs/golden @ {args.data_dir}")
    print(f"  R={R} B={B} N={N} S={S} D={D} dtype={args.dtype}")
    print(f"  Per-rank input shape: [B={B}, N={N}, S={S}, D={D}]")
    print(f"  Per-rank output shape: [B={B}, S={S}, N={N}, D={D}]")


if __name__ == '__main__':
    main()
