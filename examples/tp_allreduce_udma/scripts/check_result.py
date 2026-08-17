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

from gen_data import add_common_arguments, get_case_id, get_dtype, validate_common_arguments


def load_file(path, dtype, elements):
    if not path.exists():
        raise FileNotFoundError(path)
    tensor = np.fromfile(path, dtype=dtype)
    if tensor.size != elements:
        raise ValueError(f"{path} has {tensor.size} elements, expected {elements}")
    return tensor


def check_rank(rank, case_dir, output_dir, dtype, elements, rtol, atol):
    output = load_file(output_dir / f"output_{rank}.bin", dtype, elements)
    golden = load_file(case_dir / f"rank_{rank}" / "golden.bin", dtype, elements)

    if np.issubdtype(dtype, np.integer):
        ok = np.array_equal(output, golden)
    else:
        ok = np.allclose(output.astype(np.float32), golden.astype(np.float32), rtol=rtol, atol=atol)

    if not ok:
        diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
        idx = int(np.argmax(diff))
        raise AssertionError(
            f"rank {rank} mismatch at {idx}: output={output[idx]}, golden={golden[idx]}, abs_diff={diff[idx]}"
        )


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--golden-root", type=Path, default=Path("golden"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    validate_common_arguments(args)
    dtype = get_dtype(args.dtype)
    case_dir = args.golden_root / get_case_id(args.pes, args.elements, args.tp_size, args.dtype)
    for rank in range(args.pes):
        check_rank(rank, case_dir, args.output_dir, dtype, args.elements, args.rtol, args.atol)

    print(
        f"tp_allreduce_udma golden check passed: case_dir={case_dir}, output_dir={args.output_dir}, "
        f"pes={args.pes}, elements={args.elements}, dtype={args.dtype}"
    )


if __name__ == "__main__":
    main()
