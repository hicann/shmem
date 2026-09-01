#!/usr/bin/env python3
import argparse
import os
import sys
import ast
import numpy as np


def dtype_from_name(name):
    if name in ("int", "int32_t"):
        return np.int32
    if name in ("float16_t", "fp16"):
        return np.float16
    if name in ("float32_t", "float", "fp32"):
        return np.float32
    raise ValueError(f"unsupported dtype: {name}")


def read_bin(path: str, dtype, expected_elems: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=dtype)
    if raw.size < expected_elems:
        raise RuntimeError(
            f"{path} has {raw.size} elements, need at least {expected_elems}"
        )
    return raw[:expected_elems]


def _fmt(val: float, width: int = 12) -> str:
    if np.isnan(val):
        return f"{'nan':>{width}}"
    if np.isposinf(val):
        return f"{'inf':>{width}}"
    if np.isneginf(val):
        return f"{'-inf':>{width}}"
    return f"{val:>{width}.6f}"


def _print_d_tensor(t: np.ndarray, title: str, indent: str = "") -> None:
    B, S, N, D = t.shape
    print()
    print(f"{indent}{'=' * 78}")
    print(f"{indent}{title}  shape = [B={B}, S={S}, N={N}, D={D}]  dtype = {t.dtype}")
    print(f"{indent}{'=' * 78}")
    print(f"{indent}array([")
    for b in range(B):
        print(f"{indent}  [")
        for s in range(S):
            print(f"{indent}    [")
            for n in range(N):
                row = " ".join(_fmt(float(t[b, s, n, d])) for d in range(D))
                comma = "," if (n != N - 1) or (s != S - 1) or (b != B - 1) else ""
                print(f"{indent}      [{row}],")
            close_s = "    ]" + ("," if (s != S - 1) or (b != B - 1) else "")
            print(f"{indent}{close_s}")
        close_b = "  ]" + ("," if (b != B - 1) else "")
        print(f"{indent}{close_b}")
    print(f"{indent}])")
    print()


def compare_exact(golden: np.ndarray, actual: np.ndarray) -> tuple:
    """Element-wise bit-exact comparison for this pure data movement op."""
    assert golden.shape == actual.shape, (golden.shape, actual.shape)
    assert golden.dtype == actual.dtype, (golden.dtype, actual.dtype)

    golden_contiguous = np.ascontiguousarray(golden)
    actual_contiguous = np.ascontiguousarray(actual)
    itemsize = golden.dtype.itemsize
    golden_bytes = golden_contiguous.view(np.uint8).reshape(golden.shape + (itemsize,))
    actual_bytes = actual_contiguous.view(np.uint8).reshape(actual.shape + (itemsize,))
    mismatch = np.any(golden_bytes != actual_bytes, axis=-1)
    mismatch_idx = np.argwhere(mismatch)
    if mismatch_idx.size > 0:
        print(f"--- {mismatch_idx.shape[0]} mismatched positions (bit-exact compare) ---")
        for coord in mismatch_idx[:64]:
            b, s, n, d = int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])
            expected_bits = golden_bytes[b, s, n, d].tobytes().hex()
            actual_bits = actual_bytes[b, s, n, d].tobytes().hex()
            print(f"  coord [B={b} S={s} N={n} D={d}]  "
                  f"expected={golden[b, s, n, d]} bits=0x{expected_bits}  "
                  f"actual={actual[b, s, n, d]} bits=0x{actual_bits}")
        if mismatch_idx.shape[0] > 64:
            print(f"  ... and {mismatch_idx.shape[0] - 64} more")
    else:
        print("--- 0 mismatched positions ---")

    error_num = int(mismatch.sum())
    total = int(golden.size)
    print(f"error_num: {error_num} / total: {total}")
    passed = error_num == 0
    return passed, error_num


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-pattern", type=str, default="./out/rank_{}_input.bin",
                   help="Pattern for per-rank input files")
    p.add_argument("--golden-pattern", type=str, default="./out/rank_{}_golden.bin",
                   help="Pattern for per-rank golden files")
    p.add_argument("--actual-pattern", type=str, default="./out/rank_{}_output.bin",
                   help="Pattern for per-rank actual output files")
    p.add_argument("--B",         type=int, required=True, help="B per-rank batch size")
    p.add_argument("--N",         type=int, required=True, help="N per rank")
    p.add_argument("--S",         type=int, required=True, help="S seq length")
    p.add_argument("--D",         type=int, required=True, help="D head dim")
    p.add_argument("--rankSize",  type=int, required=True, help="AllToAll rank count")
    p.add_argument("--dtype",     type=str, default="float16_t",
                   help="float16_t | float32_t | int32_t")
    p.add_argument("--print-only", action="store_true",
                   help="Only print the tensors; skip PASS/ERROR comparison exit code.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print input, golden, and actual tensors.")
    args = p.parse_args()

    B = args.B
    N = args.N
    S = args.S
    D = args.D
    R = args.rankSize

    assert B % R == 0, f"B={B} must be divisible by rankSize={R}"

    dtype = dtype_from_name(args.dtype)
    per_rank_inputs = []
    per_rank_goldens = []
    per_rank_actuals = []

    for r in range(R):
        input_path = args.input_pattern.format(r)
        golden_path = args.golden_pattern.format(r)
        actual_path = args.actual_pattern.format(r)

        if not os.path.isfile(input_path):
            print(f"[ERROR] Input file missing for rank {r}: {input_path}", file=sys.stderr)
            sys.exit(2)
        if not os.path.isfile(golden_path):
            print(f"[ERROR] Golden file missing for rank {r}: {golden_path}", file=sys.stderr)
            sys.exit(2)
        if not os.path.isfile(actual_path):
            print(f"[ERROR] Actual file missing for rank {r}: {actual_path}", file=sys.stderr)
            sys.exit(2)

        input_elems = B * N * S * D
        orig = read_bin(input_path, dtype, input_elems).reshape(B, N, S, D)
        per_rank_inputs.append(orig)

        output_elems = B * S * N * D
        golden = read_bin(golden_path, dtype, output_elems).reshape(B, S, N, D)
        per_rank_goldens.append(golden)

        actual = read_bin(actual_path, dtype, output_elems).reshape(B, S, N, D)
        per_rank_actuals.append(actual)

    print(f"transpose_alltoall verification "
          f"(B={B}, N={N}, S={S}, D={D}, rankSize={R}, dtype={args.dtype})")
    print("-" * 70)

    for r in range(R):
        print(f"\n=== Rank {r} ===")
        if args.verbose or args.print_only:
            print(f"Input shape: [B={B}, N={N}, S={S}, D={D}]")
            _print_d_tensor(per_rank_inputs[r], "INPUT")

            print(f"\nExpected output shape: [B={B}, S={S}, N={N}, D={D}]")
            _print_d_tensor(per_rank_goldens[r], "EXPECTED")

            print(f"\nActual output shape: [B={B}, S={S}, N={N}, D={D}]")
            _print_d_tensor(per_rank_actuals[r], "ACTUAL")

    print("\n" + "-" * 70)
    print("COMPARISON RESULTS")
    print("-" * 70)

    all_passed = True
    for r in range(R):
        print(f"\n--- Rank {r} ---")
        rank_ok, _ = compare_exact(per_rank_goldens[r], per_rank_actuals[r])
        all_passed = all_passed and rank_ok

    if args.print_only:
        print("\nPASS (print-only mode)")
        sys.exit(0)

    if all_passed:
        print("\n\033[32mPASS\033[0m")
        sys.exit(0)
    else:
        print("\n\033[31mERROR\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        with open(__file__, "rb") as f:
            ast.parse(f.read())
        print("[self-test] AST parse OK")
        np.random.seed(42)
        shape = (2, 2, 2, 3)
        for test_dtype in (np.float16, np.float32, np.int32):
            golden = np.ones(shape, dtype=test_dtype)
            ok, n = compare_exact(golden, golden.copy())
            assert ok and n == 0, (test_dtype, ok, n)
        print("[self-test] identical fp16/fp32/int32 tensors bit-exact PASS")

        for test_dtype in (np.float16, np.float32):
            golden = np.ones(shape, dtype=test_dtype)
            actual = golden.copy()
            actual[0, 0, 0, 0] = np.nextafter(
                test_dtype(1.0), test_dtype(2.0), dtype=test_dtype)
            ok, n = compare_exact(golden, actual)
            assert (not ok) and n == 1, (test_dtype, ok, n)
        print("[self-test] one-ULP fp16/fp32 perturbations must FAIL  OK")

        golden = np.zeros(shape, dtype=np.float32)
        actual = golden.copy()
        actual[0, 0, 0, 0] = -0.0
        ok, n = compare_exact(golden, actual)
        assert (not ok) and n == 1, (ok, n)
        print("[self-test] +0.0 vs -0.0 bit mismatch must FAIL  OK")

        golden = np.zeros(shape, dtype=np.float32)
        golden[0, 0, 0, 0] = np.nan
        actual = golden.copy()
        ok, n = compare_exact(golden, actual)
        assert ok and n == 0, (ok, n)
        actual.view(np.uint32)[0, 0, 0, 0] ^= np.uint32(1)
        ok, n = compare_exact(golden, actual)
        assert (not ok) and n == 1, (ok, n)
        print("[self-test] identical NaN bits PASS, changed NaN payload FAIL  OK")
        sys.exit(0)

    main()
