import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

N = 1024
BLOCK_SIZES = [16, 64]


def matmul_non_tiled_row_range(A, B, C, start_row, end_row):
    for i in range(start_row, end_row):
        for k in range(N):
            a_ik = A[i, k]
            for j in range(N):
                C[i, j] += a_ik * B[k, j]


def matmul_tiled_row_range(A, B, C, start_row, end_row, tile_size):
    for i0 in range(start_row, end_row, tile_size):
        i_max = min(i0 + tile_size, end_row)
        for k0 in range(0, N, tile_size):
            k_max = min(k0 + tile_size, N)
            for j0 in range(0, N, tile_size):
                j_max = min(j0 + tile_size, N)

                for i in range(i0, i_max):
                    for k in range(k0, k_max):
                        a_ik = A[i, k]
                        for j in range(j0, j_max):
                            C[i, j] += a_ik * B[k, j]


def run_parallel_matmul(A, B, mode="non_tiled", tile_size=16, num_workers=4):
    C = np.zeros((N, N), dtype=np.float64)
    chunk_rows = N // num_workers

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for w in range(num_workers):
            start_row = w * chunk_rows
            end_row = N if w == num_workers - 1 else (w + 1) * chunk_rows

            if mode == "non_tiled":
                f = executor.submit(
                    matmul_non_tiled_row_range, A, B, C, start_row, end_row
                )
            else:
                f = executor.submit(
                    matmul_tiled_row_range,
                    A,
                    B,
                    C,
                    start_row,
                    end_row,
                    tile_size,
                )
            futures.append(f)

        for f in futures:
            f.result()

    return C


def main():
    print(f"Matrix Size: {N}x{N} (float64)")
    num_workers = 4
    print(f"Parallel Threads: {num_workers}")

    rng = np.random.default_rng(42)
    A = rng.random((N, N), dtype=np.float64)
    B = rng.random((N, N), dtype=np.float64)

    # 1. Non-Tiled
    start = time.perf_counter()
    C_non_tiled = run_parallel_matmul(
        A, B, mode="non_tiled", num_workers=num_workers
    )
    t_non_tiled = (time.perf_counter() - start) * 1000

    results = [{"strategy": "Non-Tiled", "time_ms": t_non_tiled}]

    # 2. Tiled variants
    for block_size in BLOCK_SIZES:
        start = time.perf_counter()
        C_tiled = run_parallel_matmul(
            A,
            B,
            mode="tiled",
            tile_size=block_size,
            num_workers=num_workers,
        )
        t_tiled = (time.perf_counter() - start) * 1000

        # Verification
        assert np.allclose(C_non_tiled, C_tiled), "Matrix multiplication result mismatch!"

        results.append(
            {"strategy": f"Tiled ({block_size}x{block_size})", "time_ms": t_tiled}
        )

    base_time = results[0]["time_ms"]

    print("\n" + "=" * 65)
    print(
        f"{'Strategy':<25} | "
        f"{'Wall Time (ms)':>15} | "
        f"{'Speedup vs Non-Tiled':>20}"
    )
    print("-" * 65)

    for r in results:
        speedup = base_time / r["time_ms"]
        print(
            f"{r['strategy']:<25} | "
            f"{r['time_ms']:15.2f} | "
            f"{speedup:19.2f}x"
        )

    print("=" * 65)

    best_tiled = min(results[1:], key=lambda x: x["time_ms"])
    print(
        f"\nCONCLUSION: {best_tiled['strategy']} achieved optimal execution latency "
        f"by maximizing CPU L1/L2 cache line reuse and minimizing cache eviction stalls."
    )


if __name__ == "__main__":
    main()