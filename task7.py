import math
import os
import time
from multiprocessing import Pool
import numpy as np

N = 30_000_000


def kernel_a_chunk(chunk):
    return np.copy(chunk)


def kernel_b_chunk(chunk):
    return np.sin(chunk) * np.cos(chunk) + np.tan(chunk)


def benchmark_kernel(kernel_fn, data, workers, pool):
    chunks = np.array_split(data, workers)
    start = time.perf_counter()
    pool.map(kernel_fn, chunks)
    elapsed = time.perf_counter() - start
    return elapsed * 1000


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted(list(set([1, 2, max_cores])))

    print(f"Array size: {N:,} elements (float64)")
    print(f"Detected logical cores: {max_cores}")

    rng = np.random.default_rng(42)
    data = rng.random(N, dtype=np.float64)

    results_a = []
    results_b = []

    with Pool(processes=max_cores) as pool:
        for w in worker_counts:
            ms = benchmark_kernel(kernel_a_chunk, data, w, pool)
            results_a.append({"workers": w, "time_ms": ms})

        for w in worker_counts:
            ms = benchmark_kernel(kernel_b_chunk, data, w, pool)
            results_b.append({"workers": w, "time_ms": ms})

    base_a = results_a[0]["time_ms"]
    base_b = results_b[0]["time_ms"]

    print("\n" + "=" * 70)
    print("KERNEL A: Vector Copy B[i] = A[i] (Memory-Bound)")
    print("-" * 70)
    print(
        f"{'Cores':>8} | "
        f"{'Wall Time (ms)':>15} | "
        f"{'Speedup':>10} | "
        f"{'Scaling Efficiency':>20}"
    )
    print("-" * 70)

    for r in results_a:
        w = r["workers"]
        ms = r["time_ms"]
        speedup = base_a / ms
        efficiency = speedup / w
        print(
            f"{w:8d} | "
            f"{ms:15.2f} | "
            f"{speedup:10.2f}x | "
            f"{efficiency:19.2%}"
        )

    print("\n" + "=" * 70)
    print(
        "KERNEL B: Trig Functions B[i] = sin(A[i])*cos(A[i])+tan(A[i]) (Compute-Bound)"
    )
    print("-" * 70)
    print(
        f"{'Cores':>8} | "
        f"{'Wall Time (ms)':>15} | "
        f"{'Speedup':>10} | "
        f"{'Scaling Efficiency':>20}"
    )
    print("-" * 70)

    for r in results_b:
        w = r["workers"]
        ms = r["time_ms"]
        speedup = base_b / ms
        efficiency = speedup / w
        print(
            f"{w:8d} | "
            f"{ms:15.2f} | "
            f"{speedup:10.2f}x | "
            f"{efficiency:19.2%}"
        )

    print("=" * 70)

    eff_a_max = (base_a / results_a[-1]["time_ms"]) / results_a[-1]["workers"]
    eff_b_max = (base_b / results_b[-1]["time_ms"]) / results_b[-1]["workers"]

    print("\nCONCLUSION:")
    if eff_a_max < eff_b_max:
        print(
            f"Kernel A (Memory-Bound) stalled in scaling (Efficiency: {eff_a_max:.1%}) "
            f"due to RAM memory channel bandwidth saturation."
        )
        print(
            f"Kernel B (Compute-Bound) maintained higher scaling efficiency ({eff_b_max:.1%}) "
            f"because its bottleneck is CPU ALU computation, not memory bus speed."
        )
    else:
        print(
            "Kernel B experienced higher overhead relative to its computational workload."
        )


if __name__ == "__main__":
    main()