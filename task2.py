import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

STUDENT_ID = "230103037"
N = 20_000_000
P = 0.95


def sum_chunk(chunk):
    """Calculate the sum of one array chunk."""
    return np.sum(chunk, dtype=np.float64)


def parallel_sum(data, workers):
    """Split array and calculate partial sums using worker threads."""
    chunks = np.array_split(data, workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        partial_sums = list(executor.map(sum_chunk, chunks))

    return math.fsum(float(value) for value in partial_sums)


def amdahl_upper_bound(workers):
    """Theoretical speedup according to Amdahl's Law."""
    return 1.0 / ((1.0 - P) + P / workers)


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")

    max_cores = os.cpu_count() or 1
    print(f"Logical cores detected: {max_cores}")

    # Generate deterministic data using Student ID as the seed
    rng = np.random.default_rng(int(STUDENT_ID))
    data = rng.random(N, dtype=np.float64)

    # Required worker counts
    worker_counts = sorted(set([1, 2, 4, max_cores, max_cores * 2]))

    results = []

    # Benchmark each worker count
    for workers in worker_counts:
        start = time.perf_counter()

        total = parallel_sum(data, workers)

        elapsed = time.perf_counter() - start

        results.append({"workers": workers, "time": elapsed, "sum": total})

    # Baseline time for one worker
    baseline_time = next(
        result["time"] for result in results if result["workers"] == 1
    )

    print(f"\nArray size: {N:,}")
    print("Data type: float64")
    print(f"Seed: {STUDENT_ID}")
    print(f"Sum: {results[0]['sum']:.12f}")

    print("\n" + "=" * 65)
    print(
        f"{'Cores':>8} | "
        f"{'Wall Time (ms)':>15} | "
        f"{'Speedup':>10} | "
        f"{'Amdahl Upper Bound':>19}"
    )
    print("-" * 65)

    for result in results:
        workers = result["workers"]
        elapsed = result["time"]

        speedup = baseline_time / elapsed
        upper_bound = amdahl_upper_bound(workers)

        print(
            f"{workers:8d} | "
            f"{elapsed * 1000:15.3f} | "
            f"{speedup:10.3f} | "
            f"{upper_bound:19.3f}"
        )

    print("=" * 65)


if __name__ == "__main__":
    main()