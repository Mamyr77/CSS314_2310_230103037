import math
import os
import time
from multiprocessing import Pool

A = 2.5
B = 1.7
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]


def saxpy_chunk(chunk):
    return [A * x + B for x in chunk]


def serial_saxpy(x):
    return [A * val + B for val in x]


def parallel_saxpy(x, pool, num_workers):
    chunk_size = math.ceil(len(x) / num_workers)
    chunks = [x[i : i + chunk_size] for i in range(0, len(x), chunk_size)]
    results = pool.map(saxpy_chunk, chunks)
    return [val for chunk in results for val in chunk]


def benchmark_n(n, pool, num_workers, runs=5):
    x = list(range(n))

    # Warmup
    serial_saxpy(x[: min(100, n)])

    serial_times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = serial_saxpy(x)
        serial_times.append(time.perf_counter() - start)
    t_serial = min(serial_times) * 1000

    parallel_times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = parallel_saxpy(x, pool, num_workers)
        parallel_times.append(time.perf_counter() - start)
    t_parallel = min(parallel_times) * 1000

    return t_serial, t_parallel


def main():
    num_cores = os.cpu_count() or 1
    print(f"Cores detected: {num_cores}")
    print("Vector Operation: y[i] = a * x[i] + b")

    results = []
    crossover_found = None

    with Pool(processes=num_cores) as pool:
        for i, n in enumerate(SIZES):
            t_ser, t_par = benchmark_n(n, pool, num_cores)
            is_faster = t_par < t_ser

            results.append(
                {
                    "N": n,
                    "serial_ms": t_ser,
                    "parallel_ms": t_par,
                    "faster": is_faster,
                }
            )

            if crossover_found is None and is_faster:
                prev_n = SIZES[i - 1] if i > 0 else 0
                crossover_found = (prev_n, n)

    print("\n" + "=" * 65)
    print(
        f"{'N':>12} | "
        f"{'Serial (ms)':>12} | "
        f"{'Parallel (ms)':>13} | "
        f"{'Faster?':>8}"
    )
    print("-" * 65)

    for r in results:
        status = "YES" if r["faster"] else "NO"
        print(
            f"{r['N']:12,d} | "
            f"{r['serial_ms']:12.3f} | "
            f"{r['parallel_ms']:13.3f} | "
            f"{status:>8}"
        )

    print("=" * 65)

    if crossover_found:
        print(
            f"\nCrossover Point: Multi-threading becomes faster between "
            f"N = {crossover_found[0]:,} and N = {crossover_found[1]:,}"
        )
    else:
        print(
            "\nCrossover Point: Parallel execution remained slower for all tested N."
        )


if __name__ == "__main__":
    main()