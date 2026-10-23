import concurrent.futures
import math
import os
import time

NUM_ITEMS = 1_000
NUM_WORKERS = 4
DYNAMIC_CHUNK_SIZE = 5


def compute_workload(i):
    cost = 0.0001 * (i % 20)
    time.sleep(cost)
    return i


def static_worker(items):
    start_time = time.perf_counter()
    for item in items:
        compute_workload(item)
    end_time = time.perf_counter()
    return start_time, end_time


def dynamic_worker(work_queue):
    start_time = time.perf_counter()
    while True:
        try:
            chunk = work_queue.pop(0)
        except IndexError:
            break
        for item in chunk:
            compute_workload(item)
    end_time = time.perf_counter()
    return start_time, end_time


def run_strategy_a():
    items = list(range(NUM_ITEMS))
    chunk_size = math.ceil(len(items) / NUM_WORKERS)
    static_chunks = [
        items[i : i + chunk_size] for i in range(0, len(items), chunk_size)
    ]

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:
        futures = [
            executor.submit(static_worker, chunk) for chunk in static_chunks
        ]
        results = [f.result() for f in futures]
    wall_end = time.perf_counter()

    finish_times = [end for _, end in results]
    min_max_delta = (max(finish_times) - min(finish_times)) * 1000
    total_time = (wall_end - wall_start) * 1000

    return total_time, min_max_delta


def run_strategy_b():
    items = list(range(NUM_ITEMS))
    work_queue = [
        items[i : i + DYNAMIC_CHUNK_SIZE]
        for i in range(0, len(items), DYNAMIC_CHUNK_SIZE)
    ]

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:
        futures = [
            executor.submit(dynamic_worker, work_queue)
            for _ in range(NUM_WORKERS)
        ]
        results = [f.result() for f in futures]
    wall_end = time.perf_counter()

    finish_times = [end for _, end in results]
    min_max_delta = (max(finish_times) - min(finish_times)) * 1000
    total_time = (wall_end - wall_start) * 1000

    return total_time, min_max_delta


def main():
    print(f"Items: {NUM_ITEMS}")
    print(f"Parallel Workers: {NUM_WORKERS}")
    print(f"Dynamic Chunk Size: {DYNAMIC_CHUNK_SIZE}")

    time_a, delta_a = run_strategy_a()
    time_b, delta_b = run_strategy_b()

    print("\n" + "=" * 65)
    print(
        f"{'Strategy':<30} | "
        f"{'Total Time (ms)':>15} | "
        f"{'Min/Max Delta (ms)':>15}"
    )
    print("-" * 65)
    print(f"{'Strategy A (Static Chunk)':<30} | {time_a:15.2f} | {delta_a:15.2f}")
    print(f"{'Strategy B (Dynamic/Stealing)':<30} | {time_b:15.2f} | {delta_b:15.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()