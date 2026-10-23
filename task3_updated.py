import os
import time
from multiprocessing import Process, RawArray
from ctypes import c_uint64

ITERATIONS = 50_000_000



def unpadded_worker(shared_arr, thread_id):
    for _ in range(ITERATIONS):
        shared_arr[thread_id] += 1


def local_worker(shared_arr, thread_id):
    local_count = 0
    for _ in range(ITERATIONS):
        local_count += 1
    shared_arr[thread_id] = local_count


def main():
    num_cores = os.cpu_count() or 4
    print(f"Physical/Logical cores detected: {num_cores}")
    print(f"Iterations per thread: {ITERATIONS:,}")


    shared_counts_unpadded = RawArray(c_uint64, num_cores)

    processes = []
    start_unpadded = time.perf_counter()

    for i in range(num_cores):
        p = Process(target=unpadded_worker, args=(shared_counts_unpadded, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_unpadded = time.perf_counter()
    unpadded_ms = (end_unpadded - start_unpadded) * 1000


    shared_counts_padded = RawArray(c_uint64, num_cores)

    processes = []
    start_padded = time.perf_counter()

    for i in range(num_cores):
        p = Process(target=local_worker, args=(shared_counts_padded, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_padded = time.perf_counter()
    padded_ms = (end_padded - start_padded) * 1000


    ratio = unpadded_ms / padded_ms if padded_ms > 0 else 0

    print("\n" + "=" * 50)
    print(f"Unpadded Shared Time : {unpadded_ms:.3f} ms")
    print(f"Padded/Local Time   : {padded_ms:.3f} ms")
    print(f"Execution Ratio      : {ratio:.3f}x faster (Local vs Unpadded)")
    print("=" * 50)


if __name__ == "__main__":
    main()