import threading
import time

TARGET_INCREMENTS = 2_000_000
NUM_THREADS = 4
EXPECTED_TOTAL = TARGET_INCREMENTS * NUM_THREADS

unsynchronized_counter = 0
synchronized_counter = 0
lock = threading.Lock()


def unsafe_worker():
    global unsynchronized_counter
    for _ in range(TARGET_INCREMENTS):

        unsynchronized_counter += 1


def safe_worker():
    global synchronized_counter
    for _ in range(TARGET_INCREMENTS):
        with lock:
            synchronized_counter += 1


def main():
    global unsynchronized_counter, synchronized_counter

    print(f"Threads: {NUM_THREADS}")
    print(f"Increments per thread: {TARGET_INCREMENTS:,}")
    print(f"Expected final value: {EXPECTED_TOTAL:,}")


    threads = []
    start_unsafe = time.perf_counter()

    for _ in range(NUM_THREADS):
        t = threading.Thread(target=unsafe_worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_unsafe = time.perf_counter()
    time_unsafe = (end_unsafe - start_unsafe) * 1000

    actual_unsafe = unsynchronized_counter
    lost_updates = EXPECTED_TOTAL - actual_unsafe
    loss_percentage = (
        (lost_updates / EXPECTED_TOTAL) * 100 if EXPECTED_TOTAL > 0 else 0
    )


    threads = []
    start_safe = time.perf_counter()

    for _ in range(NUM_THREADS):
        t = threading.Thread(target=safe_worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_safe = time.perf_counter()
    time_safe = (end_safe - start_safe) * 1000

    actual_safe = synchronized_counter
    slowdown_ratio = time_safe / time_unsafe if time_unsafe > 0 else 0

    print("\n" + "=" * 60)
    print(f"{'Metric':<35} | {'Value':>20}")
    print("-" * 60)
    print(
        f"{'Unsynchronized Final Value':<35} | {actual_unsafe:20,d}"
    )
    print(f"{'Lost Updates':<35} | {lost_updates:20,d}")
    print(f"{'Data Loss Percentage':<35} | {loss_percentage:19.2f}%")
    print("-" * 60)
    print(f"{'Synchronized Final Value':<35} | {actual_safe:20,d}")
    print(f"{'Unsynchronized Time (ms)':<35} | {time_unsafe:20.2f}")
    print(f"{'Synchronized Time (ms)':<35} | {time_safe:20.2f}")
    print(
        f"{'Synchronization Lock Penalty':<35} | {slowdown_ratio:19.2f}x"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()