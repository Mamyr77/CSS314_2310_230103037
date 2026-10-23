import os
import platform
import time
import numpy as np

STUDENT_ID = "230103037"
N = 20_000_000

def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical CPU cores: {os.cpu_count()}")


    seed = int(STUDENT_ID)
    rng = np.random.default_rng(seed)
    data = rng.random(N, dtype=np.float64)


    start = time.perf_counter()
    total = float(np.sum(data, dtype=np.float64))
    std_dev = float(np.std(data, dtype=np.float64, ddof=0))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"Array elements: {N}")
    print("Data type: float64")
    print(f"Seed: {seed}")
    print(f"Mathematical sum: {total:.12f}")
    print(f"Population standard deviation: {std_dev:.12f}")
    print(f"Serial computation wall-clock time (ms): {elapsed_ms:.3f}")

if __name__ == "__main__":
    main()
