import argparse
import multiprocessing as mp
import re
import sys
import time
from contextlib import redirect_stdout
import os

import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sa_solver import solve_sa

# Regex to parse progress lines from sa_loop
PROGRESS_RE = re.compile(r"SA_LOOP_PROGRESS: Iter:\s*(\d+).*?BestNormScore:\s*([-0-9.eE]+)")


def _worker(n, iterations, seed, log_interval, queue):
    """Run solve_sa and forward progress lines to the queue."""

    class QueueWriter:
        def __init__(self, q, idx):
            self.q = q
            self.idx = idx

        def write(self, msg):
            msg = msg.strip()
            if msg:
                self.q.put((self.idx, msg))

        def flush(self):
            pass

    with redirect_stdout(QueueWriter(queue, seed)):
        solve_sa(n, iterations=iterations, seed=seed, log_interval_sa_loop=log_interval)
    queue.put((seed, "DONE"))


def visualize_sa(n, iterations, runs, log_interval=100):
    queue = mp.Queue()
    processes = []
    seeds = [42 + i for i in range(runs)]

    for seed in seeds:
        p = mp.Process(target=_worker, args=(n, iterations, seed, log_interval, queue))
        p.start()
        processes.append(p)

    data = {seed: {"iter": [], "score": []} for seed in seeds}

    finished = set()

    while len(finished) < runs:
        sid, msg = queue.get()
        if msg == "DONE":
            finished.add(sid)
            continue
        m = PROGRESS_RE.search(msg)
        if m:
            it = int(m.group(1))
            score = float(m.group(2))
            data[sid]["iter"].append(it)
            data[sid]["score"].append(score)

    for p in processes:
        p.join()

    fig, ax = plt.subplots()
    for seed in seeds:
        ax.plot(data[seed]["iter"], data[seed]["score"], label=f"seed {seed}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Norm Score")
    ax.legend()
    ax.set_title("SA Progress per Thread")

    fig.savefig("sa_progress.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SA solver progress")
    parser.add_argument("n", type=int, help="number of players")
    parser.add_argument("iterations", type=int, help="iterations per thread")
    parser.add_argument("runs", type=int, default=2, nargs="?", help="parallel runs")
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    visualize_sa(args.n, args.iterations, args.runs, args.log_interval)
