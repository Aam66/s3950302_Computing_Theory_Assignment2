# TSP timing comparison: Exact (brute force) vs 2-approx (MST/double-tree) vs Nearest-Neighbor.

# Now with live progress:
# - Brute force runs in a separate process with a heartbeat every PROGRESS_INTERVAL seconds.
# - If the run exceeds MAX_BF_TIME, we terminate it and skip BF for larger n.

import os, sys, time, math, random, csv, multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.append(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    import question4_files.Brute_Force_TSP as BF
    import question4_files.Approx_MST_TSP as MST
    import question4_files.Nearest_Neighbor_TSP as NN
except ModuleNotFoundError:
    import Brute_Force_TSP as BF
    import Approx_MST_TSP as MST
    import Nearest_Neighbor_TSP as NN

# ----------------- configuration -----------------
SEED = 42                    # fixed seed so runs are reproducible
MAX_BF_TIME = 30 * 60        # 30 minutes cap for brute force (change if you like)
AIM_TOTAL_FAST_SMALL = 0.50  # repeat fast algos until ~this total per n (small n)
AIM_TOTAL_FAST_LARGE = 0.50  # same for large n
PROGRESS_INTERVAL = 5        # heartbeat interval (seconds) while BF is running

# Choose n-values you want to see. Brute force will auto-skip once too slow.
N_SMALL = (5, 7, 9, 11, 13, 14, 15)   # candidates where we *attempt* brute force
N_LARGE = (50, 100, 200, 500, 1000, 2000, 5000, 10000)  # fast methods only

OUT_CSV = os.path.join(HERE, "results_tsp_timings.csv")

# -------------- data generation --------------
def generate_graph(n, seed=SEED, low=1, high=100):
    """Dense symmetric integer-weighted graph in [low, high]."""
    rnd = random.Random(seed + n)  # vary by n but deterministically
    g = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            w = rnd.randint(low, high)
            g[i][j] = w
            g[j][i] = w
    return g

# -------------- timing helpers --------------
def time_once(fn, *args):
    t0 = time.perf_counter()
    val = fn(*args)
    dt = time.perf_counter() - t0
    return dt, val

def time_repeated_total(fn, graph, min_total):
    """
    Repeat fn(graph) until total_time >= min_total (or at least 1 run).
    Returns (total_time, runs, last_value).
    """
    total = 0.0
    runs = 0
    last = None
    # Warm-up single run to get a per-call estimate:
    t_one, last = time_once(fn, graph)
    total += t_one
    runs += 1
    if t_one < min_total:
        remaining = max(0.0, min_total - t_one)
        per = max(t_one, 1e-6)
        extra_runs = int(math.ceil(remaining / per))
        for _ in range(extra_runs):
            t, last = time_once(fn, graph)
            total += t
            runs += 1
    return total, runs, last

# -------------- brute-force with heartbeat & timeout --------------
def _bf_worker(graph, q):
    """Run BF in a separate process and put (cost,) in q on success."""
    try:
        cost = BF.tsp_min_cost(graph)
        q.put(("ok", cost))
    except Exception as e:
        q.put(("err", str(e)))

def run_bf_with_progress(n, graph, cap_seconds, interval=PROGRESS_INTERVAL):
    """
    Returns (elapsed_s, cost or None, exceeded_cap: bool)
    - If exceeds cap, terminates the worker and returns (elapsed, None, True).
    """
    q = mp.Queue()
    p = mp.Process(target=_bf_worker, args=(graph, q))
    start = time.perf_counter()
    p.start()
    next_ping = interval
    exceeded = False
    cost = None

    while True:
        # Check queue without blocking
        if not q.empty():
            status, payload = q.get()
            elapsed = time.perf_counter() - start
            if status == "ok":
                cost = payload
            else:
                cost = None
            p.join(timeout=0.1)
            return elapsed, cost, False

        elapsed = time.perf_counter() - start

        # Heartbeat
        if elapsed >= next_ping:
            print(f"  n={n} | BF … {int(elapsed)}s", flush=True)
            next_ping += interval

        # Cap check
        if elapsed >= cap_seconds:
            exceeded = True
            print(f"  n={n} | BF exceeded {cap_seconds}s cap — terminating.", flush=True)
            try:
                p.terminate()
                p.join(timeout=1.0)
            except Exception:
                pass
            return elapsed, None, True

        time.sleep(0.05)

# -------------- runner --------------
def run_suite():
    print("TSP timing comparison (same random graph per n)")
    print(f"Seed = {SEED}, BF cap = {MAX_BF_TIME}s\n")

    rows = []
    largest_n_under_cap = None

    # Try brute force on the small set until it would exceed the cap.
    bf_still_ok = True
    for n in N_SMALL:
        g = generate_graph(n)

        # --- brute force (exact) with heartbeat ---
        bf_time = None
        bf_cost = None
        if bf_still_ok:
            print(f"Starting n={n} …", flush=True)
            bf_time, bf_cost, exceeded = run_bf_with_progress(n, g, MAX_BF_TIME, PROGRESS_INTERVAL)
            if exceeded or bf_cost is None:
                bf_still_ok = False
                print(f"Brute force exceeded cap at n={n}; skipping BF from here on.\n", flush=True)
            else:
                largest_n_under_cap = n
                print(f"  n={n} | BF done in {bf_time:.4f}s (cost={bf_cost})\n", flush=True)

        # --- fast methods (repeat to ~AIM_TOTAL_FAST_SMALL seconds total) ---
        mst_total, mst_runs, _ = time_repeated_total(MST.approx_tsp, g, AIM_TOTAL_FAST_SMALL)
        nn_total,  nn_runs,  _ = time_repeated_total(NN.nearest_neighbor_cost, g, AIM_TOTAL_FAST_SMALL)

        rows.append({
            "n": n,
            "bf_time_s": bf_time,
            "mst_total_s": mst_total,
            "mst_runs": mst_runs,
            "nn_total_s": nn_total,
            "nn_runs": nn_runs
        })

    # Large n block: only fast methods (still repeated to ~AIM_TOTAL_FAST_LARGE seconds).
    for n in N_LARGE:
        g = generate_graph(n)
        print(f"Fast-only n={n} …", flush=True)
        mst_total, mst_runs, _ = time_repeated_total(MST.approx_tsp, g, AIM_TOTAL_FAST_LARGE)
        nn_total,  nn_runs,  _ = time_repeated_total(NN.nearest_neighbor_cost, g, AIM_TOTAL_FAST_LARGE)
        rows.append({
            "n": n,
            "bf_time_s": None,
            "mst_total_s": mst_total,
            "mst_runs": mst_runs,
            "nn_total_s": nn_total,
            "nn_runs": nn_runs
        })

    # ---- terminal output ----
    print("\nResults Table (totals for fast methods; they are repeated to avoid 0.0000 artifacts)")
    print("| n | Brute-Force (s) | 2-Approx MST total (s) [runs] | Nearest-Neighbor total (s) [runs] |")
    print("|---|------------------|-------------------------------|------------------------------------|")
    for r in rows:
        bf_str = "—" if r["bf_time_s"] is None else f"{r['bf_time_s']:.4f}"
        print(f"| {r['n']:>2} | {bf_str:>16} | {r['mst_total_s']:>27.4f} [{r['mst_runs']:>3}] | "
              f"{r['nn_total_s']:>34.4f} [{r['nn_runs']:>3}] |")

    if largest_n_under_cap is None:
        print("\nBrute force exceeded the time cap at the very first attempted n.")
    else:
        print(f"\nLargest n solved under the {MAX_BF_TIME}s cap by brute force: n = {largest_n_under_cap}")

    # ---- write CSV ----
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "brute_force_time_s", "approx_mst_total_time_s", "approx_mst_runs",
                    "nearest_neighbor_total_time_s", "nearest_neighbor_runs"])
        for r in rows:
            w.writerow([r["n"],
                        "" if r["bf_time_s"] is None else f"{r['bf_time_s']:.6f}",
                        f"{r['mst_total_s']:.6f}", r["mst_runs"],
                        f"{r['nn_total_s']:.6f}", r["nn_runs"]])
    print(f"\nWrote {OUT_CSV}")

if __name__ == "__main__":
    # On Windows, multiprocessing requires this guard.
    run_suite()
