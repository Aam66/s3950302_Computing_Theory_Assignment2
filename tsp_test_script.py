
# Simple timing harness for three TSP methods:

# - Brute force (exact) — only for very small n.
# - MST 2-approx (double-tree preorder).
# - Nearest neighbour heuristic.

# Design choices:
# - Same random symmetric graph shared by all algorithms per n (fairness).
# - For tiny n, run fast algos many times and report the total to avoid "0.0000".
# - Output mirrors your preferred, simple console style.


import time, math, random
import Brute_Force_TSP as BF
import Approx_MST_TSP as MST
import Nearest_Neighbor_TSP as NN

def generate_graph(n, seed=42, low=1, high=100):
    # Dense symmetric integer-weighted graph in [low, high].
    rnd = random.Random(seed)
    g = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            w = rnd.randint(low, high)
            g[i][j] = w
            g[j][i] = w
    return g

def time_once(fn, *args):
    # High-resolution timing for a single call.
    t0 = time.perf_counter()
    val = fn(*args)
    return time.perf_counter() - t0, val

def time_amplified_total(fn, graph, runs):
    # Total time of repeated calls (helps tiny n avoid printing 0.0000).
    total = 0.0
    last_val = None
    for _ in range(runs):
        dt, last_val = time_once(fn, graph)
        total += dt
    return total, last_val

def run_suite(
    n_values_small=(5,7,9,11),
    n_values_big=(50,100,200,500,1000,2000,5000,10000),
    seed=42,
    amplify_small_fast=2000,
    fast_only_threshold=12,
):
    print("TSP timing comparison (same random graph per n)\n")
    rows = []

    # Small n block (shows your preferred "Testing with n = ..." lines)
    for n in n_values_small:
        g = generate_graph(n, seed)
        t_bf, cost_bf = time_once(BF.tsp_min_cost, g)
        print(f"Testing with n = {n}")
        print(f"Minimum Cost is: {cost_bf}\n")

        # Run fast algos many times so totals are clearly > 0
        t_mst, _ = time_amplified_total(MST.approx_tsp, g, amplify_small_fast)
        t_nn,  _ = time_amplified_total(NN.nearest_neighbor_cost, g, amplify_small_fast)
        rows.append((n, t_bf, t_mst, t_nn))

    # Larger n (skip brute force; factorial time would blow up)
    for n in n_values_big:
        g = generate_graph(n, seed)
        t_bf = float("nan")
        t_mst, _ = time_once(MST.approx_tsp, g)
        t_nn,  _ = time_once(NN.nearest_neighbor_cost, g)
        rows.append((n, t_bf, t_mst, t_nn))

    # Console table in the same simple style
    print("Results Table:")
    print("| n (Cities) | Brute-Force Time (s) | 2-Approx MST Time (s) | Nearest-Neighbor Time (s) |")
    print("|------------|----------------------|-----------------------|---------------------------|")
    for n, t_bf, t_mst, t_nn in rows:
        bf_str = f"{t_bf:.4f}" if not math.isnan(t_bf) else "—"
        print(f"| {n:<10} | {bf_str:>22} | {t_mst:>23.4f} | {t_nn:>25.4f} |")

if __name__ == "__main__":
    run_suite()
