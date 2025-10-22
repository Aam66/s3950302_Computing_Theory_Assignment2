# Runs the study: loops over (N, p), repeats R times, logs CSVs, makes two plots.

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from life_core import simulate_once

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="30,50,80", help="Comma-separated N values.")
    ap.add_argument("--densities", default="0.1,0.2,0.3,0.4", help="Comma-separated p values.")
    ap.add_argument("--runs", type=int, default=200, help="Runs per (N,p).")
    ap.add_argument("--steps", type=int, default=500, help="Generations per run (T).")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    ap.add_argument("--out", default="results", help="Output directory.")
    ap.add_argument("--glider_cdf_choice", default="auto",
                    help='Which (N,p) to draw the glider CDF for, format "N,p" or "auto".')
    return ap.parse_args()

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def run_condition(N, p, R, T, rng):
    rows = []
    for r in range(R):
        outcome = simulate_once(N, p, T, rng)
        rows.append({
            "N": N,
            "p": p,
            "run_id": r,
            "outcome": outcome.outcome,
            "t_event": outcome.t_event,
            "period": outcome.period,
            "glider_seen": outcome.glider_seen,
            "t_glider": outcome.t_glider
        })
    return rows

def summarise_outcomes(df_cond: pd.DataFrame):
    # Percentages by outcome
    total = len(df_cond)
    counts = df_cond["outcome"].value_counts().to_dict()
    pct = {k: 100.0 * counts.get(k, 0) / total for k in ["extinct", "still", "oscillating", "active_end"]}

    # Glider stats
    g_seen = int(df_cond["glider_seen"].sum())
    g_pct = 100.0 * g_seen / total
    g_times = df_cond.loc[df_cond["glider_seen"] == 1, "t_glider"]
    g_avg = float(g_times.mean()) if not g_times.empty else None

    # Period distribution among oscillating runs
    periods = df_cond.loc[df_cond["outcome"] == "oscillating", "period"].dropna().astype(int).tolist()
    return pct, g_pct, g_avg, periods

def main():
    args = parse_args()
    Ns = [int(x) for x in args.sizes.split(",") if x]
    Ps = [float(x) for x in args.densities.split(",") if x]
    R  = args.runs
    T  = args.steps

    ensure_dir(args.out)

    rng = np.random.default_rng(args.seed)
    all_rows = []

    # 1) Run all conditions and collect per-run rows
    for N in Ns:
        for p in Ps:
            print(f"Running N={N}, p={p} ...")
            rows = run_condition(N, p, R, T, rng)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out, "runs_raw.csv"), index=False)
    print(f"Wrote {os.path.join(args.out, 'runs_raw.csv')}")

    # 2) Build Table 1: outcome percentages + glider stats per (N,p)
    tbl1_rows = []
    periods_all = []  # for Table 2
    for N in Ns:
        for p in Ps:
            sub = df[(df["N"] == N) & (df["p"] == p)]
            pct, g_pct, g_avg, periods = summarise_outcomes(sub)
            periods_all.extend(periods)
            tbl1_rows.append({
                "N": N,
                "p": p,
                "Extinct_pct": round(pct["extinct"], 2),
                "Still_pct": round(pct["still"], 2),
                "Oscillating_pct": round(pct["oscillating"], 2),
                "ActiveEnd_pct": round(pct["active_end"], 2),
                "Glider_seen_pct": round(g_pct, 2),
                "Avg_t_glider": (round(g_avg, 2) if g_avg is not None else "")
            })
    tbl1 = pd.DataFrame(tbl1_rows)
    tbl1.to_csv(os.path.join(args.out, "table_outcomes.csv"), index=False)
    print(f"Wrote {os.path.join(args.out, 'table_outcomes.csv')}")

    # 3) Build Table 2: oscillator period distribution (overall)
    period_counts = Counter(periods_all)
    tbl2 = pd.DataFrame(
        {"Period": list(period_counts.keys()),
         "Count": list(period_counts.values())}
    ).sort_values("Period")
    total_osc = sum(period_counts.values()) or 1
    tbl2["Pct_of_oscillators"] = (100.0 * tbl2["Count"] / total_osc).round(2)
    tbl2.to_csv(os.path.join(args.out, "table_periods.csv"), index=False)
    print(f"Wrote {os.path.join(args.out, 'table_periods.csv')}")

    # 4) Plot 1: outcome probabilities vs density, per N
    # (extinct, still, oscillating, active_end, glider-seen)
    for N in Ns:
        sub = tbl1[tbl1["N"] == N].sort_values("p")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sub["p"], sub["Extinct_pct"], marker="o", label="Extinct %")
        ax.plot(sub["p"], sub["Still_pct"], marker="o", label="Still-life %")
        ax.plot(sub["p"], sub["Oscillating_pct"], marker="o", label="Oscillating %")
        ax.plot(sub["p"], sub["ActiveEnd_pct"], marker="o", label="ActiveEnd %")
        ax.plot(sub["p"], sub["Glider_seen_pct"], marker="o", label="Glider-seen %")
        ax.set_title(f"Outcome probabilities vs density (N={N})")
        ax.set_xlabel("Initial live-cell density p")
        ax.set_ylabel("Percentage of runs (%)")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(args.out, f"plot_prob_vs_density_N{N}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {out_path}")

    # 5) Plot 2: time-to-first-glider CDF for one condition
    # Choose (N,p) either from args or pick the first (N,p) with any gliders.
    choice = None
    if args.glider_cdf_choice != "auto":
        try:
            N0, p0 = args.glider_cdf_choice.split(",")
            choice = (int(N0), float(p0))
        except:
            pass

    if choice is None:
        for N in Ns:
            for p in Ps:
                sub = df[(df["N"] == N) & (df["p"] == p) & (df["glider_seen"] == 1)]
                if len(sub) > 0:
                    choice = (N, p)
                    break
            if choice is not None:
                break

    if choice is not None:
        N0, p0 = choice
        sub = df[(df["N"] == N0) & (df["p"] == p0) & (df["glider_seen"] == 1)]
        times = sorted(sub["t_glider"].astype(int).tolist())
        if len(times) > 0:
            cdf_y = np.arange(1, len(times) + 1) / len(times)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.step(times, cdf_y, where="post")
            ax.set_title(f"Time-to-first-glider CDF (N={N0}, p={p0})")
            ax.set_xlabel("t (generations)")
            ax.set_ylabel("Cumulative fraction of runs")
            fig.tight_layout()
            out_path = os.path.join(args.out, f"plot_glider_cdf_N{N0}_p{p0}.png")
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Wrote {out_path}")
    else:
        print("No condition produced any gliders; skipping CDF plot.")

    print("Done.")

if __name__ == "__main__":
    main()
