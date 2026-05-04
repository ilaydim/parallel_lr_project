"""
analysis_comparison.py
Bcast vs Scatterv karşılaştırma analizi.

Input:
  timing_output.csv  — Scatterv (N, P, par_time_total, compute_time, comm_time)
  timing_bcast.csv   — Bcast    (N, P, par_time_total, compute_time, comm_time)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_sc = pd.read_csv("timing_output.csv")
df_bc = pd.read_csv("timing_bcast.csv")

for df in [df_sc, df_bc]:
    df.columns = df.columns.str.strip()

df_sc = df_sc.groupby(["N", "P"]).median().reset_index()
df_bc = df_bc.groupby(["N", "P"]).median().reset_index()

N_values = sorted(df_sc["N"].unique())
P_values = sorted(df_sc["P"].unique())
N_labels = {n: f"{n // 1_000_000}M" for n in N_values}

# ── Baseline: her N için P=1'deki compute_time ───────────────────────
for df in [df_sc, df_bc]:
    for n in N_values:
        p1 = df[(df["N"] == n) & (df["P"] == 1)]
        if p1.empty:
            continue
        t1_compute = p1["compute_time"].values[0]
        t1_total   = p1["par_time_total"].values[0]
        mask = df["N"] == n
        df.loc[mask, "speedup_compute"] = t1_compute / df.loc[mask, "compute_time"]
        df.loc[mask, "speedup_total"]   = t1_total   / df.loc[mask, "par_time_total"]

def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches="tight")
    print(f"Saved: {name}")
    plt.close(fig)

# ── PLOT 1: Communication Time ────────────────────────────────────────
fig, axes = plt.subplots(1, len(N_values), figsize=(5 * len(N_values), 4))
if len(N_values) == 1: axes = [axes]
for ax, n in zip(axes, N_values):
    sub_sc = df_sc[df_sc["N"] == n].sort_values("P")
    sub_bc = df_bc[df_bc["N"] == n].sort_values("P")
    ax.plot(sub_sc["P"], sub_sc["comm_time"], marker="o", color="steelblue", label="Scatterv (comm)")
    ax.plot(sub_bc["P"], sub_bc["comm_time"], marker="s", color="tomato",    label="Bcast (comm)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("P", fontsize=11)
    ax.set_ylabel("Comm Time (s)", fontsize=11)
    ax.set_title(f"N = {N_labels[n]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
fig.suptitle("Communication Time: Bcast vs Scatterv  —  MPI Linear Regression",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "plot_comparison_commtime.png")

# ── PLOT 2: Speedup (compute only) ───────────────────────────────────
fig, axes = plt.subplots(1, len(N_values), figsize=(5 * len(N_values), 4))
if len(N_values) == 1: axes = [axes]
for ax, n in zip(axes, N_values):
    sub_sc = df_sc[df_sc["N"] == n].sort_values("P")
    sub_bc = df_bc[df_bc["N"] == n].sort_values("P")
    ax.plot(sub_sc["P"], sub_sc["speedup_compute"], marker="o", color="steelblue", label="Scatterv")
    ax.plot(sub_bc["P"], sub_bc["speedup_compute"], marker="s", color="tomato",    label="Bcast")
    ax.plot(P_values, P_values, "k--", linewidth=1, label="Ideal")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("P", fontsize=11)
    ax.set_ylabel("Speedup (S)", fontsize=11)
    ax.set_title(f"N = {N_labels[n]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
fig.suptitle("Speedup (Compute Only): Bcast vs Scatterv  —  MPI Linear Regression",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "plot_comparison_speedup.png")

# ── PLOT 3: Total Execution Time ──────────────────────────────────────
fig, axes = plt.subplots(1, len(N_values), figsize=(5 * len(N_values), 4))
if len(N_values) == 1: axes = [axes]
for ax, n in zip(axes, N_values):
    sub_sc = df_sc[df_sc["N"] == n].sort_values("P")
    sub_bc = df_bc[df_bc["N"] == n].sort_values("P")
    # Sequential baseline: Scatterv'in P=1'deki par_time_total
    t1 = df_sc[(df_sc["N"] == n) & (df_sc["P"] == 1)]["par_time_total"].values[0]
    ax.plot(sub_sc["P"], sub_sc["par_time_total"], marker="o", color="steelblue", label="Scatterv (total)")
    ax.plot(sub_bc["P"], sub_bc["par_time_total"], marker="s", color="tomato",    label="Bcast (total)")
    ax.axhline(t1, color="black", linestyle="--", linewidth=1, label="Sequential (P=1)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("P", fontsize=11)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title(f"N = {N_labels[n]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
fig.suptitle("Total Execution Time: Bcast vs Scatterv  —  MPI Linear Regression",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "plot_comparison_totaltime.png")

print("\nAll comparison plots done.")