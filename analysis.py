"""
analysis.py
Speedup / Efficiency / Scalability analysis for Parallel Linear Regression (MPI).

Input:   timing_output.csv  (columns: N, P, seq_time, par_time)
Outputs:
  - speedup_efficiency_table.csv   (machine-readable table)
  - plot_speedup.png
  - plot_efficiency.png
  - plot_scalability.png

Usage:
  python3 analysis.py [--csv timing_output.csv]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="timing_output.csv")
args = parser.parse_args()

# ── load ─────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)
df.columns = df.columns.str.strip()
df = df.groupby(["N", "P"]).median().reset_index()

N_values = sorted(df["N"].unique())
P_values = sorted(df["P"].unique())
N_labels = {n: f"{n // 1_000_000}M" for n in N_values}
COLORS   = plt.cm.tab10.colors

# ── speedup & efficiency ─────────────────────────────────────────────
records = []
for n in N_values:
    sub = df[df["N"] == n].sort_values("P")
    t1_rows = sub[sub["P"] == 1]
    if t1_rows.empty:
        continue
    t1 = t1_rows["par_time"].values[0]   # T(N,1) = baseline
    for _, row in sub.iterrows():
        p  = int(row["P"])
        tp = row["par_time"]
        S  = round(t1 / tp, 2)
        E  = round(S / p,   2)
        records.append(dict(N=n, N_label=N_labels[n],
                            P=p, T1=t1, Tp=tp,
                            Speedup=S, Efficiency=E))

res = pd.DataFrame(records)

# ── print table (same style as HW2) ─────────────────────────────────
print("\n" + "="*72)
print("  Speedup (S) and Efficiency (E)  —  Parallel Linear Regression (MPI)")
print("="*72)
header = f"{'N':>6}  {' ':>2}" + "".join(f"  {p:>6}" for p in P_values)
print(header)
print("-"*72)
for n in N_values:
    sub = res[res["N"] == n]
    def val(metric, p):
        rows = sub[sub["P"] == p]
        return f"{rows[metric].values[0]:>6.2f}" if not rows.empty else f"{'—':>6}"
    print(f"{N_labels[n]:>6}  {'S':>2}" + "".join(f"  {val('Speedup',    p)}" for p in P_values))
    print(f"{'':>6}  {'E':>2}" + "".join(f"  {val('Efficiency', p)}" for p in P_values))

res.to_csv("speedup_efficiency_table.csv", index=False)
print("\nSaved: speedup_efficiency_table.csv")

# ── helper ───────────────────────────────────────────────────────────
def setup_ax(ax, title, xlabel, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks(P_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches="tight")
    print(f"Saved: {name}")
    plt.close(fig)

# ── PLOT 1: Speedup ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for idx, n in enumerate(N_values):
    sub = res[res["N"] == n].sort_values("P")
    ax.plot(sub["P"], sub["Speedup"], marker="o", color=COLORS[idx],
            label=f"N={N_labels[n]}")
ax.plot(P_values, P_values, "k--", linewidth=1, label="Ideal")
setup_ax(ax, "Speedup — Parallel Linear Regression (MPI)",
         "Number of Processes (P)", "Speedup (S)")
save(fig, "plot_speedup.png")

# ── PLOT 2: Efficiency ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for idx, n in enumerate(N_values):
    sub = res[res["N"] == n].sort_values("P")
    ax.plot(sub["P"], sub["Efficiency"], marker="o", color=COLORS[idx],
            label=f"N={N_labels[n]}")
ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal")
setup_ax(ax, "Efficiency — Parallel Linear Regression (MPI)",
         "Number of Processes (P)", "Efficiency (E = S/P)")
save(fig, "plot_efficiency.png")

# ── PLOT 3: Execution time (scalability) for large N ────────────────
large_Ns = [n for n in N_values if n >= 100_000_000]
fig, axes = plt.subplots(1, len(large_Ns),
                          figsize=(5 * len(large_Ns), 4),
                          sharey=False)
if len(large_Ns) == 1:
    axes = [axes]

for ax, n in zip(axes, large_Ns):
    sub = res[res["N"] == n].sort_values("P")
    t1  = sub["T1"].values[0]
    ax.plot(sub["P"], sub["Tp"], marker="s", color="steelblue", label="Parallel")
    ax.axhline(t1, color="tomato", linestyle="--", label="Sequential (T1)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("P", fontsize=11)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title(f"N = {N_labels[n]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Execution Time vs. Number of Processes — MPI Linear Regression",
             fontsize=12, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "plot_scalability.png")

print("\nAll done.")
