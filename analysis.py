"""
analysis.py
Speedup / Efficiency / Scalability analysis for Parallel Linear Regression (MPI).

Input:   timing_output.csv  (N, P, par_time_total, compute_time, comm_time)
Outputs: plot_speedup.png, plot_efficiency.png, plot_scalability.png,
         plot_comm_vs_compute.png, plot_table.png, speedup_efficiency_table.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv("timing_output.csv")
df.columns = df.columns.str.strip()
df = df.groupby(["N", "P"]).median().reset_index()

N_values = sorted(df["N"].unique())
P_values = sorted(df["P"].unique())
N_labels = {n: f"{n // 1_000_000}M" for n in N_values}
COLORS   = plt.cm.tab10.colors

# ── Baseline: her N için P=1'deki değerler ───────────────────────────
for n in N_values:
    p1 = df[(df["N"] == n) & (df["P"] == 1)]
    if p1.empty:
        continue
    t1_compute = p1["compute_time"].values[0]
    t1_total   = p1["par_time_total"].values[0]
    mask = df["N"] == n
    df.loc[mask, "speedup_compute"]    = t1_compute / df.loc[mask, "compute_time"]
    df.loc[mask, "speedup_total"]      = t1_total   / df.loc[mask, "par_time_total"]
    df.loc[mask, "efficiency_compute"] = df.loc[mask, "speedup_compute"] / df.loc[mask, "P"]
    df.loc[mask, "efficiency_total"]   = df.loc[mask, "speedup_total"]   / df.loc[mask, "P"]

# ── Tabloyu yazdır ───────────────────────────────────────────────────
print("\n" + "="*80)
print("  Speedup (S) and Efficiency (E)  —  Parallel Linear Regression (MPI)")
print("  Baseline: par_time_total and compute_time at P=1 for each N")
print("="*80)
header = f"{'N':>6}  {'':>2}" + "".join(f"  {p:>6}" for p in P_values)
print(header)
print("-"*80)
for n in N_values:
    sub = df[df["N"] == n]
    def val(col, p):
        rows = sub[sub["P"] == p]
        return f"{rows[col].values[0]:>6.2f}" if not rows.empty else f"{'—':>6}"
    print(f"{N_labels[n]:>6}  {'S':>2}" + "".join(f"  {val('speedup_compute',    p)}" for p in P_values))
    print(f"{'':>6}  {'E':>2}" + "".join(f"  {val('efficiency_compute', p)}" for p in P_values))

df.to_csv("speedup_efficiency_table.csv", index=False)
print("\nSaved: speedup_efficiency_table.csv")

# ── Helpers ───────────────────────────────────────────────────────────
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

# ── PLOT 1: Speedup ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for idx, n in enumerate(N_values):
    sub = df[df["N"] == n].sort_values("P")
    axes[0].plot(sub["P"], sub["speedup_compute"], marker="o", color=COLORS[idx], label=f"N={N_labels[n]}")
    axes[1].plot(sub["P"], sub["speedup_total"],   marker="s", color=COLORS[idx], label=f"N={N_labels[n]}")
for ax in axes:
    ax.plot(P_values, P_values, "k--", linewidth=1, label="Ideal")
setup_ax(axes[0], "Speedup — Compute Only\n(T1_compute / compute_time)",
         "Number of Processes (P)", "Speedup (S)")
setup_ax(axes[1], "Speedup — Total (incl. comm)\n(T1_total / par_time_total)",
         "Number of Processes (P)", "Speedup (S)")
fig.suptitle("Speedup: Compute vs Total  —  MPI Linear Regression",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "plot_speedup.png")

# ── PLOT 2: Efficiency ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for idx, n in enumerate(N_values):
    sub = df[df["N"] == n].sort_values("P")
    ax.plot(sub["P"], sub["efficiency_compute"], marker="o", color=COLORS[idx], label=f"N={N_labels[n]}")
ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal")
setup_ax(ax, "Efficiency — Parallel Linear Regression (MPI)",
         "Number of Processes (P)", "Efficiency (E = S/P)")
save(fig, "plot_efficiency.png")

# ── PLOT 3: Comm vs Compute stacked bar ──────────────────────────────
large_Ns = [n for n in N_values if n >= 50_000_000]
fig, axes = plt.subplots(1, len(large_Ns), figsize=(5 * len(large_Ns), 5))
if len(large_Ns) == 1: axes = [axes]
for ax, n in zip(axes, large_Ns):
    sub = df[df["N"] == n].sort_values("P")
    ps  = sub["P"].values
    ct  = sub["compute_time"].values
    cm  = sub["comm_time"].values
    x   = np.arange(len(ps))
    ax.bar(x, ct, label="Compute", color="steelblue")
    ax.bar(x, cm, bottom=ct, label="Comm (Scatterv+Reduce)", color="tomato", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P={p}" for p in ps], fontsize=9)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title(f"N = {N_labels[n]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
fig.suptitle("Compute vs Communication Time  —  MPI Linear Regression",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "plot_comm_vs_compute.png")

# ── PLOT 4: Scalability ───────────────────────────────────────────────
fig, axes = plt.subplots(1, len(large_Ns), figsize=(5 * len(large_Ns), 4))
if len(large_Ns) == 1: axes = [axes]
for ax, n in zip(axes, large_Ns):
    sub = df[df["N"] == n].sort_values("P")
    t1  = df[(df["N"] == n) & (df["P"] == 1)]["par_time_total"].values[0]
    ax.plot(sub["P"], sub["par_time_total"], marker="s", color="steelblue", label="Parallel (total)")
    ax.plot(sub["P"], sub["compute_time"],   marker="o", color="seagreen",  label="Compute only")
    ax.axhline(t1, color="tomato", linestyle="--", label="Sequential (P=1)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("P", fontsize=11)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title(f"N = {N_labels[n]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
fig.suptitle("Execution Time vs Processes  —  MPI Linear Regression",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "plot_scalability.png")

# ── PLOT 5: Tablo ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
col_labels = ["N"] + [f"P={p}" for p in P_values]
table_data = []
for n in N_values:
    sub = df[df["N"] == n]
    S_row = [N_labels[n]] + [
        f"{sub[sub['P']==p]['speedup_compute'].values[0]:.2f}"
        if not sub[sub['P']==p].empty else "—" for p in P_values
    ]
    E_row = [""] + [
        f"{sub[sub['P']==p]['efficiency_compute'].values[0]:.2f}"
        if not sub[sub['P']==p].empty else "—" for p in P_values
    ]
    table_data.append(S_row)
    table_data.append(E_row)
row_labels = []
for n in N_values:
    row_labels.append("S")
    row_labels.append("E")
tbl = ax.table(cellText=table_data, colLabels=col_labels,
               rowLabels=row_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#1F4E79")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")
for i in range(len(N_values)):
    for j in range(-1, len(col_labels)):
        tbl[(i*2+1, j)].set_facecolor("#D6E4F0")
        tbl[(i*2+2, j)].set_facecolor("#FFFFFF")
fig.suptitle("Speedup (S) and Efficiency (E) — Compute Only",
             fontsize=13, fontweight="bold")
save(fig, "plot_table.png")

print("\nAll done.")