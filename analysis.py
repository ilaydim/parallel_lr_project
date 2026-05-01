"""
analysis.py
Speedup / Efficiency / Scalability analysis for Parallel Linear Regression (MPI).

Input:   timing_output.csv
  Columns: N, P, serial_time, compute_time, comm_time, par_time_total,
            speedup_compute, speedup_total, efficiency_compute, efficiency_total

Outputs:
  - speedup_efficiency_table.csv
  - plot_speedup.png
  - plot_efficiency.png
  - plot_scalability.png
  - plot_comm_vs_compute.png

Usage:
  python3 analysis.py [--csv timing_output.csv]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

# ── FIX 1: T_serial(N) — P=1 satırından al, tüm P'ler için sabit baseline ──
# Her N için yalnızca P=1 çalışmasının serial_time'ı kullanılır.
# Bu sayede P=1'de speedup=1.0, ders notlarıyla birebir uyumlu.
serial_baseline = {}
for n in N_values:
    p1 = df[(df["N"] == n) & (df["P"] == 1)]
    if not p1.empty:
        serial_baseline[n] = p1["serial_time"].values[0]
    else:
        serial_baseline[n] = df[df["N"] == n]["serial_time"].median()

# Speedup ve efficiency'yi sabit baseline ile yeniden hesapla
for n in N_values:
    t_s  = serial_baseline[n]
    mask = df["N"] == n
    df.loc[mask, "speedup_compute"]    = t_s / df.loc[mask, "compute_time"]
    df.loc[mask, "speedup_total"]      = t_s / df.loc[mask, "par_time_total"]
    df.loc[mask, "efficiency_compute"] = df.loc[mask, "speedup_compute"] / df.loc[mask, "P"]
    df.loc[mask, "efficiency_total"]   = df.loc[mask, "speedup_total"]   / df.loc[mask, "P"]

# ── print & save main table (compute only — ders notlarıyla uyumlu) ──
print("\n" + "="*80)
print("  Speedup (S) and Efficiency (E)  —  Parallel Linear Regression (MPI)")
print("  Baseline: serial_time at P=1 for each N")
print("="*80)
print()
print(f"  {'N':>6}  {'T_serial(P=1)':>14}")
for n in N_values:
    print(f"  {N_labels[n]:>6}  {serial_baseline[n]:>14.6f} s")
print()

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

# ── helpers ───────────────────────────────────────────────────────────
def setup_ax(ax, title, xlabel, ylabel, p_vals):
    ax.set_xscale("log", base=2)
    ax.set_xticks(p_vals)
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

# ── PLOT 1: Speedup (compute vs total) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for idx, n in enumerate(N_values):
    sub = df[df["N"] == n].sort_values("P")
    axes[0].plot(sub["P"], sub["speedup_compute"], marker="o", color=COLORS[idx],
                 label=f"N={N_labels[n]}")
    axes[1].plot(sub["P"], sub["speedup_total"],   marker="s", color=COLORS[idx],
                 label=f"N={N_labels[n]}")

for ax in axes:
    ax.plot(P_values, P_values, "k--", linewidth=1, label="Ideal")

setup_ax(axes[0], "Speedup — Compute Only\n(serial / compute_time)",
         "Number of Processes (P)", "Speedup (S)", P_values)
setup_ax(axes[1], "Speedup — Total (incl. comm)\n(serial / par_time_total)",
         "Number of Processes (P)", "Speedup (S)", P_values)

fig.suptitle("Speedup: Compute vs Total  —  MPI Linear Regression",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "plot_speedup.png")

# ── PLOT 2: Efficiency (compute only — ana grafik, ders notlarıyla uyumlu) ──
fig, ax = plt.subplots(figsize=(7, 5))

for idx, n in enumerate(N_values):
    sub = df[df["N"] == n].sort_values("P")
    ax.plot(sub["P"], sub["efficiency_compute"], marker="o", color=COLORS[idx],
            label=f"N={N_labels[n]}")

ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal")
setup_ax(ax, "Efficiency — Parallel Linear Regression (MPI)",
         "Number of Processes (P)", "Efficiency (E = S/P)", P_values)
save(fig, "plot_efficiency.png")

# ── PLOT 3: Comm vs Compute time (stacked bar) ───────────────────────
large_Ns = [n for n in N_values if n >= 50_000_000]
fig, axes = plt.subplots(1, len(large_Ns),
                          figsize=(5 * len(large_Ns), 5),
                          sharey=False)
if len(large_Ns) == 1:
    axes = [axes]

for ax, n in zip(axes, large_Ns):
    sub = df[df["N"] == n].sort_values("P")
    ps  = sub["P"].values
    ct  = sub["compute_time"].values
    cm  = sub["comm_time"].values
    x   = np.arange(len(ps))

    ax.bar(x, ct, label="Compute",               color="steelblue")
    ax.bar(x, cm, bottom=ct,
           label="Comm (Scatterv+Reduce)",        color="tomato", alpha=0.85)

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

# ── PLOT 4: Scalability — FIX 2: serial_baseline[n] kullan ──────────
fig, axes = plt.subplots(1, len(large_Ns),
                          figsize=(5 * len(large_Ns), 4),
                          sharey=False)
if len(large_Ns) == 1:
    axes = [axes]

for ax, n in zip(axes, large_Ns):
    sub = df[df["N"] == n].sort_values("P")
    # FIX 2: serial_t'yi sub'dan değil, serial_baseline dict'inden al
    serial_t = serial_baseline[n]
    ax.plot(sub["P"], sub["par_time_total"], marker="s", color="steelblue",
            label="Parallel (total)")
    ax.plot(sub["P"], sub["compute_time"],   marker="o", color="seagreen",
            label="Compute only")
    ax.axhline(serial_t, color="tomato", linestyle="--", label="Sequential")
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

print("\nAll done.")
