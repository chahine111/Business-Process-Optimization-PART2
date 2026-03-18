"""
src/generate_analysis_report.py

Task 1.2 — Comprehensive Analysis Report Generator.

Reads the simulation output CSVs produced by run_evaluation.py and generates
a multi-page PDF with:
    - Side-by-side metric comparison across all allocation methods
    - Cycle-time distribution plots (box plots)
    - Per-resource occupation heatmaps
    - Workforce optimisation impact (fired-employees scenario)
    - Written conclusions with sanity checks ("do the numbers make sense?")

All charts use real data from the simulated logs.  Missing config logs are
silently skipped so the script works even if not all configs have finished.

Usage
-----
    python src/generate_analysis_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from evaluation import (
    load_log,
    compute_cycle_times,
    compute_resource_occupation,
    compute_waiting_times,
    compute_all_metrics,
    compute_fairness,
    print_comparison_table,
)

# ── Palette ──────────────────────────────────────────────────────────────────
TUM_BLUE   = "#0065BD"
TUM_LIGHT  = "#64A0C8"
DARK_GRAY  = "#333333"
MID_GRAY   = "#666666"
LIGHT_BG   = "#F7F9FC"
WHITE      = "#FFFFFF"
GREEN      = "#27AE60"
AMBER      = "#E67E22"
RED        = "#C0392B"

CONFIG_COLORS = {
    "r_rma":     "#0065BD",
    "r_rra":     "#64A0C8",
    "r_shq":     "#27AE60",
    "kbatch":    "#E67E22",
    "park_song": "#8E44AD",
}
CONFIG_LABELS = {
    "r_rma":     "R-RMA\n(Random)",
    "r_rra":     "R-RRA\n(Round-Robin)",
    "r_shq":     "R-SHQ\n(Shortest Queue)",
    "kbatch":    "K-Batch\n(k=5)",
    "park_song": "Park & Song\n(Prediction)",
}

SIM_START = pd.Timestamp("2016-01-01", tz="UTC")
SIM_END   = pd.Timestamp("2017-01-01", tz="UTC")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_page_style(fig: plt.Figure) -> None:
    """Apply uniform background and tight layout."""
    fig.patch.set_facecolor(WHITE)
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.96])


def _title(ax: plt.Axes, text: str, sub: str = "") -> None:
    ax.set_title(text, fontsize=13, fontweight="bold", color=DARK_GRAY, pad=10)
    if sub:
        ax.annotate(
            sub, xy=(0.5, 1.01), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=8, color=MID_GRAY,
        )


def _header_bar(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    """Draw a TUM-blue header band at the top of a figure."""
    fig.add_axes([0, 0.93, 1, 0.07]).set_axis_off()
    fig.axes[-1].set_facecolor(TUM_BLUE)
    fig.axes[-1].text(
        0.02, 0.5, title,
        transform=fig.axes[-1].transAxes,
        ha="left", va="center",
        fontsize=14, fontweight="bold", color=WHITE,
    )
    if subtitle:
        fig.axes[-1].text(
            0.98, 0.5, subtitle,
            transform=fig.axes[-1].transAxes,
            ha="right", va="center",
            fontsize=9, color=WHITE, alpha=0.85,
        )


def _footer(fig: plt.Figure, page: int, total: int) -> None:
    fig.text(
        0.5, 0.01,
        f"Business Process Simulation — Task 1.2 Evaluation  |  Page {page}/{total}",
        ha="center", va="bottom", fontsize=7, color=MID_GRAY,
    )


def _bar_chart(
    ax: plt.Axes,
    configs: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    subtitle: str = "",
    best_is_low: bool = True,
    pct: bool = False,
) -> None:
    """Horizontal bar chart with best-value highlighted."""
    colors = [CONFIG_COLORS.get(c, MID_GRAY) for c in configs]
    bars = ax.barh(
        [CONFIG_LABELS.get(c, c) for c in configs],
        values,
        color=colors,
        edgecolor=WHITE,
        linewidth=0.6,
        height=0.55,
    )
    # Highlight the best bar
    best_idx = int(np.argmin(values) if best_is_low else np.argmax(values))
    bars[best_idx].set_edgecolor(GREEN)
    bars[best_idx].set_linewidth(2.5)

    for bar, v in zip(bars, values):
        label = f"{v * 100:.1f}%" if pct else f"{v:.2f}"
        ax.text(
            v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8, color=DARK_GRAY,
        )

    ax.set_xlabel(ylabel, fontsize=9, color=MID_GRAY)
    ax.invert_yaxis()
    ax.set_facecolor(LIGHT_BG)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=8)
    _title(ax, title, subtitle)
    if pct:
        ax.set_xlim(0, max(values) * 1.15)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_results(
    outputs_dir: Path,
    configs: List[str],
) -> Dict[str, Dict]:
    """
    Load metrics for every config whose output CSV exists.
    Returns only configs that were found.
    """
    results: Dict[str, Dict] = {}
    for config in configs:
        csv = outputs_dir / f"eval_{config}.csv"
        if csv.exists():
            print(f"  loading metrics for {config} …")
            results[config] = compute_all_metrics(
                str(csv), sim_start=SIM_START, sim_end=SIM_END
            )
        else:
            print(f"  [SKIP] {config}: no CSV found.")
    return results


def load_cycle_time_distributions(
    outputs_dir: Path,
    configs: List[str],
) -> Dict[str, pd.Series]:
    """Load raw per-case cycle times (hours) for box-plot comparisons."""
    dists: Dict[str, pd.Series] = {}
    for config in configs:
        csv = outputs_dir / f"eval_{config}.csv"
        if csv.exists():
            df = load_log(str(csv))
            dists[config] = compute_cycle_times(df)
    return dists


def load_resource_occupation_detail(
    outputs_dir: Path,
    config: str,
) -> Optional[pd.DataFrame]:
    """Load per-resource occupation for a single config (for heatmap)."""
    csv = outputs_dir / f"eval_{config}.csv"
    if not csv.exists():
        return None
    df = load_log(str(csv))
    return compute_resource_occupation(df, SIM_START, SIM_END)


# ══════════════════════════════════════════════════════════════════════════════
# Page Renderers
# ══════════════════════════════════════════════════════════════════════════════

def page_title(pdf: PdfPages, results: Dict, configs_found: List[str]) -> None:
    """Title / executive-summary page."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(TUM_BLUE)

    # Main title
    fig.text(0.5, 0.78, "Business Process Simulation", ha="center",
             fontsize=26, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.70, "Task 1.2 — Resource Allocation Evaluation",
             ha="center", fontsize=18, color=WHITE, alpha=0.9)
    fig.text(0.5, 0.62, "BPI 2017 Loan Application Process  |  Full-Year Simulation 2016",
             ha="center", fontsize=12, color=WHITE, alpha=0.75)

    # Quick summary box
    ax = fig.add_axes([0.10, 0.15, 0.80, 0.40])
    ax.set_facecolor("#003F75")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.88, "Executive Summary", ha="center", fontsize=13,
            fontweight="bold", color=WHITE, transform=ax.transAxes)

    if results:
        # Find best method for cycle time
        ct_vals  = {c: results[c]["avg_cycle_time_h"]   for c in configs_found}
        occ_vals = {c: results[c]["avg_resource_occ"]   for c in configs_found}
        fair_vals= {c: results[c]["fairness"]            for c in configs_found}

        best_ct   = min(ct_vals, key=ct_vals.get)
        best_fair = min(fair_vals, key=fair_vals.get)

        lines = [
            f"  Configs evaluated : {len(configs_found)}  ({', '.join(configs_found)})",
            f"  Simulation window : 2016-01-01 → 2017-01-01 (365 days)",
            f"  Best Avg Cycle Time : {CONFIG_LABELS[best_ct].replace(chr(10),' ')}  "
            f"({ct_vals[best_ct]:.1f} h)",
            f"  Avg Resource Occ.   : "
            f"{np.mean(list(occ_vals.values())) * 100:.1f}% across all methods",
            f"  Fairest Allocation  : {CONFIG_LABELS[best_fair].replace(chr(10),' ')}  "
            f"(deviation {fair_vals[best_fair]:.4f})",
        ]
        for i, line in enumerate(lines):
            ax.text(0.04, 0.72 - i * 0.14, line, ha="left", fontsize=9.5,
                    color=WHITE, alpha=0.9, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No simulation results found.\nRun run_evaluation.py first.",
                ha="center", va="center", fontsize=12, color=WHITE, alpha=0.7,
                transform=ax.transAxes)

    fig.text(0.5, 0.06, "Technical University of Munich  |  Chair of BPM",
             ha="center", fontsize=9, color=WHITE, alpha=0.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_metric_bars(pdf: PdfPages, results: Dict, configs: List[str],
                     page: int, total: int) -> None:
    """3-panel bar chart: avg cycle time / resource occupation / fairness."""
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(WHITE)

    ct_vals   = [results[c]["avg_cycle_time_h"]   for c in configs]
    occ_vals  = [results[c]["avg_resource_occ"]   for c in configs]
    fair_vals = [results[c]["fairness"]            for c in configs]

    _bar_chart(axes[0], configs, ct_vals,
               "Hours", "Avg Cycle Time",
               "↓ lower = faster cases", best_is_low=True)

    _bar_chart(axes[1], configs, occ_vals,
               "Fraction", "Avg Resource Occupation",
               "fraction of simulation window", best_is_low=False, pct=True)

    _bar_chart(axes[2], configs, fair_vals,
               "MAD", "Resource Fairness (weighted)",
               "↓ lower = more equal load", best_is_low=True)

    _header_bar(fig, "Metric Comparison — Basic Metrics",
                "Basic Metrics 1–3 (required by assignment)")
    _footer(fig, page, total)
    _set_page_style(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_advanced_metrics(pdf: PdfPages, results: Dict, configs: List[str],
                          page: int, total: int) -> None:
    """P90 cycle time / waiting time / throughput."""
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(WHITE)

    p90_vals  = [results[c]["p90_cycle_time_h"]   for c in configs]
    wait_vals = [results[c]["avg_waiting_time_h"]  for c in configs]
    tput_vals = [results[c]["throughput_per_day"]  for c in configs]

    _bar_chart(axes[0], configs, p90_vals,
               "Hours", "P90 Cycle Time",
               "↓ lower = better tail behaviour", best_is_low=True)

    _bar_chart(axes[1], configs, wait_vals,
               "Hours", "Avg Waiting Time",
               "↓ shorter queue wait", best_is_low=True)

    _bar_chart(axes[2], configs, tput_vals,
               "Cases / Day", "Throughput",
               "↑ higher = more productive", best_is_low=False)

    _header_bar(fig, "Metric Comparison — Advanced Metrics",
                "Advanced Metrics 4–6 (own contributions)")
    _footer(fig, page, total)
    _set_page_style(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_cycle_time_distributions(
    pdf: PdfPages,
    dists: Dict[str, pd.Series],
    configs: List[str],
    page: int,
    total: int,
) -> None:
    """Box plots of per-case cycle time for each method."""
    fig, (ax_box, ax_ecdf) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(WHITE)

    # ── Box plot ──────────────────────────────────────────────────────
    data   = [dists[c].dropna().values for c in configs if c in dists]
    labels = [CONFIG_LABELS.get(c, c) for c in configs if c in dists]
    colors = [CONFIG_COLORS.get(c, MID_GRAY) for c in configs if c in dists]

    bp = ax_box.boxplot(
        data, vert=True, patch_artist=True,
        medianprops=dict(color=WHITE, linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
        labels=labels,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax_box.set_ylabel("Cycle Time (hours)", fontsize=9)
    ax_box.set_facecolor(LIGHT_BG)
    ax_box.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_box.tick_params(axis="x", labelsize=8)
    _title(ax_box, "Cycle Time Distribution per Method",
           "Boxes = IQR, line = median, dots = outliers")

    # ── ECDF ─────────────────────────────────────────────────────────
    for c in configs:
        if c not in dists:
            continue
        vals = np.sort(dists[c].dropna().values)
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        ax_ecdf.plot(vals, ecdf, color=CONFIG_COLORS.get(c, MID_GRAY),
                     linewidth=1.8, label=CONFIG_LABELS.get(c, c).replace("\n", " "))

    ax_ecdf.axhline(0.9, color=DARK_GRAY, linestyle="--", linewidth=0.8,
                    label="P90 threshold")
    ax_ecdf.set_xlabel("Cycle Time (hours)", fontsize=9)
    ax_ecdf.set_ylabel("Cumulative Fraction of Cases", fontsize=9)
    ax_ecdf.set_facecolor(LIGHT_BG)
    ax_ecdf.grid(linestyle="--", linewidth=0.5, alpha=0.6)
    ax_ecdf.legend(fontsize=7, loc="lower right")
    _title(ax_ecdf, "Empirical CDF of Cycle Times",
           "Leftward shift = faster cases overall")

    _header_bar(fig, "Cycle Time Distributions", "All completed cases in 2016")
    _footer(fig, page, total)
    _set_page_style(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_resource_heatmap(
    pdf: PdfPages,
    outputs_dir: Path,
    configs: List[str],
    page: int,
    total: int,
) -> None:
    """Occupation heatmap: resources × configs."""
    # Collect occupation per resource per config
    all_resources: set = set()
    occ_by_config: Dict[str, pd.DataFrame] = {}
    for c in configs:
        odf = load_resource_occupation_detail(outputs_dir, c)
        if odf is not None:
            occ_by_config[c] = odf
            all_resources.update(odf["resource"].tolist())

    if not occ_by_config:
        return

    resources = sorted(all_resources)
    matrix = np.full((len(resources), len(configs)), np.nan)
    for j, c in enumerate(configs):
        if c not in occ_by_config:
            continue
        odf = occ_by_config[c].set_index("resource")
        for i, r in enumerate(resources):
            if r in odf.index:
                matrix[i, j] = odf.at[r, "occupation"]

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(WHITE)

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Occupation (0 = idle, 1 = always busy)",
                 fraction=0.03, pad=0.03)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(
        [CONFIG_LABELS.get(c, c).replace("\n", " ") for c in configs],
        rotation=20, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels(resources, fontsize=6.5)
    ax.set_xlabel("Allocation Method", fontsize=9)
    ax.set_ylabel("Resource (anonymised ID)", fontsize=9)

    _title(ax, "Resource Occupation Heatmap",
           "Per-resource occupation fraction across all allocation methods")

    _header_bar(fig, "Resource-Level Analysis",
                "Heatmap: each cell = fraction of simulation window the resource was busy")
    _footer(fig, page, total)
    _set_page_style(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_workforce_optimisation(
    pdf: PdfPages,
    outputs_dir: Path,
    baseline_results: Dict,
    page: int,
    total: int,
) -> None:
    """Fired-employees metric delta page."""
    fired_csv = outputs_dir / "eval_r_rma_fired2.csv"
    if not fired_csv.exists():
        return  # Skip if the fired-employees run hasn't happened yet

    fired_metrics = compute_all_metrics(
        str(fired_csv), sim_start=SIM_START, sim_end=SIM_END
    )
    base_metrics = baseline_results.get("r_rma", {})
    if not base_metrics:
        return

    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(WHITE)

    metrics_to_show = [
        ("avg_cycle_time_h",   "Avg Cycle Time (h)", True),
        ("avg_resource_occ",   "Avg Resource Occ",   False),
        ("throughput_per_day", "Throughput (c/day)", False),
    ]

    for ax, (key, label, low_good) in zip(axes, metrics_to_show):
        before = base_metrics.get(key, 0.0)
        after  = fired_metrics.get(key, 0.0)
        categories = ["Before\n(Baseline)", "After\n(−2 staff)"]
        vals = [before, after]

        if key == "avg_resource_occ":
            vals = [v * 100 for v in vals]
            label_sfx = "%"
        else:
            label_sfx = ""

        delta = vals[1] - vals[0]
        bar_colors = [TUM_BLUE, RED if (low_good and delta > 0) or (not low_good and delta < 0) else GREEN]
        bars = ax.bar(categories, vals, color=bar_colors, width=0.45,
                      edgecolor=WHITE, linewidth=0.7)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.2f}{label_sfx}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=DARK_GRAY)

        sign = "+" if delta > 0 else ""
        arrow_color = RED if (low_good and delta > 0) or (not low_good and delta < 0) else GREEN
        ax.annotate(
            f"{sign}{delta:.2f}{label_sfx}",
            xy=(0.5, max(vals) * 1.07), ha="center",
            fontsize=11, fontweight="bold", color=arrow_color, xycoords="data",
        )
        ax.set_ylabel(f"{label}", fontsize=9)
        ax.set_facecolor(LIGHT_BG)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_ylim(0, max(vals) * 1.20)
        _title(ax, label, "before vs after removing 2 resources")

    _header_bar(fig, "Workforce Optimisation — 'Fire Two Employees'",
                "Impact of removing the 2 least-contributing resources")
    _footer(fig, page, total)
    _set_page_style(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_conclusions(
    pdf: PdfPages,
    results: Dict,
    configs: List[str],
    page: int,
    total: int,
) -> None:
    """
    Written conclusions page:
    - What the numbers mean
    - Which method is recommended and why
    - Sanity checks (do the numbers make sense?)
    """
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(WHITE)

    ax = fig.add_axes([0.04, 0.06, 0.92, 0.85])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(WHITE)

    # ── Compute conclusion data ────────────────────────────────────────
    ct    = {c: results[c]["avg_cycle_time_h"]   for c in configs}
    p90   = {c: results[c]["p90_cycle_time_h"]   for c in configs}
    occ   = {c: results[c]["avg_resource_occ"]   for c in configs}
    fair  = {c: results[c]["fairness"]            for c in configs}
    wait  = {c: results[c]["avg_waiting_time_h"]  for c in configs}
    tput  = {c: results[c]["throughput_per_day"]  for c in configs}

    best_ct   = min(ct,   key=ct.get)
    worst_ct  = max(ct,   key=ct.get)
    best_fair = min(fair, key=fair.get)
    best_tput = max(tput, key=tput.get)

    # Compare Park & Song vs best simple heuristic
    simple = ["r_rma", "r_rra", "r_shq"]
    simple_avail = [c for c in simple if c in ct]
    ps_improvement = ""
    if "park_song" in ct and simple_avail:
        best_simple_ct = min(ct[c] for c in simple_avail)
        delta = ct["park_song"] - best_simple_ct
        sign  = "IMPROVED" if delta < 0 else "WORSENED"
        ps_improvement = (
            f"Park & Song {sign} avg cycle time by "
            f"{abs(delta):.2f} h vs best simple heuristic "
            f"({CONFIG_LABELS.get(min(simple_avail, key=lambda c: ct[c]),'').replace(chr(10),' ')})."
        )

    # Sanity checks
    sanity = []
    # Cycle times should be in range 1–500 h for a loan process
    mean_ct = np.mean(list(ct.values()))
    if 1 < mean_ct < 500:
        sanity.append(("✔", GREEN, f"Avg cycle time {mean_ct:.1f} h is plausible for a loan application process (typical: 1–3 weeks)."))
    else:
        sanity.append(("✘", RED, f"Avg cycle time {mean_ct:.1f} h seems unrealistic — check simulation parameters."))

    # Occupation should be 0.01–0.99
    mean_occ = np.mean(list(occ.values()))
    if 0.01 < mean_occ < 0.99:
        sanity.append(("✔", GREEN, f"Avg occupation {mean_occ * 100:.1f}% is plausible (resources not always 100% busy)."))
    else:
        sanity.append(("✔" if mean_occ < 1.0 else "✘",
                        AMBER if mean_occ > 0.9 else RED,
                        f"Occupation {mean_occ * 100:.1f}% — check availability model."))

    # R-SHQ should generally beat R-RMA on cycle time (shorter-queue → less wait)
    if "r_shq" in ct and "r_rma" in ct:
        if ct["r_shq"] <= ct["r_rma"] * 1.05:
            sanity.append(("✔", GREEN, "R-SHQ cycle time ≤ R-RMA: shortest-queue heuristic behaves as expected."))
        else:
            sanity.append(("?", AMBER, "R-SHQ is slower than R-RMA — may indicate queue-depth calculation edge case."))

    # K-batching should reduce cycle time vs random (batch decisions)
    if "kbatch" in ct and "r_rma" in ct:
        if ct["kbatch"] < ct["r_rma"]:
            sanity.append(("✔", GREEN, "K-Batch reduces avg cycle time vs R-RMA: batch allocation is beneficial."))
        else:
            sanity.append(("?", AMBER,
                "K-Batch does not reduce cycle time vs R-RMA — batching may be "
                "too aggressive (k=5) introducing extra waiting."))

    # ── Render text ────────────────────────────────────────────────────
    y = 0.97
    def write(text, size=9.5, color=DARK_GRAY, bold=False, indent=0.0):
        nonlocal y
        ax.text(indent, y, text, ha="left", va="top",
                fontsize=size, color=color,
                fontweight="bold" if bold else "normal",
                transform=ax.transAxes)
        # Estimate line height from font size
        y -= (size / 9.5) * 0.055

    write("Conclusions & Sanity Checks", size=14, bold=True, color=TUM_BLUE)
    y -= 0.02

    write("1. Which allocation method is best?", size=11, bold=True)
    y -= 0.005
    write(
        f"   Best avg cycle time : {CONFIG_LABELS.get(best_ct,'').replace(chr(10),' ')}  "
        f"({ct[best_ct]:.2f} h)",
        color=GREEN,
    )
    write(
        f"   Worst avg cycle time: {CONFIG_LABELS.get(worst_ct,'').replace(chr(10),' ')}  "
        f"({ct[worst_ct]:.2f} h)",
        color=RED,
    )
    write(
        f"   Best fairness       : {CONFIG_LABELS.get(best_fair,'').replace(chr(10),' ')}  "
        f"(deviation {fair[best_fair]:.4f})",
        color=TUM_BLUE,
    )
    write(
        f"   Best throughput     : {CONFIG_LABELS.get(best_tput,'').replace(chr(10),' ')}  "
        f"({tput[best_tput]:.2f} cases/day)",
        color=TUM_BLUE,
    )
    y -= 0.02

    write("2. Park & Song (prediction-based) vs simple heuristics:", size=11, bold=True)
    y -= 0.005
    if ps_improvement:
        write(f"   {ps_improvement}")
        write(
            "   Strategic idling reserves resources for predicted tasks, which can "
            "reduce wait time when predictions are accurate.",
            color=MID_GRAY,
        )
    y -= 0.02

    write("3. Sanity Checks — do the numbers make sense?", size=11, bold=True)
    y -= 0.005
    for icon, color, msg in sanity:
        write(f"   {icon}  {msg}", color=color)
        y -= 0.005

    y -= 0.02
    write("4. Recommendations for management:", size=11, bold=True)
    y -= 0.005
    write(
        f"   • For minimum cycle time, use {CONFIG_LABELS.get(best_ct,'').replace(chr(10),' ')}.",
        color=TUM_BLUE,
    )
    write(
        f"   • For workload balance, use {CONFIG_LABELS.get(best_fair,'').replace(chr(10),' ')} "
        f"(fairness = {fair[best_fair]:.4f}).",
        color=TUM_BLUE,
    )
    write(
        "   • Park & Song requires scipy and a next-activity predictor but may "
        "yield benefits when predictions are reliable.",
        color=MID_GRAY,
    )
    write(
        "   • The Deep RL method (not run here) requires a pre-trained policy "
        "(rl_train.py) and may outperform all heuristics after sufficient training.",
        color=MID_GRAY,
    )

    _header_bar(fig, "Conclusions & Sanity Checks", "Do the results make sense?")
    _footer(fig, page, total)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

EVAL_CONFIGS = ["r_rma", "r_rra", "r_shq", "kbatch", "park_song"]


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    outputs_dir  = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = outputs_dir / "task_1_2_analysis_report.pdf"

    print("\n" + "=" * 60)
    print("Task 1.2 — Comprehensive Analysis Report Generator")
    print("=" * 60)
    print(f"  Output: {out_pdf}\n")

    # ── Load data ─────────────────────────────────────────────────────
    results = load_all_results(outputs_dir, EVAL_CONFIGS)

    if not results:
        print("\n  ERROR: No simulation logs found in outputs/.")
        print("  Run: python src/run_evaluation.py   (trains models + runs all configs)")
        sys.exit(1)

    configs_found = [c for c in EVAL_CONFIGS if c in results]
    print(f"\n  Found results for: {configs_found}")

    # Print console table for quick review
    print_comparison_table(results)

    dists = load_cycle_time_distributions(outputs_dir, configs_found)

    # ── Count pages dynamically ───────────────────────────────────────
    total_pages = 6
    has_fired = (outputs_dir / "eval_r_rma_fired2.csv").exists()
    if not has_fired:
        total_pages -= 1

    # ── Generate PDF ──────────────────────────────────────────────────
    with PdfPages(str(out_pdf)) as pdf:
        p = 1

        print(f"  Rendering page {p}/{total_pages} — Title / Executive Summary …")
        page_title(pdf, results, configs_found)
        p += 1

        print(f"  Rendering page {p}/{total_pages} — Basic Metrics Bar Charts …")
        page_metric_bars(pdf, results, configs_found, p, total_pages)
        p += 1

        print(f"  Rendering page {p}/{total_pages} — Advanced Metrics Bar Charts …")
        page_advanced_metrics(pdf, results, configs_found, p, total_pages)
        p += 1

        print(f"  Rendering page {p}/{total_pages} — Cycle Time Distributions …")
        page_cycle_time_distributions(pdf, dists, configs_found, p, total_pages)
        p += 1

        print(f"  Rendering page {p}/{total_pages} — Resource Occupation Heatmap …")
        page_resource_heatmap(pdf, outputs_dir, configs_found, p, total_pages)
        p += 1

        if has_fired:
            print(f"  Rendering page {p}/{total_pages} — Workforce Optimisation …")
            page_workforce_optimisation(pdf, outputs_dir, results, p, total_pages)
            p += 1

        print(f"  Rendering page {p}/{total_pages} — Conclusions & Sanity Checks …")
        page_conclusions(pdf, results, configs_found, p, total_pages)

    print(f"\n  PDF saved → {out_pdf}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
