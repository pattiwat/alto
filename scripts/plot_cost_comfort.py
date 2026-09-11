"""Figures: operating cost and comfort temperature, as-operated record vs trained SAC agent.

    python scripts/plot_cost_comfort.py [--traces report/rl/traces.csv]

Reads `report/rl/traces.csv` (written by scripts/trace_policies.py) and writes PNGs to
`report/rl/figures/`.

BOTH ARMS RAN THE SAME DAYS UNDER THE SAME REPLAYED WEATHER. The tape supplies outdoor temperature,
wetbulb, occupancy, zone setpoints, the fan schedule and the outside-air damper; only the
supervisory action differs. Panel (a) of figure 1 exists to SHOW that rather than assert it.

Every figure carries a DIFFERENCE panel. The two arms differ by a few percent on quantities that
span a factor of two across the day, so two near-coincident lines would hide the very thing the
comparison is about. The difference is also the better-determined quantity: the COP band moves the
LEVEL of the electric column by tens of percent and the DIFFERENCE between two arms by very little.

Cost is shown as ELECTRIC ENERGY WITH A COP BAND, never as a point and never in currency: this
dataset contains no chiller COP (6.4) and no tariff. Under a flat tariff, operating cost is
proportional to the plotted electric kWh.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "rl" / "figures"

# dataviz reference palette, light mode. Categorical slots 1 and 2; text and surface as published.
C = {"as_operated": "#2a78d6", "sac": "#eb6834",
     "g36_ignore_top_2": "#9ea69c", "naive_cut_20pct": "#c2c6bf"}
LABEL = {"as_operated": "As operated (record)", "sac": "SAC agent",
         "g36_ignore_top_2": "G36 trim-and-respond", "naive_cut_20pct": "Naive 20% SP cut"}
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
ARMS = ["as_operated", "sac"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 160,
})


def style(ax, title=None, xlabel=None, ylabel=None, note=None, note_y=-0.185):
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if note:
        # Wrapped, because an unwrapped caption under a left panel runs straight into the caption
        # under the right one and the reader cannot tell which sentence belongs to which chart.
        ax.text(0.0, note_y, textwrap.fill(note, 72), transform=ax.transAxes, fontsize=8,
                color=INK3, va="top", linespacing=1.5)
    ax.tick_params(length=0)
    return ax


def pair(ax, x, ya, ys, marker=None, label=True, ls_over=(0, (4, 3))):
    """Draw the two arms so a small difference stays visible: the record as a thick translucent
    band underneath, the agent as a thin dashed line on top of it."""
    kw = dict(marker=marker, ms=5.0, mec=SURFACE, mew=1.2) if marker else {}
    ax.plot(x, ya, color=C["as_operated"], lw=3.6, alpha=0.45, solid_capstyle="round",
            label=LABEL["as_operated"] if label else None, zorder=3, **kw)
    ax.plot(x, ys, color=C[ARMS[1]], lw=1.8, ls=ls_over,
            label=LABEL[ARMS[1]] if label else None, zorder=4, **kw)


def binned(df, xcol, ycol, edges, min_n=8):
    """Mean of `ycol` in bins of `xcol`, dropping bins too thin to read."""
    g = df.groupby(pd.cut(df[xcol], edges, right=False), observed=True)[ycol]
    m, n = g.mean(), g.size()
    ok = (n >= min_n).values
    return np.array([iv.mid for iv in m.index])[ok], m.values[ok]


def hourly(df, col):
    g = df.groupby("hour")[col].mean()
    return g.index.to_numpy(), g.to_numpy()


def arm_label(name: str) -> str:
    """Readable name for a matrix arm: sac_b50_s1 -> 'SAC · budget 50% (seed 1)'.

    Derived rather than hand-listed so adding an arm to the sweep needs no edit here, and so the
    budget - the thing that actually distinguishes the arms - is on the figure instead of a
    filename stem.
    """
    if not name.startswith("sac"):
        return LABEL.get(name, name)
    parts, budget, seed = name.split("_"), None, None
    for p in parts[1:]:
        if p.startswith("b") and p[1:].isdigit():
            budget = p[1:]
        elif p.startswith("s") and p[1:].isdigit():
            seed = p[1:]
    out = "SAC" if budget is None else f"SAC · budget {budget}%"
    return out if seed is None else f"{out} (seed {seed})"


def headroom(ax, frac=0.22):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * (1 + frac))


def main() -> int:
    global FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(ROOT / "report" / "rl" / "traces.csv"))
    ap.add_argument("--outdir", default=str(FIG))
    ap.add_argument("--sac-label", default="sac",
                    help="which SAC arm figures 1-3 compare against the record; the frontier "
                         "figure uses every sac* arm in the traces")
    args = ap.parse_args()
    FIG = Path(args.outdir)
    FIG.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(args.traces)
    SAC = args.sac_label
    if SAC not in set(tr.policy):
        raise SystemExit(f"--sac-label {SAC!r} not in traces; have {sorted(set(tr.policy))}")
    ARMS[:] = ["as_operated", SAC]
    for p in tr.policy.unique():
        C.setdefault(p, C["sac"])
        LABEL.setdefault(p, arm_label(p))
    sc = tr[tr.scored].copy()
    # Two different day counts, and conflating them overstates the window: the tape hands over 19
    # test days, but weekends carry no occupied step and so no scored step at all.
    all_days = tr[tr.policy == "as_operated"].date.nunique()
    days = sc.date.nunique()
    span = f"{sc.date.min()} to {sc.date.max()}"
    sub_window = (f"{all_days}-day held-out test window ({span}); {days} of them carry scored "
                  f"(occupied, analysis-clean) steps")
    h0, h1 = float(sc.hour.min()), float(sc.hour.max())
    edges = np.arange(float(np.floor(sc.T_oa.min())), float(np.ceil(sc.T_oa.max())) + 1.0, 1.0)
    A, S = sc[sc.policy == "as_operated"], sc[sc.policy == SAC]
    Af, Sf = tr[tr.policy == "as_operated"], tr[tr.policy == SAC]

    def header(fig, title, sub):
        fig.suptitle(title, x=0.006, ha="left", fontsize=14, fontweight="bold", y=0.998)
        fig.text(0.006, 0.958, sub, ha="left", fontsize=9, color=INK2)

    # ================================================================ FIGURE 1: OPERATING COST
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 8.4))

    # (a) the identical-weather claim, shown
    ax = axs[0, 0]
    xa, ya = hourly(Af, "T_oa")
    xs, ys = hourly(Sf, "T_oa")
    pair(ax, xa, ya, ys)
    ax.axvspan(h0, h1, color=C["as_operated"], alpha=0.05, lw=0, zorder=0)
    ax.text((h0 + h1) / 2, ax.get_ylim()[0], " scored window ", ha="center", va="bottom",
            fontsize=8, color=INK3)
    ax.set_xticks([0, 6, 12, 18, 24])
    style(ax, "a. Same weather, by construction", "hour of day", "outdoor dry-bulb  (°C)",
          f"The two arms replay the same {all_days} days, so the curves coincide exactly — "
          f"every difference below is the controller.")
    headroom(ax, 0.18)
    ax.legend(loc="upper left", fontsize=8.5)

    # (b) operating cost against outdoor temperature - the comparison asked for
    ax = axs[0, 1]
    for p, d in (("as_operated", A), (SAC, S)):
        x, y3 = binned(d, "T_oa", "elec_cop3_kwh", edges)
        _, y5 = binned(d, "T_oa", "elec_cop5_kwh", edges)
        ax.fill_between(x, y5 * 4, y3 * 4, color=C[p], alpha=0.13, lw=0, zorder=2)
    x, a4 = binned(A, "T_oa", "elec_cop4_kwh", edges)
    _, s4 = binned(S, "T_oa", "elec_cop4_kwh", edges)
    pair(ax, x, a4 * 4, s4 * 4, marker="o")
    style(ax, "b. Operating cost against outdoor temperature", "outdoor dry-bulb  (°C)",
          "mean electric demand  (kW)",
          "Line = COP 4.  Shading = COP 5 (low) to COP 3 (high): no chiller COP exists in this "
          "record, so the level is a band.")
    headroom(ax, 0.20)
    ax.legend(loc="upper left", fontsize=8.5)

    # (c) the difference, which the COP band barely touches
    ax = axs[1, 0]
    d3 = binned(S, "T_oa", "elec_cop3_kwh", edges)[1] - binned(A, "T_oa", "elec_cop3_kwh", edges)[1]
    d5 = binned(S, "T_oa", "elec_cop5_kwh", edges)[1] - binned(A, "T_oa", "elec_cop5_kwh", edges)[1]
    dd = (s4 - a4) * 4
    ax.fill_between(x, d5 * 4, d3 * 4, color=C[SAC], alpha=0.18, lw=0, zorder=2)
    ax.plot(x, dd, color=C[SAC], lw=2.2, marker="o", ms=5.0, mec=SURFACE, mew=1.2, zorder=4)
    ax.axhline(0, color=INK3, lw=1.2, zorder=1)
    ax.text(ax.get_xlim()[0], 0, " the record ", ha="left", va="bottom", fontsize=8, color=INK3)
    # The title states what the data does, read off the data. Never hard-code the direction: the
    # sign of this curve is exactly the question the figure is asked to answer.
    dearer = dd > 0
    if dearer.all():
        claim = "SAC costs more than the record at every outdoor temperature"
    elif (~dearer).all():
        claim = "SAC costs less than the record at every outdoor temperature"
    else:
        claim = (f"SAC costs more than the record in {int(dearer.sum())} of "
                 f"{len(dd)} temperature bins")
    style(ax, f"c. {claim}", "outdoor dry-bulb  (°C)",
          "electric demand vs the record  (kW)",
          "Above the line is dearer than the record. The COP band shifts the LEVEL in (b) by tens "
          "of percent but barely moves this difference — which is why the difference, not the "
          "level, is the reportable number.")

    # (d) per-day totals, paired on identical days
    ax = axs[1, 1]
    tot = sc.groupby(["policy", "date"])[["elec_cop4_kwh", "fan_kwh"]].sum().reset_index()
    # A day's TOTAL is only comparable to another day's total if both days were scored for a
    # comparable number of steps. The last test day contributes a couple of steps; drawn as a bar
    # beside full days it reads as a collapse in consumption rather than as a short day.
    n_steps = sc[sc.policy == "as_operated"].groupby("date").size()
    full = n_steps[n_steps >= 0.5 * n_steps.median()].index
    part = [d for d in sorted(n_steps.index) if d not in set(full)]
    tot = tot[tot.date.isin(full)]
    dts = sorted(tot.date.unique())
    xi = np.arange(len(dts))
    w = 0.40
    for k, p in enumerate(ARMS):
        d = tot[tot.policy == p].set_index("date").loc[dts]
        ax.bar(xi + (k - 0.5) * w, d.elec_cop4_kwh.to_numpy(), w * 0.88, color=C[p],
               label=LABEL[p], zorder=3)
    ax.set_xticks(xi)
    ax.set_xticklabels([d[5:] for d in dts], rotation=90, fontsize=7)
    a_ = tot[tot.policy == "as_operated"].set_index("date").loc[dts].elec_cop4_kwh.to_numpy()
    s_ = tot[tot.policy == SAC].set_index("date").loc[dts].elec_cop4_kwh.to_numpy()
    worse = int((s_ > a_).sum())
    dropped = (f" {len(part)} short day ({', '.join(part)}, under half the usual scored steps) is "
               f"left out: its total is not comparable." if part else "")
    style(ax, "d. Every scored test day, side by side", "test day (month–day)",
          "electric energy at COP 4  (kWh)",
          f"SAC costs more than the record on {worse} of {len(dts)} days "
          f"(mean {100 * np.mean(s_ / a_ - 1):+.1f}% per day).{dropped}", note_y=-0.31)
    headroom(ax, 0.26)
    ax.legend(loc="upper center", ncol=2, fontsize=8.5)

    header(fig, "Operating cost: as-operated record vs trained SAC agent",
           f"Floor 8, {sub_window}. Identical weather, occupancy, zone setpoints and fan "
           f"schedule in both arms.")
    fig.tight_layout(rect=(0, 0.01, 1, 0.945), h_pad=4.2, w_pad=3.0)
    fig.savefig(FIG / "cost_as_operated_vs_sac.png", bbox_inches="tight")
    plt.close(fig)

    # ============================================================ FIGURE 2: COMFORT TEMPERATURE
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 8.4))

    # (a) zone temperature through the day
    ax = axs[0, 0]
    xa, ya = hourly(Af, "T_zone_mean")
    _, ys = hourly(Sf, "T_zone_mean")
    pair(ax, xa, ya, ys)
    _, yap = hourly(Af, "T_zone_p95")
    _, ysp = hourly(Sf, "T_zone_p95")
    pair(ax, xa, yap, ysp, label=False)
    _, tsp = hourly(Af, "T_sp_mean")
    ax.plot(xa, tsp + 0.5, color=INK3, lw=1.4, ls=(0, (1, 2.5)), zorder=5,
            label="mean setpoint + 0.5 K deadband")
    ax.axvspan(h0, h1, color=C["as_operated"], alpha=0.05, lw=0, zorder=0)
    j = int(np.argmin(np.abs(xa - 20.5)))       # evening, clear of both the legend and the shading
    ax.annotate("95th-percentile zone", (xa[j], yap[j]), textcoords="offset points",
                xytext=(0, 9), fontsize=8, color=INK3, ha="center")
    ax.annotate("floor mean", (xa[j], ya[j]), textcoords="offset points",
                xytext=(0, -16), fontsize=8, color=INK3, ha="center")
    ax.set_xticks([0, 6, 12, 18, 24])
    style(ax, "a. Zone temperature through the day", "hour of day", "zone temperature  (°C)",
          "Two curves per arm: the floor mean of 53 zones, and the 95th-percentile zone above it.")
    headroom(ax, 0.24)
    ax.legend(loc="upper left", fontsize=8.5)

    # (b) comfort temperature against outdoor temperature
    ax = axs[0, 1]
    x, am = binned(A, "T_oa", "T_zone_mean", edges)
    _, sm = binned(S, "T_oa", "T_zone_mean", edges)
    _, ap95 = binned(A, "T_oa", "T_zone_p95", edges)
    _, sp95 = binned(S, "T_oa", "T_zone_p95", edges)
    pair(ax, x, am, sm, marker="o")
    pair(ax, x, ap95, sp95, marker="o", label=False)
    _, tsp = binned(A, "T_oa", "T_sp_mean", edges)
    ax.plot(x, tsp + 0.5, color=INK3, lw=1.4, ls=(0, (1, 2.5)), zorder=5,
            label="setpoint + deadband")
    ax.annotate("95th-percentile zone", (x[-1], ap95[-1]), textcoords="offset points",
                xytext=(-2, 9), fontsize=8, color=INK3, ha="right")
    ax.annotate("floor mean", (x[-1], am[-1]), textcoords="offset points",
                xytext=(-2, -16), fontsize=8, color=INK3, ha="right")
    style(ax, "b. Comfort temperature against outdoor temperature", "outdoor dry-bulb  (°C)",
          "zone temperature  (°C)",
          "Occupied, analysis-clean steps only. Note how far the floor mean sits BELOW the "
          "setpoint line at every outdoor temperature — the floor is not struggling to cool, it "
          "is cooling well past the target.")
    headroom(ax, 0.22)
    ax.legend(loc="upper left", fontsize=8.5)

    # (c) the difference in kelvin - the panel that makes a 0.1 K story legible
    ax = axs[1, 0]
    ax.plot(x, sm - am, color=C[SAC], lw=2.2, marker="o", ms=5.0, mec=SURFACE, mew=1.2,
            zorder=4, label="floor mean")
    ax.plot(x, sp95 - ap95, color=C[SAC], lw=1.6, ls=(0, (4, 3)), marker="s", ms=4.5,
            mec=SURFACE, mew=1.2, zorder=4, label="95th-percentile zone")
    ax.axhline(0, color=INK3, lw=1.2, zorder=1)
    ax.text(ax.get_xlim()[0], 0, " the record ", ha="left", va="bottom", fontsize=8, color=INK3)
    dm, dmax = float(np.mean(sm - am)), float(np.max(np.abs(sm - am)))
    side = "cooler" if dm < 0 else "warmer"
    if ((sm - am) < 0).all():
        claim, why = "SAC runs the floor cooler everywhere", "cooler"
    elif ((sm - am) > 0).all():
        claim, why = "SAC runs the floor warmer everywhere", "warmer"
    else:
        claim, why = f"Zone temperature tracks the record to within {dmax:.2f} K", side
    style(ax, f"c. {claim}", "outdoor dry-bulb  (°C)",
          "zone temperature vs the record  (K)",
          f"On average SAC is {abs(dm):.2f} K {why} than the record, up to {dmax:.2f} K. On a "
          f"floor that overcools, running {why} is not a comfort loss — it is zones moving back "
          f"toward their setpoint, and it is where the electricity in figure 1 was saved.")
    ax.legend(loc="best", fontsize=8.5)

    # (d) WHERE the zones sit relative to their own setpoints - both directions.
    # The old version of this panel plotted `comfort_excess_sum_k`, which is one-sided and so
    # showed only the 12.7% of zone-steps that are too warm while saying nothing about the 51%
    # that are too cold. On a floor that overcools, that is the wrong half of the picture.
    ax = axs[1, 1]
    COLD, BAND, WARM = "#2a78d6", "#b8b7b1", "#e34948"     # diverging: cool <- neutral -> warm
    rows_ = []
    for p in ARMS:
        d = sc[sc.policy == p]
        tot_ = float(d.n_too_cold.sum() + d.n_in_band.sum() + d.n_too_warm.sum())
        rows_.append((LABEL[p], 100 * d.n_too_cold.sum() / tot_,
                      100 * d.n_in_band.sum() / tot_, 100 * d.n_too_warm.sum() / tot_))
    ypos = np.arange(len(rows_))[::-1]
    left = np.zeros(len(rows_))
    for seg, colr, name in ((1, COLD, "more than 0.5 K too cold"),
                            (2, BAND, "within deadband"),
                            (3, WARM, "more than 0.5 K too warm")):
        vals = np.array([r[seg] for r in rows_])
        ax.barh(ypos, vals, left=left, height=0.5, color=colr, label=name,
                edgecolor=SURFACE, linewidth=2.0, zorder=3)   # 2px surface gap between segments
        for y, v, l in zip(ypos, vals, left):
            if v >= 7:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color=INK if colr == BAND else SURFACE, fontweight="bold")
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows_], fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    cold_a, cold_s = rows_[0][1], rows_[1][1]
    # No xlabel: the segments are labelled with their own percentages and the legend sits where
    # an axis label would, so a "share of zone-steps (%)" caption would just collide with it.
    style(ax, "d. The floor is overcooled, not overheated", None, None,
          f"Share of zone-steps, every zone against its OWN setpoint. The record leaves "
          f"{cold_a:.0f}% of them more than 0.5 K too cold — that is energy spent cooling past "
          f"the target. SAC: {cold_s:.0f}%.", note_y=-0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8,
              handlelength=1.4, columnspacing=1.2)

    header(fig, "Comfort temperature: as-operated record vs trained SAC agent",
           f"Floor 8, {sub_window}. The rogue zones stay in the statistic — a starved zone is "
           f"still a zone.")
    fig.tight_layout(rect=(0, 0.01, 1, 0.945), h_pad=4.2, w_pad=3.0)
    fig.savefig(FIG / "comfort_as_operated_vs_sac.png", bbox_inches="tight")
    plt.close(fig)

    # ===================================================== FIGURE 3: the trade-off in one picture
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    # TWO-SIDED comfort, explicitly. `cost_comfort` carries whichever metric the tracing env was
    # configured with, and on the one-sided metric this figure says SAC is ~17% WORSE on comfort -
    # because warming an overcooled zone back toward its setpoint registers as "more too-warm"
    # while the far larger too-cold reduction is invisible. Naming the column removes the
    # ambiguity and puts the axis on the quantity this comparison is actually about.
    agg = sc.groupby("policy").agg(elec4=("elec_cop4_kwh", "sum"), elec3=("elec_cop3_kwh", "sum"),
                                   elec5=("elec_cop5_kwh", "sum"),
                                   comfort=("cost_comfort_two_sided", "sum"))
    b = agg.loc["as_operated"]
    pts = {}
    for p in ["naive_cut_20pct", "g36_ignore_top_2", "as_operated", SAC]:
        if p not in agg.index:
            continue
        r = agg.loc[p]
        dx = 100 * (r.comfort - b.comfort) / b.comfort
        dy = 100 * (r.elec4 - b.elec4) / b.elec4
        lo_ = 100 * (r.elec5 - b.elec5) / b.elec5
        hi_ = 100 * (r.elec3 - b.elec3) / b.elec3
        pts[p] = (dx, dy)
        big = p in ARMS
        ax.errorbar(dx, dy, yerr=[[dy - min(lo_, hi_)], [max(lo_, hi_) - dy]], fmt="o",
                    ms=12 if big else 8, color=C[p], ecolor=C[p], elinewidth=1.6, capsize=4,
                    mec=SURFACE, mew=1.6, zorder=4 if big else 3)
    ax.axhline(0, color=INK3, lw=1.1, zorder=1)
    ax.axvline(0, color=INK3, lw=1.1, zorder=1)
    # Margins BEFORE the labels, so a label on the rightmost point still has somewhere to sit.
    ax.margins(x=0.22, y=0.22)
    x0, x1 = ax.get_xlim()
    for p, (dx, dy) in pts.items():
        right = dx > (x0 + x1) / 2          # flip the label inward near the right-hand edge
        ax.annotate(LABEL[p], (dx, dy), textcoords="offset points",
                    xytext=(-13 if right else 13, -17 if p == SAC else 9),
                    ha="right" if right else "left", fontsize=9.5 if p in ARMS else 8.5,
                    color=INK if p in ARMS else INK2,
                    fontweight="bold" if p in ARMS else "normal")
    ax.text(0.985, 0.975, "worse on both", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK3)
    ax.text(0.015, 0.025, "better on both", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=INK3)
    style(ax, None, "setpoint-tracking cost vs the record  (%, lower is better)",
          "operating cost vs the record  (%, lower is better)",
          "Comfort here is the TWO-SIDED cost — distance outside the deadband in either "
          "direction, so overcooling counts. Error bars span the COP 3–5 band; the origin is the "
          "as-operated record.", note_y=-0.135)
    fig.suptitle("Cost against setpoint tracking, every controller", x=0.012, ha="left",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.012, 0.945, "Held-out test window, identical boundary conditions in every arm.",
             ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.925))
    fig.savefig(FIG / "tradeoff_as_operated_vs_sac.png", bbox_inches="tight")
    plt.close(fig)

    # ============================================ FIGURE 4: the electricity / tracking frontier
    # Only meaningful once more than one SAC arm exists - the comfort-budget sweep is what draws
    # the curve. With a single agent there is one point and no frontier, so the figure is skipped
    # rather than drawn misleadingly.
    sac_arms = sorted(p for p in sc.policy.unique() if p.startswith("sac"))
    if len(sac_arms) >= 2:
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        base = sc[sc.policy == "as_operated"]
        b_elec = float(base.elec_cop4_kwh.sum())
        b_track = float(base.track_abs_mean_k.mean())

        def point(p):
            d = sc[sc.policy == p]
            return (float(d.track_abs_mean_k.mean()),
                    100.0 * (float(d.elec_cop4_kwh.sum()) / b_elec - 1.0))

        # Reference controllers first, so the swept curve draws on top of them.
        # The two reference controllers land close together on both axes, so their labels are
        # pushed to opposite sides rather than both trailing right off the same point.
        for k, p in enumerate(("naive_cut_20pct", "g36_ignore_top_2")):
            if p in set(sc.policy):
                tx, ty = point(p)
                ax.plot(tx, ty, "o", ms=8, color=C[p], mec=SURFACE, mew=1.6, zorder=3)
                ax.annotate(LABEL[p], (tx, ty), textcoords="offset points",
                            xytext=(10, 7) if k == 0 else (-10, -13),
                            ha="left" if k == 0 else "right", fontsize=8.5, color=INK2)
        xs_ = [point(p)[0] for p in sac_arms]
        ys_ = [point(p)[1] for p in sac_arms]
        order = np.argsort(xs_)
        ax.plot(np.array(xs_)[order], np.array(ys_)[order], "-", color=C["sac"], lw=1.8,
                alpha=0.55, zorder=3)
        # The arms can land almost on top of each other - which is itself a finding - so labels are
        # fanned out vertically with leader lines instead of being stacked on the same few pixels.
        ax.plot(xs_, ys_, "o", ms=11, color=C["sac"], mec=SURFACE, mew=1.6, zorder=5)
        span = (max(ys_) - min(ys_)) or 1.0
        for k, (p, tx, ty) in enumerate(sorted(zip(sac_arms, xs_, ys_), key=lambda r: -r[2])):
            ax.annotate(LABEL.get(p, p), (tx, ty),
                        textcoords="offset points", xytext=(26, 16 - 15 * k),
                        ha="left", fontsize=8.5, color=INK, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=INK3, lw=0.7,
                                        shrinkA=0, shrinkB=7))
        ax.plot(b_track, 0.0, "o", ms=12, color=C["as_operated"], mec=SURFACE, mew=1.6, zorder=5)
        ax.annotate("As operated (record)", (b_track, 0.0), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.axhline(0, color=INK3, lw=1.1, zorder=1)
        ax.axvline(b_track, color=INK3, lw=1.1, ls=(0, (2, 3)), zorder=1)
        ax.margins(x=0.22, y=0.26)
        # No quadrant caption here: both axis labels already carry "lower is better", and the
        # fanned arm labels need the lower-left corner.
        # The caption states what the sweep DID, read off the sweep. On this record the arms
        # collapse onto one another, and saying "tightening the budget buys tracking" when the
        # data shows it buying nothing would be the figure lying about its own content.
        t_spread = max(xs_) - min(xs_)
        t_gain = b_track - float(np.mean(xs_))
        if t_spread < 0.1 * max(t_gain, 1e-9):
            swept = (f"Every budget setting lands in the same place — {t_spread:.3f} K apart, "
                     f"against a {t_gain:.2f} K gain over the record. Tightening the comfort "
                     f"budget bought no further tracking: the agent has already taken what "
                     f"pressure and supply-air trim can give, and the remaining error is not "
                     f"reachable with those two levers.")
        else:
            swept = ("Each SAC point is one comfort-budget setting; the slope between them is "
                     "what better tracking costs in electricity.")
        style(ax, None,
              "setpoint tracking: mean |zone temp − its setpoint|  (K, lower is better)",
              "electricity vs the record  (%, lower is better)",
              swept + " Both axes are measured on the held-out window under identical weather.",
              note_y=-0.15)
        fig.suptitle("What better setpoint tracking costs in electricity", x=0.012, ha="left",
                     fontsize=13, fontweight="bold", y=0.995)
        fig.text(0.012, 0.945, f"Floor 8, {sub_window}.", ha="left", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0.02, 1, 0.925))
        fig.savefig(FIG / "frontier_cost_vs_tracking.png", bbox_inches="tight")
        plt.close(fig)
    else:
        print(f"  (only {len(sac_arms)} SAC arm in traces - frontier figure skipped)")

    print(agg.to_string(float_format=lambda v: f"{v:.1f}"))
    for f in sorted(FIG.glob("*.png")):
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
