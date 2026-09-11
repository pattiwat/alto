"""Reproduce Block A's beta from the raw CSV, dumping every intermediate table.

`grey-box-surrogate.md` §2.2 states

    f(t) = f_ref * ( SP_act(t) / SP_ref ) ** beta      beta = 0.382 / 0.232

and cites control-gap §4.2-4.3. This script is the audit trail behind those two numbers.
It adds no analysis of its own: it imports the same modules the control-gap runner uses,
so the derivation document cannot drift away from the implementation. What it adds is
the **Layer-2 pairs table**, which `scripts/run_control_gap.py` computes in memory and
throws away - and which is the table the whole headline reduces to.

    python block-a-derivation/reproduce_beta.py

Writes to block-a-derivation/data/ only. Nothing outside this folder is touched.

The assertion at the end is the point: the recomputed estimate must equal the published
one in report/elasticities.csv to 1e-12, or this script fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src import excitation, io, masks  # noqa: E402

AHUS = (1, 2)
DATA_DIR = HERE / "data"

# The same points scripts/run_control_gap.py loads for the excitation analysis. The
# override points are needed by the mask, not by the estimator.
AHU_POINTS = (
    "frequency",
    "power",
    "static_pressure",
    "static_pressure_setpoint_read",
    "override_control",
    "auto_manual_control_mode_read",  # AHU-1 only; absent points are skipped
)
WEATHER = "outdoor_weather_station__drybulb_temperature"

TOLERANCE = 1e-12


def load(config: dict) -> pd.DataFrame:
    present = set(io.all_columns(config))
    columns = [c for ahu in AHUS for c in io.ahu_columns(ahu, AHU_POINTS) if c in present]
    columns += [WEATHER] if WEATHER in present else []
    return io.load(config, columns=columns)


def contributions(pairs: pd.DataFrame, sp_col: str, hz_col: str) -> pd.DataFrame:
    """Per-episode terms of the ratio of sums, with the running estimate beside them.

    Reading down `running_beta` is watching the headline assemble one episode at a time.
    The last row of that column IS the published number - there is no further step.
    """
    out = pd.DataFrame(
        {
            "episode": pairs["episode"],
            "start": pairs["start"],
            "hour": pairs["hour"],
            "n_treated": pairs["n_treated"],
            "n_control": pairs["n_control"],
            "sp_treated": pairs[f"{sp_col}__treated"],
            "sp_control": pairs[f"{sp_col}__control"],
            "dln_sp": pairs[f"{sp_col}__dln"],
            "hz_treated": pairs[f"{hz_col}__treated"],
            "hz_control": pairs[f"{hz_col}__control"],
            "dln_f": pairs[f"{hz_col}__dln"],
        }
    )
    out["episode_ratio"] = out["dln_f"] / out["dln_sp"]
    out["cum_dln_sp"] = out["dln_sp"].cumsum()
    out["cum_dln_f"] = out["dln_f"].cumsum()
    out["running_beta"] = out["cum_dln_f"] / out["cum_dln_sp"]
    return out


def candidate_anchors(
    df: pd.DataFrame, eligible: pd.Series, sp_col: str, hz_col: str, config: dict
) -> pd.DataFrame:
    """The three candidate (SP_ref, f_ref) pairs of the derivation document §6.

    SP_ref and f_ref are not defined anywhere in this repo - they appear only in
    grey-box-surrogate.md, and the code anchors on each step's own logged pair instead.
    A closed-loop surrogate needs a fixed anchor, so these are the defensible options,
    computed rather than argued.
    """
    tol = config["control_gap"]["excitation"]["min_step_change"][masks.signal_key(sp_col)]
    base = excitation.baseline_level(df[sp_col], eligible)
    at_baseline = ((df[sp_col] - base).abs() <= tol).fillna(False) & eligible

    sp, hz = df[sp_col][eligible], df[hz_col][eligible]
    return pd.DataFrame(
        [
            {
                "anchor": "A per-step (status quo in src/effects.py)",
                "sp_ref": np.nan,
                "f_ref": np.nan,
                "n_steps": int(eligible.sum()),
                "note": "each step's own logged (SP, Hz); no fixed pair exists",
            },
            {
                "anchor": "B modal baseline (recommended)",
                "sp_ref": base,
                "f_ref": float(df[hz_col][at_baseline].mean()),
                "n_steps": int(at_baseline.sum()),
                "note": "the operating point beta was measured around",
            },
            {
                "anchor": "C episode-mask median",
                "sp_ref": float(sp.median()),
                "f_ref": float(hz.median()),
                "n_steps": int(eligible.sum()),
                "note": "precedent at run_control_gap.py:270; mixes episode steps in",
            },
        ]
    )


def main() -> int:
    config = io.load_config()
    df = load(config)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    published = pd.read_csv(HERE.parent / "report" / "elasticities.csv")
    anchors, summary = [], []

    print(f"Loaded {len(df):,} steps x {df.shape[1]} points, {df.index[0]} -> {df.index[-1]}\n")

    for ahu in AHUS:
        p = f"ahu_b8_{ahu}__"
        sp_col, hz_col, kw_col = (
            p + "static_pressure_setpoint_read",
            p + "frequency",
            p + "power",
        )

        # The EPISODE mask: on-hours & no-outage & no-override, deliberately WITHOUT the
        # settling component. An episode is defined by a setpoint change, so a mask that
        # drops steps near setpoint changes would delete the subject. Settling is applied
        # per-signal inside excitation.py instead.
        eligible = masks.analysis_mask(df, ahu, config).keep

        episodes = excitation.find_setpoint_episodes(df[sp_col], eligible, config)
        pairs = excitation.paired_response(
            df, episodes, sp_col, [hz_col, kw_col], eligible, config
        )
        terms = contributions(pairs, sp_col, hz_col)

        episodes.to_csv(DATA_DIR / f"episodes_ahu{ahu}.csv", index=False)
        pairs.to_csv(DATA_DIR / f"pairs_ahu{ahu}.csv", index=False)
        terms.to_csv(DATA_DIR / f"contributions_ahu{ahu}.csv", index=False)

        result = excitation.fit_elasticity(
            pairs, sp_col, hz_col, config,
            assumed=config["control_gap"]["excitation"]["assumed_affinity_exponent"],
            label=f"AHU-{ahu} static pressure -> fan frequency",
        )

        # The arithmetic, done three ways, all of which must agree exactly.
        by_sums = terms["dln_f"].sum() / terms["dln_sp"].sum()
        by_means = terms["dln_f"].mean() / terms["dln_sp"].mean()
        row = published[
            (published["label"] == result.label) & (published["y"] == "frequency")
        ].iloc[0]

        print(f"--- AHU-{ahu} " + "-" * 62)
        print(f"  episodes detected      {len(episodes)}")
        print(f"  paired with a control  {len(pairs)}   ({terms['n_treated'].sum()} treated steps)")
        print(f"  baseline setpoint      {episodes['baseline_level'].iloc[0]:.3f} inWG (mode)")
        print(f"  sum dln SP             {terms['dln_sp'].sum():+.15f}")
        print(f"  sum dln f              {terms['dln_f'].sum():+.15f}")
        print(f"  beta = ratio of sums   {by_sums:.16f}")
        print(f"       = ratio of means  {by_means:.16f}")
        print(f"       = fit_elasticity  {result.estimate:.16f}")
        print(f"       = report/         {row['ratio_of_sums']:.16f}")
        print(f"  95% CI                 [{result.ci_low:.5f}, {result.ci_high:.5f}]")

        for name, value in (("ratio of sums", by_sums), ("ratio of means", by_means)):
            assert abs(value - result.estimate) < TOLERANCE, (
                f"AHU-{ahu}: {name} gives {value!r}, fit_elasticity gives "
                f"{result.estimate!r}. The document's arithmetic does not match the code."
            )
        assert abs(result.estimate - row["ratio_of_sums"]) < TOLERANCE, (
            f"AHU-{ahu}: recomputed {result.estimate!r} against published "
            f"{row['ratio_of_sums']!r}. report/elasticities.csv is stale, or an input "
            "moved underneath it."
        )
        print(f"  -> reproduces the published value to better than {TOLERANCE:g}\n")

        anchors.append(
            candidate_anchors(df, eligible, sp_col, hz_col, config).assign(ahu=ahu)
        )
        summary.append(
            {
                "ahu": ahu,
                "episodes_detected": len(episodes),
                "episodes_paired": len(pairs),
                "treated_steps": int(terms["n_treated"].sum()),
                "baseline_sp_inwg": float(episodes["baseline_level"].iloc[0]),
                "sum_dln_sp": float(terms["dln_sp"].sum()),
                "sum_dln_f": float(terms["dln_f"].sum()),
                "beta": float(result.estimate),
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "excludes_affinity_half": result.excludes_assumed,
            }
        )

    pd.DataFrame(summary).to_csv(DATA_DIR / "beta_summary.csv", index=False)
    anchor_table = pd.concat(anchors, ignore_index=True)
    anchor_table.to_csv(DATA_DIR / "candidate_anchors.csv", index=False)

    print("Candidate (SP_ref, f_ref) anchors - see README §6. None of these is currently")
    print("defined in the repo; the code anchors per-step and the symbols appear only in")
    print("grey-box-surrogate.md.\n")
    print(anchor_table.to_string(index=False))
    print(f"\nWrote 8 tables to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
