"""Reproduce Block B's fan-power curve from the raw CSV, dumping every intermediate.

`grey-box-surrogate.md` §3 states

    P_fan(f) = a + b * (f/50)**3      a, b = 0.719, 6.199 (AHU-1) / 0.537, 6.806 (AHU-2)

and cites §1.9 of the data memo. This script is the audit trail behind those numbers.
It adds no estimator of its own: it imports `src.effects.fit_fan_models`, the same
function `scripts/run_control_gap.py` calls, so the derivation document cannot drift
away from the implementation.

What it adds is the **normal-equations table**, which nothing on disk currently holds.
An offset cubic fitted by least squares reduces to five sums; shipping those five
numbers is what lets a reader recompute `a` and `b` by hand rather than take them on
trust. It also refits under four different row masks, because the reason three
different versions of `a` and `b` are in circulation turns out to be the mask, not the
estimator (README §7-8).

    python block-b-derivation/reproduce_fan_power.py

Writes to block-b-derivation/data/ only. Nothing outside this folder is touched.

Two assertions are the point:

  * the evaluation-mask fit must equal report/fan_models.csv to 1e-9, or this fails;
  * the `complete` mask component must be non-binding for the fan fit, which is what
    makes "settling costs Block B a third of its rows" a statement about settling
    alone.

The gap to grey-box-surrogate.md §3 is REPORTED, not asserted. Those numbers are not
reproducible under any mask in this repo - that is README §8's subject, and a script
that hard-failed on arrival would be a worse audit trail than one that shows the gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src import benchmarks, effects, io, masks  # noqa: E402

AHUS = (1, 2)
DATA_DIR = HERE / "data"

# The same points scripts/run_control_gap.py loads. `frequency` and `power` are the fit;
# the setpoints drive the settling component; the override points drive the mask.
AHU_POINTS = (
    "frequency",
    "power",
    "static_pressure_setpoint_read",
    "supply_air_temperature_setpoint_read",
    "override_control",
    "auto_manual_control_mode_read",  # AHU-1 only; absent points are skipped
)
VAV_POINTS = ("air_flow_rate", "maximum_air_flow_rate_setpoint_read")

TOLERANCE = 1e-9

# Transcribed from grey-box-surrogate.md §3 and its §8.1 parameter table. Held here so
# the comparison in the output is against the document as written, not against a
# remembered version of it. Reported, never asserted - see the module docstring.
PUBLISHED_IN_SPEC = {
    (1, "offset_cubic"): {"a": 0.719, "b": 6.199, "r2": 0.903, "n": 2679},
    (2, "offset_cubic"): {"a": 0.537, "b": 6.806, "r2": 0.960, "n": 2677},
    (1, "power_law"): {"a": np.nan, "b": 1.42, "r2": 0.655, "n": 2679},
    (2, "power_law"): {"a": np.nan, "b": 1.73, "r2": 0.793, "n": 2677},
}


def load(config: dict) -> pd.DataFrame:
    present = set(io.all_columns(config))
    columns = [c for ahu in AHUS for c in io.ahu_columns(ahu, AHU_POINTS) if c in present]
    columns += io.vav_columns(config, VAV_POINTS)
    return io.load(config, columns=columns)


def setpoint_columns(ahu: int) -> list[str]:
    """The signals whose changes the settling component excludes. Same as the runner."""
    return io.ahu_columns(
        ahu, ["static_pressure_setpoint_read", "supply_air_temperature_setpoint_read"]
    )


def build_masks(df: pd.DataFrame, ahu: int, config: dict) -> dict[str, pd.Series]:
    """The four row sets whose disagreement README §8 is about.

    `evaluation` is what the shipped numbers came from; it is `analysis` plus a
    completeness requirement on the priced inputs, including delivered airflow.
    `no_settling` removes the one component that has no business in a static fit, and
    `all_usable` is every row the fan was running for.
    """
    analysis = masks.analysis_mask(df, ahu, config, setpoint_columns(ahu))

    p = f"ahu_b8_{ahu}__"
    flow = benchmarks.delivered_airflow(df, benchmarks.qc_box_flows(df, ahu, config))
    needed = pd.DataFrame(
        {
            "hz": df[p + "frequency"],
            "kw": df[p + "power"],
            "sp": df[p + "static_pressure_setpoint_read"],
            "flow": flow,
        }
    )
    complete = needed.notna().all(axis=1) & (needed > 0).all(axis=1)
    evaluation = masks.AnalysisMask(
        f"AHU-{ahu} evaluation set",
        [*analysis.components, masks.MaskComponent("complete", complete, "priced inputs")],
    )

    return {
        "evaluation": evaluation.keep,
        "analysis": analysis.keep,
        "no_settling": masks.analysis_mask(df, ahu, config).keep,
        "all_usable": pd.Series(True, index=df.index),
    }


def mask_components(df: pd.DataFrame, ahu: int, config: dict) -> pd.DataFrame:
    """Each mask component as its own column, so a reader can see why a row was dropped."""
    analysis = masks.analysis_mask(df, ahu, config, setpoint_columns(ahu))
    return pd.DataFrame({c.name: c.keeps for c in analysis.components})


def normal_equations(f: np.ndarray, p: np.ndarray, rated: float) -> dict:
    """The sufficient statistics of all three fits.

    An offset cubic is an ordinary least-squares line of P on c = (f/50)^3, so it is
    five sums and two divisions:

        b = (n*Scp - Sc*Sp) / (n*Scc - Sc^2)
        a = (Sp - b*Sc) / n

    The ideal cubic is the same regression forced through the origin, b = Scp / Scc.
    The power law is the same line again, in logs. Nothing else is going on.
    """
    c = (f / rated) ** 3
    lf, lp = np.log(f), np.log(p)
    return {
        "n": len(f),
        "sum_c": float(c.sum()),
        "sum_c2": float((c * c).sum()),
        "sum_p": float(p.sum()),
        "sum_cp": float((c * p).sum()),
        "sum_lnf": float(lf.sum()),
        "sum_lnf2": float((lf * lf).sum()),
        "sum_lnp": float(lp.sum()),
        "sum_lnflnp": float((lf * lp).sum()),
    }


def solve_from_sums(row: dict) -> tuple[float, float]:
    """`a` and `b` recomputed from the shipped sums alone - the hand-check of §5."""
    n, sc, scc, sp, scp = row["n"], row["sum_c"], row["sum_c2"], row["sum_p"], row["sum_cp"]
    b = (n * scp - sc * sp) / (n * scc - sc * sc)
    return (sp - b * sc) / n, b


def usable(df: pd.DataFrame, ahu: int, keep: pd.Series) -> pd.DataFrame:
    """The rows `fit_fan_models` actually regresses: masked, non-null, both positive."""
    frame = pd.DataFrame(
        {"hz": df[f"ahu_b8_{ahu}__frequency"], "kw": df[f"ahu_b8_{ahu}__power"]}
    )[keep].dropna()
    return frame[(frame["hz"] > 0) & (frame["kw"] > 0)]


def binned_curve(df: pd.DataFrame, ahu: int, models: dict[str, effects.FanModel]) -> pd.DataFrame:
    """Measured power against each fitted model, per frequency bin.

    The first row is the fan-off population, which is the whole of README §2: the plant
    draws nothing at f = 0, while every fitted offset predicts 0.5-1.1 kW there. The
    remaining bins show where the fit is excellent (32-45 Hz) and where the models part
    company from the plant and from each other.
    """
    hz, kw = df[f"ahu_b8_{ahu}__frequency"], df[f"ahu_b8_{ahu}__power"]
    both = hz.notna() & kw.notna()

    off = both & (hz == 0)
    rows = [
        {
            "bin": "f == 0 (fan off)",
            "n": int(off.sum()),
            "f_mean": 0.0,
            "p_mean": float(kw[off].mean()),
            "p_median": float(kw[off].median()),
            "p_sd": float(kw[off].std()),
        }
    ]

    on = both & (hz > 0) & (kw > 0)
    edges = [0, 10, 20, 25, 30, 32, 34, 36, 38, 40, 42, 45, 50.1]
    grouped = pd.DataFrame({"f": hz[on], "p": kw[on]}).groupby(
        pd.cut(hz[on], edges), observed=True
    )
    for interval, g in grouped:
        rows.append(
            {
                "bin": str(interval),
                "n": len(g),
                "f_mean": float(g["f"].mean()),
                "p_mean": float(g["p"].mean()),
                "p_median": float(g["p"].median()),
                "p_sd": float(g["p"].std()),
            }
        )

    out = pd.DataFrame(rows)
    for name, model in models.items():
        out[f"pred_{name}"] = model.predict(out["f_mean"].to_numpy())
        out[f"resid_{name}"] = out["p_mean"] - out[f"pred_{name}"]
    return out


def local_elasticity(model: effects.FanModel, hz: float) -> float:
    """d ln P / d ln f of a fitted curve at one frequency.

    For the offset cubic this is 3bc/(a+bc) with c = (f/50)^3 - strictly below 3, and
    the more so the larger the offset. For the ideal cubic it is 3 by construction, and
    for the power law it is the exponent. This is the quantity a saving depends on,
    which is why README §9 compares it against the episodes rather than comparing R².
    """
    if model.kind == "power_law":
        return model.b
    c = (hz / model.rated_hz) ** 3
    return 3 * model.b * c / (model.a + model.b * c)


def main() -> int:
    config = io.load_config()
    df = load(config)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    shipped = pd.read_csv(HERE.parent / "report" / "fan_models.csv")
    elasticities = pd.read_csv(HERE.parent / "report" / "elasticities.csv")
    rated = float(config["control_gap"]["effects"]["rated_hz"])

    print(f"Loaded {len(df):,} steps x {df.shape[1]} points, {df.index[0]} -> {df.index[-1]}\n")

    model_rows, sum_rows, marginal_rows, saving_rows = [], [], [], []

    for ahu in AHUS:
        keeps = build_masks(df, ahu, config)
        fits: dict[str, dict[str, effects.FanModel]] = {}

        print(f"--- AHU-{ahu} " + "-" * 62)
        for mask_name, keep in keeps.items():
            frame = usable(df, ahu, keep)
            models = {m.kind: m for m in effects.fit_fan_models(df, ahu, keep, config)}
            fits[mask_name] = models

            sums = normal_equations(
                frame["hz"].to_numpy(), frame["kw"].to_numpy(), rated
            )
            a_hand, b_hand = solve_from_sums(sums)
            oc = models["offset_cubic"]

            # The five sums must rebuild the fitted coefficients, or README §5's
            # hand-check is a story rather than a derivation.
            assert abs(a_hand - oc.a) < TOLERANCE and abs(b_hand - oc.b) < TOLERANCE, (
                f"AHU-{ahu}/{mask_name}: normal equations give ({a_hand!r}, {b_hand!r}), "
                f"polyfit gives ({oc.a!r}, {oc.b!r})."
            )
            sum_rows.append({"ahu": ahu, "mask": mask_name, **sums,
                             "a_from_sums": a_hand, "b_from_sums": b_hand})

            for model in models.values():
                model_rows.append(
                    {"ahu": ahu, "mask": mask_name, "kind": model.kind, "name": model.name,
                     "a": model.a, "b": model.b, "rated_hz": model.rated_hz,
                     "r2": model.r2, "n": model.n, "formula": model.formula()}
                )

            print(f"  {mask_name:<12} n={oc.n:5,}  {oc.formula():<34} R2={oc.r2:.4f}"
                  f"  | power law n={models['power_law'].b:.3f}")

        # ---- assertion 1: the shipped CSV is reproducible from the raw data today
        for kind, model in fits["evaluation"].items():
            row = shipped[(shipped["ahu"] == ahu) & (shipped["kind"] == kind)].iloc[0]
            for field in ("a", "b", "r2"):
                assert abs(getattr(model, field) - row[field]) < TOLERANCE, (
                    f"AHU-{ahu} {kind}.{field}: recomputed {getattr(model, field)!r} "
                    f"against published {row[field]!r}. report/fan_models.csv is stale, "
                    "or an input moved underneath it."
                )
            assert model.n == row["n"], f"AHU-{ahu} {kind}: n {model.n} vs {row['n']}"

        # ---- assertion 2: `complete` drops no row the fan fit would have used, so the
        # 2678 -> 1807 loss belongs to settling alone (README §7).
        assert fits["evaluation"]["offset_cubic"].n == fits["analysis"]["offset_cubic"].n, (
            f"AHU-{ahu}: the completeness component is now binding on the fan fit "
            f"({fits['evaluation']['offset_cubic'].n} vs "
            f"{fits['analysis']['offset_cubic'].n}). README §7 attributes the row loss "
            "to settling alone and would need revisiting."
        )
        print(f"  -> reproduces report/fan_models.csv to better than {TOLERANCE:g}")
        print(f"  -> `complete` is non-binding; settling alone drops "
              f"{fits['no_settling']['offset_cubic'].n - fits['evaluation']['offset_cubic'].n:,} "
              f"of {fits['no_settling']['offset_cubic'].n:,} rows\n")

        # ---- the audit inputs, and the curve they imply
        frame = usable(df, ahu, keeps["all_usable"]).copy()
        frame["cube"] = (frame["hz"] / rated) ** 3
        components = mask_components(df, ahu, config)
        for name in components.columns:
            frame[f"mask_{name}"] = components[name].reindex(frame.index)
        for mask_name, keep in keeps.items():
            frame[f"in_{mask_name}"] = keep.reindex(frame.index)
        frame.to_csv(DATA_DIR / f"fit_inputs_ahu{ahu}.csv", index_label="timestamp")

        binned_curve(
            df, ahu,
            {"offset_cubic_shipped": fits["evaluation"]["offset_cubic"],
             "offset_cubic_all": fits["all_usable"]["offset_cubic"],
             "ideal_cubic_all": fits["all_usable"]["ideal_cubic"],
             "power_law_shipped": fits["evaluation"]["power_law"]},
        ).to_csv(DATA_DIR / f"binned_curve_ahu{ahu}.csv", index=False)

        # ---- level fit vs marginal response (README §9)
        hz_ref = float(df[f"ahu_b8_{ahu}__frequency"][keeps["evaluation"]].median())
        d_hz = float(elasticities[
            elasticities["label"] == f"AHU-{ahu} static pressure -> fan frequency"
        ]["mean_dln_y"].iloc[0])
        d_kw = float(elasticities[
            elasticities["label"] == f"AHU-{ahu} static pressure -> fan power"
        ]["mean_dln_y"].iloc[0])
        measured = d_kw / d_hz

        for mask_name, models in fits.items():
            for kind, model in models.items():
                implied = local_elasticity(model, hz_ref)
                marginal_rows.append(
                    {"ahu": ahu, "mask": mask_name, "kind": kind, "hz_ref": hz_ref,
                     "measured_d_ln_hz": d_hz, "measured_d_ln_kw": d_kw,
                     "measured_d_lnP_d_lnf": measured, "model_d_lnP_d_lnf": implied,
                     "gap": measured - implied}
                )

        print(f"  measured d lnP/d lnf = {measured:.3f} at {hz_ref:.2f} Hz "
              f"(ideal cube law = 3.000)")
        for kind in ("power_law", "offset_cubic", "ideal_cubic"):
            print(f"    {kind:<14} shipped fit implies "
                  f"{local_elasticity(fits['evaluation'][kind], hz_ref):.3f}")
        print()

        # ---- what the model choice is worth (README §10)
        for kind, model in fits["evaluation"].items():
            for exponent, source in (
                (float(elasticities[
                    elasticities["label"] == f"AHU-{ahu} static pressure -> fan frequency"
                ]["ratio_of_sums"].iloc[0]), "measured"),
                (config["control_gap"]["excitation"]["assumed_affinity_exponent"], "assumed"),
            ):
                s = effects.naive_cut_saving(
                    df, ahu, keeps["evaluation"], model, exponent, config,
                    exponent_source=source,
                )
                saving_rows.append(
                    {"ahu": ahu, "kind": kind, "model": s.model, "exponent": s.exponent,
                     "exponent_source": source, "cut_fraction": s.cut_fraction, "n": s.n,
                     "baseline_kwh": s.baseline_kwh,
                     "counterfactual_kwh": s.counterfactual_kwh,
                     "saving_kwh": s.saving_kwh, "saving_fraction": s.saving_fraction}
                )

    models_by_mask = pd.DataFrame(model_rows)
    models_by_mask.to_csv(DATA_DIR / "models_by_mask.csv", index=False)
    pd.DataFrame(sum_rows).to_csv(DATA_DIR / "normal_equations.csv", index=False)
    pd.DataFrame(marginal_rows).to_csv(DATA_DIR / "marginal_vs_level.csv", index=False)
    savings = pd.DataFrame(saving_rows)
    savings.to_csv(DATA_DIR / "savings_by_model.csv", index=False)

    # ---- the gap to the specification: reported, never asserted
    print("=" * 76)
    print("grey-box-surrogate.md sec.3 against every mask in this repo")
    print("=" * 76)
    for (ahu, kind), spec in PUBLISHED_IN_SPEC.items():
        stated = (
            f"a={spec['a']:.3f} b={spec['b']:.3f}" if np.isfinite(spec["a"])
            else f"exponent={spec['b']:.3f}"
        )
        print(f"\n  AHU-{ahu} {kind}  -  spec says "
              f"{stated} R2={spec['r2']:.3f} n={spec['n']:,}")
        sub = models_by_mask[(models_by_mask["ahu"] == ahu) & (models_by_mask["kind"] == kind)]
        for _, r in sub.iterrows():
            print(f"    {r['mask']:<12} a={r['a']:.3f} b={r['b']:.3f} "
                  f"R2={r['r2']:.3f} n={r['n']:,}")
    print("\n  No mask reproduces the specification's (a, b, R2, n) jointly. "
          "See README sec.8.")

    print(f"\nWrote 8 tables to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
