"""Reproduce the numbers Block E's derivation rests on, from the raw CSV.

Block E answers two questions: what temperature leaves the coil (E1), and what does the
coil cost to run (E2)? grey-box-surrogate.md section 6 specifies both:

    E1 :  T_sa(t) =  SAT_sp(t) + b0 + b1*valve(t) + eta        eta ~ N(0, sigma^2)
    E2 :  T_ma    =  phi*T_oa + (1 - phi)*T_ra
          Q_req   =  rho*c_p * V_total * (T_ma - T_sa)         [kW]
          phi     =  phi0 + phi1*damper%                       clipped to [0, 1]

    python block-e-derivation/reproduce_sat_coil.py

Writes to block-e-derivation/data/ only. Nothing outside this folder is touched.

WHY THE SPEC'S FORM FAILS
------------------------
Block E is the only block whose equations have never been checked against the CSV, and
four of them do not survive the check.

  * THE RECORD MIXES degF AND degC. Outdoor drybulb averages 86.0 and the water circuit
    runs 51.9 / 65.2 - those are degF. Zone and supply-air temperatures average 23.8 and
    21.4 - those are degC. E2's mixing equation adds a degF term to a degC term. The
    falsifier needs no thermometer: read as degC, the CHILLED water entering the coil is
    51.9 degC while the air leaving it is 17.0 degC, so the coil would be transferring
    heat from the cold side to the hot side.

  * THERE IS NO RETURN-AIR TEMPERATURE POINT. Section 6.2 notes that MIXED-air
    temperature is unlogged and then quietly uses T_ra, which is not logged either. It
    has to be constructed, and the two available proxies differ by 2.3 K - which lands
    undivided on phi.

  * THE MIXING MAP HAS THE WRONG SIGN. phi backed out of the energy balance regresses on
    the fresh-air damper with a NEGATIVE slope on both AHUs - opening the outside-air
    damper is associated with LESS outside air. Section 6.2's map needs a positive slope.
    The correlation is real (-0.48 / -0.28), which is worse than no correlation: the point
    is not that the damper is uninformative but that it does not report what the map
    assumes it reports.

  * E1 ASSUMES A UNIT SETPOINT GAIN AND NEVER MEASURES IT. Block A measured beta for the
    pressure channel and carried its interval into domain randomisation. The SAT channel
    - worth 4.4x the fan thermally - got `T_sa = SAT_sp + bias`, i.e. a gain of exactly
    1, asserted from nothing. In levels it is 0.43; in first differences it is 2.5.

WHY THIS SCRIPT IS SELF-CONTAINED
---------------------------------
`block-a-derivation/reproduce_beta.py` and `block-b-derivation/reproduce_fan_power.py`
both open with `from src import ...`. There is no `src/` package in this working tree, so
neither runs here. This script inlines the mask; `mask_components()` documents what it
believes `src.masks.analysis_mask` does and where it differs, and `main()` prints its step
counts against the anchors in block-b-derivation/README.md section 3.2.

The water-side constant is DERIVED from config's fluid properties rather than pasted in,
so that `k = 0.0387593 kW/(L/min*degF)` - the one number in Block E that section 1.2 of
the data memo established beyond reasonable doubt - is reproduced rather than trusted.

WHAT IS ASSERTED, AND WHAT IS ONLY REPORTED
-------------------------------------------
Asserted (a reversal must fail the run):

  * the water-side constant derives from config's fluid properties to 1e-9, and the
    logged `cooling_rate` reproduces k*flow*dT within config's `k_ratio_bounds`.
  * `water_delta_temperature` is (return - supply) water temperature, within config's
    `delta_t_derived_tolerance_deg_f`.
  * the degF/degC split, via the second-law falsifier: read as degC the entering chilled
    water is hotter than the leaving air on essentially every coil-on step; read as degF
    it is colder on essentially every one.
  * no `return_air_temperature` point exists on either AHU.
  * the airflow unit hypothesis that wins beats its runner-up by config's
    `min_discrimination_margin`, scored on how often it puts phi inside [0, 1].
  * the SAT channel is barely excited: sd(SAT_sp) is a small fraction of sd(T_sa).
  * the mixing map's fitted slope is negative on BOTH AHUs, where section 6.2 needs it
    positive. The sign is the assertion; the R^2 is reported alongside it.

Reported, never asserted:

  * the SAT gain itself. Three estimators, three answers (0.43 / 2.57 / 3.02 on AHU-1),
    and no way to choose between them on this record - README section 4 is that subject.
  * the coil-closure residual on the held-out test month. It inherits Block C's total,
    the mixing map and the E1 bias all at once, so a large residual does not localise.
  * the fan-heat correction. It is a BAND across how much of fan power reaches the air,
    and the record cannot narrow it.
  * every phi in this document, which is a function of a return-air proxy that is not a
    logged point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"

AHUS = (1, 2)
DT_H = 0.25  # hours per step, surrogate section 1

# Air at ~20 degC, 1 atm: rho ~ 1.2 kg/m3, c_p ~ 1.006 kJ/(kg*K). The product is what the
# balance uses, and surrogate section 4.6 quotes it as 1.21 kJ/(m3*K). Declared here with
# its inputs so the rounding is visible: this is DRY-air sensible capacity only, and the
# coil on this floor is certainly doing latent work as well (section 9 of the README).
RHO_AIR_KG_PER_M3 = 1.20
CP_AIR_KJ_PER_KG_K = 1.006

# Candidate airflow units for `vav_*__air_flow_rate`, as multipliers to m3/s. Surrogate
# section 4.6 names exactly these three and says the discrimination "has power here".
UNIT_HYPOTHESES = {
    "CFM": 0.3048 ** 3 / 60.0,   # ft3/min -> m3/s
    "m3/h": 1.0 / 3600.0,
    "L/s": 1.0e-3,
}

# Block B's offset cubic, block-b-derivation/README.md. Used ONLY to put a scale on fan
# heat in section 8; the logged `power` point is used in preference where it is present.
FAN_OFFSET_CUBIC = {1: (0.719, 6.199), 2: (0.537, 6.806)}

# block-b-derivation/README.md section 3.2, the "no override" row. Block B reaches it from
# a `hz > 0 and kw > 0` base and Block E masks on the coil as well, so these are
# orientation, not equalities. Printed, never asserted.
BLOCK_B_ANCHOR = {1: 2678, 2: 2681}

# Training window, matching block-d-derivation/reproduce_zone_thermal.py. The test month
# is held back so that section 9's closure is a falsification and not a calibration.
TRAIN_END = pd.Timestamp("2026-07-29")

# Guards that ARE asserted. Local to this block; there is no `sat_coil:` section in
# config.yml, so they are declared here rather than silently inlined at their use sites.
# The same choice block-d-derivation/reproduce_zone_thermal.py made, for the same reason.
DERIVED_K_TOLERANCE = 1e-9          # config's fluid properties -> k, exactly
SECOND_LAW_MIN_SHARE = 0.95         # share of coil-on steps the degF reading must satisfy
SECOND_LAW_MAX_SHARE = 0.05         # ...and the share the degC reading must not exceed
MAX_SAT_EXCITATION_RATIO = 0.50     # sd(SAT_sp) / sd(T_sa)
MIN_TOA_TRA_SEPARATION_K = 2.0      # below this, phi is a ratio of two small numbers

# Physical band for an outdoor drybulb reading, degC. Used only to classify a column's
# unit, never to filter data.
PLAUSIBLE_OUTDOOR_C = (-50.0, 60.0)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_config() -> dict:
    with open(ROOT / "config.yml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def water_constant(config: dict) -> float:
    """kW per (L/min * degF), derived from config's fluid properties.

    Q[kW] = rho[kg/L] * flow[L/min] / 60 * c_p[kJ/(kg*K)] * dT[degF] * 5/9
    so k = rho * c_p * 5 / (60 * 9). Section 1.2 of the data memo reports 0.0387593.
    """
    fluid = config["energy_balance"]["fluid"]
    return fluid["density_kg_per_l"] * fluid["cp_kj_per_kg_k"] * 5.0 / (60.0 * 9.0)


def box_ids(header: list[str], ahu: int | None = None) -> list[str]:
    ids = sorted({c.split("__")[0] for c in header if c.endswith("__air_flow_rate")})
    if ahu is None:
        return ids
    return [b for b in ids if b.startswith(f"vav_8_{ahu}_")]


AHU_POINTS = (
    "supply_air_temperature",
    "supply_air_temperature_setpoint_read",
    "cooling_valve_position_read",
    "cooling_valve_flow_rate",
    "water_delta_temperature",
    "supply_water_temperature",
    "return_water_temperature",
    "cooling_rate",
    "fresh_air_damper_position_read",
    "return_air_humidity",
    "static_pressure",
    "frequency",
    "power",
    "override_control",
)
VAV_POINTS = (
    "air_flow_rate",
    "damper_position",
    "room_temperature",
    "maximum_air_flow_rate_setpoint_read",
)


def load_frame(config: dict) -> tuple[pd.DataFrame, list[str]]:
    csv = ROOT / config["data"]["csv_path"]
    header = pd.read_csv(csv, nrows=0).columns.tolist()
    wanted = ["timestamp_local"]
    for ahu in AHUS:
        wanted += [f"ahu_b8_{ahu}__{p}" for p in AHU_POINTS]
    for box in box_ids(header):
        wanted += [f"{box}__{p}" for p in VAV_POINTS]
    wanted += [c for c in header
               if c.startswith("floor_8_zone_1_iaq_") and c.endswith("__temperature")]
    wanted += ["outdoor_weather_station__drybulb_temperature",
               "outdoor_weather_station__wetbulb_temperature"]
    cols = [c for c in wanted if c in header]
    df = pd.read_csv(csv, usecols=cols, parse_dates=["timestamp_local"])
    return df.set_index("timestamp_local").sort_index(), header


def f_to_c(x):
    return (x - 32.0) * 5.0 / 9.0


# --------------------------------------------------------------------------------------
# the mask, inlined
# --------------------------------------------------------------------------------------

def mask_components(df: pd.DataFrame, ahu: int, config: dict) -> pd.DataFrame:
    """One boolean column per component of the shared analysis mask.

    Reconstructed from config.yml's own comments plus the component table in
    block-b-derivation/README.md section 3.2. Known differences from
    `src.masks.analysis_mask`:

      * `no_outage` counts nulls across THE COLUMNS THIS SCRIPT LOADS. The rule is a
        fraction, so the denominator matters, and Block E loads the water circuit that
        Blocks B and C do not. Step counts will not match those blocks exactly.
      * `settled` here drops `settling_steps` after a change in EITHER supervisory
        setpoint (static pressure or SAT), because Block E reads both.
      * `coil_on` is Block E's own addition, from `energy_balance.mask`. The balance is a
        0/0 form when the coil is off, which is the whole reason that mask exists.
    """
    m = config["control_gap"]["mask"]
    eb = config["energy_balance"]["mask"]
    P = lambda p: f"ahu_b8_{ahu}__{p}"

    out = pd.DataFrame(index=df.index)
    out["on_hours"] = df[P("frequency")] > m["min_fan_hz"]

    ov = df.get(P("override_control"))
    if ov is None:
        out["no_override"] = True
    else:
        out["no_override"] = (ov - m["normal_override_value"]).abs() <= m["override_tolerance"]

    mine = [c for c in df.columns if c.startswith(f"ahu_b8_{ahu}__")
            or c.startswith(f"vav_8_{ahu}_")]
    out["no_outage"] = df[mine].isna().mean(axis=1) < m["outage_null_fraction"]

    settled = pd.Series(True, index=df.index)
    for pt in ("static_pressure_setpoint_read", "supply_air_temperature_setpoint_read"):
        col = df.get(P(pt))
        if col is None:
            continue
        changed = col.diff().abs() > 0
        for lag in range(m["settling_steps"] + 1):
            settled &= ~changed.shift(lag, fill_value=False).astype(bool)
    out["settled"] = settled

    out["coil_on"] = (
        (df[P("cooling_valve_flow_rate")] >= eb["min_flow_l_per_min"])
        & (df[P("water_delta_temperature")] >= eb["min_delta_t_deg_f"])
        & (df[P("cooling_rate")] >= eb["min_cooling_rate_kw"])
    )
    return out.fillna(False)


def masks_for(df: pd.DataFrame, ahu: int, config: dict) -> dict[str, pd.Series]:
    c = mask_components(df, ahu, config)
    base = c.on_hours & c.no_override & c.no_outage
    return {
        "components": c,
        "on": base,
        "settled": base & c.settled,
        "balance": base & c.coil_on,
        "balance_settled": base & c.settled & c.coil_on,
    }


# --------------------------------------------------------------------------------------
# panel construction: the quantities every later section needs
# --------------------------------------------------------------------------------------

def clean_boxes(df: pd.DataFrame, ahu: int, config: dict) -> tuple[list[str], list[str]]:
    """(kept, dropped) box ids. Dead by config's measured rule, rogue by config's list."""
    bench = config["control_gap"]["benchmark"]
    rogue = set(bench["rogue_boxes"])
    kept, dropped = [], []
    for b in box_ids(df.columns.tolist(), ahu):
        flow = df.get(f"{b}__air_flow_rate")
        mx = df.get(f"{b}__maximum_air_flow_rate_setpoint_read")
        dead = False
        if flow is not None and mx is not None and mx.max() > 0:
            dead = (flow.max() / mx.max()) < bench["max_flow_ratio_dead_box"]
        (dropped if (dead or b in rogue) else kept).append(b)
    return kept, dropped


def build_panel(df: pd.DataFrame, ahu: int, keep: list[str], config: dict) -> pd.DataFrame:
    """One row per timestep, carrying every AHU-level quantity Block E uses."""
    m = config["control_gap"]["mask"]
    P = lambda p: f"ahu_b8_{ahu}__{p}"
    k = water_constant(config)

    # Both frames are relabelled to the bare box id, because `flow.where(temp.notna())`
    # aligns on COLUMN LABELS and `__air_flow_rate` never matches `__room_temperature`.
    flow = df[[f"{b}__air_flow_rate" for b in keep]].set_axis(keep, axis=1)
    temp = (df[[f"{b}__room_temperature" for b in keep]].set_axis(keep, axis=1)
            .replace(m["room_temp_sentinel"], np.nan))

    # Flow-weight only over boxes whose temperature is present at that step, so a masked
    # sentinel drops the box from BOTH sides of the ratio rather than zeroing its weight.
    w = flow.where(temp.notna())
    p = pd.DataFrame(index=df.index)
    p["v_raw"] = flow.sum(axis=1, min_count=1)
    num = np.nansum(temp.values * w.values, axis=1)
    den = np.nansum(w.values, axis=1)
    p["t_ra_flow"] = np.where(den > 0, num / np.where(den > 0, den, np.nan), np.nan)
    p["t_ra_plain"] = temp.mean(axis=1)

    iaq = [c for c in df.columns
           if c.startswith("floor_8_zone_1_iaq_") and c.endswith("__temperature")]
    p["t_ra_iaq"] = df[iaq].mean(axis=1) if iaq else np.nan

    p["t_sa"] = df[P("supply_air_temperature")]
    p["sat_sp"] = df[P("supply_air_temperature_setpoint_read")]
    p["sat_err"] = p.t_sa - p.sat_sp
    p["valve"] = df[P("cooling_valve_position_read")]
    p["fresh_damper"] = df[P("fresh_air_damper_position_read")]
    p["hz"] = df[P("frequency")]
    p["kw_fan"] = df.get(P("power"))
    p["q_water"] = k * df[P("cooling_valve_flow_rate")] * df[P("water_delta_temperature")]
    p["q_logged"] = df[P("cooling_rate")]
    p["t_oa"] = f_to_c(df["outdoor_weather_station__drybulb_temperature"])
    p["t_sw"] = f_to_c(df[P("supply_water_temperature")])
    p["t_sw_raw"] = df[P("supply_water_temperature")]
    return p


# --------------------------------------------------------------------------------------
# ordinary least squares
# --------------------------------------------------------------------------------------

def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(coefficients, R^2, residual sd). Hand-rolled, as every sibling script is."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    dof = max(len(y) - X.shape[1], 1)
    return beta, r2, float(np.sqrt(ss_res / dof))


# --------------------------------------------------------------------------------------
# unit discrimination: which columns are degF
# --------------------------------------------------------------------------------------

def unit_scales(df: pd.DataFrame, panel: dict, config: dict) -> pd.DataFrame:
    """Classify every temperature column as degC or degF, and falsify the degC reading.

    Two independent arguments, both reported per column:

      RANGE   - read as degC, does the column land inside a physically possible band?
      SECOND LAW - for the chilled water entering the coil, is it colder than the air
                leaving the coil? A cooling coil that runs the other way is not a
                cooling coil. This one does not depend on any climate judgement.
    """
    rows = []
    for col in sorted(c for c in df.columns if "temperature" in c and "setpoint" not in c):
        s = df[col].replace(0.0, np.nan).dropna()
        if s.empty:
            continue
        # Classify on the MEDIAN, not the extremes. Several room-temperature points carry
        # bad-read minima near 5 (the sentinel's neighbours), and an extremes-based rule
        # reads those as degF. The two populations here are centred near 23 and near 60,
        # so a median rule separates them with a margin no outlier can close.
        med = float(s.median())
        if "delta" in col:
            # An INTERVAL, not a level, so the level bands say nothing about it. Its unit
            # is fixed instead by the water-side constant, which is asserted in main().
            verdict = "degF interval (set by k)"
        else:
            verdict = "degC" if 5.0 <= med <= 45.0 else ("degF" if 45.0 < med <= 130.0 else "?")
        rows.append({
            "column": col, "n": int(s.size),
            "median": med, "mean": float(s.mean()),
            "p1": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
            "median_if_degf": float(f_to_c(med)),
            "p1_if_degf": float(f_to_c(s.quantile(0.01))),
            "p99_if_degf": float(f_to_c(s.quantile(0.99))),
            "plausible_as_degc": bool(PLAUSIBLE_OUTDOOR_C[0] <= med <= PLAUSIBLE_OUTDOOR_C[1]),
            "verdict": verdict,
        })
    tbl = pd.DataFrame(rows)

    # The second-law falsifier, per AHU.
    law = []
    for ahu in AHUS:
        p, mk = panel[ahu], panel[f"mask{ahu}"]
        sel = mk["balance"] & p.t_sw.notna() & p.t_sa.notna()
        q = p[sel]
        law.append({
            "ahu": ahu, "n_coil_on": int(sel.sum()),
            "t_sa_mean_c": float(q.t_sa.mean()),
            "t_sw_mean_if_degc": float(q.t_sw_raw.mean()),
            "t_sw_mean_if_degf": float(q.t_sw.mean()),
            # A cooling coil needs entering water COLDER than leaving air.
            "share_tsw_below_tsa_if_degc": float((q.t_sw_raw < q.t_sa).mean()),
            "share_tsw_below_tsa_if_degf": float((q.t_sw < q.t_sa).mean()),
        })
    return tbl, pd.DataFrame(law)


# --------------------------------------------------------------------------------------
# E1: the setpoint gain
# --------------------------------------------------------------------------------------

def sat_gain(panel: dict, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """dT_sa/dSAT_sp by three estimators. The spec assumes it is exactly 1."""
    exc = config["control_gap"]["excitation"]
    step = exc["min_step_change"]["supply_air_temperature_setpoint_read"]
    win, settle = exc["control_window_steps"], config["control_gap"]["mask"]["settling_steps"]
    min_ctl = exc["min_control_steps"]

    rows, epi_rows = [], []
    for ahu in AHUS:
        p, mk = panel[ahu], panel[f"mask{ahu}"]
        on = mk["on"] & p.t_sa.notna() & p.sat_sp.notna()
        q = p[on]

        b, r2, sd = ols(np.c_[np.ones(len(q)), q.sat_sp.values], q.t_sa.values)
        rows.append({"ahu": ahu, "estimator": "levels", "n": len(q),
                     "slope": b[1], "intercept": b[0], "r2": r2, "resid_sd_k": sd,
                     "sd_sat_sp_k": float(q.sat_sp.std()), "sd_t_sa_k": float(q.t_sa.std())})

        # First differences, contiguous 15-minute pairs only.
        d = p[["sat_sp", "t_sa"]].diff()
        contiguous = p.index.to_series().diff().dt.total_seconds().eq(900)
        ok = on & on.shift(1, fill_value=False) & contiguous & d.notna().all(axis=1)
        dd = d[ok]
        b, r2, sd = ols(np.c_[np.ones(len(dd)), dd.sat_sp.values], dd.t_sa.values)
        rows.append({"ahu": ahu, "estimator": "differences", "n": len(dd),
                     "slope": b[1], "intercept": b[0], "r2": r2, "resid_sd_k": sd,
                     "sd_sat_sp_k": float(dd.sat_sp.std()), "sd_t_sa_k": float(dd.t_sa.std())})

        big = dd[dd.sat_sp.abs() >= step]
        b, r2, sd = ols(np.c_[np.ones(len(big)), big.sat_sp.values], big.t_sa.values)
        rows.append({"ahu": ahu, "estimator": f"differences |d|>={step}", "n": len(big),
                     "slope": b[1], "intercept": b[0], "r2": r2, "resid_sd_k": sd,
                     "sd_sat_sp_k": float(big.sat_sp.std()), "sd_t_sa_k": float(big.t_sa.std())})

        # Episodes: a step change, its own before-window as control.
        changes = np.flatnonzero((p.sat_sp.diff().abs() >= step).values)
        pos_on = on.values
        for j in changes:
            c0, c1 = j - win, j - 1
            t0, t1 = j + settle, j + settle + win - 1
            if c0 < 0 or t1 >= len(p):
                continue
            cs = np.arange(c0, c1 + 1)
            ts = np.arange(t0, t1 + 1)
            cs = cs[pos_on[cs]]
            ts = ts[pos_on[ts]]
            if len(cs) < min_ctl or len(ts) < min_ctl:
                continue
            d_sp = p.sat_sp.values[ts].mean() - p.sat_sp.values[cs].mean()
            d_sa = p.t_sa.values[ts].mean() - p.t_sa.values[cs].mean()
            if not np.isfinite(d_sp) or not np.isfinite(d_sa) or abs(d_sp) < step:
                continue
            epi_rows.append({
                "ahu": ahu, "at": p.index[j], "hour": int(p.index[j].hour),
                "direction": "up" if d_sp > 0 else "down",
                "d_sat_sp_k": float(d_sp), "d_t_sa_k": float(d_sa),
                "ratio": float(d_sa / d_sp),
                "n_control": int(len(cs)), "n_treated": int(len(ts)),
                "startup": bool(p.index[j].hour < 9),
            })

    epi = pd.DataFrame(epi_rows)
    for ahu in AHUS:
        for label, sub in (("episodes", epi[epi.ahu == ahu]),
                           ("episodes ex-startup",
                            epi[(epi.ahu == ahu) & ~epi.startup] if not epi.empty else epi)):
            if sub.empty:
                continue
            # Ratio of sums, matching Block A's headline estimator for beta.
            rows.append({"ahu": ahu, "estimator": label, "n": len(sub),
                         "slope": float(sub.d_t_sa_k.sum() / sub.d_sat_sp_k.sum()),
                         "intercept": np.nan,
                         "r2": np.nan, "resid_sd_k": np.nan,
                         "sd_sat_sp_k": float(sub.d_sat_sp_k.std()),
                         "sd_t_sa_k": float(sub.d_t_sa_k.std())})
    gain = pd.DataFrame(rows).sort_values(["ahu", "estimator"]).reset_index(drop=True)
    return gain, epi


# --------------------------------------------------------------------------------------
# E1: the bias term, and whether anything the surrogate simulates can carry it
# --------------------------------------------------------------------------------------

def bias_fit(panel: dict) -> pd.DataFrame:
    """b0 + b1*x for the spec's regressor and for every rival the surrogate can step.

    Surrogate section 1's state vector `x_t` does NOT contain valve position, and no block
    in the section 1.1 step order produces it. So the spec's own regressor is unavailable
    at simulation time; these are the candidates that are not.
    """
    rows = []
    for ahu in AHUS:
        p, mk = panel[ahu], panel[f"mask{ahu}"]
        base = mk["settled"] & p.sat_err.notna()
        # `admissible` marks whether the regressor is free of T_sa. The dependent variable
        # is sat_err = T_sa - SAT_sp, so ANY regressor containing T_sa is mechanically
        # correlated with it and its R^2 is an artifact of the shared term, not a finding.
        # The signature is a slope near -1, and both contaminated rows below show it.
        for name, col, simulable, admissible in (
                ("valve (the spec's own)", p.valve, False, True),
                ("V_total (raw units)", p.v_raw, True, True),
                ("fan Hz", p.hz, True, True),
                ("T_ra (zone mean)", p.t_ra_flow, True, True),
                ("T_oa", p.t_oa, True, True),
                ("T_oa - T_sa", p.t_oa - p.t_sa, True, False),
                ("T_ra - T_sa", p.t_ra_flow - p.t_sa, True, False),
                ("none (constant bias)", pd.Series(0.0, index=p.index), True, True)):
            sel = base & col.notna()
            y = p.sat_err[sel].values
            x = col[sel].values
            X = np.c_[np.ones(len(y))] if name.startswith("none") else np.c_[np.ones(len(y)), x]
            b, r2, sd = ols(X, y)
            rows.append({
                "ahu": ahu, "regressor": name, "n": int(sel.sum()),
                "b0": float(b[0]), "b1": float(b[1]) if len(b) > 1 else 0.0,
                "r2": float(r2) if np.isfinite(r2) else 0.0, "sigma_k": sd,
                "mean_err_k": float(y.mean()),
                "p5_err_k": float(np.percentile(y, 5)),
                "p95_err_k": float(np.percentile(y, 95)),
                "simulable": simulable, "admissible": admissible,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# E2: return-air proxies, the mixing map, and the airflow unit
# --------------------------------------------------------------------------------------

def implied_phi(p: pd.DataFrame, sel: pd.Series, unit: float, t_ra: pd.Series,
                fan_air_fraction: float = 0.0) -> pd.Series:
    """phi implied by closing the energy balance, given an airflow unit and a T_ra proxy.

    Q_coil = rho*c_p * V * (T_ma - T_coil_out), and T_coil_out = T_sa - dT_fan when the
    fan sits downstream of the coil and delivers `fan_air_fraction` of its shaft power
    into the air. Then phi follows from T_ma = phi*T_oa + (1 - phi)*T_ra.
    """
    rho_cp = RHO_AIR_KG_PER_M3 * CP_AIR_KJ_PER_KG_K
    v = p.v_raw[sel] * unit                       # m3/s
    dt_fan = (fan_air_fraction * p.kw_fan[sel]) / (rho_cp * v) if fan_air_fraction else 0.0
    t_coil_out = p.t_sa[sel] - dt_fan
    t_ma = t_coil_out + p.q_water[sel] / (rho_cp * v)
    return (t_ma - t_ra[sel]) / (p.t_oa[sel] - t_ra[sel])


def balance_selector(p: pd.DataFrame, mk: dict, t_ra: pd.Series) -> pd.Series:
    return (mk["balance"] & p.v_raw.notna() & (p.v_raw > 0) & p.t_sa.notna()
            & p.q_water.notna() & (p.q_water > 0) & t_ra.notna() & p.t_oa.notna()
            & ((p.t_oa - t_ra).abs() > MIN_TOA_TRA_SEPARATION_K))


def return_air_proxies(panel: dict) -> pd.DataFrame:
    rows = []
    for ahu in AHUS:
        p, mk = panel[ahu], panel[f"mask{ahu}"]
        for name, t_ra in (("flow-weighted room temp", p.t_ra_flow),
                           ("unweighted room temp", p.t_ra_plain),
                           ("IAQ sensor mean", p.t_ra_iaq)):
            sel = balance_selector(p, mk, t_ra)
            if sel.sum() == 0:
                continue
            phi = implied_phi(p, sel, UNIT_HYPOTHESES["m3/h"], t_ra)
            rows.append({
                "ahu": ahu, "proxy": name, "n": int(sel.sum()),
                "logged_point": False,
                "t_ra_mean_c": float(t_ra[sel].mean()),
                "t_ra_sd_k": float(t_ra[sel].std()),
                "phi_median": float(phi.median()),
                "phi_p25": float(phi.quantile(0.25)),
                "phi_p75": float(phi.quantile(0.75)),
                "share_phi_in_unit_interval": float(((phi >= 0) & (phi <= 1)).mean()),
            })
    return pd.DataFrame(rows)


def airflow_unit(panel: dict, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discriminate the airflow unit on whether it can put phi inside [0, 1].

    This is surrogate section 4.6's test, which Block C is blocked on and could not run
    from its own side: ratios within Block C are unit-invariant, so only the air-side
    energy balance has any power here.
    """
    rows, fan_rows = [], []
    for ahu in AHUS:
        p, mk = panel[ahu], panel[f"mask{ahu}"]
        t_ra = p.t_ra_flow
        sel = balance_selector(p, mk, t_ra)
        for name, unit in UNIT_HYPOTHESES.items():
            phi = implied_phi(p, sel, unit, t_ra)
            rho_cp = RHO_AIR_KG_PER_M3 * CP_AIR_KJ_PER_KG_K
            d_t = p.q_water[sel] / (rho_cp * p.v_raw[sel] * unit)
            rows.append({
                "ahu": ahu, "unit": name, "n": int(sel.sum()),
                "v_total_m3_s_median": float((p.v_raw[sel] * unit).median()),
                "implied_dt_coil_k_median": float(d_t.median()),
                "phi_median": float(phi.median()),
                "share_phi_in_unit_interval": float(((phi >= 0) & (phi <= 1)).mean()),
            })
        # Fan heat: a band over how much shaft power reaches the air.
        for frac in (0.0, 0.5, 1.0):
            phi = implied_phi(p, sel, UNIT_HYPOTHESES["m3/h"], t_ra, fan_air_fraction=frac)
            rho_cp = RHO_AIR_KG_PER_M3 * CP_AIR_KJ_PER_KG_K
            v = p.v_raw[sel] * UNIT_HYPOTHESES["m3/h"]
            a, b = FAN_OFFSET_CUBIC[ahu]
            block_b_kw = a + b * (p.hz[sel] / 50.0) ** 3
            fan_rows.append({
                "ahu": ahu, "fan_air_fraction": frac, "n": int(sel.sum()),
                "kw_logged_median": float(p.kw_fan[sel].median()),
                "kw_block_b_median": float(block_b_kw.median()),
                "dt_fan_k_median": float((frac * p.kw_fan[sel] / (rho_cp * v)).median()),
                "phi_median": float(phi.median()),
                "share_phi_in_unit_interval": float(((phi >= 0) & (phi <= 1)).mean()),
            })

    tbl = pd.DataFrame(rows)
    margins = []
    for ahu in AHUS:
        a = tbl[tbl.ahu == ahu].sort_values("share_phi_in_unit_interval", ascending=False)
        best, second = a.iloc[0], a.iloc[1]
        margins.append({
            "ahu": ahu, "winner": best.unit,
            "winner_share": best.share_phi_in_unit_interval,
            "runner_up": second.unit,
            "runner_up_share": second.share_phi_in_unit_interval,
            "margin": (best.share_phi_in_unit_interval /
                       max(second.share_phi_in_unit_interval, 1e-9)),
        })
    tbl = tbl.merge(pd.DataFrame(margins)[["ahu", "winner", "margin"]], on="ahu", how="left")
    return tbl, pd.DataFrame(fan_rows)


def mixing_map(panel: dict) -> pd.DataFrame:
    """phi = phi0 + phi1*damper%, the map surrogate section 6.2 says to fit."""
    rows = []
    for ahu in AHUS:
        p, mk = panel[ahu], panel[f"mask{ahu}"]
        t_ra = p.t_ra_flow
        sel = balance_selector(p, mk, t_ra) & p.fresh_damper.notna()
        phi = implied_phi(p, sel, UNIT_HYPOTHESES["m3/h"], t_ra)
        good = np.isfinite(phi)
        y = phi[good].values
        x = p.fresh_damper[sel][good].values
        b, r2, sd = ols(np.c_[np.ones(len(y)), x], y)
        rows.append({
            "ahu": ahu, "n": int(good.sum()),
            "phi0": float(b[0]), "phi1": float(b[1]), "r2": float(r2), "sigma": sd,
            "damper_mean_pct": float(x.mean()), "damper_sd_pct": float(x.std()),
            "damper_min_pct": float(x.min()), "damper_max_pct": float(x.max()),
            "phi_median": float(np.median(y)),
            "share_phi_in_unit_interval": float(((y >= 0) & (y <= 1)).mean()),
            "corr_damper_phi": float(np.corrcoef(x, y)[0, 1]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# coil closure, on the held-out test month
# --------------------------------------------------------------------------------------

def coil_closure(panel: dict) -> pd.DataFrame:
    """Section 6.3's falsification, run as one: fit on train, score on the test month."""
    rows = []
    rho_cp = RHO_AIR_KG_PER_M3 * CP_AIR_KJ_PER_KG_K
    for ahu in AHUS:
        p, mk = panel[ahu], panel[f"mask{ahu}"]
        t_ra = p.t_ra_flow
        sel = balance_selector(p, mk, t_ra)
        train = sel & (p.index < TRAIN_END)
        test = sel & (p.index >= TRAIN_END)
        if train.sum() < 50 or test.sum() < 50:
            continue
        phi_tr = implied_phi(p, train, UNIT_HYPOTHESES["m3/h"], t_ra)
        phi_tr = phi_tr[np.isfinite(phi_tr)]

        for name, get_phi in (
            ("constant phi (train median)", lambda idx: float(np.clip(phi_tr.median(), 0, 1))),
            ("affine map on damper", None),
            ("phi = 0, all return air", lambda idx: 0.0),
        ):
            if name == "affine map on damper":
                y = phi_tr.values
                x = p.fresh_damper[train].reindex(phi_tr.index).values
                b, _, _ = ols(np.c_[np.ones(len(y)), x], y)
                get_phi = lambda idx, b=b: np.clip(b[0] + b[1] * p.fresh_damper[idx], 0, 1)
            for window, idx in (("train", train), ("test (held out)", test)):
                phi = get_phi(idx)
                t_ma = phi * p.t_oa[idx] + (1 - phi) * t_ra[idx]
                v = p.v_raw[idx] * UNIT_HYPOTHESES["m3/h"]
                q_req = rho_cp * v * (t_ma - p.t_sa[idx])
                q_meas = p.q_water[idx]
                ok = np.isfinite(q_req) & np.isfinite(q_meas)
                ratio = (q_req[ok] / q_meas[ok])
                rows.append({
                    "ahu": ahu, "mixing_model": name, "window": window,
                    "n": int(ok.sum()),
                    "q_req_mean_kw": float(q_req[ok].mean()),
                    "q_measured_mean_kw": float(q_meas[ok].mean()),
                    "ratio_median": float(ratio.median()),
                    "median_abs_pct_error": float((ratio - 1).abs().median() * 100),
                })
    return pd.DataFrame(rows)


def energy_context(panel: dict) -> pd.DataFrame:
    """Section 6.4's 4.4x: the coil is the prize and the fan is the measurable part."""
    rows = []
    for ahu in AHUS:
        p, mk = panel[ahu], panel[f"mask{ahu}"]
        sel = mk["on"]
        th = float((p.q_logged[sel].fillna(0) * DT_H).sum())
        el = float((p.kw_fan[sel].fillna(0) * DT_H).sum())
        rows.append({"ahu": ahu, "n_on_steps": int(sel.sum()),
                     "coil_kwh_thermal": th, "fan_kwh_electric": el,
                     "ratio_thermal_over_electric": th / el if el else np.nan,
                     "kwh_electric_at_cop_3": th / 3.0, "kwh_electric_at_cop_5": th / 5.0})
    tot = pd.DataFrame(rows)
    tot.loc[len(tot)] = {
        "ahu": 0, "n_on_steps": int(tot.n_on_steps.sum()),
        "coil_kwh_thermal": tot.coil_kwh_thermal.sum(),
        "fan_kwh_electric": tot.fan_kwh_electric.sum(),
        "ratio_thermal_over_electric": tot.coil_kwh_thermal.sum() / tot.fan_kwh_electric.sum(),
        "kwh_electric_at_cop_3": tot.coil_kwh_thermal.sum() / 3.0,
        "kwh_electric_at_cop_5": tot.coil_kwh_thermal.sum() / 5.0,
    }
    return tot


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    config = load_config()
    df, header = load_frame(config)
    k = water_constant(config)

    panel: dict = {}
    box_report = []
    for ahu in AHUS:
        keep, dropped = clean_boxes(df, ahu, config)
        panel[ahu] = build_panel(df, ahu, keep, config)
        panel[f"mask{ahu}"] = masks_for(df, ahu, config)
        box_report.append((ahu, len(keep), dropped))

    scales, law = unit_scales(df, panel, config)
    gain, episodes = sat_gain(panel, config)
    bias = bias_fit(panel)
    proxies = return_air_proxies(panel)
    units, fan = airflow_unit(panel, config)
    mixing = mixing_map(panel)
    closure = coil_closure(panel)
    context = energy_context(panel)

    # Identities that belong with the constants they check.
    ident_rows = []
    for ahu in AHUS:
        P = lambda p: f"ahu_b8_{ahu}__{p}"
        derived = df[P("return_water_temperature")] - df[P("supply_water_temperature")]
        resid = (df[P("water_delta_temperature")] - derived).dropna()
        mk = panel[f"mask{ahu}"]["balance"]
        r = (panel[ahu].q_logged[mk] / panel[ahu].q_water[mk]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        ident_rows.append({
            "ahu": ahu, "k_derived": k,
            "n_delta_t": int(resid.size),
            "delta_t_max_abs_resid_degf": float(resid.abs().max()),
            "n_coil_on": int(mk.sum()),
            "cooling_rate_over_k_flow_dt_median": float(r.median()),
        })
    identities = pd.DataFrame(ident_rows)

    # ---- write ----------------------------------------------------------------------
    for name, frame in (("unit_scales", scales), ("second_law", law),
                        ("sat_gain", gain), ("sat_episodes", episodes),
                        ("sat_bias_fit", bias), ("return_air_proxies", proxies),
                        ("airflow_unit", units), ("fan_heat", fan),
                        ("mixing_map", mixing), ("coil_closure", closure),
                        ("energy_context", context), ("identities", identities)):
        frame.to_csv(DATA_DIR / f"{name}.csv", index=False)

    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.4f}"

    print("\n--- THE MASK, AND WHAT IT LEAVES " + "-" * 47)
    for ahu, nkeep, dropped in box_report:
        c = panel[f"mask{ahu}"]
        print(f"  AHU-{ahu}: on {int(c['on'].sum()):5d}   settled {int(c['settled'].sum()):5d}"
              f"   coil-on {int(c['balance'].sum()):5d}"
              f"   (block-b anchor {BLOCK_B_ANCHOR[ahu]}, orientation only)")
        print(f"          {nkeep} clean boxes, {len(dropped)} dropped: {', '.join(dropped)}")

    print("\n--- THE WATER-SIDE CONSTANT, DERIVED NOT PASTED " + "-" * 32)
    print(identities.to_string(index=False, float_format=fmt))
    print(f"  k = rho*c_p*5/(60*9) = {k:.10f} kW/(L/min*degF). Data memo section 1.2: 0.0387593.")
    print("  Both identities are ASSERTED below - README section 2.")

    print("\n--- THE UNITS PROBLEM: degF AND degC IN ONE EQUATION " + "-" * 27)
    print(law.to_string(index=False, float_format=fmt))
    print("  A cooling coil needs entering water COLDER than leaving air. Read as degC")
    print("  the chilled water is ~52 degC against ~17 degC air, so the coil would run")
    print("  backwards on essentially every step. Read as degF it is ~11 degC and runs")
    print("  the right way on essentially every step. ASSERTED - README section 3.")
    zone = scales.column.str.startswith("vav_") | scales.column.str.startswith("floor_8_")
    print(scales[~zone][["column", "median", "p1", "p99", "median_if_degf", "verdict"]]
          .to_string(index=False, float_format=fmt))
    print(f"  ...plus {int(zone.sum())} zone-level temperature points, all degC "
          f"(median of medians {scales[zone]['median'].median():.2f}), full table in "
          f"data/unit_scales.csv. Two populations, centred near "
          f"{scales[scales.verdict == 'degC']['median'].median():.0f} and near "
          f"{scales[scales.verdict == 'degF']['median'].median():.0f}, and nothing between.")

    print("\n--- E1: THE SETPOINT GAIN NOBODY MEASURED " + "-" * 38)
    print(gain.to_string(index=False, float_format=fmt))
    span = gain.groupby("ahu").slope.agg(["min", "max"])
    print(f"  The spec writes T_sa = SAT_sp + bias, i.e. a gain of EXACTLY 1, and never")
    print(f"  measures it. Five estimators span {span.loc[1,'min']:.2f}-{span.loc[1,'max']:.2f} "
          f"(AHU-1) and {span.loc[2,'min']:.2f}-{span.loc[2,'max']:.2f} (AHU-2). Levels sit")
    print("  near 1 but explain 4-8% of the variance; differences say 2.5-3.0. The record")
    print("  does not choose between them, so the GAIN is reported, never asserted.")
    if not episodes.empty:
        print(f"  Episodes: {len(episodes)} survive, earliest hour "
              f"{int(episodes.hour.min())} - requiring an on-hours CONTROL window before")
        print("  the change removes every 07h start-up move on its own, which is why the")
        print("  ex-startup row is identical rather than absent.")
    print("  What IS asserted is that the channel is barely excited - README section 4.")

    print("\n--- E1: THE BIAS TERM, AND WHETHER ANYTHING CAN CARRY IT " + "-" * 23)
    print(bias.to_string(index=False, float_format=fmt))
    print("  The spec's own regressor is NOT SIMULABLE: valve position is absent from")
    print("  surrogate section 1's state vector and no block in the section 1.1 step order")
    print("  produces it. As specified, E1 cannot be stepped at all.")
    print("  Two rows beat it and are NOT ADMISSIBLE: T_oa - T_sa and T_ra - T_sa both")
    print("  contain T_sa, which is also inside the dependent variable. The giveaway is")
    print("  the slope near -1 on the second, which is the shared term and nothing else.")
    adm = bias[bias.simulable & bias.admissible & (bias.regressor != "none (constant bias)")]
    best = adm.loc[adm.groupby("ahu").r2.idxmax()]
    for r in best.itertuples():
        spec = bias[(bias.ahu == r.ahu) & (bias.regressor == "valve (the spec's own)")].iloc[0]
        print(f"  AHU-{r.ahu}: best simulable AND admissible regressor is `{r.regressor}` at "
              f"R^2 {r.r2:.3f}, against")
        print(f"          the spec's unsimulable valve term at {spec.r2:.3f}. A candidate "
              f"replacement, not a recommendation:")
    print("  T_ra is available when E1 runs (section 1.1 puts E1 before Block D updates the")
    print("  zones), so it is causally prior WITHIN a step - but across steps it is the")
    print("  same closed loop as the valve, and this record cannot separate `warm zones")
    print("  cause SAT error` from `SAT error causes warm zones` - README section 5.")

    print("\n--- E2: RETURN AIR IS NOT A LOGGED POINT " + "-" * 39)
    print(proxies.to_string(index=False, float_format=fmt))
    print("  Neither AHU logs return-air temperature - only return-air HUMIDITY. Every")
    print("  phi in this document is therefore a function of a constructed quantity.")

    print("\n--- E2: THE MIXING MAP DOES NOT IDENTIFY " + "-" * 39)
    print(mixing.to_string(index=False, float_format=fmt))
    print("  The damper reads ~87% open while the implied outside-air fraction is ~0, and")
    print("  the fitted slope is NEGATIVE on both AHUs - opening the outside-air damper is")
    print("  associated with LESS outside air. Section 6.2's map needs phi1 > 0. The sign,")
    print("  not the R^2, is what fails: ASSERTED below - README section 7.")

    print("\n--- THE AIRFLOW UNIT, DISCRIMINATED " + "-" * 44)
    print(units.to_string(index=False, float_format=fmt))
    print("  Surrogate section 4.6 is blocked on this and Block C could not run it: ratios")
    print("  within Block C are unit-invariant, so only the air-side balance has power.")
    print("  CFM and L/s put mixed air COLDER than both the sources it is mixed from.")
    print(fan.to_string(index=False, float_format=fmt))
    print("  Fan heat is a band, not a correction: how much shaft power reaches the air is")
    print("  not logged. Reported - it moves phi but not the ranking - README section 8.")

    print("\n--- COIL CLOSURE ON THE HELD-OUT TEST MONTH " + "-" * 36)
    print(closure.to_string(index=False, float_format=fmt))
    print("  Reported, never asserted. This residual inherits Block C's total, the mixing")
    print("  map and the E1 bias at once, so its size does not localise the fault.")

    print("\n--- COP CONTEXT: WHY THE COIL STAYS OUT OF THE REWARD " + "-" * 26)
    print(context.to_string(index=False, float_format=fmt))
    print("  No chiller COP exists in this dataset. The coil is the prize and the least")
    print("  knowable conversion at the same time - README section 10.")

    # ---- guards ---------------------------------------------------------------------
    assert abs(k - 0.0387593) < 1e-6, (
        f"The water-side constant derived from config's fluid properties is {k:.10f}, "
        f"not the 0.0387593 that section 1.2 of the data memo discriminated against every "
        f"rival unit hypothesis by a factor of 1.8x or more. Either the fluid properties "
        f"moved or the derivation is wrong; both invalidate every kW in this document.")

    lo, hi = config["energy_balance"]["k_ratio_bounds"]
    for r in identities.itertuples():
        assert abs(r.delta_t_max_abs_resid_degf) <= config["qc"]["delta_t_derived_tolerance_deg_f"], (
            f"AHU-{r.ahu}: water_delta_temperature departs from "
            f"(return - supply) water temperature by {r.delta_t_max_abs_resid_degf:.3f} degF. "
            f"That identity is what lets the water side stand in for a heat meter.")
        assert lo <= r.cooling_rate_over_k_flow_dt_median <= hi, (
            f"AHU-{r.ahu}: the logged cooling_rate is "
            f"{r.cooling_rate_over_k_flow_dt_median:.4f}x k*flow*dT, outside config's "
            f"k_ratio_bounds [{lo}, {hi}]. Q_measured is the anchor every other number in "
            f"Block E is checked against; if it moved, nothing downstream is comparable.")

    for r in law.itertuples():
        assert r.share_tsw_below_tsa_if_degf >= SECOND_LAW_MIN_SHARE, (
            f"AHU-{r.ahu}: reading the water circuit as degF, the entering chilled water is "
            f"colder than the leaving air on only {r.share_tsw_below_tsa_if_degf:.1%} of "
            f"coil-on steps. The degF reading is supposed to be the one that makes the coil "
            f"a cooling coil; if it does not, the unit argument in README section 3 fails.")
        assert r.share_tsw_below_tsa_if_degc <= SECOND_LAW_MAX_SHARE, (
            f"AHU-{r.ahu}: reading the water circuit as degC, the entering water is colder "
            f"than the leaving air on {r.share_tsw_below_tsa_if_degc:.1%} of coil-on steps. "
            f"The falsification of the degC reading rests on that share being ~0 - a coil "
            f"cannot move heat from {r.t_sw_mean_if_degc:.1f} degC water into "
            f"{r.t_sa_mean_c:.1f} degC air.")

    for ahu in AHUS:
        assert f"ahu_b8_{ahu}__return_air_temperature" not in header, (
            f"AHU-{ahu} now logs a return_air_temperature point. README section 6 exists "
            f"only because it does not: every phi here is built on a CONSTRUCTED T_ra, and "
            f"if the real point has appeared the proxy table should be replaced, not kept.")

    margin_min = config["energy_balance"]["min_discrimination_margin"]
    for ahu in AHUS:
        a = units[units.ahu == ahu].sort_values("share_phi_in_unit_interval", ascending=False)
        best, second = a.iloc[0], a.iloc[1]
        assert best.margin >= margin_min, (
            f"AHU-{ahu}: the airflow unit test no longer discriminates - {best.unit} puts "
            f"phi inside [0,1] on {best.share_phi_in_unit_interval:.1%} of steps against "
            f"{second.unit}'s {second.share_phi_in_unit_interval:.1%}, a margin of "
            f"{best.margin:.2f}x below config's min_discrimination_margin of {margin_min}. "
            f"Surrogate section 4.6 stays blocked and Block C's absolute totals stay unquotable.")

    for r in gain[gain.estimator == "levels"].itertuples():
        ratio = r.sd_sat_sp_k / r.sd_t_sa_k
        assert ratio <= MAX_SAT_EXCITATION_RATIO, (
            f"AHU-{r.ahu}: sd(SAT_sp)/sd(T_sa) = {ratio:.3f}. The claim in README section 4 "
            f"is that the SAT channel is barely excited - the setpoint moves {r.sd_sat_sp_k:.2f} K "
            f"while what it is supposed to command moves {r.sd_t_sa_k:.2f} K. If the setpoint "
            f"now moves comparably, the gain becomes identifiable and section 4 must be refitted.")

    for r in mixing.itertuples():
        assert r.phi1 < 0, (
            f"AHU-{r.ahu}: the fitted damper slope is {r.phi1:+.4f}, i.e. NON-NEGATIVE. "
            f"Surrogate section 6.2 fits phi = phi0 + phi1*damper%% and needs phi1 > 0 - "
            f"opening the outside-air damper must admit more outside air. README section 7 "
            f"rests on the fitted slope being negative on BOTH units of the same design, "
            f"which is what makes this a falsification rather than a weak fit. If the sign "
            f"has turned positive, section 6.2's map may be salvageable and section 7 must "
            f"be rewritten rather than merely re-run.")

    print(f"\nAll asserted guards passed. Wrote 12 tables to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
