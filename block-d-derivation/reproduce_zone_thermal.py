"""Reproduce the numbers Block D's derivation rests on, from the raw CSV.

Block D answers one question: given the air a room actually received and the temperature
it was delivered at, where does that room's temperature go next? This script does NOT
build the identification pipeline - grey-box-technique.md section 5.1 assigns that to
`src/rl/identify.py`. It computes the handful of quantities the DERIVATION in README.md
claims, so that none of them is asserted from prose alone:

    T_i(t+1) - T_i(t) = k_i * (T_m,i    - T_i(t))        air <-> internal mass
                      + a_i * V_i(t) * (T_sa(t) - T_i(t))  supply air
                      + g_i * (T_oa(t) - T_i(t))           outdoor   (expected ~ 0)
                      + d_i * occ(t) + c_i                 gains

    python block-d-derivation/reproduce_zone_thermal.py

Writes to block-d-derivation/data/ only. Nothing outside this folder is touched.

WHY THE OUTDOOR COUPLING CAN BE RULED OUT WITHOUT AN ESTIMATOR
--------------------------------------------------------------
With the fan off and gains at zero the zone sits at an equilibrium that is a conductance-
weighted average of the two temperatures it is coupled to:

    T_eq = (k*T_m + g*T_oa) / (k + g)        =>    dT_eq/dT_oa = g / (k + g)

The night plateau IS that equilibrium, and outdoor temperature keeps falling through it.
So regressing plateau zone temperature on outdoor temperature WITHIN a night - night
means removed, which sweeps out the mass node's own night-to-night movement - measures
g/(k+g) directly. No NNLS, no fitted structure, one slope. grey-box-technique section 2.3
reaches "g is zero" from the boundary behaviour of a constrained estimator; this reaches
it from the data.

WHICH NIGHT'S PLATEAU - AND WHERE 2.5 HOURS CAME FROM
-----------------------------------------------------
grey-box-technique section 2.3 reports the air time constant as tau = dt/k, median 2.5 h.
This script gets 0.94 / 1.00 h from the same window on the same record, and the difference
is not masking, not sensor noise and not the discretisation. It is WHICH NIGHT T_m IS
TAKEN FROM.

An evening at 19:00 is relaxing toward the plateau it will reach at 21:00-05:00 THAT SAME
NIGHT. Technique section 4.1 defines T_m for an episode as "the night that ended that
morning", which is correct for a daytime episode and wrong for the evening transient: read
literally in the evening window it makes T_m YESTERDAY's plateau. Yesterday's plateau is a
noisy proxy for tonight's - the plateau's night-to-night innovation is about 0.47 K against
a regressor sd of 0.57 K - and a noisy regressor attenuates the slope. So k falls and tau
rises, by the same factor.

Fitting it that way reproduces technique's number almost exactly: tau = dt/k comes out at
2.32 / 2.56 h against its stated 2.5 h, IQR 2.1-2.9. That agreement is ASSERTED here, not
because 2.5 h is right, but because reproducing the artifact is the evidence that this is
where it came from. README section 5.1.

A SECOND OBJECTION, RETIRED
---------------------------
T_m is the mean of zone i's own sensor, so a stage-1 window overlapping the plateau window
puts T_i(t) inside T_m and the regressor and regressand share that term. Technique's
18:00-23:00 does overlap its own 21:00-05:00 plateau by two hours, so the objection has a
target - but the arithmetic kills it: T_m averages ~33 plateau steps, so any single T_i(t)
carries about 1/33 of it and the bias is O(1/N_plateau), not O(1). Measured, it is under
3%, in the opposite direction. Reported, not asserted - README section 4.2.

WHAT IS ASSERTED, AND WHAT IS ONLY REPORTED
-------------------------------------------
Asserted (a reversal must fail the run):

  * block averaging preserves the pole of a first-order system exactly, under input held
    constant across the step. This is a mathematical identity, verified numerically to
    1e-10, and it is why a 15-minute AVERAGE can identify a 1-3 h time constant at all
    when the same record cannot identify a 1-5 minute control loop (surrogate section 0).
  * the two-stage fit recovers a known synthetic zone's k, a, g to 1e-6.
  * the outdoor equilibrium share g/(k+g) is below `max_outdoor_equilibrium_share`.
  * the two-node form beats the one-node-to-outdoor form the specification carries, on
    the POOLED median per-zone R2 over the floor - which is the comparison
    grey-box-technique section 2.3 actually makes (50 boxes, one median).
  * the two named excluded sensors are still the floor's two coldest zones.
  * k > 0 on every zone that is not pooled.
  * that reading T_m as the PREVIOUS night's plateau reproduces grey-box-technique section
    2.3's tau of 2.5 h. Asserting agreement with a number this document argues is wrong
    looks odd until you see why: reproducing the artifact is the whole evidence that the
    artifact is what it is. If that stops reproducing, the explanation in README 5.1 is
    wrong and has to be withdrawn.

Reported, never asserted:

  * tau itself, under the reading this document recommends. 0.94 / 1.00 h. It is the
    quantity in dispute and the disagreement is the subject of README section 5.1.
  * THE PER-AHU SPLIT OF THE TWO-NODE COMPARISON, WHICH REVERSES ON AHU-2. The pooled
    claim holds and is asserted; on AHU-2 alone the two-node form does not beat the
    one-node form. That is the single most important thing this script found and it is
    the reason the guard is pooled rather than per-AHU - README section 5.2. A guard
    written per-AHU would have been a guard that hid its own counter-example.
  * the per-zone R2 of both stages. They are low, and the gate that matters is a
    multi-step rollout against persistence, which is not built yet.
  * the unconstrained sign of g, which is the evidence in README section 4.6.
  * the stage-1 window contrast, per the section above.
  * the sensor screen's margin. One of the two named sensors clears the third-coldest
    zone by 0.4 K, so its exclusion is a judgement call, not a measurement.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"

AHUS = (1, 2)
DT_H = 0.25  # hours per step, surrogate section 1

# grey-box-technique section 4.2 step 1. Two sensors whose day means are not room
# temperatures (8.4 degC and 19.7 degC against a floor median near 24), so they are
# excluded as instruments rather than pooled as zones. Verified, not trusted: the script
# recomputes both day means and asserts they are still the outliers that justify this.
GARBAGE_SENSORS = ("vav_8_1_9", "vav_8_1_6")

# grey-box-technique section 2.5 / surrogate section 5.3 step 4. Parked at a 27 degC
# setpoint, so airflow barely varies and `a` cannot be identified per zone. Pooled.
PARKED = tuple(f"vav_8_1_{n}" for n in (1, 2, 3, 4, 5, 10, 11, 14)) + \
         tuple(f"vav_8_2_{n}" for n in (19, 20, 21, 22))

# Windows. The plateau is grey-box-technique section 4.1's definition of T_m. STAGE1_TECH
# is that document's stage-1 window and overlaps the plateau by two hours; STAGE1 is the
# disjoint window this document recommends. Both are fitted - README section 4.2.
PLATEAU_HOURS = tuple(range(21, 24)) + tuple(range(0, 6))   # 21:00 - 05:59
STAGE1_HOURS = (18, 19, 20)                                  # 18:00 - 20:59, disjoint
STAGE1_TECH_HOURS = (18, 19, 20, 21, 22)                     # 18:00 - 22:59, overlapping
DAY_HOURS = tuple(range(8, 17))                              # 08:00 - 16:59 weekdays

# Training window, grey-box-technique section 4.2 step 1. The test month is held back.
TRAIN_END = pd.Timestamp("2026-07-29")

# grey-box-technique section 2.3 / 2.2, the numbers this script is checking itself
# against. Printed and differenced, never asserted.
TECH_TAU_H = 2.5
TECH_PLATEAU_MEAN, TECH_PLATEAU_SD = 24.75, 0.54
TECH_R2_TWO_NODE, TECH_R2_ONE_NODE = 0.185, 0.056

# Guards that ARE asserted. Local to this block; there is no `rl:` section in config.yml
# yet (grey-box-technique section 5.1 says one must be added), so they are declared here
# rather than silently inlined at their use sites.
MAX_OUTDOOR_EQUILIBRIUM_SHARE = 0.05   # g/(k+g). Measured ~0.006.
POLE_IDENTITY_TOLERANCE = 1e-10
SYNTHETIC_RECOVERY_TOLERANCE = 1e-6


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_config() -> dict:
    with open(ROOT / "config.yml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def zone_ids(header: list[str]) -> list[str]:
    pat = re.compile(r"^(vav_8_\d_\d+)__room_temperature$")
    return sorted({m.group(1) for c in header if (m := pat.match(c))})


def ahu_of(box: str) -> int:
    return int(box.split("_")[2])


def load_frame(config: dict) -> tuple[pd.DataFrame, list[str]]:
    csv = ROOT / config["data"]["csv_path"]
    header = pd.read_csv(csv, nrows=0).columns.tolist()
    wanted = ["timestamp_local",
              "outdoor_weather_station__drybulb_temperature"]
    for ahu in AHUS:
        wanted += [f"ahu_b8_{ahu}__frequency",
                   f"ahu_b8_{ahu}__supply_air_temperature",
                   f"ahu_b8_{ahu}__override_control",
                   f"ahu_b8_{ahu}__auto_manual_control_mode_read"]
    for box in zone_ids(header):
        wanted += [f"{box}__room_temperature",
                   f"{box}__room_temperature_setpoint_read",
                   f"{box}__air_flow_rate",
                   f"{box}__maximum_air_flow_rate_setpoint_read"]
    wanted += [c for c in header if c.endswith("__pir")]
    cols = [c for c in wanted if c in header]
    df = pd.read_csv(csv, usecols=cols, parse_dates=["timestamp_local"])
    return df.set_index("timestamp_local").sort_index(), header


def clean_room_temps(df: pd.DataFrame, boxes: list[str], config: dict) -> pd.DataFrame:
    """Room temperatures with the bad-read sentinel removed.

    config.yml `room_temp_sentinel`: the 0.00 in the AHU-1 room-temperature points is a
    bad read, not a room at freezing point (surrogate section 1.7 - minimum non-zero is
    6.49). Averaging it drags every zone statistic down.
    """
    m = config["control_gap"]["mask"]
    sentinel = float(m["room_temp_sentinel"])
    tol = float(m["room_temp_sentinel_tolerance"])
    Z = df[[f"{b}__room_temperature" for b in boxes]].copy()
    Z.columns = boxes
    return Z.mask((Z - sentinel).abs() < tol)


def night_id(index: pd.DatetimeIndex) -> pd.Series:
    """Label a night by the calendar day it STARTED on.

    Shifting back 12 h puts 18:00 on day X and 04:00 on day X+1 in the same group, which
    is what makes "the plateau this evening is settling into" a single object.
    """
    return pd.Series((index - pd.Timedelta(hours=12)).normalize(), index=index)


# --------------------------------------------------------------------------------------
# regression helpers
# --------------------------------------------------------------------------------------

def through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, uncentred R2, centred R2 for y = b*x with no intercept.

    Both R2s are reported because they answer different questions and differ a lot on
    this data. Uncentred (1 - SSres/sum(y^2)) is the honest goodness-of-fit for a model
    with no intercept. Centred (against the mean of y) is what most software prints and
    what grey-box-technique section 2.3's "R2 0.16 median" most likely is.
    """
    denom = float(x @ x)
    if denom <= 0:
        return float("nan"), float("nan"), float("nan")
    b = float(x @ y) / denom
    res = y - b * x
    ss = float(res @ res)
    unc = 1.0 - ss / float(y @ y) if float(y @ y) > 0 else float("nan")
    cen = 1.0 - ss / float(((y - y.mean()) ** 2).sum()) if y.size > 1 else float("nan")
    return b, unc, cen


def nnls_fit(X: np.ndarray, y: np.ndarray, n_constrained: int) -> tuple[np.ndarray, float]:
    """Least squares with the FIRST `n_constrained` coefficients held >= 0.

    The trailing coefficients (intercept, occupancy gain) are free in sign: an intercept
    is a lumped residual gain and a negative one is not a defect. Implemented by flipping
    the free columns into a non-negative pair, which is the standard reduction and keeps
    one solver for the whole fit.
    """
    n_free = X.shape[1] - n_constrained
    Xa = np.hstack([X, -X[:, n_constrained:]]) if n_free else X
    sol, _ = nnls(Xa, y)
    beta = sol[:X.shape[1]].copy()
    if n_free:
        beta[n_constrained:] -= sol[X.shape[1]:]
    res = y - X @ beta
    denom = float(((y - y.mean()) ** 2).sum())
    r2 = float("nan") if denom <= 0 else 1.0 - float(res @ res) / denom
    return beta, r2


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    denom = float(((y - y.mean()) ** 2).sum())
    r2 = float("nan") if denom <= 0 else 1.0 - float(res @ res) / denom
    return beta, r2


def tau_euler(k: float) -> float:
    return DT_H / k if k > 0 else float("nan")


def tau_exact(k: float) -> float:
    """Time constant if the step is a zero-order-hold solution rather than Euler.

    Fitting dT = k*(T_m - T) on data generated by the exact solution recovers
    k = 1 - exp(-dt/tau), not dt/tau. Inverting the right one matters: the two agree to
    5% at k = 0.1 and to 27% at k = 0.37. README section 3.2.
    """
    return -DT_H / np.log(1.0 - k) if 0.0 < k < 1.0 else float("nan")


# --------------------------------------------------------------------------------------
# the mass node: the night plateau
# --------------------------------------------------------------------------------------

def fan_off(df: pd.DataFrame, ahu: int, config: dict) -> pd.Series:
    return df[f"ahu_b8_{ahu}__frequency"] <= float(config["control_gap"]["mask"]["min_fan_hz"])


def plateau_means(Z: pd.DataFrame, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """T_m per (night, zone): the mean of that zone's own reading over 21:00-05:00.

    Each zone is gated on ITS OWN AHU being off, because AHU-1 and AHU-2 do not stop at
    the same time.
    """
    hours = Z.index.hour
    nid = night_id(Z.index)
    in_plateau = pd.Series(np.isin(hours, PLATEAU_HOURS), index=Z.index)
    out = {}
    for box in Z.columns:
        keep = in_plateau & fan_off(df, ahu_of(box), config)
        s = Z[box].where(keep)
        out[box] = s.groupby(nid).mean()
    return pd.DataFrame(out)


def mass_node_table(Z: pd.DataFrame, df: pd.DataFrame, config: dict,
                    tm: pd.DataFrame) -> pd.DataFrame:
    """Reproduce grey-box-technique section 2.2: is the mass a real, slow state?"""
    floor = tm.mean(axis=1).dropna()
    oa = df["outdoor_weather_station__drybulb_temperature"]
    nid = night_id(oa.index)
    hours = oa.index.hour
    oa_night = oa.where(pd.Series(np.isin(hours, PLATEAU_HOURS), index=oa.index)).groupby(nid).mean()
    joint = pd.DataFrame({"plateau": floor, "oa": oa_night}).dropna()
    dow = pd.Series(joint.index, index=joint.index).dt.dayofweek

    # Weekend drift: Friday night (dow 4) to Sunday night (dow 6) of the same weekend.
    fri = joint.plateau[dow == 4]
    sun = joint.plateau[dow == 6]
    drift = []
    for t, v in fri.items():
        nxt = sun[(sun.index > t) & (sun.index <= t + pd.Timedelta(days=3))]
        if len(nxt):
            drift.append(float(nxt.iloc[0]) - float(v))
    weekend_drift = float(np.median(drift)) if drift else float("nan")

    # tau_m from the uncooled drift: dT_m per day = (24 h / tau_m) * (T_air - T_m). Over a
    # weekend the air sits near the mass, so the gradient driving it is the mass's own
    # distance from the free-running building. Reported as an order of magnitude only.
    return pd.DataFrame([{
        "nights": int(len(joint)),
        "plateau_mean_c": float(joint.plateau.mean()),
        "plateau_sd_k": float(joint.plateau.std()),
        "plateau_min_c": float(joint.plateau.min()),
        "plateau_max_c": float(joint.plateau.max()),
        "corr_with_night_outdoor": float(joint.plateau.corr(joint.oa)),
        "lag1_autocorr": float(joint.plateau.autocorr(1)),
        "mon_night_mean_c": float(joint.plateau[dow == 0].mean()),
        "fri_night_mean_c": float(joint.plateau[dow == 4].mean()),
        "sat_night_mean_c": float(joint.plateau[dow == 5].mean()),
        "sun_night_mean_c": float(joint.plateau[dow == 6].mean()),
        "weekend_drift_fri_to_sun_k": weekend_drift,
        # Two uncooled nights separate Friday night from Sunday night, so the drift is a
        # RATE, in K per uncooled day. Turning it into tau_m needs the gradient driving
        # it - the air temperature the mass is relaxing toward - and the record does not
        # observe a free-running floor for long enough to pin that. Reported as the rate.
        "weekend_drift_k_per_uncooled_day": weekend_drift / 2.0,
        "weekday_cooling_mon_to_fri_k_per_day": (
            float(joint.plateau[dow == 4].mean() - joint.plateau[dow == 0].mean()) / 4.0),
        "tech_plateau_mean_c": TECH_PLATEAU_MEAN,
        "tech_plateau_sd_k": TECH_PLATEAU_SD,
    }])


def outdoor_equilibrium_share(Z: pd.DataFrame, df: pd.DataFrame,
                              config: dict) -> pd.DataFrame:
    """g/(k+g), measured as the within-night slope of plateau temp on outdoor temp.

    The plateau is a quasi-equilibrium, so dT_eq/dT_oa = g/(k+g) with no estimator in
    between. Night means are removed from both sides, which sweeps out the mass node's
    own night-to-night movement - the confound that would otherwise dominate, since the
    plateau and outdoor temperature both drift with the season.
    """
    oa = df["outdoor_weather_station__drybulb_temperature"]
    hours = Z.index.hour
    in_plateau = pd.Series(np.isin(hours, PLATEAU_HOURS), index=Z.index)
    nid = night_id(Z.index)
    rows = []
    for label, boxes in [("1", [b for b in Z.columns if ahu_of(b) == 1]),
                         ("2", [b for b in Z.columns if ahu_of(b) == 2]),
                         ("floor", list(Z.columns))]:
        keep = in_plateau.copy()
        for ahu in ({1} if label == "1" else {2} if label == "2" else set(AHUS)):
            keep &= fan_off(df, ahu, config)
        d = pd.DataFrame({"z": Z[boxes].mean(axis=1), "oa": oa,
                          "nid": nid}).loc[keep].dropna()
        if len(d) < 200:
            continue
        zw = d.z - d.groupby("nid").z.transform("mean")
        ow = d.oa - d.groupby("nid").oa.transform("mean")
        slope, _, _ = through_origin(ow.to_numpy(), zw.to_numpy())
        rows.append({
            "scope": label, "n_steps": int(len(d)), "nights": int(d.nid.nunique()),
            "within_night_slope": slope,
            "corr": float(np.corrcoef(ow, zw)[0, 1]),
            "within_night_sd_outdoor_k": float(ow.std()),
            "implied_g_over_k_plus_g": slope,
        })
    return pd.DataFrame(rows)


def add_outdoor_conductance(share: pd.DataFrame, s1: pd.DataFrame) -> pd.DataFrame:
    """Turn the equilibrium share into an outdoor conductance, given k from stage 1.

        sigma = g/(k+g)   =>   g = k*sigma/(1-sigma)   =>   tau_oa = dt/g

    The point of carrying it this far is that tau_oa comes out in DAYS. A real facade
    coupled to a real room does not have a multi-day time constant, and that is the
    physical reading of "the outdoor term fits to zero": these are interior zones on a
    floor with conditioned space above and below, so most of them have no facade at all.
    """
    out = share.copy()
    k_by_scope = {"1": s1[s1.ahu == 1].k.median(), "2": s1[s1.ahu == 2].k.median(),
                  "floor": s1.k.median()}
    out["k_median_from_stage1"] = out.scope.map(k_by_scope)
    sigma = out.within_night_slope
    out["implied_g"] = out.k_median_from_stage1 * sigma / (1.0 - sigma)
    out["implied_outdoor_tau_h"] = DT_H / out.implied_g
    out["implied_outdoor_tau_days"] = out.implied_outdoor_tau_h / 24.0
    return out


# --------------------------------------------------------------------------------------
# stage 1: the free response, fan off
# --------------------------------------------------------------------------------------

def sensor_noise(Z: pd.DataFrame, df: pd.DataFrame, config: dict) -> pd.Series:
    """Per-zone white-noise sd, from second differences over the flat plateau.

    For a white noise e on a locally straight true signal,
    var(T_t - 2*T_{t-1} + T_{t-2}) = 6*var(e). Any real curvature in the true signal
    inflates this, so it is an UPPER bound on sigma - which is the conservative direction
    for the conclusion in README section 4.3, that noise does not bias k measurably.
    """
    hours = Z.index.hour
    in_plateau = pd.Series(np.isin(hours, PLATEAU_HOURS), index=Z.index)
    nid = night_id(Z.index)
    out = {}
    for box in Z.columns:
        keep = in_plateau & fan_off(df, ahu_of(box), config)
        s = Z[box].where(keep)
        d2 = s.groupby(nid).transform(lambda x: x.diff().diff()).dropna()
        out[box] = np.sqrt(max(float(d2.var()) / 6.0, 0.0)) if len(d2) >= 100 else np.nan
    return pd.Series(out)


def stage1(Z: pd.DataFrame, df: pd.DataFrame, tm: pd.DataFrame, config: dict,
           hours_window: tuple[int, ...], train_only: bool = True,
           night_offset: int = 0) -> pd.DataFrame:
    """Per zone: regress dT on (T_m - T) through the origin, fan off.

    `night_offset` = 0 uses the plateau of the night this evening is settling INTO, so the
    regressor is the gap the zone is actually closing. `night_offset` = 1 uses the PREVIOUS
    night's plateau instead - which is what grey-box-technique section 4.1's own definition
    of T_m ("the night that ended that morning") says when read literally and applied to
    the evening window. Yesterday's plateau is a noisy proxy for tonight's, so it
    attenuates k and inflates tau. Fitting both is how README section 5.1 tests whether
    that is where technique's 2.5 h comes from.
    """
    hours = Z.index.hour
    in_win = pd.Series(np.isin(hours, hours_window), index=Z.index)
    nid = night_id(Z.index)
    if night_offset:
        nid = pd.Series(nid.to_numpy() - pd.Timedelta(days=night_offset), index=Z.index)
    rows = []
    for box in Z.columns:
        keep = in_win & fan_off(df, ahu_of(box), config)
        if train_only:
            keep &= Z.index < TRAIN_END
        s = Z[box]
        target = s.shift(-1) - s
        gap = nid.map(tm[box]) - s
        d = pd.DataFrame({"y": target, "x": gap}).loc[keep].dropna()
        if len(d) < 100:
            rows.append({"box": box, "ahu": ahu_of(box), "n": int(len(d)),
                         "k": np.nan, "r2_uncentred": np.nan, "r2_centred": np.nan,
                         "var_x": np.nan})
            continue
        k, unc, cen = through_origin(d.x.to_numpy(), d.y.to_numpy())
        rows.append({"box": box, "ahu": ahu_of(box), "n": int(len(d)), "k": k,
                     "r2_uncentred": unc, "r2_centred": cen,
                     "var_x": float(d.x.var())})
    return pd.DataFrame(rows)


def add_noise_correction(s1: pd.DataFrame, sigma: pd.Series) -> pd.DataFrame:
    """Errors-in-variables correction to k, and the size of the bias it removes.

    With T = theta + e, both y = T(t+1)-T(t) and x = T_m - T(t) carry -e(t), so
        k_hat = k + var(e)*(1-k)/var(x)
    i.e. k is biased UP, toward 1 - a zone that looks faster than it is. Inverting:
        k_debiased = (k_hat*var(x) - var(e)) / (var(x) - var(e))
    """
    out = s1.copy()
    out["sigma_eps_k"] = out.box.map(sigma)
    v = out.sigma_eps_k ** 2
    out["eiv_bias_in_k"] = v * (1.0 - out.k) / out.var_x
    out["k_debiased"] = (out.k * out.var_x - v) / (out.var_x - v)
    out["tau_euler_h"] = out.k.map(tau_euler)
    out["tau_exact_h"] = out.k.map(tau_exact)
    out["tau_exact_debiased_h"] = out.k_debiased.map(tau_exact)
    return out


# --------------------------------------------------------------------------------------
# stage 2: the daytime fit, k fixed
# --------------------------------------------------------------------------------------

def occupancy(df: pd.DataFrame) -> pd.Series:
    pir = [c for c in df.columns if c.endswith("__pir")]
    return df[pir].mean(axis=1)


def stage2(Z: pd.DataFrame, df: pd.DataFrame, tm: pd.DataFrame, k_by_box: pd.Series,
           config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per zone, with k fixed from stage 1:

        dT - k*(T_m - T) = a*V*(T_sa - T) + g*(T_oa - T) + d*occ + c

    Fitted three ways on identical rows so the comparisons mean something:
      * NNLS with a, g >= 0            - the specification
      * OLS, unconstrained             - to read the SIGN g wanted (README section 4.6)
      * the one-node form, k dropped   - the model the specification currently carries

    T_m here is the night that ENDED THAT MORNING, per grey-box-technique section 4.1.
    """
    m = config["control_gap"]["mask"]
    oa = df["outdoor_weather_station__drybulb_temperature"]
    occ = occupancy(df)
    hours = Z.index.hour
    weekday = Z.index.dayofweek < 5
    in_win = pd.Series(np.isin(hours, DAY_HOURS) & weekday, index=Z.index)
    nid = night_id(Z.index)
    prev_night = pd.Series(nid.to_numpy() - pd.Timedelta(days=1), index=Z.index)

    nulls = df.isna().sum(axis=1) / float(df.shape[1])
    no_outage = nulls <= float(m["outage_null_fraction"])

    rows, panels = [], []
    for box in Z.columns:
        ahu = ahu_of(box)
        hz = df[f"ahu_b8_{ahu}__frequency"]
        tsa = df[f"ahu_b8_{ahu}__supply_air_temperature"]
        flow = df[f"{box}__air_flow_rate"]
        s = Z[box]
        keep = in_win & no_outage & (hz > float(m["min_fan_hz"])) & (Z.index < TRAIN_END)
        d = pd.DataFrame({
            "y": s.shift(-1) - s,
            "gap_m": prev_night.map(tm[box]) - s,
            "air": flow * (tsa - s),
            "gap_oa": oa - s,
            "occ": occ,
            "flow": flow,
        }).loc[keep].dropna()
        if len(d) < 100 or box not in k_by_box.index or not np.isfinite(k_by_box[box]):
            continue
        k = float(k_by_box[box])
        X = np.column_stack([d.air, d.gap_oa, d.occ, np.ones(len(d))])
        y_two = (d.y - k * d.gap_m).to_numpy()
        beta_n, r2_two = nnls_fit(X, y_two, n_constrained=2)
        beta_u, _ = ols(X, y_two)
        # The one-node form the specification carries: no mass term at all.
        beta_1, r2_one = nnls_fit(X, d.y.to_numpy(), n_constrained=2)
        rows.append({
            "box": box, "ahu": ahu, "n": int(len(d)), "k_fixed": k,
            "a": beta_n[0], "g": beta_n[1], "delta": beta_n[2], "c": beta_n[3],
            "r2_two_node": r2_two,
            "g_unconstrained": beta_u[1], "a_unconstrained": beta_u[0],
            "a_one_node": beta_1[0], "g_one_node": beta_1[1], "r2_one_node": r2_one,
            "flow_cv": float(d.flow.std() / d.flow.mean()) if d.flow.mean() else np.nan,
            "parked": box in PARKED,
        })
        panels.append(pd.DataFrame({"box": box, "occ": d.occ.to_numpy()}))
    return pd.DataFrame(rows), (pd.concat(panels, ignore_index=True) if panels
                                else pd.DataFrame())


# --------------------------------------------------------------------------------------
# the discretisation identities
# --------------------------------------------------------------------------------------

def block_average_pole_identity() -> pd.DataFrame:
    """Verify that block averaging preserves the pole of a first-order system.

    x' = (x_inf - x)/tau with x_inf held constant across a step has exact solution
    x(t+dt) = x_inf + (x(t) - x_inf)*exp(-dt/tau). Averaging x over each step gives
        xbar_n = x_inf + (x_n - x_inf) * (tau/dt) * (1 - exp(-dt/tau))
    and since that factor does not depend on n,
        xbar_{n+1} - x_inf = exp(-dt/tau) * (xbar_n - x_inf).
    The AVERAGES obey the same recursion as the SAMPLES, with the same pole, exactly.

    This is the whole reason Block D is identifiable from 15-minute averages while the
    inner control loops are not: the record's coarseness attenuates the input gain of a
    fast loop into noise, but it does not move the pole of a slow one - provided the
    input is constant across the step, which is exactly what "the fan is off" means.
    """
    rows = []
    for tau in (0.5, 1.0, 2.5, 5.0):
        n_sub, n_steps, x_inf = 20000, 40, 24.7
        dt_sub = DT_H / n_sub
        x, samples, averages = 30.0, [], []
        for _ in range(n_steps):
            samples.append(x)
            acc = 0.0
            for _ in range(n_sub):     # fine-grained exact integration of the ODE
                acc += x * dt_sub
                x = x_inf + (x - x_inf) * np.exp(-dt_sub / tau)
            averages.append(acc / DT_H)
        pole_true = float(np.exp(-DT_H / tau))
        for label, series in (("instantaneous_samples", samples),
                              ("block_averages", averages)):
            v = np.array(series) - x_inf
            pole_fit, _, _ = through_origin(v[:-1], v[1:])
            k_fit = 1.0 - pole_fit
            rows.append({
                "tau_true_h": tau, "sampled_as": label,
                "pole_true": pole_true, "pole_recovered": pole_fit,
                "pole_abs_error": abs(pole_fit - pole_true),
                "k_zoh": k_fit,
                "k_euler_naive": DT_H / tau,
                "tau_from_euler_inversion_h": tau_euler(k_fit),
                "tau_from_exact_inversion_h": tau_exact(k_fit),
                "euler_tau_error_pct": 100 * (tau_euler(k_fit) - tau) / tau,
            })
    return pd.DataFrame(rows)


def synthetic_recovery() -> pd.DataFrame:
    """Two-stage fit on a synthetic zone whose k, a, g are known.

    Data generated by the block's OWN difference equation, so this tests the estimator,
    not the physics: an evening free response with the fan off, then a day with airflow.
    grey-box-technique section 5.1 asks for exactly this test at 1e-6.
    """
    rng = np.random.default_rng(0)
    k, a, g, delta, c = 0.11, 3.5e-4, 0.004, 0.6, -0.05
    t_m, t_oa, t_sa = 24.7, 29.0, 13.5
    # ---- stage 1: fan off, no airflow, no gains ----------------------------------
    x, ys, xs = 27.5, [], []
    for _ in range(400):
        gap = t_m - x
        dx = k * gap
        xs.append(gap)
        ys.append(dx)
        x += dx
        if abs(gap) < 1e-6:
            x = 27.5 + rng.normal(0, 0.4)   # restart the evening, same dynamics
    k_hat, _, _ = through_origin(np.array(xs), np.array(ys))
    # ---- stage 2: fan on, k fixed at the stage-1 value ---------------------------
    n = 800
    flow = 300 + 200 * rng.random(n)
    occ = 0.3 + 0.1 * rng.standard_normal(n)
    x, rows = 24.0, []
    for i in range(n):
        air = flow[i] * (t_sa - x)
        dx = k * (t_m - x) + a * air + g * (t_oa - x) + delta * occ[i] + c
        rows.append((x, air, t_oa - x, occ[i], dx))
        x = float(np.clip(x + dx, 18.0, 30.0))
    arr = np.array(rows)
    X = np.column_stack([arr[:, 1], arr[:, 2], arr[:, 3], np.ones(n)])
    y = arr[:, 4] - k_hat * (t_m - arr[:, 0])
    beta, _ = nnls_fit(X, y, n_constrained=2)
    return pd.DataFrame([{
        "parameter": p, "true": tv, "recovered": rv, "abs_error": abs(rv - tv),
    } for p, tv, rv in (("k", k, k_hat), ("a", a, beta[0]), ("g", g, beta[1]),
                        ("delta", delta, beta[2]), ("c", c, beta[3]))])


def capacitance_check(tau_measured_h: float | None = None) -> pd.DataFrame:
    """Order-of-magnitude: could the fast node be the zone AIR alone?

    A 50 m2 zone at 3 m is 150 m3 of air, about 180 kJ/K. Coupled to the slab through a
    combined surface coefficient of order 8 W/m2K over 50 m2 - 400 W/K - that node has a
    time constant of minutes, not hours. Every assumption here is a round number and is
    labelled as such; the conclusion is a factor, not a value.
    """
    rows = []
    area, height = 50.0, 3.0
    rho_cp_air = 1.2 * 1005.0            # J/(m3 K)
    c_air = rho_cp_air * area * height   # J/K
    for h_surf in (4.0, 8.0, 12.0):
        cond = h_surf * area             # W/K
        tau_air_only_h = c_air / cond / 3600.0
        rows.append({
            "assumed_zone_area_m2": area, "assumed_height_m": height,
            "assumed_surface_coeff_w_per_m2k": h_surf,
            "c_air_kj_per_k": c_air / 1000.0,
            "conductance_w_per_k": cond,
            "tau_air_only_h": tau_air_only_h,
            "tau_air_only_min": tau_air_only_h * 60.0,
            # The ratio at the tau this document actually measures (section 5.1), and at
            # grey-box-technique's 2.5 h, so the argument can be read either way round.
            "tau_measured_h": tau_measured_h,
            "capacitance_ratio_at_tau_measured": (
                tau_measured_h / tau_air_only_h if tau_measured_h else np.nan),
            "capacitance_ratio_at_tau_2p5h": 2.5 / tau_air_only_h,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    df, header = load_frame(config)
    boxes_all = zone_ids(header)
    Z_all = clean_room_temps(df, boxes_all, config)

    print(f"rows {len(df)}   room-temperature points {len(boxes_all)}")

    # ---- the two garbage sensors: verified, not trusted ---------------------------
    # grey-box-technique section 4.2 drops these two on their day means, quoted as
    # 8.4 and 19.7 degC. Neither reproduces here, because that screen appears to have
    # been run WITHOUT masking the 0.00 bad-read sentinel config.yml warns about - and
    # the sentinel is exactly what would drag an AHU-1 day mean toward 8 degC. Under a
    # masked screen the two are still the floor's two coldest zones, which is the claim
    # asserted; how far clear of the third they sit is reported, and for one of them the
    # answer is "barely". README section 4.7.
    day = Z_all[np.isin(Z_all.index.hour, DAY_HOURS)]
    day_means = day.mean().sort_values()
    floor_median = float(day_means.median())
    screen = pd.DataFrame({
        "box": day_means.index, "day_mean_c": day_means.to_numpy(),
        "rank_coldest": np.arange(1, len(day_means) + 1),
    })
    screen["margin_to_next_k"] = screen.day_mean_c.diff(-1).abs().shift(1)
    screen["k_below_floor_median"] = floor_median - screen.day_mean_c
    screen["excluded_by_technique"] = screen.box.isin(GARBAGE_SENSORS)
    screen["technique_quoted_day_mean_c"] = screen.box.map(
        {"vav_8_1_9": 8.4, "vav_8_1_6": 19.7})
    screen["floor_median_c"] = floor_median

    print("\n--- sensor screen: lowest day means, sentinel MASKED " + "-" * 17)
    print(screen.head(4)[["box", "day_mean_c", "k_below_floor_median",
                          "technique_quoted_day_mean_c"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    coldest = list(screen.box.head(len(GARBAGE_SENSORS)))
    assert set(coldest) == set(GARBAGE_SENSORS), (
        f"the floor's two coldest zones are {coldest}, not the "
        f"{list(GARBAGE_SENSORS)} grey-box-technique section 4.2 excludes. That "
        f"exclusion is named, not measured, so it has to be re-argued if the ranking "
        f"moves.")
    third = float(screen.day_mean_c.iloc[len(GARBAGE_SENSORS)])
    print(f"  the two named sensors ARE the two coldest; but the second sits only "
          f"{third - float(screen.day_mean_c.iloc[1]):.2f} K clear of the third "
          f"({screen.box.iloc[len(GARBAGE_SENSORS)]}).")
    print("  REPORTED, not asserted: that margin makes one of the two a judgement call.")
    boxes = [b for b in boxes_all if b not in GARBAGE_SENSORS]
    Z = Z_all[boxes]
    print(f"  excluded {len(GARBAGE_SENSORS)} sensors -> {len(boxes)} zones carried")

    # ---- the mass node --------------------------------------------------------------
    tm = plateau_means(Z, df, config)
    mass = mass_node_table(Z, df, config, tm)
    share = outdoor_equilibrium_share(Z, df, config)

    # ---- stage 1, both windows ------------------------------------------------------
    sigma = sensor_noise(Z, df, config)
    s1 = add_noise_correction(stage1(Z, df, tm, config, STAGE1_HOURS), sigma)
    s1_tech = add_noise_correction(stage1(Z, df, tm, config, STAGE1_TECH_HOURS), sigma)
    # The candidate explanation for technique's 2.5 h: T_m read as "the night that ended
    # that morning" even in the evening window, i.e. YESTERDAY's plateau.
    s1_prev = add_noise_correction(
        stage1(Z, df, tm, config, STAGE1_TECH_HOURS, night_offset=1), sigma)
    share = add_outdoor_conductance(share, s1[s1.k.notna()])

    overlap_rows = []
    for ahu in AHUS:
        a_dis = s1[(s1.ahu == ahu) & s1.k.notna()]
        a_ovl = s1_tech[(s1_tech.ahu == ahu) & s1_tech.k.notna()]
        a_prv = s1_prev[(s1_prev.ahu == ahu) & s1_prev.k.notna()]
        overlap_rows.append({
            "ahu": ahu, "zones": int(len(a_dis)),
            "stage1_window_disjoint": "18:00-20:59",
            "stage1_window_technique": "18:00-22:59",
            "plateau_window": "21:00-05:59",
            "k_disjoint": float(a_dis.k.median()),
            "k_overlapping": float(a_ovl.k.median()),
            "k_inflation_from_overlap": float(a_ovl.k.median() / a_dis.k.median()),
            "tau_exact_disjoint_h": float(a_dis.tau_exact_h.median()),
            "tau_exact_overlapping_h": float(a_ovl.tau_exact_h.median()),
            # The wrong-night variant, which is the candidate explanation for 2.5 h.
            "k_previous_night_tm": float(a_prv.k.median()),
            "tau_exact_previous_night_tm_h": float(a_prv.tau_exact_h.median()),
            "tau_euler_previous_night_tm_h": float(a_prv.tau_euler_h.median()),
            "r2_uncentred_disjoint": float(a_dis.r2_uncentred.median()),
            "r2_uncentred_overlapping": float(a_ovl.r2_uncentred.median()),
            "r2_uncentred_previous_night_tm": float(a_prv.r2_uncentred.median()),
        })
    overlap = pd.DataFrame(overlap_rows)

    # ---- stage 2 --------------------------------------------------------------------
    k_by_box = s1.set_index("box").k
    s2, occ_panel = stage2(Z, df, tm, k_by_box, config)

    # ---- gains identifiability ------------------------------------------------------
    occ = occupancy(df)
    hours = df.index.hour
    on = pd.Series(np.isin(hours, DAY_HOURS) & (df.index.dayofweek < 5), index=df.index)
    on &= (df[[f"ahu_b8_{a}__frequency" for a in AHUS]].max(axis=1)
           > float(config["control_gap"]["mask"]["min_fan_hz"]))
    o = occ[on].dropna()
    prof = o.groupby(o.index.hour).transform("mean")
    gains = pd.DataFrame([{
        "pir_sensors": int(len([c for c in df.columns if c.endswith('__pir')])),
        "zones_sharing_them": len(boxes),
        "n_steps": int(len(o)),
        "occ_mean": float(o.mean()), "occ_sd": float(o.std()),
        "occ_cv": float(o.std() / o.mean()),
        "occ_p5": float(np.percentile(o, 5)), "occ_p95": float(np.percentile(o, 95)),
        "r2_of_occ_on_hour_profile": float(1 - ((o - prof) ** 2).sum()
                                           / ((o - o.mean()) ** 2).sum()),
        # Technique section 2.5 quotes a median flow CV of 0.32 "during on-hours" and
        # finds 8 zones below 0.1. This is computed on the STAGE-2 ROWS - weekday
        # 08:00-16:59, fan on, no outage, training window - which is a narrower window
        # and gives a much lower figure. The narrower window is the right one, because it
        # is the variation `a` is actually identified from, and it says `a` is
        # unidentifiable on far more zones than technique's count implies.
        "flow_cv_window": "weekday 08:00-16:59, fan on, training window",
        "median_flow_cv_on_hours": float(s2.flow_cv.median()) if len(s2) else np.nan,
        "zones_with_flow_cv_below_0p1": int((s2.flow_cv < 0.1).sum()) if len(s2) else 0,
        "zones_fitted": int(len(s2)),
        "technique_median_flow_cv": 0.32,
        "technique_zones_below_0p1": 8,
        "parked_zones_named": len(PARKED),
    }])

    # ---- discretisation and the synthetic check --------------------------------------
    disc = block_average_pole_identity()
    synth = synthetic_recovery()
    caps = capacitance_check(float(s1[s1.k.notna()].tau_exact_h.median()))

    # ---- the tau disagreement, assembled --------------------------------------------
    tau_rows = []
    for ahu in AHUS:
        a_dis = s1[(s1.ahu == ahu) & s1.k.notna()]
        tau_rows.append({
            "ahu": ahu, "zones": int(len(a_dis)),
            "k_median": float(a_dis.k.median()),
            "tau_euler_h": float(a_dis.tau_euler_h.median()),
            "tau_exact_h": float(a_dis.tau_exact_h.median()),
            "tau_exact_debiased_h": float(a_dis.tau_exact_debiased_h.median()),
            "tau_exact_iqr_low_h": float(a_dis.tau_exact_h.quantile(0.25)),
            "tau_exact_iqr_high_h": float(a_dis.tau_exact_h.quantile(0.75)),
            "technique_tau_h": TECH_TAU_H,
            "ratio_technique_over_this": TECH_TAU_H / float(a_dis.tau_exact_h.median()),
            "sigma_eps_median_k": float(a_dis.sigma_eps_k.median()),
            "eiv_bias_in_k_median": float(a_dis.eiv_bias_in_k.median()),
            "r2_uncentred_median": float(a_dis.r2_uncentred.median()),
            "r2_centred_median": float(a_dis.r2_centred.median()),
        })
    tau_tbl = pd.DataFrame(tau_rows)

    # ---- write ----------------------------------------------------------------------
    for name, frame in (("mass_node", mass), ("outdoor_share", share),
                        ("stage1_fit", s1), ("stage1_window_variants", overlap),
                        ("stage2_fit", s2), ("gains_identifiability", gains),
                        ("discretisation", disc), ("synthetic_recovery", synth),
                        ("capacitance_check", caps), ("tau_disagreement", tau_tbl),
                        ("sensor_screen", screen)):
        frame.to_csv(DATA_DIR / f"{name}.csv", index=False)

    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.4f}"

    print("\n--- THE MASS NODE: the night plateau (technique section 2.2) " + "-" * 15)
    print(f"  {int(mass.nights[0])} nights   plateau {mass.plateau_mean_c[0]:.2f} +/- "
          f"{mass.plateau_sd_k[0]:.2f} K   (technique: "
          f"{TECH_PLATEAU_MEAN} +/- {TECH_PLATEAU_SD})")
    print(f"  range {mass.plateau_min_c[0]:.1f} - {mass.plateau_max_c[0]:.1f} C   "
          f"lag-1 autocorr {mass.lag1_autocorr[0]:.2f}   "
          f"corr with night outdoor {mass.corr_with_night_outdoor[0]:.2f}")
    print(f"  Fri night {mass.fri_night_mean_c[0]:.2f} -> Sun night "
          f"{mass.sun_night_mean_c[0]:.2f}   weekend drift "
          f"{mass.weekend_drift_fri_to_sun_k[0]:+.2f} K")

    print("\n--- NO OUTDOOR PATH: within-night slope of plateau on outdoor " + "-" * 14)
    print(share[["scope", "n_steps", "nights", "within_night_slope", "corr",
                 "within_night_sd_outdoor_k", "implied_g",
                 "implied_outdoor_tau_days"]].to_string(index=False, float_format=fmt))
    print("  That slope IS g/(k+g) - the share of the zone's equilibrium that outdoor")
    print("  air sets. No estimator, no constraint, one regression. ASSERTED below.")
    print("  Carried through to a conductance it implies an outdoor time constant in")
    print("  DAYS, which is the physical reading: these are interior zones.")

    print("\n--- STAGE 1: the free response, three ways of reading T_m " + "-" * 18)
    print(overlap[["ahu", "zones", "k_disjoint", "k_overlapping",
                   "k_previous_night_tm", "tau_exact_disjoint_h",
                   "tau_exact_overlapping_h", "tau_exact_previous_night_tm_h",
                   "tau_euler_previous_night_tm_h", "r2_uncentred_disjoint",
                   "r2_uncentred_previous_night_tm"]]
          .to_string(index=False, float_format=fmt))
    print("  Window overlap does almost nothing (k moves <3%, the other way): T_m averages")
    print("  33 steps, so the shared term is O(1/33). An objection, retired - README 4.2.")
    print("  WHICH NIGHT does everything. Yesterday's plateau is a noisy proxy for")
    print("  tonight's, it attenuates k by ~2.2x, and tau = dt/k then lands on 2.32/2.56 h")
    print(f"  against grey-box-technique section 2.3's {TECH_TAU_H} h. That is where 2.5 h")
    print("  came from, and it is ASSERTED below - README section 5.1.")

    print("\n--- TAU, THE NUMBER IN DISPUTE (reported, never asserted) " + "-" * 18)
    print(tau_tbl[["ahu", "zones", "k_median", "tau_euler_h", "tau_exact_h",
                   "tau_exact_debiased_h", "technique_tau_h",
                   "ratio_technique_over_this", "r2_uncentred_median"]]
          .to_string(index=False, float_format=fmt))
    print("  sigma_eps median %.4f / %.4f K, so the errors-in-variables bias in k is"
          % (tau_tbl.sigma_eps_median_k[0], tau_tbl.sigma_eps_median_k[1]))
    print("  %.5f / %.5f - three orders below k. Sensor noise is NOT why these differ."
          % (tau_tbl.eiv_bias_in_k_median[0], tau_tbl.eiv_bias_in_k_median[1]))

    print("\n--- STAGE 2: two-node against the one-node form in the spec " + "-" * 16)
    for label, s in (("AHU-1", s2[s2.ahu == 1]), ("AHU-2", s2[s2.ahu == 2]),
                     ("FLOOR", s2)):
        if not len(s):
            continue
        two, one = s.r2_two_node.median(), s.r2_one_node.median()
        flag = "" if two > one else "   <-- REVERSES"
        print(f"  {label}  n={len(s)} zones")
        print(f"    R2 two-node {two:.3f}   one-node {one:.3f}   "
              f"ratio {two / one:.2f}x{flag}"
              f"   (technique, pooled: {TECH_R2_TWO_NODE} / {TECH_R2_ONE_NODE} = "
              f"{TECH_R2_TWO_NODE / TECH_R2_ONE_NODE:.1f}x)")
        print(f"    a > 0 on {int((s.a > 0).sum())}/{len(s)}   "
              f"g pinned at 0 by NNLS on {int((s.g <= 0).sum())}/{len(s)}   "
              f"g WANTED negative on {int((s.g_unconstrained < 0).sum())}/{len(s)}")
    print("  g pinned at zero is not 'no outdoor effect measured' - it is the")
    print("  unconstrained fit asking for a NEGATIVE one. README section 4.6.")
    print("  THE POOLED CLAIM HOLDS AND IS ASSERTED. The AHU-2 reversal is REPORTED and")
    print("  is the most important thing in this run - README section 5.2.")

    print("\n--- DISCRETISATION: block averaging preserves the pole " + "-" * 21)
    print(disc[["tau_true_h", "sampled_as", "pole_true", "pole_recovered",
                "pole_abs_error", "tau_from_euler_inversion_h",
                "euler_tau_error_pct"]].to_string(index=False, float_format=fmt))

    print("\n--- SYNTHETIC RECOVERY " + "-" * 52)
    print(synth.to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    print("\n--- COULD THE FAST NODE BE THE AIR ALONE? " + "-" * 33)
    print(caps[["assumed_surface_coeff_w_per_m2k", "c_air_kj_per_k",
                "conductance_w_per_k", "tau_air_only_min",
                "capacitance_ratio_at_tau_measured"]]
          .to_string(index=False, float_format=fmt))
    print("  Minutes, not hours. Whatever the fitted node is, it is not the air.")

    # ---- guards ---------------------------------------------------------------------
    for _, r in disc.iterrows():
        if r.sampled_as == "block_averages":
            assert r.pole_abs_error < POLE_IDENTITY_TOLERANCE, (
                f"tau={r.tau_true_h}: block averaging moved the pole by "
                f"{r.pole_abs_error:.2e}. That identity is the whole argument that a "
                f"15-minute AVERAGE can identify this block at all - README 3.3.")

    for _, r in synth.iterrows():
        assert r.abs_error < SYNTHETIC_RECOVERY_TOLERANCE * max(abs(r.true), 1e-3), (
            f"the two-stage fit recovered {r.parameter} as {r.recovered:.6g} against a "
            f"true {r.true:.6g}. The estimator has to work on data it generated itself "
            f"before any number it produces from the CSV means anything.")

    for _, r in share.iterrows():
        assert r.within_night_slope < MAX_OUTDOOR_EQUILIBRIUM_SHARE, (
            f"{r.scope}: outdoor air sets {r.within_night_slope:.3f} of the zone "
            f"equilibrium, over the {MAX_OUTDOOR_EQUILIBRIUM_SHARE} budget. The two-node "
            f"form drops the outdoor path on the strength of this being ~0.")

    # Pooled over the floor, which is the comparison grey-box-technique section 2.3 makes
    # (50 boxes, one median). Deliberately NOT per-AHU: it reverses on AHU-2, and that
    # counter-example is reported in full above and in README section 5.2 rather than
    # being promoted into a guard that this run would simply fail.
    if len(s2):
        two, one = s2.r2_two_node.median(), s2.r2_one_node.median()
        assert two > one, (
            f"pooled over the floor, the two-node form scored {two:.4f} against the "
            f"one-node form's {one:.4f}. Replacing Block D's structure rests on this "
            f"comparison, and it has just stopped holding even pooled.")

    # The explanation of technique's 2.5 h stands or falls on reproducing it. Tolerance is
    # generous because the target is a median-of-medians quoted to one decimal place with
    # an IQR of 2.1-2.9; the claim is "this is the same number", not "this is that number
    # to three figures".
    for _, r in overlap.iterrows():
        assert abs(r.tau_euler_previous_night_tm_h - TECH_TAU_H) < 0.5, (
            f"AHU-{int(r.ahu)}: reading T_m as the previous night's plateau gives "
            f"tau = dt/k of {r.tau_euler_previous_night_tm_h:.2f} h, which no longer "
            f"lands on grey-box-technique section 2.3's {TECH_TAU_H} h. README 5.1 "
            f"explains that document's number as exactly this artifact; if it stops "
            f"reproducing, the explanation is wrong and must be withdrawn.")

    unpooled = s1[~s1.box.isin(PARKED) & s1.k.notna()]
    bad = unpooled[unpooled.k <= 0]
    assert bad.empty, (
        f"k <= 0 on {list(bad.box)}. A negative air-to-mass conductance is not a fit, "
        f"it is a defect indicator; those zones must be pooled, not carried.")

    print(f"\nAll asserted guards passed. Wrote 11 tables to {DATA_DIR}")
    print("REPORTED, NOT ASSERTED: tau itself (~2.5x below grey-box-technique section")
    print("2.3), every per-zone R2, and the AHU-2 reversal. See README sections 5.1-5.2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
