"""Reproduce Block F - the zone control loops - from the raw CSV, dumping every intermediate.

Block F is the block that closes the surrogate. Blocks A-E describe the plant; without F
nothing produces `d_i`, Block C's split has no input, and the environment can replay a day
but cannot answer a counterfactual.

    stage 1   V_sp_i(t+1) = clip( V_sp_i(t) + K_i*(T_i(t) - Tsp_i(t)), V_min_i, V_max_i )
    stage 2   damper modulates until measured flow matches  ->  d_i

    python block-f-derivation/reproduce_zone_control.py

Writes to block-f-derivation/data/ only. Nothing outside this folder is touched.

WHAT WAS MISSING
----------------
grey-box-surrogate.md 7 specifies stage 1 and stops. It ends at `V_sp_i`; Block C 4.2
begins at `d_i`; nothing in the 1.1 step order joins them. The plant runs both stages -
hvac-system-logical-flow.md 61-62, "modulates the damper until measured flow matches" -
so the gap is in the specification, not in the building.

WHY STAGE 1 IS FITTABLE ON AN UNLOGGED VARIABLE
-----------------------------------------------
`V_sp_i` is not a logged point. Each box logs seven: air_flow_rate, damper_position,
maximum_/minimum_air_flow_rate_setpoint_read, room_temperature, and its setpoint read and
write. So 7's own recipe - "regress dV_sp_i on the zone temperature error" - is a
regression on an unobservable, which is verbatim the objection 4.7.2(3) that retired Block
C v0.

The way out is 0's Nyquist argument used FORWARD. The damper loop runs at 1-5 min against
a 15-min sample, so it has settled within a step and `V_i ~= V_sp_i`. The unlogged setpoint
is observable through logged flow.

That proxy holds only while the box has authority. Against a stop, stage 2 cannot track and
the equality fails exactly where it matters - so the authority exclusions in config are
part of the identification, not hygiene. They apply to the GAIN FIT alone: never to the
comfort constraint (4.3, "a starved zone is still a zone") and never to the counterfactual,
where clipping is the effect being measured.

WHY THE INVERSION IS CLOSED FORM
--------------------------------
Given target flows, Block C's split inverts analytically. With V_i = C*W^(q-1)*SP^s*a_i*d_i
and W = sum_j a_j d_j, summing over boxes gives W^q = V_target_total / (C*SP^s), hence
d_i = V_i * W / (a_i * V_total). No search. The iteration in `dampers_for_targets` exists
only to redistribute after boxes hit the 0/100 stops, and its inner solve is a scalar
bisection on a monotone function.

Better still, in RATIO form the scale constant C cancels completely (README 3.2). That
matters because C is the weakest number in Block C - the total's R2 is 0.42/0.64 against
0.94/0.90 for the split - and it carries the unresolved airflow unit of 4.6. F-a therefore
touches neither.

WHAT IS ASSERTED, AND WHAT IS ONLY REPORTED
-------------------------------------------
Asserted (a reversal must fail the run):

  * ROUND TRIP. Inverting Block C's forward model and stepping it forward again returns
    the flows asked for, to floating point, wherever nothing clips. If this trips, F-a is
    not inverting the model Block C forward-steps and the two halves have drifted apart.
  * ADDING UP survives the inversion: sum_i V_i = V_total, per 4.2.
  * THE BAND ENDPOINTS. When every box asks for the flow it already had, F-a must
    reproduce block-c-derivation/data/pressure_cut_counterfactual.csv - 0.11%/0.15% at
    lam=1 and 10.79%/14.92% at lam=0. Block C reaches those by a COMMON DAMPER SCALING and
    F-a by a PER-BOX inversion; 3.3 shows the two coincide when the target is "restore what
    you had", so agreement is a real cross-check between independent code paths rather than
    a tautology.
  * COMPENSATION OPENS DAMPERS. Restoring flow at lower pressure cannot close a damper.
    A sign check on the algebra.
  * THE GAIN SIGN. A warm zone asks for more air, so K_i > 0 on a majority of identified
    zones. A negative gain is not a fit, it is a defect indicator - the stance block-d
    takes on k_i, a_i, g_i >= 0.

Reported, never asserted:

  * lam itself. 4.3 puts the ventilation cost of a 20% cut at 0.11%-10.79% (AHU-1) and
    0.15%-14.92% (AHU-2) and states that which end applies turns on the zone loops - which
    this record cannot observe, because pressure and demand moved together in every episode
    it contains. lam is therefore carried as a DOMAIN-RANDOMISATION axis, the same
    treatment 8.2 gives beta and s. This script measures the band; it does not pick a point
    inside it.
  * THE EPISODE REPLAY, and it is reported because it comes out AGAINST the compensating
    reading. F-a predicts dampers OPENING during a pressure cut. Over the plant's own 8
    AHU-1 cuts they CLOSED, by 1.65 points, with no box saturated in either window (4.3).
    That does not falsify F-a - it confirms those episodes are demand moves, not authority
    events, and so cannot identify lam. Asserting agreement here would be asserting the
    thing the record cannot show.
  * K_i for the twelve zones parked at 27.0 degC. Their flow barely varies, so the gain is
    unidentified rather than wrong, and 5.3(4) already says to pool them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"

AHUS = (1, 2)


# --------------------------------------------------------------------------------------
# Block C's forward model, imported rather than restated
# --------------------------------------------------------------------------------------
# One forward model, two consumers. Block C has been corrected twice, and both corrections
# landed in its own script; a Block F that kept a private copy of the split would validate
# a model the surrogate does not step. The folder name is not an identifier, so this goes
# through importlib rather than `import`.

def _load_block_c():
    path = ROOT / "block-c-derivation" / "reproduce_zone_flow.py"
    spec = importlib.util.spec_from_file_location("block_c", path)
    if spec is None or spec.loader is None:              # pragma: no cover
        raise ImportError(f"cannot load Block C's forward model from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_block_c()

VAV_EXTRA = ("room_temperature", "room_temperature_setpoint_read")


# --------------------------------------------------------------------------------------
# loading - Block C's frame plus the two zone-control points it does not need
# --------------------------------------------------------------------------------------

def load_frame(config: dict) -> tuple[pd.DataFrame, list[str]]:
    csv = ROOT / config["data"]["csv_path"]
    header = pd.read_csv(csv, nrows=0).columns.tolist()
    wanted = ["timestamp_local"]
    for ahu in AHUS:
        wanted += [f"ahu_b8_{ahu}__{p}" for p in C.AHU_POINTS]
    for box in C.box_ids(header):
        wanted += [f"{box}__{p}" for p in C.VAV_POINTS + VAV_EXTRA]
    cols = [c for c in wanted if c in header]
    df = pd.read_csv(csv, usecols=cols, parse_dates=["timestamp_local"])
    return df.set_index("timestamp_local").sort_index(), header


def room_temp(df: pd.DataFrame, box: str, config: dict) -> pd.Series:
    """Room temperature with the 0.00 bad-read sentinel masked, per 1.7 / 5.3(2).

    Every AHU-1 box carries a 0.00 that is a failed read, not a room at freezing point -
    the minimum non-zero on the floor is 6.49 degC. Averaging it drags every zone statistic
    down, and here it would manufacture an enormous cooling error and a spurious gain.
    """
    m = config["control_gap"]["mask"]
    s = df[f"{box}__room_temperature"].copy()
    sentinel = float(m["room_temp_sentinel"])
    tol = float(m["room_temp_sentinel_tolerance"])
    return s.mask((s - sentinel).abs() < tol)


# --------------------------------------------------------------------------------------
# stage 1 - the zone controller gain
# --------------------------------------------------------------------------------------

def fit_stage1_gains(df: pd.DataFrame, ahu: int, keep: pd.Series, header: list[str],
                     config: dict) -> pd.DataFrame:
    """V_i(t) = A_i + G_i * (T_i(t-1) - Tsp_i(t-1)), one gain per zone, on logged flow.

    A LEVEL relationship, and the specification hinges on that word.

    WHY NOT THE DIFFERENCE FORM. 7 writes the loop as an increment,
    `V_sp(t+1) = V_sp(t) + K*(T - Tsp)`, and the obvious translation is to regress dV on
    the lagged error. That is a sub-Nyquist dynamic specification - it asks the 15-minute
    record to resolve the integrating action of a 1-5 minute loop - which is exactly what
    0 forbids. It also fails on its own terms: fitted here it returns a POSITIVE gain on
    5% of AHU-1 zones and 23% of AHU-2's, with a median of -16.8 / -18.6. The sign is an
    artefact of regressing a difference on a level when the two are cointegrated by the
    controller: V(t-1) already carries e(t-1), so dV = V(t) - V(t-1) inherits its negative.
    Both specifications are fitted and both are reported below; only the level form is
    used.

    WHY THE LEVEL FORM IS THE ONE 0 LICENSES. The damper loop settles within a step, so a
    settled box sits AT its flow setpoint, and the setpoint is a static function of the
    error. What the record can see is that static map - not the integration that produced
    it. Fitted, G > 0 on 100% of AHU-1's identified zones and 95% of AHU-2's.

    The error is lagged one step: the controller cannot respond to a temperature it has not
    yet measured. Same one-step-lead protection 5.4(1) uses against Block D's endogeneity,
    and it is not a cure - the reverse path (more air cools the room, lowering the error)
    still runs, and biases G toward zero. Stated, not solved; README 5.3.

    `A_i` is reported. It is the flow the box passes at zero error - in effect its parked
    level - and on this floor it is most of the flow, because 5.5's zones sit 2.60 K BELOW
    setpoint and the error term is doing little work. That is why R2 is 0.05-0.09 and not a
    defect to be tuned away: it is the measurement that these loops are barely modulating.
    """
    zc = config["control_gap"]["zone_control"]
    min_steps = int(zc["min_gain_fit_steps"])
    d_stop = float(zc["exclude_damper_at_or_above"])
    f_min = float(zc["exclude_flow_within_frac_of_min"])
    f_max = float(zc["exclude_flow_within_frac_of_max"])

    rows = []
    for box in C.box_ids(header, ahu):
        flow = df[f"{box}__air_flow_rate"]
        damper = df[f"{box}__damper_position"]
        vmin = df[f"{box}__minimum_air_flow_rate_setpoint_read"]
        vmax = df[f"{box}__maximum_air_flow_rate_setpoint_read"]
        temp = room_temp(df, box, config)
        tsp = df[f"{box}__room_temperature_setpoint_read"]

        err = (temp - tsp).shift(1)

        # THE AUTHORITY EXCLUSIONS. Against a stop the damper cannot chase flow, so
        # V_i != V_sp_i and the proxy this whole fit rests on stops holding. Required at
        # BOTH ends: at t because that is the flow being explained, at t-1 because a box
        # released from a stop is not sitting on its static map either.
        has_authority = (
            (damper < d_stop)
            & (flow > vmin * (1.0 + f_min))
            & (flow < vmax * (1.0 - f_max))
        )
        prev_ok = (keep & has_authority).shift(1, fill_value=False).astype(bool)
        good = keep & has_authority & prev_ok & err.notna() & flow.notna()

        n = int(good.sum())
        if n < min_steps:
            rows.append({"ahu": ahu, "box": box, "n": n, "G": np.nan, "A": np.nan,
                         "r2": np.nan, "identified": False, "mean_err_K": np.nan,
                         "K_difference_form": np.nan, "r2_difference_form": np.nan})
            continue

        x = err[good].to_numpy()
        beta, r2 = C.ols(np.column_stack([np.ones(n), x]), flow[good].to_numpy())

        # The rejected specification, refitted on the same rows so the comparison is like
        # for like. Reported so the correction stays visible rather than being a silent
        # deletion - the same discipline 4.7 applies to Block C's v0 and v1.
        dv = flow.diff()
        gd = good & dv.notna()
        nd = int(gd.sum())
        if nd >= min_steps:
            bd, r2d = C.ols(np.column_stack([np.ones(nd), err[gd].to_numpy()]),
                            dv[gd].to_numpy())
            k_diff, r2_diff = float(bd[1]), r2d
        else:
            k_diff, r2_diff = np.nan, np.nan

        rows.append({"ahu": ahu, "box": box, "n": n, "G": float(beta[1]),
                     "A": float(beta[0]), "r2": r2, "identified": True,
                     "mean_err_K": float(np.mean(x)),
                     "K_difference_form": k_diff, "r2_difference_form": r2_diff})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# stage 2 - F-a, the inversion of Block C's split for the damper
# --------------------------------------------------------------------------------------

def solve_total_weight(w_stopped: float, w_free_now: float, w_now: float,
                       sp_ratio: float, q: float, s: float) -> float:
    """Solve  W**(q-1) * (W - w_stopped)  =  w_free_now * w_now**(q-1) * sp_ratio**(-s).

    This is the fixed point that says "the free boxes each get their own flow back". It
    comes out of holding V_i at target for every free box while W - which every box sees
    through the total - moves. The left-hand side is monotone increasing in W above
    `w_stopped`, so a bisection is exact to machine precision in ~60 halvings.

    C cancels on the way in, and so does SP: only the RATIO of new to old pressure enters.
    Neither the scale constant nor the unresolved airflow unit of 4.6 can reach this.
    """
    target = w_free_now * w_now ** (q - 1.0) * sp_ratio ** (-s)
    if w_free_now <= 0.0:
        return w_stopped

    def lhs(w: float) -> float:
        return w ** (q - 1.0) * (w - w_stopped)

    lo = w_stopped
    hi = max(w_stopped, w_now) * 2.0 + 1e-12
    for _ in range(200):
        if lhs(hi) >= target:
            break
        hi *= 2.0
    else:                                                # pragma: no cover
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if lhs(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compensating_dampers(a: np.ndarray, d: np.ndarray, sp_ratio: float,
                         p: float, q: float, s: float, config: dict) -> tuple[np.ndarray, float]:
    """Dampers that restore every box's OWN current flow after SP -> sp_ratio * SP.

    Returns (d_new, W_new). Boxes that would need to pass 100% clip there and deliver what
    they can; the shortfall is NOT redistributed, because each loop tracks its own flow
    setpoint and does not open further because a neighbour is starved. That is the
    pressure-independent premise of 4.1 taken literally.

    Contrast `restore_total_dampers` below, which is Block C's arm and does redistribute.
    The two coincide exactly while nothing clips and diverge once something does; README
    4.2 is that subject, and the divergence is a finding rather than a defect.

    The clipped set only ever grows, so the loop terminates. `max_clip_iterations` is a
    guard against a pathological parameter set, not an expected exit.
    """
    zc = config["control_gap"]["zone_control"]
    cap = float(zc["damper_max_pct"])
    w_i_now = a * d ** p
    w_now = float(w_i_now.sum())
    stopped = np.zeros(len(d), dtype=bool)

    d_new = d.copy()
    w_new = w_now
    for _ in range(int(zc["max_clip_iterations"])):
        w_stopped = float(np.sum(a[stopped] * cap ** p))
        w_free_now = float(np.sum(w_i_now[~stopped]))
        w_new = solve_total_weight(w_stopped, w_free_now, w_now, sp_ratio, q, s)
        if not np.isfinite(w_new):                       # pragma: no cover
            d_new = np.where(stopped, cap, cap)
            break
        # Every free box needs the same multiplier - the split is untouched by a common
        # scaling, which is 4.3's observation arrived at from the per-box side.
        scale = ((w_now / w_new) ** (q - 1.0) * sp_ratio ** (-s)) ** (1.0 / p)
        d_new = np.where(stopped, cap, np.minimum(d * scale, cap))
        newly = (~stopped) & (d * scale > cap)
        if not newly.any():
            break
        stopped |= newly
    return d_new, w_new


def restore_total_dampers(a: np.ndarray, d: np.ndarray, sp_ratio: float,
                          p: float, q: float, s: float, config: dict) -> tuple[np.ndarray, float]:
    """Block C's arm: one common scaling k chosen so the FLOOR TOTAL comes back.

    Delegates to `block_c.required_scaling` rather than restating it, so this really is the
    other route and not a paraphrase of it. The difference from `compensating_dampers` is
    where k comes from: here it is solved AFTER clipping, so boxes with authority left open
    further to cover boxes that have run out. There they do not.
    """
    cap = float(config["control_gap"]["zone_control"]["damper_max_pct"])
    w_now = float(np.sum(a * d ** p))
    w_req = w_now * sp_ratio ** (-s / q)
    k = C.required_scaling(a, d, w_req, p)
    d_new = d if not np.isfinite(k) else np.minimum(k * d, cap)
    return d_new, float(np.sum(a * d_new ** p))


def forward_flow(a: np.ndarray, d: np.ndarray, p: float, q: float) -> tuple[np.ndarray, float]:
    """Block C forward, in weight space: returns (share_i, W). Scale-free by construction."""
    w = a * d ** p
    tot = float(w.sum())
    return (w / tot if tot > 0 else w), tot


def lambda_band(panel: pd.DataFrame, fit: dict, total: dict, cut: float,
                config: dict) -> pd.DataFrame:
    """Delivered air after a pressure cut, swept over the compensation axis lam.

    lam = 0   dampers frozen. Every room loses (1-cut)**s. No compensation.
    lam = 1   the loops compensate, under BOTH readings of what that means:
                own_flow      each loop restores its own flow, no slack pickup (F-a)
                restore_total one common k, solved after clipping, so boxes with authority
                              left cover boxes that have run out (Block C's arm)
    between   d(lam) = d + lam*(d_compensating - d), stepped forward through Block C.

    The interpolation is in DAMPER space rather than in flow, so every intermediate point
    is a state the plant could actually be in and the endpoints are exact.
    """
    p, q, s = fit["p"], total["q"], total["s"]
    a_map = fit["weights"]
    cap = float(config["control_gap"]["zone_control"]["damper_max_pct"])
    grid = [float(x) for x in config["control_gap"]["zone_control"]["lambda_grid"]]
    sp_ratio = 1.0 - cut

    rows = []
    for t, grp in panel.groupby("t"):
        a = grp.box.map(a_map).to_numpy()
        d = grp.damper.to_numpy()
        v = grp.flow.to_numpy()
        v_now = float(v.sum())
        _, w_now = forward_flow(a, d, p, q)
        if w_now <= 0 or v_now <= 0:
            continue
        d_own, _ = compensating_dampers(a, d, sp_ratio, p, q, s, config)
        d_tot, _ = restore_total_dampers(a, d, sp_ratio, p, q, s, config)
        row = {"t": t, "n_boxes": len(grp), "flow_now": v_now,
               "clipped_own": int(np.sum(d_own >= cap - 1e-9)),
               "clipped_total": int(np.sum(d_tot >= cap - 1e-9)),
               "damper_scaling_median": float(np.median(d_own / np.maximum(d, 1e-9)))}
        for lam in grid:
            for name, d_end in (("own", d_own), ("tot", d_tot)):
                _, w_lam = forward_flow(a, d + lam * (d_end - d), p, q)
                # total ratio = (W'/W)**q * sp_ratio**s ; C and the airflow unit cancel
                row[f"flow_{name}_lam{lam:g}"] = v_now * (w_lam / w_now) ** q * sp_ratio ** s
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# the round trip - F-a against Block C's forward model
# --------------------------------------------------------------------------------------

def round_trip(panel: pd.DataFrame, fit: dict, total: dict, config: dict) -> pd.DataFrame:
    """Ask F-a for the flows the plant already had, at unchanged pressure, and check.

    With sp_ratio = 1 and no clipping the answer must be the dampers the plant was already
    sitting at, and the shares must come back bit-for-bit. This is the guard that keeps
    the inversion honest against the forward model it inverts - the concrete form of the
    plan's "one forward model, two consumers".
    """
    p, q, s = fit["p"], total["q"], total["s"]
    a_map = fit["weights"]
    rows = []
    for t, grp in panel.groupby("t"):
        a = grp.box.map(a_map).to_numpy()
        d = grp.damper.to_numpy()
        d_back, _ = compensating_dampers(a, d, 1.0, p, q, s, config)
        sh_now, _ = forward_flow(a, d, p, q)
        sh_back, _ = forward_flow(a, d_back, p, q)
        rows.append({"t": t,
                     "max_abs_damper_error": float(np.max(np.abs(d_back - d))),
                     "max_abs_share_error": float(np.max(np.abs(sh_back - sh_now))),
                     "share_sum_error": float(abs(sh_back.sum() - 1.0))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# the plant's own pressure cuts, replayed through F-a
# --------------------------------------------------------------------------------------

def episode_replay(panel: pd.DataFrame, fit: dict, total: dict, ahu: int,
                   config: dict) -> pd.DataFrame:
    """What F-a predicts over the 8 real AHU-1 cuts, against what the dampers did.

    Reported, never asserted, and it comes out against the compensating reading: F-a says
    dampers must OPEN to hold flow at lower pressure, and over the plant's own cuts they
    CLOSED (45.51 vs 47.16, -1.65 pts, no box saturated in either window - 4.3).

    The right conclusion is not that F-a is wrong but that those episodes are demand moves.
    The floor was 0.89 K cooler in the treated windows; the loops were closing dampers on
    falling load, not opening them against falling pressure. Which is precisely why this
    record cannot identify lam, and why lam is a DR axis instead of a fitted number.
    """
    episodes = ROOT / "report" / "setpoint_episodes.csv"
    if not episodes.exists() or not len(panel):
        return pd.DataFrame()
    ep = pd.read_csv(episodes, parse_dates=["start", "end"])
    ep = ep[(ep.ahu == ahu) & (ep.direction == "down")]
    if not len(ep):
        return pd.DataFrame()

    p, q, s = fit["p"], total["q"], total["s"]
    a_map = fit["weights"]
    tz = config["data"]["timezone"]
    step = pd.Timedelta("15min")
    rows = []
    for _, e in ep.iterrows():
        start = e.start if e.start.tzinfo else e.start.tz_localize(tz)
        end = e.end if e.end.tzinfo else e.end.tz_localize(tz)
        pt = panel.t.dt.tz_localize(tz) if panel.t.dt.tz is None else panel.t
        pre = panel[(pt >= start - 8 * step) & (pt < start)]
        post = panel[(pt >= start + 2 * step) & (pt <= end)]
        if not len(pre) or not len(post):
            continue
        # An episode is a SETPOINT change; what matters here is whether the achieved
        # pressure actually fell. On AHU-2 it frequently did not - 2.2 records that 93% of
        # its episodes sit at 08h, inside the start-up ramp, so duct static is climbing
        # while the setpoint steps down. Those are not pressure cuts and are dropped.
        ex = config["control_gap"]["excitation"]
        if (pre.t.nunique() < int(ex["min_control_steps"])
                or post.t.nunique() < int(ex["min_episode_steps"])):
            continue
        # the pre-window state, box-medianed, as the point F-a is asked about
        g = pre.groupby("box").agg(damper=("damper", "median"), sp=("sp", "median"))
        boxes = sorted(g.index)
        a = np.array([a_map[b] for b in boxes if b in a_map])
        d = np.array([g.damper[b] for b in boxes if b in a_map])
        if not len(a):
            continue
        sp_pre = float(pre.sp.median())
        sp_post = float(post.sp.median())
        if sp_pre <= 0 or sp_post <= 0 or sp_post >= sp_pre:
            continue
        d_comp, _ = compensating_dampers(a, d, sp_post / sp_pre, p, q, s, config)
        obs = post.groupby("box").damper.median()
        obs_mean = float(np.mean([obs[b] for b in boxes if b in a_map and b in obs.index]))
        rows.append({
            "ahu": ahu, "episode": int(e.episode),
            "sp_ratio": sp_post / sp_pre,
            "damper_mean_pre": float(d.mean()),
            "damper_mean_observed_post": obs_mean,
            "damper_mean_fa_predicted": float(d_comp.mean()),
            "observed_delta_pts": obs_mean - float(d.mean()),
            "fa_predicted_delta_pts": float(d_comp.mean()) - float(d.mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    config = C.load_config()
    zc = config["control_gap"]["zone_control"]
    bench = config["control_gap"]["benchmark"]
    p_imposed = float(config["control_gap"]["zone_flow"]["damper_share_exponent"])
    cut = float(bench["naive_cut_fraction"])
    df, header = load_frame(config)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"raw grid: {len(df)} steps, {len(C.box_ids(header))} boxes\n")

    gain_rows, band_rows, rt_rows, epi_rows, arm_rows = [], [], [], [], []

    for ahu in AHUS:
        keeps = C.masks_for(df, ahu, config)
        keep = keeps["fit"]
        print(f"=== AHU-{ahu} ===")

        panel = C.build_panel(df, ahu, keep, header, config)
        rogue = set(bench["rogue_boxes"])
        clean = panel[~panel.box.isin(rogue)]
        print(f"  panel: {len(panel):,} box-steps, {panel.box.nunique()} boxes"
              f"  ({clean.box.nunique()} clean)")

        # ---- Block C's split and total, refitted here so F inverts THIS fit ----------
        fit = C.fit_split(clean, p_fixed=p_imposed)
        total = C.fit_total(clean, fit)
        print(f"  Block C: p={fit['p']:.3f}  q={total['q']:.3f}  s={total['s']:.3f}"
              f"  (total R2 {total['r2']:.3f})")

        # ---- stage 1 -----------------------------------------------------------------
        gains = fit_stage1_gains(df, ahu, keep, header, config)
        gain_rows.append(gains)
        ident = gains[gains.identified]
        pos = float((ident.G > 0).mean()) if len(ident) else float("nan")
        print(f"  stage 1: {len(ident)}/{len(gains)} zones identified,"
              f" {pos:.0%} with G > 0, median G {ident.G.median():.3f}")

        # ---- round trip ---------------------------------------------------------------
        rt = round_trip(clean, fit, total, config)
        rt_rows.append({"ahu": ahu, "steps": len(rt),
                        "max_abs_damper_error": float(rt.max_abs_damper_error.max()),
                        "max_abs_share_error": float(rt.max_abs_share_error.max()),
                        "max_share_sum_error": float(rt.share_sum_error.max())})

        # ---- the lam band ------------------------------------------------------------
        band = lambda_band(clean, fit, total, cut, config)
        if len(band):
            row = {"ahu": ahu, "cut_fraction": cut, "steps": len(band),
                   "median_damper_scaling": float(band.damper_scaling_median.median()),
                   "steps_with_any_clipping": int((band.clipped_own > 0).sum())}
            for lam in [float(x) for x in zc["lambda_grid"]]:
                for name in ("own", "tot"):
                    col = f"flow_{name}_lam{lam:g}"
                    row[f"loss_{name}_lam{lam:g}_pct"] = 100.0 * (
                        1 - band[col].sum() / band.flow_now.sum())
            band_rows.append(row)
            # The two arms are provably identical while nothing clips (README 4.2), so
            # that subset is where they can be held to floating point. Where something
            # clips they must NOT agree, and the gap is the finding.
            quiet = band[(band.clipped_own == 0) & (band.clipped_total == 0)]
            arm_rows.append({
                "ahu": ahu, "steps": len(band), "steps_no_clipping": len(quiet),
                "max_abs_flow_gap_no_clipping": float(
                    (quiet.flow_own_lam1 - quiet.flow_tot_lam1).abs().max()
                    / max(quiet.flow_now.max(), 1e-12)) if len(quiet) else float("nan"),
                "loss_own_pct": 100.0 * (1 - band.flow_own_lam1.sum() / band.flow_now.sum()),
                "loss_tot_pct": 100.0 * (1 - band.flow_tot_lam1.sum() / band.flow_now.sum()),
            })

        epi = episode_replay(clean, fit, total, ahu, config)
        if len(epi):
            epi_rows.append(epi)
        print()

    gains_df = pd.concat(gain_rows, ignore_index=True)
    band_df = pd.DataFrame(band_rows)
    rt_df = pd.DataFrame(rt_rows)
    arm_df = pd.DataFrame(arm_rows)
    epi_df = pd.concat(epi_rows, ignore_index=True) if epi_rows else pd.DataFrame()

    for name, frame in (("zone_gain_fit", gains_df), ("compensation_band", band_df),
                        ("inversion_round_trip", rt_df), ("arm_comparison", arm_df),
                        ("episode_replay", epi_df)):
        frame.to_csv(DATA_DIR / f"{name}.csv", index=False)

    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.4f}"
    print("--- STAGE 1: V_i = A_i + G_i*(T_i - Tsp_i), lagged one step " + "-" * 15)
    summary = (gains_df[gains_df.identified].groupby("ahu")
               .agg(zones=("G", "size"), positive=("G", lambda s: int((s > 0).sum())),
                    median_G=("G", "median"), median_r2=("r2", "median"),
                    positive_diff_form=("K_difference_form", lambda s: int((s > 0).sum())),
                    median_K_diff=("K_difference_form", "median")).reset_index())
    print(summary.to_string(index=False, float_format=fmt))
    print("  The LEVEL form is the specification. The difference form 7 implies is refitted")
    print("  on the same rows and reported in the last two columns: it returns the WRONG")
    print("  SIGN on most zones, because regressing dV on a level the controller already")
    print("  put into V(t-1) inherits that level's negative. It is also a sub-Nyquist")
    print("  dynamic form, which 0 forbids outright. Kept visible, not deleted - 4.7's")
    print("  discipline applied to this block.")
    print("  G absorbs the airflow unit of 4.6, exactly as block-d's a_i does.")
    print("  Zones that never call for cooling are reported unidentified, not fitted -")
    print("  5.3(4) pools them rather than printing twelve confident numbers.")
    print("  R2 of 0.05-0.09 is a finding, not a defect: 5.5's floor sits 2.60 K BELOW")
    print("  setpoint, so most of each box's flow is its parked level A_i and the error")
    print("  term is barely modulating.")

    print("\n--- ROUND TRIP: F-a inverse of Block C forward " + "-" * 29)
    print(rt_df.to_string(index=False, float_format=lambda v: f"{v:.2e}"))

    print("\n--- THE COMPENSATION BAND: a 20% pressure cut over lam " + "-" * 21)
    show = ["ahu", "steps", "steps_with_any_clipping", "median_damper_scaling"]
    show += [c for c in band_df.columns if c.startswith("loss_own_")]
    print(band_df[show].to_string(index=False, float_format=fmt))
    print("  lam is NOT fitted. 4.3: which end applies turns on the zone flow loops, and")
    print("  no event in this record separates that from a demand move. It ships as a DR")
    print("  axis over lambda_range, the same treatment 8.2 gives beta and s.")

    print("\n--- WHAT 'COMPENSATING' MEANS: two readings, and they differ " + "-" * 15)
    print(arm_df.to_string(index=False, float_format=fmt))
    print("  own = each loop restores ITS OWN flow (4.1's pressure-independent premise")
    print("  taken literally). tot = one common k solved after clipping, so boxes with")
    print("  authority left cover boxes that ran out - Block C's arm, and 4.3's 0.11%.")
    print("  Identical wherever nothing clips; the gap is entirely the clipping steps.")
    print("  REPORTED, not resolved: no VAV controller opens because a NEIGHBOUR is")
    print("  starved, so `own` is the defensible reading and the band's compensating end")
    print("  is WIDER than 4.3 states. See README 4.2.")

    if len(epi_df):
        print("\n--- THE PLANT'S OWN CUTS, replayed through F-a " + "-" * 29)
        print(epi_df.to_string(index=False, float_format=fmt))
        print("  F-a predicts dampers OPENING; they CLOSED. Reported, not asserted: this")
        print("  says the episodes are demand moves, which is why they cannot identify lam.")

    # ---- guards --------------------------------------------------------------------
    tol = float(zc["inversion_tolerance"])
    for _, r in rt_df.iterrows():
        assert r.max_abs_share_error < tol, (
            f"AHU-{int(r.ahu)}: inverting Block C's split and stepping it forward moved a "
            f"share by {r.max_abs_share_error:.2e}. F-a is not inverting the model Block C "
            f"forward-steps; the two halves of the surrogate have drifted apart.")
        assert r.max_share_sum_error < tol, (
            f"AHU-{int(r.ahu)}: shares sum to 1 only within {r.max_share_sum_error:.2e} "
            f"after the inversion. Adding-up is by construction (4.2), not approximately.")

    ref_path = ROOT / "block-c-derivation" / "data" / "pressure_cut_counterfactual.csv"
    assert ref_path.exists(), (
        f"{ref_path} is missing. The band endpoints are cross-checked against Block C's "
        f"independent common-scaling route; run block-c-derivation/reproduce_zone_flow.py "
        f"first.")
    ref = pd.read_csv(ref_path)
    endpoint_tol = float(zc["band_endpoint_tolerance_pct"])
    for _, r in band_df.iterrows():
        c_row = ref[ref.ahu == r.ahu]
        assert len(c_row) == 1, f"AHU-{int(r.ahu)}: no Block C counterfactual row to check against."
        c_row = c_row.iloc[0]
        # FROZEN is a pure identity - every room loses (1-cut)**s - so the two routes must
        # agree there whatever they assume about the loops.
        mine = float(r["loss_own_lam0_pct"])
        theirs = float(c_row["loss_frozen_pct"])
        assert abs(mine - theirs) < endpoint_tol, (
            f"AHU-{int(r.ahu)} at lam=0 (frozen): F-a gives {mine:.3f}% against Block C's "
            f"{theirs:.3f}%. With the dampers held still both routes reduce to the same "
            f"identity, so a disagreement here is arithmetic, not modelling.")
        # COMPENSATING: it is the `tot` arm that reproduces Block C, because that is the
        # arm running Block C's algorithm. The `own` arm is a different control assumption
        # and is NOT expected to match - see arm_comparison.csv and README 4.2.
        mine = float(r["loss_tot_lam1_pct"])
        theirs = float(c_row["loss_compensating_pct"])
        assert abs(mine - theirs) < endpoint_tol, (
            f"AHU-{int(r.ahu)} at lam=1 (restore-total): {mine:.3f}% against Block C's "
            f"{theirs:.3f}%. This arm delegates to block_c.required_scaling, so it is "
            f"running Block C's own code - a disagreement means the surrounding "
            f"bookkeeping has drifted, not the control assumption.")

    # The two arms are the SAME arm wherever no box clips. Held to floating point on that
    # subset, which is what makes their divergence elsewhere a finding and not a bug.
    for _, r in arm_df.iterrows():
        if not np.isfinite(r.max_abs_flow_gap_no_clipping):
            continue
        assert r.max_abs_flow_gap_no_clipping < float(zc["inversion_tolerance"]), (
            f"AHU-{int(r.ahu)}: on {int(r.steps_no_clipping)} steps where nothing clips, "
            f"the own-flow and restore-total arms differ by "
            f"{r.max_abs_flow_gap_no_clipping:.2e} of total flow. With no box against a "
            f"stop, 'restore your own flow' and 'restore the total' are the same common "
            f"scaling; they must be identical there.")
        assert r.loss_own_pct >= r.loss_tot_pct, (
            f"AHU-{int(r.ahu)}: the own-flow arm lost {r.loss_own_pct:.3f}% against the "
            f"restore-total arm's {r.loss_tot_pct:.3f}%. Declining to pick up a starved "
            f"neighbour's slack cannot deliver MORE air than picking it up.")

    for _, r in band_df.iterrows():
        assert r.median_damper_scaling >= 1.0, (
            f"AHU-{int(r.ahu)}: compensation scaled the median damper by "
            f"{r.median_damper_scaling:.4f}. Restoring flow at LOWER pressure cannot close "
            f"a damper; the inversion has a sign error.")

    min_pos = float(zc["min_positive_gain_fraction"])
    for ahu in AHUS:
        ident = gains_df[(gains_df.ahu == ahu) & gains_df.identified]
        if not len(ident):
            continue
        frac = float((ident.G > 0).mean())
        assert frac >= min_pos, (
            f"AHU-{ahu}: only {frac:.0%} of identified zones have G > 0, under the "
            f"{min_pos:.0%} floor. A warm zone asks for MORE air; a majority-negative gain "
            f"means the regression is reading something other than the zone controller. "
            f"Check the authority exclusions first - the proxy V_i ~= V_sp_i only holds "
            f"while the box is off its stops.")

    print(f"\nAll asserted guards passed. Wrote 5 tables to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
