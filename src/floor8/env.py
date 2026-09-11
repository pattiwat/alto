"""Floor8SupervisoryEnv - the Gymnasium API over the grey-box plant.

    obs, info                       = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(action)

The Gymnasium *API*, not the Gymnasium *package* - the contract is the five-tuple and `reset(seed)`,
and this repo has no external RL framework installed. `to_gymnasium()` at the foot of the file wraps
it if one is ever wanted.

THE ENV IS THIN ON PURPOSE. It holds four things - state, params, cursor, rng - and every equation
lives under `plant/`, which does not import this module. Three reasons, all specific to this repo:

  * 4.5's specific-fan-power test and block-d 7.1's 8-step open-loop rollout must run WITHOUT an
    environment. If the physics lived in `step()`, each falsifier would re-implement it and the
    divergence would be undetectable.
  * Block C has been corrected twice, both times from its derivation script. Script and simulator
    must call the same function or the audit trail validates something the agent never runs.
  * rl-environment-design 6 requires bit-exact determinism under a fixed seed.

WHAT THIS ENV DELIBERATELY DOES NOT DO
--------------------------------------
    the Lagrangian   - it emits RAW constraint costs in `info` and dual ascent lives in the training
                       loop, so one env serves SAC-Lagrangian, plain SAC and the MPC arm of 2.3
                       without three environments that can silently diverge
    fitting          - parameters arrive frozen from `build_params`
    the frontier assertion  - that belongs to the comparison, not the reward
    reading the CSV  - the tape does that
"""

from __future__ import annotations

import numpy as np

from .plant.params import PlantParams
from .plant.state import PlantState
from .plant.step import step as plant_step
from .tape import STEPS_PER_DAY, DayTape, EpisodeTape

# 9: the observed ranges of the logged setpoints, per AHU. Not engineering limits.
BOUNDS = {
    "sp_lo": np.array([0.30, 0.40]), "sp_hi": np.array([0.64, 0.78]), "sp_rate": 0.02,
    # AHU-1's SAT span is 1.27 K - four steps at the rate limit. Any AHU-1 SAT-reset saving rests
    # on that width, and the agent will sit on a bound. Stated in 9 rather than left as
    # "observed span".
    "sat_lo": np.array([17.50, 16.00]), "sat_hi": np.array([18.77, 19.00]), "sat_rate": 0.3,
}

OBS_NAMES: tuple[str, ...] = (
    "T_oa", "T_wb", "occ", "sin_h", "cos_h", "sin_dow", "cos_dow",
    *[f"{p}_{a}" for a in (1, 2) for p in
      ("SP_sp", "SP_act", "SAT_sp", "T_sa", "f", "P_fan", "V_total", "agg_cond",
       "n_requests", "damper_2nd", "damper_3rd", "n_clipped", "err_max", "err_mean", "T_p95")],
    "flag_fan_off", "flag_outage", "flag_unscored",
)
# Declared PHYSICAL ranges, not dataset statistics (3). This CSV mixes degC, degF, inWG, L/min and
# Hz on the same AHU; a scaler derived from data silently changes when the mask changes.
OBS_RANGE = {
    "T_oa": (15.0, 45.0), "T_wb": (10.0, 35.0), "occ": (0.0, 1.0),
    "SP_sp": (0.2, 0.9), "SP_act": (0.0, 0.9), "SAT_sp": (14.0, 20.0), "T_sa": (8.0, 30.0),
    "f": (0.0, 50.0), "P_fan": (0.0, 12.0), "V_total": (0.0, 40000.0),
    "agg_cond": (0.0, 120.0), "n_requests": (0.0, 30.0),
    "damper_2nd": (0.0, 100.0), "damper_3rd": (0.0, 100.0), "n_clipped": (0.0, 30.0),
    "err_max": (-8.0, 8.0), "err_mean": (-8.0, 8.0), "T_p95": (18.0, 32.0),
}


class Floor8SupervisoryEnv:
    """One env, both AHUs - the comfort constraint is floor-wide and the zones partition across."""

    action_dim = 4          # (dSP_1, dSAT_1, dSP_2, dSAT_2), each in [-1, 1]
    action_low, action_high = -1.0, 1.0

    def __init__(self, tape: EpisodeTape, params: PlantParams, day_indices: list[int],
                 param_sampler=None, comfort_deadband_k: float = 0.5,
                 # `move` is in units of the rate limit and reaches 4.0 when every lever moves at
                 # full rate, while a step's actual energy is ~1.5 kWh. At 0.5 the smoothness term
                 # was 1.3x the entire objective and the agent optimised for standing still. Set so
                 # a full-rate move costs ~5% of a step's energy: a nudge against actuator chatter,
                 # not a competing objective.
                 lam_move: float = 0.02, cop: float = 4.0, horizon: int = STEPS_PER_DAY,
                 # Which comfort statistic the CONSTRAINT is written on. Both are always computed
                 # and both always reach `info`; this only picks what `cost_comfort` returns.
                 #   "one_sided"  max(0, T_i - Tsp - db)   - too warm only. The original.
                 #   "two_sided"  max(0, |T_i - Tsp| - db) - holds the setpoint, so OVERCOOLING
                 #                                          is a violation rather than free.
                 # Defaulted to the original so every existing script reproduces its own numbers;
                 # the switch has to be asked for.
                 comfort_mode: str = "one_sided"):
        if comfort_mode not in ("one_sided", "two_sided"):
            raise ValueError(f"comfort_mode must be 'one_sided' or 'two_sided', got "
                             f"{comfort_mode!r}")
        self.tape, self.base_params, self.day_indices = tape, params, list(day_indices)
        self.param_sampler = param_sampler
        self.comfort_deadband_k, self.lam_move, self.cop = comfort_deadband_k, lam_move, cop
        self.comfort_mode = comfort_mode
        self.horizon = horizon
        self.obs_dim = len(OBS_NAMES)
        self._scale_lo, self._scale_hi = self._build_scaler()
        self.np_random = np.random.default_rng(0)
        self.params: PlantParams = params
        self.day: DayTape | None = None
        self.state: PlantState | None = None
        self.t = 0

    # ------------------------------------------------------------------ Gymnasium API
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        i = int(self.np_random.choice(self.day_indices)) if options is None or "day" not in options \
            else int(options["day"])
        self.day = self.tape.day(i)
        # DR is drawn HERE - not in the plant, which must be deterministic given params, and not in
        # a wrapper, which cannot reach params cleanly. The draw is echoed to info for audit.
        self.params, dr = (self.param_sampler(self.base_params, self.np_random)
                           if self.param_sampler else (self.base_params, {}))
        self.t = 0
        self.state = self._initial_state()
        return self._obs(None), {"day_index": i, "date": str(self.day.date), "dr": dr}

    def step(self, action: np.ndarray):
        if self.state is None:
            raise RuntimeError("step() before reset()")
        if self.t >= self.horizon:
            raise RuntimeError(
                f"step() called after truncation (t={self.t} of {self.horizon}). The episode is "
                f"over - call reset(). Continuing would silently replay the last exogenous row and "
                f"credit the agent with a day that has no boundary conditions.")
        a = np.clip(np.asarray(action, dtype=float).reshape(4), -1.0, 1.0)
        d_sp = np.array([a[0], a[2]]) * BOUNDS["sp_rate"]
        d_sat = np.array([a[1], a[3]]) * BOUNDS["sat_rate"]
        prev = self.state

        noise = self.np_random.normal(0.0, np.array(self.params.E.sigma), size=2)
        self.state, diag = plant_step(prev, self.day.exog(self.t), (d_sp, d_sat),
                                      self.params, noise, BOUNDS, self.comfort_deadband_k)

        scored = bool(self.day.score_mask[self.t])
        move = float(np.abs(self.state.SP_sp - prev.SP_sp).sum() / BOUNDS["sp_rate"]
                     + np.abs(self.state.SAT_sp - prev.SAT_sp).sum() / BOUNDS["sat_rate"])

        # The coil MUST be in the objective or the optimum is degenerate: with fan-only reward the
        # agent cuts SP, the zones warm, and it restores comfort by lowering SAT - moving cost onto
        # the UNMETERED coil, which is 4.4x the fan thermally (6.4). Reported as a COP band, never
        # as a point.
        e_fan = diag.E_fan_kwh
        e_coil = diag.Q_coil_kwh_th / self.cop
        reward = -(e_fan + e_coil) - self.lam_move * move if scored else 0.0

        # A non-finite or absurd reward means the physics is wrong, not that the agent found
        # something. Fail loudly here: a silent NaN propagates into alpha and the multipliers and
        # the run still "trains", just on nothing. This fired once already - the fan-model band
        # switching functional form while keeping the other form's coefficients.
        if not np.isfinite(reward) or abs(reward) > 1e4:
            raise FloatingPointError(
                f"implausible reward {reward!r} at t={self.t} on {self.day.date.date()}: "
                f"fan {e_fan:.3g} kWh, coil {diag.Q_coil_kwh_th:.3g} kWh-th, f={self.state.f}, "
                f"V_total={diag.V_total}, fan model {self.params.B.kind!r}. "
                f"A 15-minute step cannot use this much energy; check the block that produced it.")

        # Constraint costs go out RAW. Dual ascent is the training loop's job.
        # BOTH comfort statistics are always computed; `comfort_mode` only decides which one the
        # constraint is written on, so a run can be re-scored on the other without re-simulating.
        c_one = float(np.sum(diag.comfort_excess_k ** 2)) if scored else 0.0
        c_two = float(np.sum(diag.comfort_dev_k ** 2)) if scored else 0.0
        cost_comfort = c_two if self.comfort_mode == "two_sided" else c_one
        cost_vent = float(diag.vent_infeasible.sum()) if scored else 0.0

        self.t += 1
        truncated = self.t >= self.horizon
        info = {
            "scored": scored, "cost_comfort": cost_comfort, "cost_vent": cost_vent,
            "cost_comfort_one_sided": c_one, "cost_comfort_two_sided": c_two,
            "comfort_mode": self.comfort_mode,
            "e_fan_kwh": e_fan if scored else 0.0,
            "q_coil_kwh_th": diag.Q_coil_kwh_th if scored else 0.0,
            "q_coil_ahu_kwh_th": diag.Q_coil_kw * self.params.dt_h if scored else np.zeros(2),
            "v_total": diag.V_total.copy(), "sfp": diag.sfp.copy(),
            "n_clipped": diag.n_clipped.copy(), "sp_shortfall": diag.sp_shortfall.copy(),
            "comfort_excess_k": diag.comfort_excess_k.copy(),
            "comfort_dev_k": diag.comfort_dev_k.copy(),
            "v_min_violation": diag.v_min_violation.copy(),
            "P_fan": diag.P_fan.copy(), "T_sa": self.state.T_sa.copy(),
            "f": self.state.f.copy(), "T_i": self.state.T_i.copy(),
        }
        # `terminated` is ALWAYS False. Gymnasium semantics: terminated means an MDP-absorbing
        # state, and SAC zeroes the bootstrap target on it - marking the end of a day terminal
        # teaches the critic the world ends at midnight. End-of-horizon is `truncated`.
        return self._obs(diag), float(reward), False, bool(truncated), info

    # ---------------------------------------------------------------------- internals
    def _initial_state(self) -> PlantState:
        P, d = self.params, self.day
        lg = d.logged
        n = P.n_boxes
        T0 = np.where(np.isfinite(lg["T_i"][0]), lg["T_i"][0], d.T_m)
        d0 = np.where(np.isfinite(lg["d_i"][0]), lg["d_i"][0], 50.0)
        v0 = np.where(np.isfinite(lg["V_i"][0]), lg["V_i"][0], 0.0)
        first = lambda arr, fb: np.where(np.isfinite(arr[0]), arr[0], fb)
        return PlantState(
            T_i=T0, V_sp_i=v0.copy(), d_i=d0, V_i=v0,
            T_sa=first(lg["T_sa"], np.array([18.0, 18.0])),
            SP_act=first(lg["SP_act"], np.array([0.55, 0.60])),
            f=first(lg["f"], np.array([35.0, 35.0])),
            SP_sp=np.clip(first(lg["SP_sp"], np.array([0.55, 0.60])),
                          BOUNDS["sp_lo"], BOUNDS["sp_hi"]),
            SAT_sp=np.clip(first(lg["SAT_sp"], np.array([18.0, 17.5])),
                           BOUNDS["sat_lo"], BOUNDS["sat_hi"]),
        )

    def _build_scaler(self):
        lo = np.array([OBS_RANGE.get(n.rsplit("_", 1)[0] if n[-1] in "12" else n, (0.0, 1.0))[0]
                       for n in OBS_NAMES])
        hi = np.array([OBS_RANGE.get(n.rsplit("_", 1)[0] if n[-1] in "12" else n, (0.0, 1.0))[1]
                       for n in OBS_NAMES])
        return lo, hi

    def _obs(self, diag) -> np.ndarray:
        P, d, st, t = self.params, self.day, self.state, min(self.t, self.horizon - 1)
        hour = t * 0.25
        dow = d.date.dayofweek
        row = [d.T_oa[t], d.T_wb[t], d.occ[t],
               np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
               np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
        Tsp = d.T_sp_zone[t]
        for ai in (0, 1):
            m = P.mask_ahu(ai)
            # The DEMAND statistic excludes the rogues (the G36 ignore-top-I rule); the COMFORT
            # statistic below does not - 4.3, "a starved zone is still a zone".
            demand = m & ~P.rogue
            dd = np.sort(st.d_i[demand])[::-1] if demand.any() else np.zeros(3)
            dd = np.pad(dd, (0, max(0, 3 - len(dd))))
            err = st.T_i[m] - Tsp[m]
            row += [st.SP_sp[ai], st.SP_act[ai], st.SAT_sp[ai], st.T_sa[ai], st.f[ai],
                    0.0 if diag is None else diag.P_fan[ai],
                    0.0 if diag is None else diag.V_total[ai],
                    float(np.sum(P.C.a_i[m] * st.d_i[m])),
                    float(np.sum(st.d_i[demand] >= 90.0)), dd[1], dd[2],
                    0.0 if diag is None else float(diag.n_clipped[ai]),
                    float(np.nanmax(err)) if err.size else 0.0,
                    float(np.nanmean(err)) if err.size else 0.0,
                    float(np.nanpercentile(st.T_i[m], 95)) if m.any() else 24.0]
        row += [float(not d.fan_on[t].any()), 0.0, float(not d.score_mask[t])]
        x = np.nan_to_num(np.array(row, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(2.0 * (x - self._scale_lo) / (self._scale_hi - self._scale_lo) - 1.0,
                       -5.0, 5.0)


def default_param_sampler(cfg: dict):
    """8.2's domain randomisation: the sensitivity axes ARE the training distribution.

    A policy trained across these is robust to the uncertainty that was actually MEASURED, and the
    tornado chart becomes a description of the environment rather than a post-hoc defence.

    No DR parameter reaches the observation. That is the point of randomising: a policy that could
    see beta would learn to exploit the easy draws instead of becoming robust to all of them.
    """
    import dataclasses

    def sample(base: PlantParams, rng: np.random.Generator):
        A, B, C, F = base.A, base.B, base.C, base.F
        beta = tuple(float(rng.uniform(lo, hi)) for lo, hi in A.beta_ci)
        s = tuple(float(rng.uniform(0.5, 0.9)) for _ in (0, 1))          # 4.3, 4.7
        lam = float(rng.uniform(0.0, 1.0))                               # 7.3 - block-f 4
        kind = str(rng.choice(["offset_cubic", "power_law", "ideal_cubic"]))
        cop = float(rng.uniform(3.0, 5.0))                               # 6.4 - absent from data
        p = base.with_(A=dataclasses.replace(A, beta=beta),
                       B=dataclasses.replace(B, kind=kind),
                       C=dataclasses.replace(C, s=s),
                       F=dataclasses.replace(F, lam=lam))
        return p, {"beta": beta, "s": s, "lam": lam, "fan_model": kind, "cop": cop}
    return sample
