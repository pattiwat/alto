"""The controllers being compared. All four are plain callables over the SAME environment.

    policy(obs, env) -> action in [-1, 1]**4

Not special-cased inside the env, because rl-environment-design 5.2 requires every policy to be
scored through an identical mask - "if the agent is scored on more steps than the baseline, the mask
treatment alone manufactures the saving." Keeping them outside as callables is what makes that
assertion mean something rather than being a promise.

The action is a DELTA on the current setpoint, rate-limited and clipped inside the plant (9). So a
policy that wants an absolute setpoint has to steer toward it, exactly as the agent does - which is
what makes the comparison fair. `_toward` does that steering.
"""

from __future__ import annotations

import numpy as np

from .env import BOUNDS


def _toward(current: np.ndarray, target: np.ndarray, rate: float) -> np.ndarray:
    """Unit-scaled action that moves `current` toward `target` as fast as the rate limit allows."""
    return np.clip((target - current) / rate, -1.0, 1.0)


def as_operated(obs, env) -> np.ndarray:
    """Replay the logged setpoints. THE reference arm.

    This is the policy the record itself ran, so simulating it is what tests whether the surrogate
    reproduces the plant under the plant's own actions. If this arm does not track the measured fan
    power, supply air temperature and zone temperatures, no comparison against it means anything.
    """
    t = min(env.t, env.horizon - 1)
    lg = env.day.logged
    sp_t = np.nan_to_num(lg["SP_sp"][t], nan=0.55)
    sat_t = np.nan_to_num(lg["SAT_sp"][t], nan=18.0)
    a_sp = _toward(env.state.SP_sp, np.clip(sp_t, BOUNDS["sp_lo"], BOUNDS["sp_hi"]),
                   BOUNDS["sp_rate"])
    a_sat = _toward(env.state.SAT_sp, np.clip(sat_t, BOUNDS["sat_lo"], BOUNDS["sat_hi"]),
                    BOUNDS["sat_rate"])
    return np.array([a_sp[0], a_sat[0], a_sp[1], a_sat[1]])


def make_naive_cut(fraction: float = 0.20):
    """The screwdriver: a blunt fixed reduction of the static-pressure setpoint, no logic behind it.

    Set to the size of cut the plant itself demonstrated during its setpoint episodes (~21%), so the
    modelled saving and the measured response describe the same intervention. SAT is left alone.
    """
    def policy(obs, env) -> np.ndarray:
        t = min(env.t, env.horizon - 1)
        lg = env.day.logged
        target_sp = np.clip(np.nan_to_num(lg["SP_sp"][t], nan=0.55) * (1.0 - fraction),
                            BOUNDS["sp_lo"], BOUNDS["sp_hi"])
        target_sat = np.clip(np.nan_to_num(lg["SAT_sp"][t], nan=18.0),
                             BOUNDS["sat_lo"], BOUNDS["sat_hi"])
        a_sp = _toward(env.state.SP_sp, target_sp, BOUNDS["sp_rate"])
        a_sat = _toward(env.state.SAT_sp, target_sat, BOUNDS["sat_rate"])
        return np.array([a_sp[0], a_sat[0], a_sp[1], a_sat[1]])
    return policy


def make_g36(ignore_top: int = 2, request_damper_pct: float = 90.0,
             trim: float = 0.02, respond: float = 0.03) -> callable:
    """ASHRAE Guideline 36 trim-and-respond on static pressure.

    Every step, count the boxes REQUESTING more air (damper at or above `request_damper_pct`),
    ignore the top `ignore_top`, and: no requests -> trim down; requests -> respond up. The
    ignore-top-I rule is what stops one stuck damper holding the whole floor at high pressure, and
    the rogue exclusion of 4.3 is the same idea applied to boxes that are stuck permanently.

    This is the arm that matters most for the verdict. The real supervisory action IS discrete and
    IS this table (rl-environment-design 2.4); if a learned policy cannot beat it, the honest report
    says the deployable controller is a trim table.
    """
    def policy(obs, env) -> np.ndarray:
        P, st = env.params, env.state
        a = np.zeros(4)
        for ai in (0, 1):
            m = P.mask_ahu(ai) & ~P.rogue
            d = np.sort(st.d_i[m])[::-1]
            requests = int(np.sum(d[ignore_top:] >= request_damper_pct))
            delta = respond if requests > 0 else -trim
            target = np.clip(st.SP_sp[ai] + delta, BOUNDS["sp_lo"][ai], BOUNDS["sp_hi"][ai])
            a[2 * ai] = np.clip((target - st.SP_sp[ai]) / BOUNDS["sp_rate"], -1.0, 1.0)
            t = min(env.t, env.horizon - 1)
            sat_t = np.nan_to_num(env.day.logged["SAT_sp"][t], nan=18.0)[ai]
            a[2 * ai + 1] = np.clip(
                (np.clip(sat_t, BOUNDS["sat_lo"][ai], BOUNDS["sat_hi"][ai]) - st.SAT_sp[ai])
                / BOUNDS["sat_rate"], -1.0, 1.0)
        return a
    return policy


def make_sac_policy(agent, deterministic: bool = True):
    """The trained agent, evaluated at the mean of its Gaussian rather than sampled."""
    def policy(obs, env) -> np.ndarray:
        return agent.act(obs, deterministic=deterministic)
    return policy
