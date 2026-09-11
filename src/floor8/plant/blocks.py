"""Blocks A-F as pure functions. No Gymnasium, no RNG, no I/O.

One module rather than six files, because the six are twenty lines each and splitting them would
cost more in import ceremony than it buys - but each function is 1:1 with a `block-*-derivation/`
folder and says which, so a correction still has exactly one home.

The forward models here are the ones the derivation scripts fit against. Where a derivation script
already owns an algorithm - Block C's split, Block F's damper inversion - the intent is that the
script imports from here rather than the reverse, so the model the agent steps and the model the
audit trail validates cannot drift apart. That was the concrete lesson of Block C being wrong twice.

UNITS. Airflow is m3/h throughout, per block-e-derivation/data/airflow_unit.csv (m3/h wins the
discrimination by a margin of 25.0 on AHU-1 and 16.5 on AHU-2). Temperatures degC, pressure inWG,
power kW, frequency Hz, energy kWh.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# Block A - static pressure setpoint -> fan frequency        block-a-derivation
# --------------------------------------------------------------------------------------

def fan_frequency(sp_act: np.ndarray, f_ref: np.ndarray, sp_ref: np.ndarray,
                  beta: np.ndarray) -> np.ndarray:
    """f = f_ref * (SP_act/SP_ref)**beta.

    `beta` is a REDUCED-FORM elasticity measured from the plant's own setpoint episodes, not the
    textbook affinity 1/2. It already contains the damper response (2.2), which is why Block C must
    not be wired into this - doing so double-counts it. 1.1 keeps them as independent projections
    and 4.5 is the test that keeps them honest.
    """
    return f_ref * np.power(np.maximum(sp_act, 1e-9) / sp_ref, beta)


def required_frequency_and_achieved_sp(sp_sp: np.ndarray, f_ref: np.ndarray, sp_ref: np.ndarray,
                                       beta: np.ndarray, f_max: float = 50.0
                                       ) -> tuple[np.ndarray, np.ndarray]:
    """4.4 - when the fan cannot make pressure, the setpoint is not achieved.

    AHU-2 falls below 0.8x its setpoint on 12% of steps and its SAT loop is saturated on 29.6% of
    large-error steps: part of it is genuinely short of capacity. A surrogate that always grants the
    requested pressure would let an agent write setpoints AHU-2 cannot deliver and then bank comfort
    that never arrived.
    """
    f_req = fan_frequency(sp_sp, f_ref, sp_ref, beta)
    f = np.minimum(f_req, f_max)
    sp_act = np.where(f_req > f_max, sp_ref * np.power(f / f_ref, 1.0 / beta), sp_sp)
    return f, sp_act


# --------------------------------------------------------------------------------------
# Block B - frequency -> fan power                           block-b-derivation
# --------------------------------------------------------------------------------------

def fan_power(f: np.ndarray, a: np.ndarray, b: np.ndarray, kind: str = "offset_cubic",
              rated_hz: float = 50.0) -> np.ndarray:
    """P = a + b*(f/50)**3 (headline), or the two band arms.

    The three models are a DR axis, not a choice: reporting the ideal cubic alone would overstate
    savings by roughly 2x (3). The offset term is the fixed motor/VSD loss, which is why a naive
    power law comes out near 1.5 rather than 3 over this floor's narrow 33-41 Hz band.
    """
    x = f / rated_hz
    if kind == "offset_cubic":
        return a + b * x ** 3
    if kind == "ideal_cubic":
        return b * x ** 3
    if kind == "power_law":
        return a * np.power(np.maximum(f, 1e-9), b)
    raise ValueError(f"unknown fan model {kind!r}")


# --------------------------------------------------------------------------------------
# Block C - static pressure + dampers -> delivered air        block-c-derivation
# --------------------------------------------------------------------------------------

def split_weights(a_i: np.ndarray, d_i: np.ndarray, p: float = 1.0) -> np.ndarray:
    """w_i = a_i * d_i**p. Static pressure is absent BY CONSTRUCTION (4.2)."""
    return a_i * np.power(np.maximum(d_i, 0.0), p)


def deliver(a_i: np.ndarray, d_i: np.ndarray, sp_act: float, ln_C: float, q: float, s: float,
            p: float = 1.0) -> tuple[np.ndarray, float]:
    """One AHU: returns (V_i per box, V_total).

    Pressure cancels from the split - it is common to every box at a step, so it divides away. Which
    room gets what air is purely relative damper opening; pressure only sets how much there is to
    divide. Adding-up therefore holds to floating point rather than approximately.
    """
    w = split_weights(a_i, d_i, p)
    W = float(w.sum())
    if W <= 0.0:
        return np.zeros_like(w), 0.0
    V_total = float(np.exp(ln_C) * W ** q * max(sp_act, 1e-9) ** s)
    return V_total * w / W, V_total


def dampers_for_targets(a_i: np.ndarray, v_target: np.ndarray, sp_act: float, ln_C: float,
                        q: float, s: float, p: float = 1.0, cap: float = 100.0,
                        max_iter: int = 40) -> np.ndarray:
    """INVERSE of `deliver`: the dampers that would deliver `v_target` at `sp_act`.

    Closed form. From V_i = C*W**(q-1)*SP**s*a_i*d_i**p, summing over boxes gives
    W**q = V_target_total/(C*SP**s), hence d_i = (V_i*W/(a_i*V_total))**(1/p). No search.

    The loop exists only to redistribute after boxes hit the 100% stop. A clipped box delivers what
    it can and its shortfall is NOT taken up by its neighbours - each loop tracks its own flow
    setpoint, which is 4.1's pressure-independent premise taken literally. See
    block-f-derivation/README.md 4.2: the other reading, in which neighbours cover the shortfall,
    is what 4.3 quoted, and the two differ once anything clips.
    """
    v_tot = float(np.sum(v_target))
    if v_tot <= 0.0:
        return np.zeros_like(a_i)
    safe_a = np.where(a_i > 0, a_i, np.inf)          # a dead box (a_i = 0) can ask for any damper
    stopped = np.zeros(len(a_i), dtype=bool)
    d = np.zeros(len(a_i))
    for _ in range(max_iter):
        w_stopped = float(np.sum(a_i[stopped] * cap ** p))
        free = ~stopped
        v_free = float(np.sum(v_target[free]))
        if v_free <= 0.0:
            d = np.where(stopped, cap, 0.0)
            break
        W = _solve_W(w_stopped, v_free, sp_act, ln_C, q, s)
        # Each free box holds ITS OWN target: from V_i = C*W**(q-1)*SP**s*a_i*d_i**p,
        #     a_i*d_i**p = V_target_i / (C * W**(q-1) * SP**s).
        # Note the denominator uses the ACHIEVED W, not the requested total - once a box clips, the
        # achieved total falls below what was asked for, and dividing by the request instead would
        # quietly hand the free boxes the clipped box's share.
        denom = np.exp(ln_C) * W ** (q - 1.0) * max(sp_act, 1e-9) ** s
        w_free = v_target / max(denom, 1e-300)        # a_i * d_i**p for each free box
        with np.errstate(divide="ignore", invalid="ignore"):
            d_free = np.power(np.maximum(w_free, 0.0) / safe_a, 1.0 / p)
        d = np.where(stopped, cap, np.minimum(d_free, cap))
        newly = free & (d_free > cap)
        if not newly.any():
            break
        stopped |= newly
    return d


def _solve_W(w_stopped: float, v_free: float, sp_act: float, ln_C: float, q: float,
             s: float) -> float:
    """Solve  W**(q-1) * (W - w_stopped)  =  v_free / (C * SP**s)  for W >= w_stopped.

    Monotone increasing in W above `w_stopped`, so bisection is exact to machine precision in ~80
    halvings. This is the only non-closed-form step in the whole plant.
    """
    target = v_free / (np.exp(ln_C) * max(sp_act, 1e-9) ** s)
    lhs = lambda W: W ** (q - 1.0) * (W - w_stopped)
    lo = w_stopped
    hi = max(w_stopped, 1.0) * 2.0
    for _ in range(200):
        if lhs(hi) >= target:
            break
        hi *= 2.0
    else:                                             # pragma: no cover
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if lhs(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# Block D - zone thermal response                            block-d-derivation
# --------------------------------------------------------------------------------------

def zone_step(T_i, T_m, V_i, T_sa_of_box, T_oa, occ, k, a, g, delta, c) -> np.ndarray:
    """T_i(t+1), the fitted two-node map. NOT integrated - see the module note below.

        T(t+1) - T(t) = k*(T_m - T) + a*V*(T_sa - T) + g*(T_oa - T) + delta*occ + c

    `k, a, g` ALREADY CARRY dt = 0.25 h (k = dt/(R_m*C) and so on), and the implied time constant is
    recovered by tau = -dt/ln(1 - k - a*V - g), the exact zero-order-hold inversion rather than an
    Euler approximation. Handing these to an ODE solver would double-count dt; and RK4 would need
    T_oa, occ and T_sa at t+h/2, which 15-minute AVERAGES do not contain. One vectorised affine map
    is both the correct discretisation and what makes 1e6 SAC steps affordable.

    `T_m` is replayed from the night plateau (block-d 5.1), not simulated: the mass node moves at
    0.65 K per uncooled day, far slower than an episode.
    """
    dT = (k * (T_m - T_i)
          + a * V_i * (T_sa_of_box - T_i)
          + g * (T_oa - T_i)
          + delta * occ
          + c)
    return T_i + dT


# --------------------------------------------------------------------------------------
# Block E - supply air temperature and coil load             block-e-derivation
# --------------------------------------------------------------------------------------

def supply_air_temperature(sat_sp: np.ndarray, b0: np.ndarray, b1: np.ndarray,
                           valve_proxy: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """T_sa = SAT_sp + b0 + b1*x + eta.

    The obvious choice T_sa = SAT_sp is wrong and expensive: the loop's mean tracking error is
    -1.02 K on AHU-1, with 574 of 588 large-error steps at a MID-RANGE valve - mis-tuned, not
    starved (control-gap 3). Assuming perfect tracking hands the agent credit for a commissioning
    fix it never performed.

    b1 SHIPS AT ZERO. 6.1's own regressor is valve position, which no block produces and which
    block-e 5 therefore rules unsimulable; of the admissible simulable alternatives none beat a
    constant bias by enough to justify the endogeneity it brings. So this is a constant per AHU
    carried with its residual sd (1.79 / 2.38 K), and `T_ra` is offered as a DR arm instead.
    """
    return sat_sp + b0 + b1 * valve_proxy + noise


def mixed_air(T_oa: float, T_ra: np.ndarray, damper_pct: np.ndarray, phi0: np.ndarray,
              phi1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T_ma = phi*T_oa + (1-phi)*T_ra, with phi an affine map of the OA damper, clipped to [0,1].

    Damper position is not linear in flow fraction, so phi is not read off it directly (6.2). The
    fitted phi1 is NEGATIVE on both AHUs (-0.0179 / -0.0132), which has no physical reading - more
    open should admit more outside air - and block-e 7 reports it as such rather than repairing it.
    Clipping to [0,1] keeps the mixing physical while leaving that defect visible in the artifact.
    """
    phi = np.clip(phi0 + phi1 * damper_pct, 0.0, 1.0)
    return phi * T_oa + (1.0 - phi) * T_ra, phi


def coil_duty(V_total_m3_h: np.ndarray, T_ma: np.ndarray, T_sa: np.ndarray,
              rho_cp: float = 1.21) -> np.ndarray:
    """Q = rho*cp * V * (T_ma - T_sa), kW THERMAL.

    Thermal, and it stays thermal in `info`. 6.4: no chiller COP exists in this dataset - the plant
    is off-floor and unmetered - so any electric number here is a labelled band, never a constant
    folded into a scalar reward.
    """
    return rho_cp * (V_total_m3_h / 3600.0) * (T_ma - T_sa)


# --------------------------------------------------------------------------------------
# Block F - the zone controllers                             block-f-derivation
# --------------------------------------------------------------------------------------

def zone_flow_setpoint(T_i: np.ndarray, T_sp: np.ndarray, G: np.ndarray, A: np.ndarray,
                       v_min: np.ndarray, v_max: np.ndarray) -> np.ndarray:
    """Stage 1: V_sp = clip(A_i + G_i*(T_i - Tsp_i), V_min_i, V_max_i). A LEVEL, not an increment.

    0's Nyquist argument used forward: the damper loop settles inside a 15-minute step, so what the
    record can see is the static map from error to flow, not the integration that produced it. The
    increment form 7 originally specified is sub-Nyquist AND returns the wrong sign on 18 of 19
    AHU-1 zones - block-f-derivation 2.3.
    """
    return np.clip(A + G * (T_i - T_sp), v_min, v_max)


def zone_dampers(v_sp: np.ndarray, a_i: np.ndarray, d_now: np.ndarray, sp_act: float,
                 ln_C: float, q: float, s: float, p: float = 1.0, lam: float = 1.0,
                 cap: float = 100.0) -> np.ndarray:
    """Stage 2, with the compensation axis: d = d_now + lam*(d_wanted - d_now).

    lam = 1 the loops have full authority and chase their setpoints.
    lam = 0 the dampers do not move at all.

    lam is NOT fitted and this is not an omission. 4.3: which end applies turns on the zone flow
    loops, and no event in this record separates that from a demand move - pressure and demand moved
    together in every episode it contains. It ships as a DR axis, exactly as beta and s do (8.2).
    """
    d_want = dampers_for_targets(a_i, v_sp, sp_act, ln_C, q, s, p, cap)
    return np.clip(d_now + lam * (d_want - d_now), 0.0, cap)
