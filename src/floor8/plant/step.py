"""One plant step: 1.1's block order, executed literally and in one place.

    step(state, exog, action, params, noise) -> (state', Diagnostics)

Pure. Same inputs, same outputs, always. The only stochastic term in the physics is Block E's
eta ~ N(0, sigma^2), and it is PASSED IN rather than drawn - so the plant is deterministic, all
randomness lives at the environment boundary, and reproducing a trajectory needs no RNG archaeology.

THE ORDER IS PART OF THE SPECIFICATION. 1.1 says so, and the reason is that it is what stops the
model being circular. Written as a comment it is a claim; written as this function it is the thing
that actually runs.

    u_t  --> clip to logged support, rate-limit          -> SP_sp', SAT_sp'
      |
      +-> E1   T_sa    = SAT_sp' + b0 + eta
      +-> F1   V_sp_i  = clip(A_i + G_i*(T_i - Tsp_i), V_min, V_max)      [stage 1, a LEVEL]
      +-> A    f, SP_act  - and 4.4's shortfall if the fan cannot make pressure
      +-> F2   d_i     = d_i + lam*(C^-1(V_sp_i, SP_act) - d_i)           [stage 2]
      +-> C    V_i, V_total from (a_i d_i, SP_act)
      +-> B    P_fan
      +-> D    T_i(t+1)
      +-> E2   T_ma, Q_coil                                    -> info only, never the reward

BLOCKS A AND C ARE TWO PROJECTIONS OF ONE EVENT, NOT A CHAIN. A gives fan speed, C gives delivered
air; neither feeds the other, because beta is a reduced-form elasticity that already contains the
damper response (2.2). Wiring C into A would double-count it. 4.5's specific-fan-power test is what
keeps them honest, and it is a real test precisely because they are computed independently here.

WHERE F2 SITS, AND WHY IT IS NOT CIRCULAR. Stage 2 needs SP_act, which comes from A, which needs
only the SETPOINT - not delivered air. So the chain SP_sp -> (f, SP_act) -> d_i -> V_i runs one way.
Block C's own output never returns to Block A.
"""

from __future__ import annotations

import numpy as np

from .blocks import (coil_duty, deliver, fan_power, mixed_air,
                     required_frequency_and_achieved_sp, supply_air_temperature, zone_dampers,
                     zone_flow_setpoint, zone_step)
from .params import PlantParams
from .state import Diagnostics, PlantState

AHUS = (0, 1)


def clip_action(state: PlantState, d_sp: np.ndarray, d_sat: np.ndarray, bounds: dict
                ) -> tuple[np.ndarray, np.ndarray]:
    """9: rate-limit, then clip to the OBSERVED range of the logged setpoints.

    Not engineering limits - the boundary of what this record can defend. Outside it every block is
    extrapolating: beta was identified from cuts of ~17-30%, the fan curve from 33-41 Hz, the zone
    models from the flows those setpoints produced. An agent allowed outside would find savings in
    the tails of fitted polynomials and report them as building physics.
    """
    sp = np.clip(state.SP_sp + np.clip(d_sp, -bounds["sp_rate"], bounds["sp_rate"]),
                 bounds["sp_lo"], bounds["sp_hi"])
    sat = np.clip(state.SAT_sp + np.clip(d_sat, -bounds["sat_rate"], bounds["sat_rate"]),
                  bounds["sat_lo"], bounds["sat_hi"])
    return sp, sat


def step(state: PlantState, exog: dict, action: tuple[np.ndarray, np.ndarray],
         params: PlantParams, noise: np.ndarray, bounds: dict,
         comfort_deadband_k: float = 0.5) -> tuple[PlantState, Diagnostics]:
    """Advance one 15-minute step. `exog` is replayed, never simulated (1.1)."""
    P, n = params, params.n_boxes
    d_sp, d_sat = action
    SP_sp, SAT_sp = clip_action(state, d_sp, d_sat, bounds)

    T_oa = float(exog["T_oa"])
    occ = float(exog["occ"])
    T_sp_zone = np.asarray(exog["T_sp_zone"], dtype=float)
    T_m = np.asarray(exog["T_m"], dtype=float)
    fan_on = np.asarray(exog["fan_on"], dtype=bool)

    beta = np.array(P.A.beta)
    f_ref = np.array(P.A.f_ref)
    sp_ref = np.array(P.A.sp_ref)

    # --- E1 : supply air temperature -------------------------------------------------
    T_sa = supply_air_temperature(SAT_sp, np.array(P.E.b0), np.array(P.E.b1),
                                  np.zeros(2), noise[:2])

    # --- F1 : stage 1, the zone controllers ------------------------------------------
    V_sp = zone_flow_setpoint(state.T_i, T_sp_zone, P.F.G, P.F.A, P.v_min, P.v_max)

    # --- A : fan frequency, and 4.4's shortfall --------------------------------------
    f, SP_act = required_frequency_and_achieved_sp(SP_sp, f_ref, sp_ref, beta)
    f = np.where(fan_on, f, 0.0)
    SP_act = np.where(fan_on, SP_act, 0.0)

    # --- F2 + C : dampers, then delivered air, per AHU --------------------------------
    d_new = np.zeros(n)
    V_i = np.zeros(n)
    V_total = np.zeros(2)
    n_clipped = np.zeros(2, dtype=int)
    vent_infeasible = np.zeros(2, dtype=bool)
    for ai in AHUS:
        m = P.mask_ahu(ai)
        if not m.any():
            continue
        if not fan_on[ai]:
            d_new[m] = state.d_i[m]
            continue
        d_new[m] = zone_dampers(V_sp[m], P.C.a_i[m], state.d_i[m], float(SP_act[ai]),
                                P.C.ln_C[ai], P.C.q[ai], P.C.s[ai], P.C.p, P.F.lam)
        V_i[m], V_total[ai] = deliver(P.C.a_i[m], d_new[m], float(SP_act[ai]),
                                      P.C.ln_C[ai], P.C.q[ai], P.C.s[ai], P.C.p)
        n_clipped[ai] = int(np.sum(d_new[m] >= 100.0 - 1e-9))
        # 4.3's hard constraint, and the only case where it can actually bind: the fan cannot
        # supply the sum of the boxes' own minima. Reported, not silently clipped away.
        vent_infeasible[ai] = V_total[ai] + 1e-9 < float(np.sum(P.v_min[m]))

    # --- B : fan power ----------------------------------------------------------------
    b_a, b_b = P.B.coeffs()          # coefficients travel WITH the kind - see BlockB.coeffs
    P_fan = fan_power(f, b_a, b_b, P.B.kind, P.B.rated_hz)
    P_fan = np.where(fan_on, P_fan, 0.0)

    # --- D : zone temperatures --------------------------------------------------------
    T_sa_of_box = T_sa[P.ahu_of_box]
    T_next = zone_step(state.T_i, T_m, V_i, T_sa_of_box, T_oa, occ,
                       P.D.k, P.D.a, P.D.g, P.D.delta, P.D.c)

    # --- E2 : mixing and coil duty  ->  info only, never the reward --------------------
    # T_ra is the zone mean at the START of the step. 1.1 places E before D updates the zones, so
    # this is causally prior rather than same-step - block-e 5's ordering caveat, honoured.
    T_ra = np.array([float(np.mean(state.T_i[P.mask_ahu(ai)])) for ai in AHUS])
    oa_damper = np.asarray(exog["oa_damper"], dtype=float)
    T_ma, phi = mixed_air(T_oa, T_ra, oa_damper, np.array(P.E.phi0), np.array(P.E.phi1))
    Q = coil_duty(V_total, T_ma, T_sa, P.E.rho_cp_kj_per_m3_k)
    Q = np.where(fan_on, np.maximum(Q, 0.0), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        sfp = np.where(V_total > 0, P_fan / np.maximum(V_total, 1e-9), 0.0)

    diag = Diagnostics(
        P_fan=P_fan, E_fan_kwh=float(P_fan.sum() * P.dt_h),
        V_total=V_total, T_ma=T_ma, Q_coil_kw=Q, Q_coil_kwh_th=float(Q.sum() * P.dt_h),
        phi=phi, sfp=sfp, n_clipped=n_clipped,
        sp_shortfall=np.maximum(SP_sp - SP_act, 0.0),
        # Rogue boxes stay IN the comfort statistic - 4.3, "a starved zone is still a zone".
        comfort_excess_k=np.maximum(state.T_i - T_sp_zone - comfort_deadband_k, 0.0),
        # Same deadband, both directions. See Diagnostics for why overcooling has to be priced.
        comfort_dev_k=np.maximum(np.abs(state.T_i - T_sp_zone) - comfort_deadband_k, 0.0),
        vent_infeasible=vent_infeasible,
        v_min_violation=np.maximum(P.v_min - V_i, 0.0),
    )
    new = state.replace(T_i=T_next, V_sp_i=V_sp, d_i=d_new, V_i=V_i,
                        T_sa=T_sa, SP_act=SP_act, f=f, SP_sp=SP_sp, SAT_sp=SAT_sp)
    return new, diag
