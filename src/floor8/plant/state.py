"""The plant's state and per-step diagnostics. Frozen: a step RETURNS a new state.

Frozen rather than mutated for three reasons that are specific to this repository:

  * The Block D falsifier is an 8-step OPEN-LOOP rollout (block-d-derivation 7.1). With an immutable
    state that is a fold over a list; with a mutable one it needs a save/restore protocol that will
    eventually be got wrong.
  * rl-environment-design 6 requires bit-exact determinism under a fixed seed. A pure
    `step(state, ...) -> state'` gives it for free.
  * Block C has been corrected twice, both times from its derivation script. Sharing one forward
    model between script and simulator is only safe if the model cannot quietly carry state.

The cost is one small allocation per step - these are 53-element arrays - which is nothing against
the 1e5-1e6 steps SAC needs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PlantState:
    """Endogenous state, 1.1. Arrays are (n_boxes,) per zone or (2,) per AHU."""
    T_i: np.ndarray            # zone temperature, degC
    V_sp_i: np.ndarray         # zone flow setpoint (Block F stage 1 output)
    d_i: np.ndarray            # damper position, %   (Block F stage 2 output)
    V_i: np.ndarray            # delivered flow       (Block C output)
    T_sa: np.ndarray           # supply air temperature, degC        (2,)
    SP_act: np.ndarray         # achieved static pressure, inWG      (2,)
    f: np.ndarray              # fan frequency, Hz                   (2,)
    SP_sp: np.ndarray          # commanded static-pressure setpoint  (2,)
    SAT_sp: np.ndarray         # commanded supply-air setpoint       (2,)

    def replace(self, **kw) -> "PlantState":
        return replace(self, **kw)


@dataclass(frozen=True)
class Diagnostics:
    """Everything `info` needs, and everything the falsifiers read.

    Carried out of the plant rather than recomputed in the env, so the env cannot disagree with the
    physics about what just happened.
    """
    P_fan: np.ndarray          # kW per AHU
    E_fan_kwh: float
    V_total: np.ndarray        # per AHU, in the Block E airflow unit (m3/h)
    T_ma: np.ndarray           # mixed-air temperature per AHU
    Q_coil_kw: np.ndarray      # per AHU, THERMAL - never folded into a scalar reward at a fixed COP
    Q_coil_kwh_th: float
    phi: np.ndarray            # outside-air fraction per AHU
    sfp: np.ndarray            # specific fan power, the 4.5 consistency check
    n_clipped: np.ndarray      # boxes pinned at 100% damper, per AHU - the observable side of lam
    sp_shortfall: np.ndarray   # SP_sp - SP_act, per AHU; > 0 is 4.4's capacity failure
    comfort_excess_k: np.ndarray   # per box, max(0, T_i - Tsp_i - deadband)      ONE-SIDED
    # Two-sided tracking error: how far the zone sits OUTSIDE its own deadband, in either
    # direction. Carried alongside `comfort_excess_k` rather than replacing it, because the two
    # measure different things and both are wanted:
    #   comfort_excess_k  "is anyone too warm?"      - 4.3's rule, what the gates reference
    #   comfort_dev_k     "is the setpoint held?"    - prices OVERCOOLING, which the one-sided
    #                                                  metric cannot see at all
    # On this record that gap is not academic: the floor runs a mean 1.46 K BELOW its own
    # setpoints and 51% of scored zone-steps are more than 0.5 K too cold, against 12.7% too
    # warm. Scored two-sided, the record's comfort cost is 31x what the one-sided metric reports,
    # and every bit of that difference is energy spent cooling past the target.
    comfort_dev_k: np.ndarray      # per box, max(0, |T_i - Tsp_i| - deadband)    TWO-SIDED
    vent_infeasible: np.ndarray    # per AHU bool: fan cannot supply sum(V_min)
    v_min_violation: np.ndarray    # per box, max(0, V_min_i - V_i)
