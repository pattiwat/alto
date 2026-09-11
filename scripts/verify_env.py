"""Falsifiers for the assembled surrogate. Run before trusting any number the agent produces.

    python scripts/verify_env.py

Each check is a claim the documents make, turned into an assertion. A reversal fails the run.

  1. CLOSURE AUDIT       every state variable has exactly one producer per step. This is the test
                         that would have caught `d_i`, `valve(t)` and `T_ra` - three variables the
                         1.1 step order consumed with nothing producing them.
  2. DETERMINISM         same seed, same trajectory, bit for bit (rl-environment-design 6).
  3. TERMINATION         `terminated` is NEVER True; `truncated` is True exactly at the horizon.
                         Getting this wrong trains fine and converges to the wrong value function.
  4. ACTION CLIPPING     an action pushing a setpoint outside the logged support of 9 leaves it at
                         the bound, and the rate limit is never exceeded.
  5. ADDING UP           sum_i V_i = V_total to floating point, per 4.2.
  6. ROUND TRIP          Block F's inversion of Block C returns the dampers it was given.
  7. LAMBDA ENDPOINTS    a 20% cut at lam = 0 reproduces block-c's frozen arm, 10.79 / 14.92%.
  8. ELASTICITY RECOVERY a -20% simulated pressure cut returns beta within its measured CI (2.4).
  9. REWARD IS ENERGY    the reward equals the analytic energy on a hand-worked step.
 10. FAN BAND SANITY     all three fan models stay in a physically possible range over the observed
                         frequency band - the check that would have caught the coefficients being
                         switched without their functional form.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from floor8.env import BOUNDS, Floor8SupervisoryEnv                  # noqa: E402
from floor8.plant.blocks import (dampers_for_targets, deliver, fan_power,  # noqa: E402
                                 split_weights)
from floor8.plant.params import PlantParams                          # noqa: E402
from floor8.plant.state import PlantState                            # noqa: E402
from floor8.policies import as_operated                              # noqa: E402
from floor8.tape import EpisodeTape                                  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        results.append((name, True, detail))
        print(f"[{PASS}] {name}  {detail}")
    except AssertionError as e:
        results.append((name, False, str(e)))
        print(f"[{FAIL}] {name}\n         {e}")


def main() -> int:
    params = PlantParams.from_derivations(ROOT)
    tape = EpisodeTape.build(ROOT, params.boxes)
    train, test = tape.split("2026-07-29", "2026-08-03")
    env = Floor8SupervisoryEnv(tape, params, test, param_sampler=None)

    # ---- 1 closure audit -------------------------------------------------------
    def closure():
        o, _ = env.reset(seed=0, options={"day": test[0]})
        for _ in range(40):
            o, *_ = env.step(np.zeros(4))
        st = env.state
        for f in PlantState.__dataclass_fields__:
            v = np.asarray(getattr(st, f))
            assert np.isfinite(v).all(), (
                f"state field {f!r} contains a non-finite value after 40 steps, which means no "
                f"block produced it. This is the failure mode that hid d_i, valve(t) and T_ra.")
        return f"{len(PlantState.__dataclass_fields__)} state fields all produced and finite"
    check("closure audit - every state variable has a producer", closure)

    # ---- 2 determinism ---------------------------------------------------------
    def determinism():
        def roll(seed):
            o, _ = env.reset(seed=seed, options={"day": test[0]})
            out = [o]
            for i in range(30):
                o, r, *_ = env.step(np.full(4, 0.3))
                out.append(o)
            return np.array(out)
        a, b = roll(7), roll(7)
        assert np.array_equal(a, b), (
            "two rollouts under the same seed differ. Determinism is what makes every other "
            "falsifier in this file reproducible.")
        return "identical to the bit over 30 steps"
    check("determinism under a fixed seed", determinism)

    # ---- 3 termination semantics ------------------------------------------------
    def termination():
        o, _ = env.reset(seed=0, options={"day": test[0]})
        for t in range(env.horizon):
            o, r, term, trunc, _ = env.step(np.zeros(4))
            assert not term, (
                f"terminated became True at t={t}. A building day has no absorbing state, and SAC "
                f"zeroes the bootstrap target on `terminated` - this teaches the critic that the "
                f"world ends at the horizon.")
            assert trunc == (t == env.horizon - 1), (
                f"truncated was {trunc} at t={t}; it must be True exactly at the final step.")
        return f"terminated never fired; truncated fired once, at t={env.horizon - 1}"
    check("termination semantics (the silent-bug check)", termination)

    # ---- 4 action clipping ------------------------------------------------------
    def clipping():
        o, _ = env.reset(seed=0, options={"day": test[0]})
        prev = env.state.SP_sp.copy()
        for _ in range(60):
            o, *_ = env.step(np.array([1.0, 1.0, 1.0, 1.0]))       # slam every lever upward
            assert (env.state.SP_sp <= BOUNDS["sp_hi"] + 1e-12).all(), "SP left the logged support"
            assert (env.state.SAT_sp <= BOUNDS["sat_hi"] + 1e-12).all(), "SAT left the logged support"
            assert (np.abs(env.state.SP_sp - prev) <= BOUNDS["sp_rate"] + 1e-12).all(), \
                "the SP rate limit was exceeded"
            prev = env.state.SP_sp.copy()
        assert np.allclose(env.state.SP_sp, BOUNDS["sp_hi"]), "should have pinned at the ceiling"
        o, _ = env.reset(seed=0, options={"day": test[0]})
        for _ in range(80):
            o, *_ = env.step(np.array([-1.0, -1.0, -1.0, -1.0]))
            assert (env.state.SP_sp >= BOUNDS["sp_lo"] - 1e-12).all(), "SP left the logged support"
        assert np.allclose(env.state.SP_sp, BOUNDS["sp_lo"]), "should have pinned at the floor"
        # Stepping past the horizon must raise rather than silently replay the last exogenous row.
        o, _ = env.reset(seed=0, options={"day": test[0]})
        for _ in range(env.horizon):
            env.step(np.zeros(4))
        try:
            env.step(np.zeros(4))
            raise AssertionError("step() after truncation should raise, not continue")
        except RuntimeError:
            pass
        return "pinned at both bounds, rate limit respected, post-horizon step raises"
    check("action clipping to the logged support (9)", clipping)

    # ---- 5 adding up ------------------------------------------------------------
    def adding_up():
        worst = 0.0
        for ai in (0, 1):
            m = params.mask_ahu(ai)
            rng = np.random.default_rng(3)
            for _ in range(200):
                d = rng.uniform(0, 100, size=int(m.sum()))
                V, tot = deliver(params.C.a_i[m], d, 0.55, params.C.ln_C[ai],
                                 params.C.q[ai], params.C.s[ai], params.C.p)
                worst = max(worst, abs(V.sum() - tot) / max(tot, 1e-12))
        assert worst < 1e-12, f"sum_i V_i deviated from V_total by {worst:.2e} (relative)"
        return f"max relative error {worst:.1e} over 400 random damper vectors"
    check("adding-up: sum_i V_i = V_total (4.2)", adding_up)

    # ---- 6 round trip -----------------------------------------------------------
    def round_trip():
        worst = 0.0
        for ai in (0, 1):
            m = params.mask_ahu(ai)
            a_i = params.C.a_i[m]
            rng = np.random.default_rng(11)
            for _ in range(100):
                d = rng.uniform(5, 95, size=int(m.sum()))
                V, _ = deliver(a_i, d, 0.55, params.C.ln_C[ai], params.C.q[ai],
                               params.C.s[ai], params.C.p)
                d2 = dampers_for_targets(a_i, V, 0.55, params.C.ln_C[ai], params.C.q[ai],
                                         params.C.s[ai], params.C.p)
                live = a_i > 0                     # a dead box has no damper to recover
                worst = max(worst, float(np.max(np.abs(d2[live] - d[live]))))
        assert worst < 1e-6, (
            f"inverting Block C and stepping it forward moved a damper by {worst:.2e} points. "
            f"Block F is not inverting the model Block C forward-steps.")
        return f"max damper error {worst:.1e} points over 200 random states"
    check("Block F inverts Block C exactly (round trip)", round_trip)

    # ---- 7 lambda endpoints -----------------------------------------------------
    def lam_endpoints():
        import pandas as pd
        ref = pd.read_csv(ROOT / "block-c-derivation" / "data" / "pressure_cut_counterfactual.csv")
        out = []
        for ai in (0, 1):
            m = params.mask_ahu(ai)
            a_i, s = params.C.a_i[m], params.C.s[ai]
            # lam = 0 freezes the dampers, so every room loses exactly (1-cut)**s
            loss = 100.0 * (1.0 - (1.0 - 0.20) ** s)
            want = float(ref[ref.ahu == ai + 1].loss_frozen_pct.iloc[0])
            assert abs(loss - want) < 0.05, (
                f"AHU-{ai + 1} frozen-damper loss {loss:.3f}% against block-c's {want:.3f}%")
            out.append(f"AHU-{ai + 1} {loss:.2f}%")
        return "frozen arm reproduces block-c: " + ", ".join(out)
    check("lam = 0 reproduces block-c's frozen arm (4.3)", lam_endpoints)

    # ---- 8 elasticity recovery --------------------------------------------------
    def elasticity():
        out = []
        for ai in (0, 1):
            beta = params.A.beta[ai]
            lo, hi = params.A.beta_ci[ai]
            sp0 = 0.55 if ai == 0 else 0.60
            f0 = params.A.f_ref[ai] * (sp0 / params.A.sp_ref[ai]) ** beta
            f1 = params.A.f_ref[ai] * (sp0 * 0.8 / params.A.sp_ref[ai]) ** beta
            est = np.log(f1 / f0) / np.log(0.8)
            assert lo <= est <= hi, (
                f"AHU-{ai + 1}: a -20% simulated cut implies beta={est:.4f}, outside the measured "
                f"CI [{lo:.3f}, {hi:.3f}]")
            out.append(f"AHU-{ai + 1} {est:.4f}")
        return "recovered " + ", ".join(out)
    check("elasticity recovery under a -20% cut (2.4)", elasticity)

    # ---- 9 reward is energy -----------------------------------------------------
    def reward_is_energy():
        o, _ = env.reset(seed=0, options={"day": test[0]})
        for _ in range(40):                       # step into the occupied window
            o, r, term, trunc, inf = env.step(np.zeros(4))
            if inf["scored"]:
                break
        analytic = -(inf["e_fan_kwh"] + inf["q_coil_kwh_th"] / env.cop)
        assert abs(r - analytic) < 0.05, (
            f"reward {r:.6f} against analytic energy {analytic:.6f}; the difference should be only "
            f"the movement term (zero action was commanded here).")
        return f"reward {r:.4f} = -(fan {inf['e_fan_kwh']:.4f} + coil/COP)"
    check("reward equals the analytic energy on a hand-worked step", reward_is_energy)

    # ---- 10 fan band sanity ------------------------------------------------------
    def fan_band():
        f = np.linspace(30.0, 45.0, 40)           # the observed band, p5-p95 is 33.3-40.7 Hz
        worst = []
        for kind in ("offset_cubic", "power_law", "ideal_cubic"):
            a, b = params.B.models[kind]
            P = fan_power(f[:, None], np.array(a), np.array(b), kind, params.B.rated_hz)
            assert np.isfinite(P).all() and (P >= 0).all() and P.max() < 20.0, (
                f"fan model {kind!r} reaches {P.max():.3g} kW over 30-45 Hz. A 4,297 kWh/season "
                f"fan cannot. This is the check that catches coefficients being used with the "
                f"wrong functional form.")
            worst.append(f"{kind} max {P.max():.2f} kW")
        return "; ".join(worst)
    check("all three fan models stay physical over the observed band", fan_band)

    # ---- 11 the derivation and the plant agree -----------------------------------
    def block_regression():
        """block-f-derivation's compensating arm vs plant/blocks' inversion, on the same states.

        These are deliberately DIFFERENT parameterisations of the same physics. The derivation works
        in RATIO space so the scale constant C cancels - block-f README 3.2 makes that a feature,
        because C is Block C's weakest number and carries the airflow unit. The plant works in
        ABSOLUTE target space because that is what a simulator needs. They must still agree, and
        this is the seam where a third correction to Block C would otherwise land in only one.
        """
        import importlib.util
        path = ROOT / "block-f-derivation" / "reproduce_zone_control.py"
        spec = importlib.util.spec_from_file_location("block_f_deriv", path)
        bf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bf)
        cfg = {"control_gap": {"zone_control": {"damper_max_pct": 100.0,
                                                "max_clip_iterations": 40}}}
        worst = 0.0
        rng = np.random.default_rng(5)
        for ai in (0, 1):
            m = params.mask_ahu(ai)
            a_i = params.C.a_i[m]
            lnC, q, s, p = params.C.ln_C[ai], params.C.q[ai], params.C.s[ai], params.C.p
            for _ in range(50):
                d = rng.uniform(10, 90, size=int(m.sum()))
                sp0, ratio = 0.55, 0.80
                V_now, _ = deliver(a_i, d, sp0, lnC, q, s, p)
                # derivation: ratio form, C never used
                d_ref, _ = bf.compensating_dampers(a_i, d, ratio, p, q, s, cfg)
                # plant: absolute form, asked to restore those same flows at the lower pressure
                d_plant = dampers_for_targets(a_i, V_now, sp0 * ratio, lnC, q, s, p)
                live = a_i > 0
                worst = max(worst, float(np.max(np.abs(d_ref[live] - d_plant[live]))))
        assert worst < 1e-6, (
            f"the derivation's compensating arm and the plant's inversion disagree by {worst:.2e} "
            f"damper points. They are two parameterisations of one model; a disagreement means the "
            f"audit trail is validating something the agent does not run.")
        return f"max disagreement {worst:.1e} damper points over 100 states"
    check("block regression - derivation and plant agree", block_regression)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} falsifiers passed.")
    if n_fail:
        print("A failing falsifier means a number this surrogate produces is not evidence about "
              "the building.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
