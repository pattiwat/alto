"""Train SAC inside the grey-box surrogate, on the TRAINING WINDOW ONLY.

    python scripts/train_sac.py [--steps 150000] [--seed 0]
                                [--comfort-mode one_sided|two_sided]
                                [--comfort-budget-frac 1.0] [--out report/rl/sac_agent.pt]

Writes the agent, `<out stem>_train_log.csv` and `report/rl/params.json`.

THE TEST MONTH IS NEVER TOUCHED HERE. rl-environment-design 5.1: train 18 May - 29 Jul, test
3 Aug - 23 Aug, the boundary falling naturally on the 4-day collector outage. A random split would
leak, because 15-minute steps are heavily autocorrelated and neighbouring steps in train and test
would be near-duplicates. The VALIDATION split used for checkpoint selection is carved out of the
training days for the same reason - selecting a checkpoint on the test month would leak it just as
surely as training on it.

Every episode draws a fresh domain-randomisation vector over beta, s, lam, the fan model and COP
(8.2). The sensitivity axes ARE the training distribution, so a policy that survives them is robust
to the uncertainty that was actually measured - and the tornado chart becomes a description of the
environment the agent was trained in rather than a post-hoc defence.

WHICH COMFORT METRIC THE CONSTRAINT IS WRITTEN ON
-------------------------------------------------
`--comfort-mode two_sided` prices OVERCOOLING as well as overheating. It matters on this record:
the floor runs a mean 1.46 K BELOW its own setpoints and 51% of scored zone-steps are more than
0.5 K too cold, against 12.7% too warm. The one-sided metric cannot see any of that, so an agent
optimising against it has no reason to stop cooling past the target - which is energy spent to
make people less comfortable, not more.

`--comfort-budget-frac` tightens the measured budget. At 1.0 the constraint reads "no worse than
the plant"; below that it reads "hold the setpoint better than the plant did, and tell me what the
electricity costs". Sweeping it is what traces the cost/tracking frontier.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from floor8.env import Floor8SupervisoryEnv, default_param_sampler   # noqa: E402
from floor8.plant.params import PlantParams                          # noqa: E402
from floor8.policies import as_operated                              # noqa: E402
from floor8.sac import SACAgent, SACConfig, seed_buffer_from_record  # noqa: E402
from floor8.tape import EpisodeTape                                  # noqa: E402

TRAIN_END, TEST_START = "2026-07-29", "2026-08-03"


def rollout_costs(env, days, policy, seed=0) -> tuple[float, float, float]:
    """Mean per-episode (energy kWh at env.cop, comfort cost, vent cost) over `days`.

    DR must be OFF in `env` - this is a measurement, and a randomised draw would make it a
    different measurement every time it is called.
    """
    e = cc = cv = 0.0
    for i in days:
        o, _ = env.reset(seed=seed + i, options={"day": i})
        for _ in range(env.horizon):
            o, r, term, trunc, inf = env.step(policy(o, env))
            e += inf["e_fan_kwh"] + inf["q_coil_kwh_th"] / env.cop
            cc += inf["cost_comfort"]
            cv += inf["cost_vent"]
    n = max(len(days), 1)
    return e / n, cc / n, cv / n


def checkpoint_score(energy: float, comfort: float, vent: float,
                     budget_c: float, budget_v: float) -> tuple:
    """Rank a candidate checkpoint: ENERGY SUBJECT TO THE CONSTRAINTS, not energy alone.

    Returned as a tuple so ordinary tuple comparison does the lexicographic work:
      (infeasible?, how badly, energy)
    A feasible checkpoint therefore always beats an infeasible one however cheap the latter is,
    infeasible ones are ranked by how close they came, and ties break on energy. Without the
    feasibility term first, selection would happily ship the policy that saved the most power by
    abandoning the comfort constraint outright.
    """
    viol = max(0.0, comfort / max(budget_c, 1e-9) - 1.0) \
        + max(0.0, (vent - budget_v) / max(budget_v, 1.0))
    return (viol > 1e-9, viol, energy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--comfort-mode", default="one_sided",
                    choices=("one_sided", "two_sided"))
    ap.add_argument("--comfort-budget-frac", type=float, default=1.0,
                    help="multiplier on the MEASURED as-operated comfort budget; 1.0 = 'no worse "
                         "than the plant', 0.5 = 'hold the setpoint twice as well'")
    ap.add_argument("--val-days", type=int, default=10,
                    help="training days held out for checkpoint selection (never the test month)")
    ap.add_argument("--out", default=str(ROOT / "report" / "rl" / "sac_agent.pt"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tag = out_path.stem

    print("building parameters from the derivation artifacts ...")
    params = PlantParams.from_derivations(ROOT)
    print(params.provenance_table().to_string(index=False))
    imputed = [p for p in params.box_provenance if p != "fitted"]
    print(f"\n{len(imputed)} of {params.n_boxes} boxes carry an imputed or pooled parameter "
          f"(see box_provenance_table); the rest are fitted.")
    params.to_json(out_path.parent / "params.json")

    print("\nbuilding the episode tape ...")
    tape = EpisodeTape.build(ROOT, params.boxes)
    train_days, test_days = tape.split(TRAIN_END, TEST_START)
    print(f"{len(tape)} full days -> {len(train_days)} train, {len(test_days)} test "
          f"(test window is NOT opened by this script)")

    # VALIDATION SPLIT, carved out of the TRAINING days. Taken as an evenly spaced stride rather
    # than the last N, so validation spans the same weather range as training instead of being
    # whichever fortnight happens to sit at the end of the window.
    stride = max(len(train_days) // max(args.val_days, 1), 1)
    val_days = train_days[::stride][:args.val_days]
    fit_days = [d for d in train_days if d not in set(val_days)]
    print(f"  -> {len(fit_days)} fit days, {len(val_days)} validation days for checkpoint "
          f"selection (the test month is untouched)")

    mode = args.comfort_mode
    env = Floor8SupervisoryEnv(tape, params, fit_days,
                               param_sampler=default_param_sampler({}), comfort_mode=mode)

    # THE COMFORT BUDGET IS MEASURED, NOT CHOSEN. It is the as-operated policy's own comfort cost
    # on the training window, so the constraint reads "no worse than what the plant already did" -
    # which is the brief's rule that savings bought with discomfort do not count.
    #
    # It has to be measured because 4.3 keeps the rogue boxes IN the comfort statistic ("a starved
    # zone is still a zone"), and those zones are starved whatever the agent does - vav_8_1_28 sits
    # at a median 28.9 degC with its damper pinned. They contribute a large constant the agent
    # cannot act on. Setting the budget from the baseline cancels that constant and leaves the
    # constraint measuring what the agent actually changed.
    seed_env = Floor8SupervisoryEnv(tape, params, fit_days, param_sampler=None, comfort_mode=mode)
    val_env = Floor8SupervisoryEnv(tape, params, val_days, param_sampler=None, comfort_mode=mode)
    _, base_c, base_v = rollout_costs(seed_env, fit_days, as_operated, args.seed)
    budget_c = base_c * args.comfort_budget_frac
    budget_v = base_v
    print(f"\ncomfort metric: {mode.upper()}"
          f"{'  (prices overcooling as well as overheating)' if mode == 'two_sided' else ''}")
    print(f"as-operated comfort cost on the fit window: {base_c:.1f} per episode")
    print(f"budget = {args.comfort_budget_frac:g} x measured             = {budget_c:.1f}")
    print(f"as-operated ventilation-infeasible AHU-steps: {budget_v:.2f} per episode")
    print("  -> these become the constraint budgets.\n")

    # The as-operated reference on the VALIDATION days, measured separately and used as the
    # budget for CHECKPOINT SELECTION.
    #
    # It must be its own measurement rather than reusing the fit-window budget: the two day sets
    # have different weather and different setpoint schedules, so they are not equally hard. The
    # validation days here run a comfort cost ~1.5x the fit days, and scoring against the fit
    # budget marked every checkpoint "infeasible" - including ones comfortably BETTER than the
    # plant on those very days. Feasibility has to be judged against what the plant itself
    # achieved on the same days, or it is measuring the split rather than the policy.
    val_e0, val_c0, val_v0 = rollout_costs(val_env, val_days, as_operated, args.seed)
    val_budget_c = val_c0 * args.comfort_budget_frac
    val_budget_v = val_v0
    print(f"as-operated on validation: energy {val_e0:.1f} kWh/ep, comfort {val_c0:.1f}")
    print(f"selection budget on validation = {val_budget_c:.1f}\n")

    cfg = SACConfig(seed=args.seed, comfort_budget=budget_c, vent_budget=budget_v)
    agent = SACAgent(env.obs_dim, env.action_dim, cfg)

    # Seed the buffer from the plant's own behaviour, with DR OFF so the seeded transitions
    # describe the record rather than a randomised variant of it.
    n_seed = seed_buffer_from_record(agent, seed_env, fit_days, as_operated)
    print(f"buffer seeded with {n_seed} transitions from the as-operated record\n")

    rows, evals = [], []
    best = None
    o, info = env.reset(seed=args.seed)
    ep_r = ep_c = ep_v = 0.0
    ep_fan = ep_coil = 0.0
    ep_i = 0
    t0 = time.time()
    for t in range(1, args.steps + 1):
        a = agent.act(o)
        o2, r, term, trunc, inf = env.step(a)
        agent.observe(o, a, r, inf["cost_comfort"], inf["cost_vent"], o2, term, trunc)
        agent.update()
        ep_r += r
        ep_c += inf["cost_comfort"]
        ep_v += inf["cost_vent"]
        ep_fan += inf["e_fan_kwh"]
        ep_coil += inf["q_coil_kwh_th"]
        o = o2
        if trunc:
            agent.note_episode_costs(ep_c, ep_v)
            lc, lv = agent.lam
            rows.append({"step": t, "episode": ep_i, "return": ep_r, "cost_comfort": ep_c,
                         "cost_vent": ep_v, "fan_kwh": ep_fan, "coil_kwh_th": ep_coil,
                         "alpha": agent.alpha, "lam_comfort": lc, "lam_vent": lv,
                         "date": info["date"], "lam_dr": info["dr"].get("lam", np.nan)})
            ep_i += 1
            if ep_i % 25 == 0:
                w = pd.DataFrame(rows[-25:])
                print(f"  step {t:>7}  ep {ep_i:>4}  return {w['return'].mean():>9.1f}  "
                      f"fan {w.fan_kwh.mean():>6.1f} kWh  comfort {w.cost_comfort.mean():>8.1f}  "
                      f"alpha {agent.alpha:.3f}  lam_c {lc:.2f}  "
                      f"({t / max(time.time() - t0, 1e-9):.0f} steps/s)")
            o, info = env.reset()
            ep_r = ep_c = ep_v = ep_fan = ep_coil = 0.0

        # ---- CHECKPOINT SELECTION -------------------------------------------------------
        # Deterministic, DR OFF, on held-out training days. The previous version of this script
        # parsed --eval-every and never used it, so it saved whichever policy the run happened to
        # end on - and with the duals oscillating, the final episode was frequently among the
        # worst of the run.
        if args.eval_every and t % args.eval_every == 0:
            ev_e, ev_c, ev_v = rollout_costs(val_env, val_days, lambda ob, en:
                                             agent.act(ob, deterministic=True), args.seed)
            sc = checkpoint_score(ev_e, ev_c, ev_v, val_budget_c, val_budget_v)
            keep = best is None or sc < best[0]
            evals.append({"step": t, "energy_kwh": ev_e, "cost_comfort": ev_c, "cost_vent": ev_v,
                          "feasible": not sc[0], "violation": sc[1], "kept": keep})
            print(f"    [eval @ {t:>7}]  energy {ev_e:7.1f} kWh/ep ({100 * (ev_e / val_e0 - 1):+5.1f}% "
                  f"vs plant)  comfort {ev_c:9.1f} ({100 * (ev_c / val_c0 - 1):+6.1f}%)  "
                  f"{'FEASIBLE' if not sc[0] else f'viol {sc[1]:.2f}'}"
                  f"{'   <- new best, saved' if keep else ''}")
            if keep:
                best = (sc, t)
                agent.save(out_path)

    # If no evaluation ever ran (a very short run), fall back to the final policy rather than
    # writing nothing at all.
    if best is None:
        agent.save(out_path)
        print("\n  (no evaluation ran - saved the final policy)")
    else:
        print(f"\n  best checkpoint was step {best[1]} "
              f"({'feasible' if not best[0][0] else f'violation {best[0][1]:.2f}'}, "
              f"energy {best[0][2]:.1f} kWh/ep)")

    pd.DataFrame(rows).to_csv(out_path.parent / f"{tag}_train_log.csv", index=False)
    pd.DataFrame(evals).to_csv(out_path.parent / f"{tag}_evals.csv", index=False)
    print(f"\ntrained {args.steps} steps in {time.time() - t0:.0f}s -> {out_path}")
    print("Nothing here is a saving. scripts/compare_to_record.py is what produces a number, and "
          "it opens the test window for the first time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
