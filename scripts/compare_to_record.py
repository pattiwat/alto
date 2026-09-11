"""Replay every controller against the record, under the record's own boundary conditions.

    python scripts/compare_to_record.py [--agent report/rl/sac_agent.pt]

Writes `report/rl/{gate_replay,policy_comparison,lambda_sweep}.csv`.

THREE STEPS, IN THIS ORDER, AND THE FIRST ONE GATES THE OTHER TWO.

  1. GATE. Replay the LOGGED setpoints through the surrogate on the held-out window and compare
     against what was measured: fan power, supply air temperature, zone temperature, delivered air,
     coil duty. If the surrogate cannot reproduce the plant under the plant's OWN actions, no
     comparison against it means anything, and the run says so instead of printing a saving.

  2. COMPARISON. Four policies - as-operated, naive 20% cut, G36 trim-and-respond, trained SAC -
     over identical days, identical weather, identical occupancy, identical zone setpoints and an
     identical mask. rl-environment-design 5.2: if the agent were scored on more steps than the
     baseline, the mask treatment alone would manufacture the saving.

  3. THE FRONTIER ASSERTION. A fan saving materially above the model-free demonstrated-achievable
     gap - 3.9% on AHU-1, 4.4% on AHU-2 - plus the policy's own stated airflow reduction means the
     agent is exploiting the surrogate, not the building. That check is model-free (no fan law, no
     exponent, no COP), which is why it is the one number an over-fitted environment cannot inflate.

WHAT THIS SCRIPT WILL NOT PRINT
-------------------------------
A single absolute kWh headline. 10.1 forbids it while control-gap 5's excess is unexplained, and
7.3 widened the band rather than narrowing it. Electric totals appear as a COP band; coil duty stays
thermal; and every SAC number is reported across the lam sweep, because lam is what decides the
ventilation cost and this record cannot locate it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from floor8.env import Floor8SupervisoryEnv                          # noqa: E402
from floor8.plant.params import PlantParams                          # noqa: E402
from floor8.policies import as_operated, make_g36, make_naive_cut, make_sac_policy  # noqa: E402
from floor8.sac import SACAgent, SACConfig                           # noqa: E402
from floor8.tape import EpisodeTape                                  # noqa: E402

TRAIN_END, TEST_START = "2026-07-29", "2026-08-03"
FRONTIER_PCT = {1: 3.9, 2: 4.4}          # control-gap 5, model-free
AGREEMENT_FACTOR = 2.0                   # config: stated in advance, not chosen after the answer


def run_policy(env, day_indices, policy, seed=0) -> dict:
    """One pass over the given days. Returns aggregates plus the per-step traces the gate needs."""
    tot = {k: 0.0 for k in ("fan_kwh", "coil_kwh_th", "comfort", "vent", "air", "steps",
                            # Both comfort metrics, always, whichever one the constraint used:
                            #   comfort_one  too warm only  - the original statistic
                            #   comfort_two  |error|        - prices overcooling too
                            # plus the setpoint-tracking counts, because on this floor the
                            # interesting failure is zones sitting BELOW target, which the
                            # one-sided number cannot show.
                            "comfort_one", "comfort_two", "abs_err_k",
                            "n_cold", "n_warm", "n_band")}
    per_ahu = {"fan_kwh": np.zeros(2), "air": np.zeros(2), "coil_kwh_th": np.zeros(2)}
    tr = {k: [] for k in ("P_fan", "T_sa", "T_i", "v_total", "f", "sfp")}
    meas = {k: [] for k in ("P_fan", "T_sa", "T_i", "V_i", "cooling_rate")}
    for i in day_indices:
        o, _ = env.reset(seed=seed + i, options={"day": i})
        d = env.day
        for t in range(env.horizon):
            a = policy(o, env)
            o, r, term, trunc, inf = env.step(a)
            if not inf["scored"]:
                continue
            tot["fan_kwh"] += inf["e_fan_kwh"]
            tot["coil_kwh_th"] += inf["q_coil_kwh_th"]
            tot["comfort"] += inf["cost_comfort"]
            tot["comfort_one"] += inf["cost_comfort_one_sided"]
            tot["comfort_two"] += inf["cost_comfort_two_sided"]
            err = inf["T_i"] - d.T_sp_zone[t]
            db = env.comfort_deadband_k
            tot["abs_err_k"] += float(np.nanmean(np.abs(err)))
            tot["n_cold"] += int(np.sum(err < -db))
            tot["n_warm"] += int(np.sum(err > db))
            tot["n_band"] += int(np.sum(np.abs(err) <= db))
            tot["vent"] += inf["cost_vent"]
            tot["air"] += float(inf["v_total"].sum()) * 0.25
            tot["steps"] += 1
            per_ahu["fan_kwh"] += inf["P_fan"] * 0.25
            per_ahu["air"] += inf["v_total"] * 0.25
            per_ahu["coil_kwh_th"] += np.asarray(inf["q_coil_ahu_kwh_th"])
            for k, v in (("P_fan", inf["P_fan"]), ("T_sa", inf["T_sa"]), ("T_i", inf["T_i"]),
                         ("v_total", inf["v_total"]), ("f", inf["f"]), ("sfp", inf["sfp"])):
                tr[k].append(np.asarray(v).copy())
            lg = d.logged
            meas["P_fan"].append(lg["P_fan"][t])
            meas["T_sa"].append(lg["T_sa"][t])
            meas["T_i"].append(lg["T_i"][t])
            meas["V_i"].append(lg["V_i"][t])
            meas["cooling_rate"].append(lg["cooling_rate"][t])
    tot["per_ahu"] = per_ahu
    tot["sim"] = {k: np.array(v) for k, v in tr.items()}
    tot["meas"] = {k: np.array(v) for k, v in meas.items()}
    return tot


def gate_table(res: dict) -> pd.DataFrame:
    """Simulated vs measured, per AHU, on the held-out window. Step 1."""
    sim, meas = res["sim"], res["meas"]
    rows = []
    for ai in (0, 1):
        m_fan = meas["P_fan"][:, ai]
        ok = np.isfinite(m_fan)
        rows.append({
            "ahu": ai + 1, "quantity": "fan power kW",
            "simulated": float(np.mean(sim["P_fan"][ok, ai])),
            "measured": float(np.mean(m_fan[ok])),
            "bias": float(np.mean(sim["P_fan"][ok, ai] - m_fan[ok])),
            "rmse": float(np.sqrt(np.mean((sim["P_fan"][ok, ai] - m_fan[ok]) ** 2))),
            "n": int(ok.sum())})
        m_sa = meas["T_sa"][:, ai]
        ok = np.isfinite(m_sa)
        rows.append({
            "ahu": ai + 1, "quantity": "supply air degC",
            "simulated": float(np.mean(sim["T_sa"][ok, ai])),
            "measured": float(np.mean(m_sa[ok])),
            "bias": float(np.mean(sim["T_sa"][ok, ai] - m_sa[ok])),
            "rmse": float(np.sqrt(np.mean((sim["T_sa"][ok, ai] - m_sa[ok]) ** 2))),
            "n": int(ok.sum())})
    # zone temperature and delivered air, floor-wide
    st, mt = sim["T_i"], meas["T_i"]
    ok = np.isfinite(mt)
    rows.append({"ahu": 0, "quantity": "zone temp degC",
                 "simulated": float(np.mean(st[ok])), "measured": float(np.mean(mt[ok])),
                 "bias": float(np.mean(st[ok] - mt[ok])),
                 "rmse": float(np.sqrt(np.mean((st[ok] - mt[ok]) ** 2))), "n": int(ok.sum())})
    sa = sim["v_total"].sum(axis=1)
    ma = np.nansum(meas["V_i"], axis=1)
    ok = np.isfinite(ma) & (ma > 0)
    rows.append({"ahu": 0, "quantity": "delivered air m3/h",
                 "simulated": float(np.mean(sa[ok])), "measured": float(np.mean(ma[ok])),
                 "bias": float(np.mean(sa[ok] - ma[ok])),
                 "rmse": float(np.sqrt(np.mean((sa[ok] - ma[ok]) ** 2))), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", nargs="*", default=[str(ROOT / "report" / "rl" / "sac_agent.pt")],
                    help="one or more agents, each 'label=path' or a bare path (labelled 'sac')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comfort-mode", default="two_sided", choices=("one_sided", "two_sided"),
                    help="which comfort statistic `comfort_cost` is written on; BOTH are always "
                         "reported in the table regardless")
    args = ap.parse_args()
    out = ROOT / "report" / "rl"
    out.mkdir(parents=True, exist_ok=True)

    params = PlantParams.from_derivations(ROOT)
    tape = EpisodeTape.build(ROOT, params.boxes)
    train_days, test_days = tape.split(TRAIN_END, TEST_START)
    print(f"\ntest window: {len(test_days)} days, "
          f"{tape.day(test_days[0]).date.date()} .. {tape.day(test_days[-1]).date.date()}")

    # Central parameters, DR OFF, for the headline. The lam sweep comes later.
    env = Floor8SupervisoryEnv(tape, params, test_days, param_sampler=None,
                               comfort_mode=args.comfort_mode)
    print(f"comfort constraint written on the {args.comfort_mode.upper()} metric "
          f"(both are reported below)")

    # ---- step 1: the gate ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 1 - DOES THE SURROGATE REPRODUCE THE RECORD UNDER THE RECORD'S OWN ACTIONS?")
    print("=" * 78)
    base = run_policy(env, test_days, as_operated, args.seed)
    gate = gate_table(base)
    gate.to_csv(out / "gate_replay.csv", index=False)
    print(gate.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    sat_sigma = max(params.E.sigma)
    fan_ok = bool((gate[gate.quantity == "fan power kW"].bias.abs() < 1.0).all())
    sat_ok = bool((gate[gate.quantity == "supply air degC"].rmse < 3.0 * sat_sigma).all())
    zone_ok = bool(gate[gate.quantity == "zone temp degC"].rmse.iloc[0] < 3.0)
    print(f"\n  fan power bias  < 1.0 kW  : {fan_ok}")
    print(f"  SAT rmse < 3 sigma ({3 * sat_sigma:.1f} K) : {sat_ok}")
    print(f"  zone temp rmse  < 3.0 K   : {zone_ok}")
    gate_pass = fan_ok and sat_ok and zone_ok
    if not gate_pass:
        print("\n  *** THE GATE DID NOT PASS. ***")
        print("  The comparison below is printed for diagnosis ONLY. A policy saving measured")
        print("  inside a surrogate that cannot reproduce the plant under the plant's own")
        print("  setpoints is not evidence about the building. 10.1's rule applies with force.")

    # ---- step 2: four policies ----------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 2 - FOUR CONTROLLERS, IDENTICAL BOUNDARY CONDITIONS, IDENTICAL MASK")
    print("=" * 78)
    policies = {"as_operated": as_operated,
                "naive_cut_20pct": make_naive_cut(0.20),
                "g36_ignore_top_2": make_g36(ignore_top=2)}
    agent = None
    for spec in args.agent:
        label, _, path = spec.rpartition("=")
        label = label or "sac"
        if not Path(path).exists():
            print(f"  (no agent at {path} - run scripts/train_sac.py first; {label} skipped)")
            continue
        # One SACAgent per label: `load` overwrites in place, so a shared object would score the
        # last-loaded policy under every label in the matrix.
        ag = SACAgent(env.obs_dim, env.action_dim, SACConfig(seed=args.seed))
        ag.load(path)
        policies[label] = make_sac_policy(ag, deterministic=True)
        agent = agent or ag          # the lam sweep below uses the first agent loaded
        print(f"  loaded {label:<12s} <- {path}")

    results, rows = {}, []
    for name, pol in policies.items():
        r = run_policy(env, test_days, pol, args.seed)
        results[name] = r
        rows.append({
            "policy": name, "steps": int(r["steps"]),
            "fan_kwh": r["fan_kwh"], "coil_kwh_th": r["coil_kwh_th"],
            "elec_cop3_kwh": r["fan_kwh"] + r["coil_kwh_th"] / 3.0,
            "elec_cop5_kwh": r["fan_kwh"] + r["coil_kwh_th"] / 5.0,
            "air_m3": r["air"], "comfort_cost": r["comfort"], "vent_infeasible": r["vent"],
            "comfort_one_sided": r["comfort_one"], "comfort_two_sided": r["comfort_two"],
            "track_abs_k": r["abs_err_k"] / max(r["steps"], 1),
            "pct_too_cold": 100.0 * r["n_cold"] / max(r["n_cold"] + r["n_warm"] + r["n_band"], 1),
            "pct_in_band": 100.0 * r["n_band"] / max(r["n_cold"] + r["n_warm"] + r["n_band"], 1),
            "fan_kwh_ahu1": r["per_ahu"]["fan_kwh"][0], "fan_kwh_ahu2": r["per_ahu"]["fan_kwh"][1],
            "air_ahu1": r["per_ahu"]["air"][0], "air_ahu2": r["per_ahu"]["air"][1]})
    comp = pd.DataFrame(rows)

    b = comp[comp.policy == "as_operated"].iloc[0]
    for col, new in (("fan_kwh", "fan_saving_pct"), ("air_m3", "air_reduction_pct"),
                     ("elec_cop3_kwh", "elec_cop3_saving_pct"),
                     ("elec_cop5_kwh", "elec_cop5_saving_pct")):
        comp[new] = 100.0 * (b[col] - comp[col]) / b[col]
    for ai in (1, 2):
        comp[f"fan_saving_pct_ahu{ai}"] = 100.0 * (
            b[f"fan_kwh_ahu{ai}"] - comp[f"fan_kwh_ahu{ai}"]) / b[f"fan_kwh_ahu{ai}"]
        comp[f"air_reduction_pct_ahu{ai}"] = 100.0 * (
            b[f"air_ahu{ai}"] - comp[f"air_ahu{ai}"]) / b[f"air_ahu{ai}"]
    comp["comfort_vs_baseline_pct"] = 100.0 * (comp.comfort_cost - b.comfort_cost) / b.comfort_cost
    comp.to_csv(out / "policy_comparison.csv", index=False)

    show = ["policy", "fan_kwh", "fan_saving_pct", "elec_cop3_saving_pct", "elec_cop5_saving_pct",
            "comfort_one_sided", "comfort_two_sided", "track_abs_k", "pct_in_band",
            "pct_too_cold", "vent_infeasible"]
    print(comp[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n  Every step count is identical across policies, which is what makes these comparable")
    print(f"  ({int(comp.steps.iloc[0])} scored steps for all {len(comp)} arms).")
    print("  ELECTRIC totals are a COP BAND, never a point: no chiller COP exists in this dataset")
    print("  (6.4), so a single electric number would be an assumption wearing a measurement's")
    print("  clothes. Coil duty is reported THERMAL alongside.")
    assert comp.steps.nunique() == 1, (
        "policies were scored on different numbers of steps - the mask is not symmetric and the "
        "comparison is void (rl-environment-design 5.2).")

    # ---- step 3: the frontier assertion --------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 3 - THE FRONTIER ASSERTION")
    print("=" * 78)
    verdicts = []
    for _, r in comp.iterrows():
        if r.policy == "as_operated":
            continue
        for ai in (1, 2):
            fan = float(r[f"fan_saving_pct_ahu{ai}"])
            air = float(r[f"air_reduction_pct_ahu{ai}"])
            ceiling = FRONTIER_PCT[ai] + max(air, 0.0)
            verdicts.append({"policy": r.policy, "ahu": ai, "fan_saving_pct": fan,
                             "air_reduction_pct": air, "frontier_ceiling_pct": ceiling,
                             "exceeds": fan > ceiling * AGREEMENT_FACTOR})
    v = pd.DataFrame(verdicts)
    print(v.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n  Ceiling = model-free frontier gap (3.9 / 4.4%) + the policy's OWN stated airflow")
    print(f"  reduction, with the stated agreement factor of {AGREEMENT_FACTOR}x. Exceeding it means")
    print("  the saving is coming from the surrogate rather than the building.")
    if len(v) and v.exceeds.any():
        print("\n  *** FRONTIER ASSERTION FIRED for: "
              f"{', '.join(sorted(set(v[v.exceeds].policy)))} ***")

    # ---- the lam sweep --------------------------------------------------------------
    if agent is not None:
        print("\n" + "=" * 78)
        print("THE LAMBDA SWEEP - because lam is what decides the ventilation cost")
        print("=" * 78)
        import dataclasses
        sweep = []
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            p2 = params.with_(F=dataclasses.replace(params.F, lam=lam))
            e2 = Floor8SupervisoryEnv(tape, p2, test_days, param_sampler=None)
            rb = run_policy(e2, test_days, as_operated, args.seed)
            rs = run_policy(e2, test_days, make_sac_policy(agent), args.seed)
            sweep.append({"lam": lam,
                          "fan_saving_pct": 100 * (rb["fan_kwh"] - rs["fan_kwh"]) / rb["fan_kwh"],
                          "air_reduction_pct": 100 * (rb["air"] - rs["air"]) / rb["air"],
                          "comfort_vs_baseline_pct":
                              100 * (rs["comfort"] - rb["comfort"]) / max(rb["comfort"], 1e-9)})
        sw = pd.DataFrame(sweep)
        sw.to_csv(out / "lambda_sweep.csv", index=False)
        print(sw.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print("\n  lam = 0 the zone loops do not compensate at all; lam = 1 they fully restore their")
        print("  own flows. 4.3 and block-f 4: no event in this record separates loop compensation")
        print("  from a demand move, so the honest report is this whole row, not a point on it.")

    print(f"\nwrote gate_replay.csv, policy_comparison.csv and lambda_sweep.csv to {out}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
