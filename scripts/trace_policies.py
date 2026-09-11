"""Per-step traces for the as-operated record and the trained SAC agent, same days, same weather.

    python scripts/trace_policies.py [--agent report/rl/sac_agent.pt]

Writes `report/rl/traces.csv` - one row per (policy, day, 15-min step) over the TEST window only.

Both arms are replayed through the SAME env instance, reset with the SAME seed on the SAME day
index, so they see an identical outdoor temperature, wetbulb, occupancy, zone setpoint schedule,
fan schedule, outside-air damper AND an identical supply-air-noise realisation. The only thing that
differs between the two rows at a given timestamp is the controller's action. That is what makes a
difference between them attributable to the controller (compare_to_record.py step 2, 5.2).

Costs stay in the units the repo permits (10.1): fan kWh metered, coil duty THERMAL, electric only
ever as a COP 3-5 band. No tariff exists anywhere in this dataset, so there is no currency column -
under a flat tariff operating cost is proportional to the electric column.
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


def trace(env, day_indices, policy, name, seed=0) -> pd.DataFrame:
    rows = []
    for i in day_indices:
        o, _ = env.reset(seed=seed + i, options={"day": i})
        d = env.day
        for t in range(env.horizon):
            a = policy(o, env)
            o, r, term, trunc, inf = env.step(a)
            Ti = inf["T_i"]
            Tsp = d.T_sp_zone[t]
            exc = inf["comfort_excess_k"]
            dev = inf["comfort_dev_k"]
            err = Ti - Tsp                      # SIGNED: negative is overcooled
            db = env.comfort_deadband_k
            e_fan = float(inf["e_fan_kwh"])
            q_th = float(inf["q_coil_kwh_th"])
            rows.append({
                "policy": name,
                "date": d.date.date().isoformat(),
                "day_index": i,
                "t": t,
                "hour": t * 0.25,
                "scored": bool(inf["scored"]),
                # --- boundary conditions: IDENTICAL between arms, carried so it can be proven so
                "T_oa": float(d.T_oa[t]),
                "T_wb": float(d.T_wb[t]),
                "occ": float(d.occ[t]),
                # --- operating cost
                "fan_kwh": e_fan,
                "coil_kwh_th": q_th,
                "elec_cop3_kwh": e_fan + q_th / 3.0,
                "elec_cop4_kwh": e_fan + q_th / 4.0,
                "elec_cop5_kwh": e_fan + q_th / 5.0,
                "P_fan_kw": float(np.sum(inf["P_fan"])),
                "P_fan_ahu1": float(inf["P_fan"][0]),
                "P_fan_ahu2": float(inf["P_fan"][1]),
                "air_m3h": float(np.sum(inf["v_total"])),
                # --- comfort temperature
                "T_zone_mean": float(np.nanmean(Ti)),
                "T_zone_p95": float(np.nanpercentile(Ti, 95)),
                "T_zone_max": float(np.nanmax(Ti)),
                "T_sp_mean": float(np.nanmean(Tsp)),
                "comfort_excess_sum_k": float(np.sum(exc)),
                "comfort_excess_max_k": float(np.max(exc)),
                "n_zones_over": int(np.sum(exc > 0.0)),
                # --- setpoint TRACKING: two-sided, so overcooling is visible
                "track_dev_sum_k": float(np.sum(dev)),
                "track_abs_mean_k": float(np.nanmean(np.abs(err))),
                "track_signed_mean_k": float(np.nanmean(err)),
                "track_rmse_k": float(np.sqrt(np.nanmean(err ** 2))),
                "n_too_cold": int(np.sum(err < -db)),
                "n_too_warm": int(np.sum(err > db)),
                "n_in_band": int(np.sum(np.abs(err) <= db)),
                "cost_comfort": float(inf["cost_comfort"]),
                "cost_comfort_one_sided": float(inf["cost_comfort_one_sided"]),
                "cost_comfort_two_sided": float(inf["cost_comfort_two_sided"]),
                "cost_vent": float(inf["cost_vent"]),
                "T_sa_ahu1": float(inf["T_sa"][0]),
                "T_sa_ahu2": float(inf["T_sa"][1]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", nargs="*", default=[str(ROOT / "report" / "rl" / "sac_agent.pt")],
                    help="one or more agents, each 'label=path' or a bare path (labelled 'sac')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report" / "rl" / "traces.csv"))
    args = ap.parse_args()

    params = PlantParams.from_derivations(ROOT)
    tape = EpisodeTape.build(ROOT, params.boxes)
    _, test_days = tape.split(TRAIN_END, TEST_START)
    # Two-sided, so the generic `cost_comfort` column carries the setpoint-tracking metric this
    # comparison is written on. Both metrics are emitted either way in their own columns; this
    # only decides what the unqualified name means, and a silently one-sided `cost_comfort` is
    # exactly the kind of ambiguity that puts the wrong series on a figure.
    env = Floor8SupervisoryEnv(tape, params, test_days, param_sampler=None,
                               comfort_mode="two_sided")
    print(f"test window: {len(test_days)} days, "
          f"{tape.day(test_days[0]).date.date()} .. {tape.day(test_days[-1]).date.date()}")

    pols = {"as_operated": as_operated,
            "naive_cut_20pct": make_naive_cut(0.20),
            "g36_ignore_top_2": make_g36(ignore_top=2)}
    for spec in args.agent:
        label, _, path = spec.rpartition("=")
        label = label or "sac"
        if not Path(path).exists():
            raise SystemExit(f"no agent at {path} - run scripts/train_sac.py first")
        # A fresh agent per label: `load` overwrites the networks in place, so reusing one object
        # across the matrix would silently trace the last-loaded policy under every label.
        ag = SACAgent(env.obs_dim, env.action_dim, SACConfig(seed=args.seed))
        ag.load(path)
        pols[label] = make_sac_policy(ag, deterministic=True)
        print(f"  loaded {label:<12s} <- {path}")

    out = pd.concat([trace(env, test_days, p, n, args.seed) for n, p in pols.items()],
                    ignore_index=True)
    out.to_csv(args.out, index=False)

    # The identical-boundary-conditions claim, asserted rather than promised.
    piv = out.pivot_table(index=["date", "t"], columns="policy", values="T_oa")
    spread = float(np.nanmax(piv.max(axis=1) - piv.min(axis=1)))
    assert spread == 0.0, f"outdoor temperature differs between arms by up to {spread} K"
    print(f"outdoor temperature identical at all {len(piv)} timestamps (max spread {spread})")

    sc = out[out.scored]
    g = sc.groupby("policy")
    s = g[["fan_kwh", "elec_cop3_kwh", "elec_cop5_kwh",
           "cost_comfort_one_sided", "cost_comfort_two_sided"]].sum()
    s["track_abs_K"] = g.track_abs_mean_k.mean()
    s["signed_K"] = g.track_signed_mean_k.mean()
    zone_steps = g.n_in_band.sum() + g.n_too_cold.sum() + g.n_too_warm.sum()
    s["in_band_%"] = 100.0 * g.n_in_band.sum() / zone_steps
    s["too_cold_%"] = 100.0 * g.n_too_cold.sum() / zone_steps
    s["steps"] = g.size()
    print()
    print(s.to_string(float_format=lambda v: f"{v:.1f}"))
    print("\n  track_abs_K = mean |zone temp - its own setpoint|, over 53 zones x scored steps.")
    print("  signed_K < 0 means the floor is running BELOW setpoint, i.e. overcooled.")
    print(f"\nwrote {args.out} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
