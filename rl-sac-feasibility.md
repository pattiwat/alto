# SAC on Floor 8 — Does It Fit, and Do You Need a Simulator?

> **What this is.** The answer to two questions asked on 8 September 2026: *if I want to use RL to
> control the HVAC setpoints, does Soft Actor-Critic work well here? And since there are only ~1,800
> usable interaction steps, do I need to model the environment to generate more steps for training?*
>
> **Companion documents.** [rl-environment-design.md](rl-environment-design.md) chooses the algorithm
> and network; [grey-box-surrogate.md](grey-box-surrogate.md) specifies the simulator equations;
> [control-gap-method.md](control-gap-method.md) is the source of every measured number. This memo
> sits in front of them: it decides *whether* to build what they describe, given where the project
> stands and how much time is left. Nothing below is implemented; where it says "checked", the check
> was run against the CSV or the code today.
>
> **Bottom line up front.** SAC fits the problem shape, but its main advantage is irrelevant here and
> its main risk is not learning failure but reward-hacking a simulator. Yes, you need a modelled
> environment: offline RL on the log collapses to behaviour cloning. And the surrogate multiplies
> *action* coverage, not *day* coverage — you still only have ~64 real weekdays. Given that Task B and
> Task C are not yet built and the deadline is 13 September, RL should be a gated extension after the
> rule-based controller, or a written design-and-rejection if the time is not there.

---

## 0. Where the project stands (checked 8 Sep 2026)

| Item | Status |
|---|---|
| Task A + control-gap analysis | Built and tested: `src/masks.py`, `src/effects.py`, `src/excitation.py`, `src/benchmarks.py`, `src/energy_balance.py`; 58 tests; 11 tables in `report/` |
| RL design memos | Written, not implemented. No `src/rl/`, no `rl:` block in `config.yml` |
| **Task B (controller + backtest)** | **Not built.** No `controller.py`, no `backtest.py` |
| **Task C (report, figures)** | **Not built.** No figures exist |
| Deadline | Day 7 from receipt = **13 Sep 2026, 23:59 Bangkok** |
| Brief's budget | 8–12 h for the whole assignment |
| Machine | Python 3.13.7, CPU only, torch 2.8 installed. `stable-baselines3 2.9.0`, `gymnasium 1.3.0`, `sb3-contrib 2.9.0`, `d3rlpy 2.8.1` all resolve on pip for this interpreter |

---

## 1. Question 1 — does SAC work well here?

**Algorithmically, yes. Practically, its selling point is moot and its risk is elsewhere.**

### 1.1 Why it fits

- The action is continuous, bounded and low-dimensional: ΔSP and ΔSAT per AHU, so 2 dims per unit,
  4 if both AHUs are controlled jointly. Episodes are one occupied weekday, 44 steps. That is
  textbook SAC / TD3 / PPO territory; all three would solve it inside a simulator.
- Off-policy replay lets the 1,807 masked logged transitions seed the buffer, so the agent starts
  from the plant's own behaviour rather than from noise.
- The maximum-entropy objective is the right prior given how the plant was run: the static-pressure
  setpoint sat at 0.55 inWG on **81% of AHU-1 on-hours steps** (checked: 122 distinct values, mode
  share 0.81). A deterministic-policy method seeded from that log collapses onto that band early;
  the entropy bonus keeps the policy spread across the admissible range long enough to find AHU-1's
  headroom.

### 1.2 Why its headline advantage does not matter

SAC's reputation rests on sample efficiency — good policies from 10⁵–10⁶ environment steps rather
than PPO's 10⁶–10⁷. That matters when environment steps are expensive. Here they are either
**impossible** (you cannot step the real building) or **free** (a numpy surrogate steps in
microseconds). In neither case is sample efficiency the binding constraint. Inside a cheap surrogate
PPO trains as well as SAC, is more robust to reward-scale and hyperparameter choices, and has no
critic-divergence failure mode. Run both; if they agree, the algorithm was never the interesting
part.

### 1.3 What actually goes wrong

The realistic failure is not that SAC fails to learn. It is that SAC learns *too well* against a
model that is wrong at the edges:

1. **Extrapolation.** Fitted curves have tails. An agent allowed outside the logged setpoint support
   (0.300–0.636 inWG on AHU-1, 0.400–0.780 on AHU-2) will find savings in a polynomial's tail and
   report them as building physics. Mitigation: hard-clip the action to the logged support
   (grey-box §9).
2. **Ventilation dressed as efficiency.** Control-gap §5: 64–73% of the modelled pressure saving is
   bought by moving less air. A scalarised comfort penalty gives the agent an exchange rate, and it
   will trade. Mitigation: comfort and minimum airflow as constraints the agent cannot cross (§4.3
   below), not as reward weights.
3. **Overfitting to the training days.** See §2.2 — this one has no mitigation beyond a held-out
   month and an honest sentence.

None of these is specific to SAC. They apply to any optimiser pointed at this surrogate, including
MPC.

---

## 2. Question 2 — do you need to model the environment?

**Yes, for any online RL including SAC.** The only alternative is offline RL on the logged
transitions, and the log does not support it.

### 2.1 What the log covers (checked)

| Lever, AHU-1 on-hours | Coverage | Verdict for offline RL |
|---|---|---|
| Static-pressure setpoint | 81% at one value; 15 paired episodes; **136 treated steps** in total | Degenerates to behaviour cloning. Any Q-function evaluated away from 0.55 inWG is extrapolating into a void |
| SAT setpoint | mode share only **13%**, 137 distinct values, on-hours range 1.27 K (3.00 K on AHU-2) | Real action coverage — richer than the pressure lever. But its comfort effect runs through zone thermal mass, which the log cannot separate from weather without a model |

Two further consequences of the near-deterministic behaviour policy, both already in
rl-environment-design §1 F2: importance-sampling off-policy evaluation is degenerate, so the standard
offline-RL validation tool is gone; and there is no way to validate a CQL-style pessimism coefficient.
IQL's failure mode is graceful — it degrades toward cloning — but "the record does not identify a
better policy" is the honest output, not a controller.

### 2.2 What a surrogate does and does not give you

This is the part the existing memos do not say plainly enough.

> **A surrogate multiplies action coverage, not day coverage.**

The grey-box replays weather, occupancy and zone setpoints from the record (grey-box §1: "an
episode *is* a real day"). That is the right choice — it puts zero modelling risk on the dominant
driver — but it means the agent can try any setpoint on the ~64 logged weekdays (checked: 64
weekdays with on-hours on each AHU, median 43 steps per day) and **cannot see a 65th day**. After a
time-based split that is roughly 50 training days and 15 test days. The realistic risk is a policy
that has memorised what works on 50 specific afternoons. Domain randomisation over β and the fan
model varies the *plant*, not the *weather*; it does not fix this.

> **Generated steps add no information about the uncertain parameters.**

The load-bearing parameter, `d ln Hz / d ln SP`, is 0.382 with 95% CI [0.241, 0.529] on AHU-1
(`report/elasticities.csv`). Stepping the surrogate ten million times does not narrow that interval.
The extra steps let the optimiser converge; they do not make the answer more certain. The number an
agent reports must therefore be presented as a *range across the DR distribution*, never as a point.

### 2.3 The airflow response is real and testable (new check)

To see whether Block C of the grey-box (pressure → delivered airflow) is a modelling argument or a
measured fact, I paired 8 AHU-1 pressure-cut episodes from `report/setpoint_episodes.csv` against
their own 8-step pre-window:

| Median ratio, treated / control | Value |
|---|---|
| Fan Hz | 0.956 |
| Delivered airflow (26 boxes summed, dead box excluded) | **0.939** |
| Fan kW | 0.917 |
| Cooling rate | 0.964 |

Delivered air *does* fall when pressure is cut, by roughly as much as fan speed. Block C is real, and
the falsification in grey-box §4.5 has data to run against. One caveat that matters for identification:
mean room temperature was **22.2 °C during cuts vs 22.8 °C in the control windows**, i.e. the plant cut
pressure when the floor was already cool. That is the confound control-gap §7 warns about, and a
surrogate identified on these episodes inherits it (β biased upward, conservative for the saving).

---

## 3. Options, with trade-offs

### 3.1 Where the training signal comes from

| | Option | Pros | Cons | Fit for the submission |
|---|---|---|---|---|
| **A1** | **Offline RL on the log** — IQL, TD3+BC via d3rlpy | No modelling risk; honest about the data; ~2 h | Collapses to cloning on the pressure lever (§2.1); no usable off-policy evaluation; produces nothing defensible | One paragraph, possibly one TD3+BC run to *show* it degenerates. Not a headline |
| **A2** | **Grey-box surrogate + online RL** — the existing design, SAC or PPO on top | Steps millions of times; every parameter traces to a `report/` table; measured uncertainty becomes the training distribution; separates efficiency from ventilation by construction | **8–15 h to build properly — the whole assignment budget again**; any kWh traces to a model, which the brief marks down; air-side units unresolved for absolute totals (grey-box §4.6) | The only route to a trained agent. Must be labelled an extension, never Task B |
| **A3** | **Black-box learned dynamics** — ensemble MLP, PETS / MBPO | Fast to write; ensemble disagreement flags out-of-distribution states | Cannot extrapolate to unseen setpoints from a log where 81% of actions are one value — which is the whole point; unauditable in the follow-up | Do not use |
| **A4** | **Generic simulators** — Sinergym, BOPTEST, EnergyPlus | Off the shelf; literature-comparable | Not this floor; no number would trace to the CSV, failing the brief's central rule | Do not use |
| **A5** | **MPC / per-step analytic bound over the grey-box** — no learned policy | Same modelling cost as A2, no training; more explainable; the constraint set is closed-form because Block C is | Inherits surrogate error with no policy-side robustness; less interesting if the goal is "RL" | Build it regardless as the oracle benchmark. If SAC matches it, that is the finding |

### 3.2 How much surrogate to build (if A2)

| | Scope | Blocks | Effort | Can answer |
|---|---|---|---|---|
| **B1** | Fan-side only | A (SP→Hz), B (Hz→kW), C (SP→airflow, damper saturation) | 3–4 h | Pressure lever only. Comfort via airflow floor and saturation count. No SAT lever, no zone temperatures |
| **B2** | Fan-side + quasi-static SAT + pooled zone RC | B1 + E₁ (SAT bias) + one pooled zone thermal model, per-zone only for the ~30 active zones | 5–7 h | Both levers; temperature-band comfort; thermal-mass effects — the only genuinely sequential content |
| **B3** | Full `grey-box-surrogate.md` | All six blocks, per-zone RC, mixing map, coil closure test | 9–12 h | Everything above plus coil attribution and the full falsification suite |

### 3.3 Constraint handling

| | Mechanism | Pros | Cons |
|---|---|---|---|
| **C1** | **Analytic safety layer.** Block C is closed-form, so the minimum pressure that keeps every non-rogue box below 100% damper and above its minimum flow is computable per step. Clip the action to it | Constraint cannot be violated; auditable; ~40 lines; **doubles as the oracle bound** (always sit on the floor) | Removes most of the agent's freedom on the pressure lever — which exposes how thin the RL content is. That is a finding, not a defect |
| C2 | Fixed penalty in the reward | Trivial in SB3 | The agent finds the exchange rate and trades comfort for kWh. The brief says those savings do not count |
| C3 | Lagrangian SAC (CMDP, learned multipliers) — the memo's choice | The principled answer | **Not in stable-baselines3 or sb3-contrib.** You write the cost critic and dual ascent yourself or add omnisafe; the multiplier schedule is one more thing to validate on 50 days |

### 3.4 Learner, if A2

| Learner | When it earns its place | Why not headline |
|---|---|---|
| **SAC** (SB3, 2×256 MLP, auto-α) | Replay seeding from the log; entropy prior against the 0.55 band; literature baseline for building control | Sample efficiency is moot inside a surrogate |
| **PPO** (SB3) | Fewest failure modes; cheapest to get stable; on-policy inefficiency costs only wall-clock here | Cannot use the logged transitions |
| QR-DQN on a discrete trim table | The real supervisory action *is* discrete (G36 trim-and-respond); masking makes constraints trivial | Action space grows across two AHUs and two levers |

Recommendation inside A2: **SAC as the named learner, PPO as the cross-check, 3 seeds each.** If they
agree, say so and move on.

---

## 4. Recommended path

**Phase 0 — mandatory, first: build the rule-based Task B.** Static-pressure reset with rogue
exclusion, plus SAT reset per AHU if time allows. Stepwise causal replay, `on_bad_data()`, the four
benchmarks, the comfort constraint. This is the submission. The existing memos already recommend it
and the brief's grading depends on it. Nothing below starts until this runs end to end from the CSV.

**Phase 1 — optional, only if ≥ ~8 h remain after Phase 0:** A2, scope B2, constraint C1, SAC
headline with PPO cross-check, reported as a **bound**:

> A near-optimal agent, on the best surrogate this record identifies, captures X% of the
> demonstrated-achievable frontier (3.9% / 4.4%) on the held-out August weeks; the oracle floor
> captures Y%; G36 trim-and-respond captures Z%. The gap between X and Z is what a learned supervisor
> would be worth on this floor.

**If fewer than ~8 h remain: do not train anything.** Write the RL section from the memos as a
design-and-rejection — why SAC fits the action space, why the log cannot train it, what the surrogate
would need, and the one model-free number that bounds any agent. That is a stronger use of the time
than a half-built simulator, and it survives the follow-up interview.

---

## 5. Implementation plan for Phase 1

### 5.1 Reuse, do not rewrite

| Module | Functions the RL code calls |
|---|---|
| [src/masks.py](../src/masks.py) | `analysis_mask`, `AnalysisMask.assert_same`, `mask_room_temp_sentinels`, `block_bootstrap_indices` |
| [src/effects.py](../src/effects.py) | `sp_to_hz`, `fit_fan_models`, `FanModel.predict` |
| [src/benchmarks.py](../src/benchmarks.py) | `qc_box_flows`, `delivered_airflow`, `percentile_achievable` — the frontier assertion |
| [src/excitation.py](../src/excitation.py) | `find_setpoint_episodes`, `paired_response`, `fit_elasticity` — the elasticity-recovery test on the finished env |
| [src/io.py](../src/io.py) | `load`, `vav_boxes`, `vav_columns`, `ahu_columns` |
| [src/units.yml](../src/units.yml) | Declared ranges for observation normalisation, never dataset statistics |

### 5.2 New files

```
src/rl/
  plant.py       Blocks A, B, C, E1, pooled zone RC. Pure numpy, vectorised over boxes.
                 Parameters read from report/*.csv, never typed in.
  identify.py    Training window only: zone RC by NNLS (scipy.optimize.nnls),
                 SAT bias b0, b1 by OLS, zone-controller gain K_i.
  env.py         Floor8Env(gymnasium.Env). One episode = one occupied weekday 07:00-18:00,
                 exogenous inputs replayed from the CSV. Obs ~28 dims per AHU
                 (rl-environment-design §4). Action Box([-0.02, -0.3], [+0.02, +0.3]).
  safety.py      Per-step analytic pressure floor from Block C; clips the action.
                 Exposes the oracle policy (always sit on the floor).
  train.py       SB3 SAC and PPO. Domain randomisation per episode over the beta CI,
                 the fan-model triple, SAT bias on/off. Everything from config.yml rl:.
  evaluate.py    Test window only (3 Aug -> 23 Aug). Mask-symmetric comparison of
                 as-operated, naive 20% cut, G36 T&R, oracle floor, SAC, PPO.
                 Saving split into efficiency share and airflow-reduction share.
                 Frontier assertion: FAIL if efficiency share > frontier gap.
scripts/run_rl.py
tests/test_rl_env.py
```

`config.yml` gains an `rl:` block: split dates, episode window, action bounds and rate limits, DR
distributions, comfort band, network sizes, total timesteps, seeds.

Install: `pip install stable-baselines3==2.9.0 gymnasium==1.3.0`, add both to `requirements.txt`.
The `gym 0.26` already on the machine is not used.

### 5.3 Tests, in order

1. Gymnasium API conformance (`gymnasium.utils.env_checker.check_env`) and bit-exact determinism
   under a fixed seed.
2. Reward equals the analytic fan kWh on a hand-worked step (offset cubic at a known Hz).
3. Action clipping: a setpoint pushed outside [0.30, 0.64] stays at the bound.
4. Safety layer: one non-rogue box at 95% damper ⇒ floor = 0.9025 × current pressure (the closed-form
   crossover in grey-box §4.2).
5. Elasticity recovery: a scripted −20% cut inside the finished env, re-estimated with
   `excitation.fit_elasticity`, returns β within its CI. Same shape as
   `tests/test_excitation.py::test_recovers_a_known_elasticity`.
6. Frontier assertion fires on a deliberately mis-specified env (ideal cubic, β = 0.5, no airflow
   loss) that permits a 25% saving.

### 5.4 Compute

SAC at 300k steps with a 2×256 MLP on this CPU is roughly 30–60 min; PPO similar. Three seeds each.
If wall-clock is short, drop the critic ensemble first, PPO second.

---

## 6. Verification

- `python -m pytest tests/ -q` passes, including the six new tests.
- `python scripts/run_rl.py` from a clean checkout with only the CSV present regenerates every RL
  table and figure, and prints the mask-symmetry confirmation and the frontier check for every policy.
- **Surrogate validation gate, before any training:** an 8-step open-loop rollout on the test month
  beats persistence on zone temperature and fan kW, and the SAT bias model reproduces the −1.02 K mean
  tracking error on AHU-1. If it fails, stop and report that — do not train on it.

---

## 7. Corrections to the existing memos (found while checking)

**7.1 The fan-model numbers are stale.** `report/fan_models.csv` no longer matches
grey-box-surrogate §3 or the data memo §1.9. The runner now fits on the 1,807-step evaluation set,
not the 2,679 coil-on steps the memos cite:

| | Memo says | `report/fan_models.csv` says |
|---|---|---|
| AHU-1 offset cubic | 0.719 + 6.199·(f/50)³, R² 0.903 | **1.097 + 5.083·(f/50)³, R² 0.816** |
| AHU-1 power law | n = 1.42, R² 0.655 | **n = 0.889, R² 0.509** |
| AHU-2 offset cubic | 0.537 + 6.806·(f/50)³, R² 0.960 | 0.496 + 6.796·(f/50)³, R² 0.947 |

The pessimistic DR arm is therefore *more* pessimistic than documented and the fan-model band is the
widest sensitivity axis. Whichever fit is intended, memos and report must quote the same one, and
`plant.py` must read it from the CSV rather than from a memo.

**7.2 Lagrangian SAC is not off the shelf.** rl-environment-design §2.1 and §6 assume it is available
in stable-baselines3 / sb3-contrib. It is not — see C3 above. C1 is the recommended replacement.

**7.3 `torchrl` is installed but reports version `0.0.0+unknown`.** Its SAC / IQL / TD3+BC / CQL loss
classes import, but treat the install as unreliable and use SB3.

---

## 8. Assumptions

- The rule-based Task B is built first and remains the headline; RL is an extension.
- Day-level time split: train 18 May → 29 Jul, test 3 Aug → 23 Aug, on the 4-day collector outage.
- Air-side flow units stay unresolved; the surrogate uses only ratios within Block C, and the
  ventilation constraint is a ratio to each box's logged minimum.
- Chiller COP never enters the reward; coil load is reported in kWh-thermal only.
