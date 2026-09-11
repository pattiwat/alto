# Reinforcement Learning on Floor 8 — What Fits This Data, and Why

> **What this is.** The answer to a specific question: *if the Floor 8 record is turned into a
> Gymnasium environment and a supervisory controller is trained on it by RL, which algorithms and
> network architectures actually fit this data?*
>
> The answer cannot start with the algorithm. Five properties of this dataset — all of them already
> measured in the existing memos — eliminate most of the RL menu before any network is chosen, and
> the eliminations are more informative than the survivor. §1 states them; everything after §1 is
> derived from them.
>
> **Companion document:** [grey-box-surrogate.md](grey-box-surrogate.md) — every equation of the
> environment, its identification from this CSV, and its falsification test. That document is the
> hard part; this one is the consequence.
>
> **Bottom line up front.** SAC with Lagrangian constraints, trained inside an identified grey-box
> with domain randomisation over the measured parameter intervals; a 2×256 MLP with LayerNorm'd
> ensemble critics; IQL as the pure-offline arm. And **it should not be submitted as Task B** — see
> §7.

---

## 1. The five facts that decide the algorithm

None of these is new work. Each is already established and traceable.

### F1 — There are ~1,800 usable transitions per AHU

| | AHU-1 | AHU-2 |
|---|---:|---:|
| Total steps | 9,305 | 9,305 |
| − off-hours (fan stopped) | −6,574 | −6,393 |
| − operator override (~8%) | −49 | −44 |
| − settling after setpoint changes | −875 | −770 |
| − incomplete inputs | −0 | −187 |
| **Evaluation set** | **1,807** | **1,911** |

Source: control-gap §2, [src/masks.py](src/masks.py). At 44 steps per occupied day (07:00–18:00),
the 2,731 on-hours steps on AHU-1 are **≈62 weekday episodes**.

**Consequence.** PPO typically needs 10⁶–10⁷ environment steps; SAC 10⁵–10⁶. You have ~10³
transitions from ~62 trajectories — **two to four orders of magnitude short**. Online model-free RL
against the building is impossible, and there is no getting around it with a cleverer optimiser.
Learning must happen either (a) inside a surrogate that can be stepped millions of times, or
(b) offline, with explicit pessimism.

> **The corollary that matters.** This constraint binds on *policy learning*, not on *system
> identification*. The surrogate has 53 zones × 1,807 steps ≈ 95,800 zone-observations for ~280
> parameters (grey-box §0). The record is simultaneously far too small to learn a policy from and
> entirely adequate to identify a plant from. That asymmetry is the entire argument for building the
> surrogate rather than doing offline RL on the raw log.

### F2 — The behaviour policy is nearly deterministic

`ahu_b8_1__static_pressure_setpoint_read` sits at **0.55 inWG on 7,362 of 8,650 steps** (85%). Where
it moves, it moves by roughly the same amount every time: treatment-intensity CV = 0.33 / 0.36 across
the setpoint episodes (control-gap §1, §4.2).

**Consequences, three of them:**

1. **Severe distributional shift for offline RL.** Any method that evaluates Q at actions the
   behaviour policy never took is extrapolating a fitted function into a void, and will
   confidently prefer the void.
2. **Importance-sampling off-policy evaluation is dead.** WIS/PDIS weights degenerate when the
   behaviour policy is near-deterministic. This removes the standard offline-RL validation tool and
   is a large part of why the surrogate is not optional.
3. **The action space must be bounded by the logged support** — grey-box §9. Not as a safety nicety;
   as the boundary of where the environment's own equations were identified.

### F3 — 15-minute averages against 1–5 minute loop dynamics

The historian buckets everything to 15 minutes; the SAT and pressure loops have time constants of
1–5 minutes. The record is **below Nyquist for the dynamics that matter to a regulatory controller**,
and averaging shrinks apparent variance by an unknown factor (control-gap §6 — the reason the
Harris/minimum-variance index was rejected there).

**Consequences.**

- The fast dynamics are unlearnable *and* unnecessary: a supervisory agent writes setpoints, it does
  not close the inner loops.
- **Recurrent policies and sequence models lose their justification.** There is nothing sub-step for
  an LSTM to recover; the information was destroyed before it reached the file.
- The surrogate is quasi-steady-state by necessity, and the only dynamic state worth carrying is the
  zone thermal mass (τ ≈ 1–3 h ≫ Δt), which is exactly why that block *is* identifiable.

### F4 — The key effect parameter is measured, with an interval wider than the effect

`d ln Hz / d ln SP` on AHU-1: **0.382, 95% CI [0.241, 0.529]**, against the textbook affinity value of
0.5. Across 36 defensible specifications the estimate spans **0.208 → 0.785**, and only 15 of 36
exclude 0.5 (control-gap §4.2–4.3).

**Consequence — and this is the constructive one.** The environment's most load-bearing parameter
comes with a *quantified* uncertainty. That is not a problem to be resolved before training; it is
the **domain-randomisation distribution**, sampled per episode. The five sensitivity axes proposed in
§2.7 of the data memo (fan model, affinity exponent, ignore-*I*, SP floor, COP) become the DR
ranges, which means the tornado chart stops being a post-hoc defence and becomes a description of
the distribution the policy was trained across.

### F5 — Most of the available prize is ventilation reduction wearing an efficiency costume

The model-free frontier — the p10 of fan power per unit airflow *within airflow bins*, a level this
plant demonstrably reached with the same fan and duct — gives a gap of **52 kWh (3.9%)** on AHU-1 and
**64 kWh (4.4%)** on AHU-2. The modelled pressure-reset saving is **190 / 175 kWh**, i.e. larger.
The reconciliation: the frontier holds delivered air fixed, and a pressure cut on this floor also
delivers *less* air, because at least one box sits at 100% damper at every occupied step. **73%
(AHU-1) and 64% (AHU-2) of the modelled saving is bought by moving less air** (control-gap §5).

**Consequence.** Comfort and ventilation are **hard constraints**, not penalty weights. A scalarised
reward with a hand-tuned λ on comfort is not a modelling shortcut here — it is an invitation, and the
agent will accept it. The brief is explicit that savings bought with discomfort do not count; that is
a feasibility statement, and it should be encoded as one.

### Two structural facts on top

- **Only setpoints are writable.** `ahu_b8_1__status_write` and `ex_b8_{1,2}__status_write` are
  constant 0 while the matching `status_read` shows the equipment running (§1.7). The agent cannot
  start or stop plant. The action space is continuous setpoint trim and nothing else — which
  incidentally removes the discrete/continuous hybrid problem entirely.
- **~8% of steps belong to a human.** `override_control` ≠ 1 on 7.85% / 8.36% of steps (§1.8). Those
  steps are neither the agent's to claim nor to learn from. They must appear in the observation as a
  flag with a defined hand-back behaviour — the RL equivalent of the brief's `on_bad_data()`
  requirement.

---

## 2. Algorithm selection

### 2.1 Headline — SAC with Lagrangian constraints, inside the grey-box, with DR over F4

Soft Actor-Critic. Five reasons, each tied to a fact above rather than to general practice:

1. **The action is continuous, bounded and 2-dimensional** (ΔSP, ΔSAT per AHU). That is SAC/TD3
   territory. DQN's discretisation is a choice here, not a requirement — see §2.4.
2. **The maximum-entropy objective is the correct prior given F2.** The logged policy sat on one
   setpoint value 85% of the time, and the replay buffer is seeded from it. A deterministic-policy
   method (DDPG, TD3) collapses onto that band early and never discovers AHU-1's headroom, which is
   the one real asymmetry in this building. The entropy bonus keeps the policy spread across the
   admissible range long enough to find it.
3. **Automatic temperature tuning removes the one hyperparameter there is no budget to tune.** With
   ~62 real episodes (F1) there is no held-out set large enough to sweep α by hand without
   overfitting the validation split.
4. **Off-policy replay lets the 1,800 real transitions seed the buffer** (SACfD-style). The agent
   starts from the plant's own behaviour rather than from noise — which matters because the
   surrogate is least trustworthy exactly where the plant has never been.
5. **It is the literature baseline** for building control (BOPTEST, Sinergym, CityLearn), so the
   result is comparable to published work rather than bespoke and unfalsifiable.

**Lagrangian, not scalarised.** Following F5, the problem is a CMDP:

```
   maximise    E[ Σ  −E_fan(t) − λ_move·|Δu(t)| ]
   subject to  zone comfort band (or |PMV| ≤ 0.5)      satisfied
               V̇_i ≥ minimum_air_flow_rate_setpoint_read_i
```

with multipliers learned by dual ascent rather than weights chosen by hand. The practical difference
is not subtle: a hand-weighted penalty trades comfort for energy at whatever exchange rate the
weight implies, and the tuner discovers that rate only after the agent has exploited it. A
constraint says the trade is not available.

### 2.2 Pure-offline arm — IQL, with TD3+BC as the baseline to beat

If the surrogate is rejected and learning must happen on the logged record alone:

**IQL (Implicit Q-Learning), not CQL.** IQL performs expectile regression on in-sample values and
**never evaluates Q at out-of-distribution actions**. Under F2 that is decisive: CQL's OOD penalty
coefficient needs a sweep that cannot be validated offline (F2 also killed the OPE that would have
validated it), and mis-set in either direction it does nothing or collapses to a clone. IQL's
failure mode is graceful — with data this narrow it degrades toward behaviour cloning, which is the
*honest* answer ("this record does not identify a better policy") rather than a confident wrong one.

**Run TD3+BC first.** It is roughly ten lines beyond TD3 — a behaviour-cloning regulariser on the
actor loss — and on datasets of this size it frequently matches everything more sophisticated. If
the elaborate method does not beat it, that is the finding, and it should be reported as one.

### 2.3 Arguably the better engineering answer — PETS / MPC over the grey-box

Worth stating plainly: **the sequential-decision content of this problem is thin.** A setpoint change
shows up in fan power within 1–2 steps; the only slow state is zone thermal mass at τ ≈ 1–3 h. With
γ ≈ 0.9 over 44-step episodes, most of what an "RL agent" would be doing is receding-horizon
optimisation against a model.

So a probabilistic-ensemble model with short rollouts (PETS), or plain MPC over the grey-box
equations, is a legitimate and possibly superior answer — it is more sample-efficient by construction,
its ensemble variance gives a principled *"do not go where the data is not"* signal, and it is far
easier to explain to someone who will be responsible for it running on a real building. Its weakness
is that it inherits the surrogate's errors with no policy-side robustness, which is exactly what DR
(F4) buys back for SAC.

**Recommendation: run both.** If MPC matches SAC, the honest report says the learned policy added
nothing, and that is a result worth having.

### 2.4 Comparison arms, not headline

| Algorithm | When it earns its place | Why not headline |
|---|---|---|
| **PPO** | The surrogate is analytic and cheap to step, so on-policy sample inefficiency (F1) costs only wall-clock, not data. Most robust of the family to reward shaping. | Cannot use the 1,800 logged transitions at all; returns are only as good as the surrogate, with no replay-seeding to anchor it. |
| **TD3** | Deterministic alternative; sometimes better final performance. | Loses reason 2 of §2.1, which is the reason that is specific to *this* data. |
| **QR-DQN / Rainbow** on discretised trim | Genuinely defensible: the real supervisory action **is** discrete — ASHRAE G36 trim-and-respond is literally `−0.02 / hold / +0.03`. Discretising collapses the exploration problem, makes constraint enforcement trivial (mask illegal actions rather than penalise them), and a **distributional critic lets you optimise CVaR on the comfort term** instead of its mean — the right risk attitude when the constraint is comfort. | Joint action space grows across two AHUs and two levers; loses the smoothness prior that setpoints ought to move gently. |

The QR-DQN arm is the one most likely to surprise. If the discretised agent matches SAC, the correct
conclusion is that the continuous action space was never needed and the deployable controller is a
trim table — which would be a *better* outcome for a building.

### 2.5 Rejected, with the reason

Rejecting a plausible method with a number is worth more than ignoring it — the same discipline
control-gap §6 applies to Harris indices and IPMVP regression.

| Method | Why not |
|---|---|
| **Decision Transformer / any sequence model** | Return-conditioned sequence modelling needs 10⁵–10⁶ trajectories. **You have 62** (F1). This is not a tuning problem. |
| **Recurrent policies (LSTM/GRU)** | F3: the sub-step information a recurrent policy would recover was destroyed by 15-minute averaging before the file was written. Frame-stacking captures what remains at a fraction of the data cost, and BPTT on 62 trajectories overfits. |
| **Deep or wide networks (≥4 layers, ≥512 units)** | Critic overfitting and value divergence is *the* dominant failure mode at 10³ transitions. Capacity here buys memorisation, not generalisation. |
| **Learning the reward** | Energy is analytic and already validated on this floor: `k = 0.0387593` kW/(L/min·°F) with every rival unit hypothesis 1.8×–59× out (§1.2), and the fan curve at R² 0.903 / 0.960 (§1.9). Replacing a measured quantity with a learned one discards the strongest asset in the repository. |
| **Importance-sampling OPE (WIS, PDIS)** | F2: near-deterministic behaviour policy ⇒ degenerate weights and unusable variance. Use the surrogate plus the frontier assertion (§5) instead, and say that this is why. |
| **Curiosity / novelty-driven exploration** | The action support is *deliberately* bounded to where the model was identified (grey-box §9). An intrinsic reward for novelty rewards leaving that region — precisely the wrong incentive. |

---

## 3. The network backbone

Small and unremarkable is the correct answer. The interesting engineering is in the features and the
constraint layer, not the architecture — and given F1, every parameter must earn its place.

| Choice | Value | Why, on this data |
|---|---|---|
| Trunk | **MLP, 2 hidden × 256** (2 × 64 for the offline arm) | Matched to ~10³ transitions; no convolution (no spatial structure), no attention (no sequence to attend over after F3) |
| Activation / head | ReLU; tanh-squashed Gaussian policy head | Standard SAC; the squash enforces the action bounds structurally |
| **LayerNorm on critic hidden layers** | yes | The single highest-value trick at this scale — it stabilises value estimation and acts as implicit pessimism against OOD actions, which is exactly F2's failure mode |
| **Critic ensemble** | 5–10, pessimistic aggregate (min or low quantile) | Buys sample efficiency via a higher update-to-data ratio, *and* supplies the epistemic uncertainty F4 says the environment genuinely has |
| History | **frame-stack 4** (= 1 h) + explicit lags of `T_oa` and zone demand | F3: cheaper, more stable, and no less informative than recurrence |
| Time encoding | sin/cos hour-of-day, sin/cos day-of-week | The BMS schedule is the dominant exogenous driver: on-fraction 0.87–0.90 from 08:00–17:00, 0.05 at 06:00 (§1.4) |
| **Observation normalisation** | fixed plausible ranges from [src/units.yml](src/units.yml), **not** dataset statistics | Load-bearing, not cosmetic: this dataset mixes °C, °F, inWG, L/min and Hz *on the same AHU* (§1.2). A scaler derived from data statistics is unauditable and silently changes when the mask changes; one derived from declared physical ranges is neither |
| Zone aggregation | hand-built rogue-excluded top-*k* statistic | See §4 |

**On the zone encoder.** The principled architecture for "53 boxes whose order is meaningless" is a
permutation-invariant set encoder — DeepSets, or attention pooling. It handles variable box counts,
learns its own demand statistic, and would deal with rogue exclusion by learning to down-weight
rather than by a hand-coded rule. It is the right upgrade **at roughly 10× this data**. At 62
episodes it is a parameter budget spent where a five-line summary statistic already captures the
finding, and the finding is known (§1.6). Note it; do not build it.

---

## 4. MDP specification

### Observation (~28 dims per AHU)

| Group | Contents |
|---|---|
| Exogenous | `T_oa`, `T_wb`, sin/cos hour, sin/cos day-of-week, PIR occupancy fraction |
| Plant | fan Hz, fan kW, SP actual + setpoint, SAT actual + setpoint, cooling-valve %, water flow, water ΔT, `cooling_rate` |
| Zone demand | count of boxes at ≥90% damper **after excluding the six rogues of §1.6**; the **2nd and 3rd highest damper positions** — i.e. the G36 ignore-top-*I* statistic made an observation rather than a rule; max and mean (zone T − setpoint); p95 zone temperature |
| Data quality | outage flag, override flag, sentinel-count flag |

The data-quality group is not decoration. The brief requires it to be obvious what a controller does
when data is missing or a sensor reads nonsense; putting the flags *in the observation* makes the
hand-back behaviour a learned, testable part of the policy rather than a wrapper around it. It fires
on the 7% outage, the `0.00` sentinel zone temperatures, and the 8% override windows.

### Action

```
   Box( low = [−0.02, −0.3] , high = [+0.02, +0.3] )        ΔSP [inWG] , ΔSAT [K] per step
   SP_sp' clipped to [0.30, 0.64] (AHU-1) / [0.40, 0.78] (AHU-2)
```

Bounds are the **observed ranges of the logged setpoints**, per grey-box §9. Outside them every
block of the environment is extrapolating.

### Reward and constraints

Per §2.1: minimise fan kWh (offset cubic, measured) plus a small actuator-movement term; comfort and
minimum airflow as **constraints with learned multipliers**. Coil load reported in `info` as
kWh-thermal; COP appears only in domain randomisation, never inside the scalar.

### Episode

One occupied weekday, 07:00–18:00 = **44 steps**. `terminated` at 18:00; `truncated` on an outage or
an operator-override window, with the override steps excluded from both the return and the
evaluation — the same rule [src/masks.py](src/masks.py) already enforces for every other analysis.

---

## 5. Evaluation protocol

This is where an RL result either survives contact with the rest of the repository or does not.

1. **Time-based split only.** Train 18 May → 29 Jul; test 3 Aug → 23 Aug. The split falls naturally
   on the 4-day collector outage (30 Jul → 3 Aug). A random split leaks: 15-minute steps are heavily
   autocorrelated, and neighbouring steps in train and test would be near-duplicates.
2. **Identical mask, asserted.** Every policy — as-operated, naive cut, G36, RL — evaluated through
   `masks.AnalysisMask.assert_same`. If the agent is scored on more steps than the baseline, the
   mask treatment alone manufactures the saving.
3. **Four benchmarks**, reusing the harness that already exists: as-operated baseline; naive fixed
   20% SP cut; G36 trim-and-respond with ignore-top-2; the demonstrated-achievable frontier.
4. **The frontier ceiling, as an assertion rather than a discussion.**

   > If the trained policy reports a fan saving materially above **3.9% (AHU-1) / 4.4% (AHU-2)**
   > plus its own explicitly-stated airflow-reduction share, **it is exploiting the surrogate, not
   > the building.** Fail the run.

   This is the single most important line in the document. The frontier is model-free — no fan law,
   no exponent, no COP — so it is the one number an over-fitted environment cannot inflate.
5. **Attribution, always split.** Every reported saving decomposed into the efficiency part and the
   ventilation part, because grey-box §4 separates them by construction. A blended number hides
   exactly the thing control-gap §5 found.
6. **Environment validation precedes agent training**, per grey-box §8: one-step RMSE, 8-step
   open-loop error against persistence, elasticity recovery at β = 0.382, coil closure against the
   validated water side. *If the surrogate cannot beat persistence at 8 steps, stop and report
   that* — do not train an agent on it.

---

## 6. What to build

```
src/rl/
├─ plant.py       # the grey-box: blocks A-F. Reuses effects.sp_to_hz, effects.fit_fan_models,
│                 # energy_balance, masks.mask_room_temp_sentinels
├─ identify.py    # NNLS zone RC, SAT bias, mixing map, zone PI gains - training window only
├─ env.py         # Floor8SupervisoryEnv(gymnasium.Env) - obs/action/reward per §4
├─ wrappers.py    # units.yml normalisation, frame stack, override hand-back, action clipping
├─ train.py       # SAC + Lagrangian (stable-baselines3 / sb3-contrib); d3rlpy for the IQL arm
└─ evaluate.py    # mask-symmetric comparison + the frontier assertion + attribution split
scripts/run_rl.py
tests/test_rl_env.py
```

`config.yml` gains an `rl:` block: episode window, action bounds and rate limits, the DR
distributions of §8.2 of the grey-box document, λ initialisation, γ, network sizes, seeds.

**Tests worth naming**, in the spirit of the leakage test already specified for the controller:

- Gymnasium API conformance and **bit-exact determinism under a fixed seed**.
- Reward equals the analytic energy on a hand-worked single step.
- Action clipping: an action pushing the setpoint outside the logged support leaves it at the bound.
- **Elasticity recovery**: the finished environment returns β = 0.382 ± CI under a simulated −20%
  cut — the same falsification shape as
  `tests/test_excitation.py::test_recovers_a_known_elasticity`.
- **Frontier assertion fires**: a deliberately mis-specified environment that permits a 25% saving
  must fail the check.

---

## 7. Positioning — and the recommendation not to submit this as Task B

**Do not make RL the Task B strategy.** The brief grades whether every number traces back to a place
in the data, and states that a modest well-defended saving scores higher than a large one that
cannot be traced. An RL policy's kWh traces to a **surrogate**, and the surrogate's most important
parameter has an interval wider than the effect it measures (F4). The rule-based options remain the
submission: static-pressure reset with rogue exclusion, or that plus SAT reset per AHU.

**Where RL does earn its place** is as a clearly-labelled extension whose contribution is a **bound,
not a saving**:

> *A near-optimal agent, given the best surrogate this record identifies, captures X% of the
> demonstrated-achievable frontier. The gap between X and the rule-based controller is what a
> learned supervisor would be worth on this floor — and the gap between X and the frontier is what
> is left for better instrumentation.*

That framing survives the follow-up interview. "RL saved 24%" does not, and on this floor it would
additionally be false — control-gap §5 already shows why.

There is one further reason to do it anyway, which has nothing to do with the number. Building the
surrogate forces every implicit assumption in the rule-based analysis to be written as an equation
with an identification route and a falsification test (grey-box §8.1). Several of them turn out not
to survive that — the SAT loop cannot be modelled as holding its setpoint, the airflow unit has
never been discriminated, and blocks A and C cannot be chained without double-counting. **Those
findings are worth more than the policy.**
