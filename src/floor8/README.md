# The surrogate, assembled — plant, environment, agent

> **What this is.** Blocks A–F wired into a steppable plant, wrapped as an RL environment, and the
> SAC agent trained inside it. The audit trail for the blocks themselves lives in the six
> `block-*-derivation/` folders; this is what happens when they are put together.
>
> **Run it:**
> ```
> python scripts/verify_env.py        # 11 falsifiers - run this first
> python scripts/train_sac.py         # training window only; never opens the test month
> python scripts/compare_to_record.py # the gate, then the four-way comparison
> ```
>
> **Three findings up front.**
>
> 1. **The surrogate closes, and it reproduces the record.** Replaying the plant's own logged
>    setpoints on the held-out window returns fan power to within **−0.07 / −0.36 kW** of measured,
>    zone temperature to **−0.33 K** bias (RMSE 1.92 K over 29,334 zone-steps), and delivered air to
>    within 6.8%. That gate had to pass before any policy number could mean anything.
> 2. **The rule-based arms trip the frontier assertion, and that is the headline.** G36
>    trim-and-respond reports a 16.1% fan saving against a model-free ceiling of ~5%. The assertion
>    exists precisely to catch this: a large number here is evidence the model is wrong, not evidence
>    the saving is real.
> 3. **Two real bugs were caught by falsifiers rather than by inspection** — the fan-model band
>    switching functional form while keeping the other form's coefficients (a 10-order-of-magnitude
>    error), and Block C's inversion crediting free boxes with a clipped neighbour's share. Both are
>    now permanent checks.

---

## 1. Layering, and why the seam is where it is

```
  policies.py     as_operated · naive_cut · g36_trim_respond · SAC     (plain callables)
  sac.py          actor · critic ensemble · replay · Lagrangian dual ascent
  wrappers        (folded into env.py: action rescale, obs normalisation)
  env.py          Floor8SupervisoryEnv    <- thin: state, params, cursor, rng
  ══════════════ nothing below this line imports the environment ══════════════
  plant/step.py   1.1's block order, executed literally
  plant/blocks.py blocks A-F as pure functions
  plant/state.py  frozen PlantState + Diagnostics
  plant/params.py frozen PlantParams, loaded from the derivation artifacts
  tape.py         EpisodeTape: CSV -> masked, day-sliced, immutable
```

**All physics lives below the Gymnasium boundary.** Three reasons, each traceable to something that
actually happened in this repository:

1. **4.5's specific-fan-power test and block-d 7.1's 8-step open-loop rollout must run without an
   environment.** If the physics lived in `step()`, every falsifier would re-implement it and the
   divergence would be undetectable.
2. **Block C has been corrected twice**, both times from its derivation script. Script and simulator
   must call the same model or the audit trail validates something the agent never runs. Falsifier
   11 is that seam, and it caught a real disagreement the first time it ran (§4.2).
3. **Determinism.** `step(state, exog, action, params, noise)` is a pure function, so a fixed seed
   reproduces a trajectory bit for bit. Block E's `η` is the only stochastic term in the physics and
   it is **passed in, not drawn**.

**The environment deliberately does not** apply the Lagrangian (it emits raw constraint costs and
dual ascent lives in the training loop, so one env serves SAC-Lagrangian, plain SAC and the MPC arm
of 2.3), fit anything, read the CSV, or evaluate the frontier assertion.

---

## 2. Parameters — where every number comes from

`PlantParams.from_derivations()` reads artifacts. It fits nothing.

| Block | Source | Reproduces in this tree? |
|---|---|:--:|
| A — β | `block-a-derivation/data/beta_summary.csv` | ✗ |
| B — fan | `block-b-derivation/data/models_by_mask.csv` | ✗ |
| C — split/total | `block-c-derivation/data/{room_weights,total_fit}.csv` | ✓ |
| D — zone thermal | `block-d-derivation/data/stage2_fit.csv` | ✓ |
| E — SAT/mixing/unit | `block-e-derivation/data/{sat_bias_fit,mixing_map,airflow_unit}.csv` | ✓ |
| F — zone control | `block-f-derivation/data/zone_gain_fit.csv` | ✓ |

> **Blocks A and B do not run here.** Both open `from src import ...` and no such package exists in
> this working tree; their `data/*.csv` were produced elsewhere. `provenance_table()` marks them
> `NOT reproduced here` and the training script prints it. They are not presented as freshly fitted.

> **Block B's mask is not a neutral choice.** §3 of the surrogate quotes
> `P = 0.719 + 6.199·(f/50)³, R² 0.903`, which is the **`all_usable`** row. Every other block is
> fitted on the `analysis` mask, whose row is `1.097 + 5.083, R² 0.816`. The plant defaults to
> `analysis` for consistency; the three-model band is a DR axis regardless.

### 2.1 The imputation layer, and why one is needed

Each block excluded what would have corrupted *it*: Block C fits `a_i` on clean boxes only (23 of 26,
25 of 27), Block D fits 51 of 53, Block F identifies 41 of 53. **A simulator cannot inherit those
exclusions**, because a zone excluded from a fit still occupies the floor — 4.3 is explicit that
rogue boxes leave the *demand statistic* the agent observes but not the comfort constraint, "a
starved zone is still a zone."

So every box gets parameters and every box carries a provenance string. **15 of 53** are imputed or
pooled; `box_provenance_table()` names them.

The `a_i` imputation is the only one that needed an argument rather than a median. `a_i` *is* air
drawn per point of damper opening, so `mean_flow/mean_damper` should be proportional to it with one
constant per AHU. Measured on the boxes Block C **did** fit, that holds almost exactly — **corr
0.9986 (AHU-1) and 0.9955 (AHU-2)**, the ratio's interquartile range spanning 0.00771–0.00779 on
AHU-1. So the scale is read off the fitted boxes and applied to the excluded ones. The tightness of
that ratio is also a small independent check that the weight means what 4.2 says it does.

For `vav_8_2_28` — 100% damper, ~50 delivered against a 1,000 setpoint — this lands near zero, which
is the behaviour the box actually exhibits rather than a workaround for it.

---

## 3. Boundary conditions

`w_t` is replayed from the record, never simulated. 1.1 calls this "the single most important
structural choice in the surrogate," and here it is a dependency direction: **the plant never touches
the CSV.**

| | Source |
|---|---|
| `T_oa`, `T_wb` | `outdoor_weather_station__*` — **converted °F → °C** (block-e §5) |
| `occ` | mean of the four `floor_8_zone_*_iaq_*__pir` |
| `T_sp_zone` | `vav_*__room_temperature_setpoint_read`, replayed incl. the parked 27.0 °C |
| `T_m` | each zone's own 21:00–05:00 mean, per night (block-d 5.1) |
| `fan_on` | `ahu_b8_*__frequency > 5 Hz` — the BMS schedule, **not** the agent's to command |
| `oa_damper` | `ahu_b8_*__fresh_air_damper_position_read` |

**Two gaps had to be handled, and they are handled differently on purpose.**

*Occupancy* is 12.5% missing. Gaps are filled from the **hour-of-day mean**, not from zero:
occupancy drives Block D's gain term, and zeroing a missing day would silently delete the load
rather than estimate it. The profile is strong enough to interpolate — 0.02 overnight against
0.24–0.27 in the occupied window.

*Outdoor temperature* is 10% missing including **six entirely blank days**. Partial gaps are bridged
across the record; a day more than half missing is **dropped**. An episode is supposed to be a real
day, and a day whose dominant driver had to be invented is not one. Ten days go, leaving **86**:
67 train, 19 test. Three of the dropped days are the collector outage the train/test boundary
already falls on.

**The agent cannot start or stop plant.** `ahu_b8_1__status_write` is constant 0 while `status_read`
shows the equipment running, so `fan_on` is replayed and the action space is setpoint trim only.

---

## 4. The two bugs the falsifiers caught

Worth recording, because both were invisible to inspection and both are now permanent checks.

### 4.1 The fan band switched form but kept the other form's coefficients

The DR axis over `{offset_cubic, power_law, ideal_cubic}` changed `kind` while `BlockB` carried a
single `(a, b)`. Read as `a + b·(f/50)³` those coefficients give ~5 kW; read as `a·f^b` they give
**1.4 × 10⁸ kW**. Training "ran" and reported a fan consumption of 4.8 × 10¹⁰ kWh per episode, with
`alpha` and both multipliers quietly NaN.

Two fixes, and the second matters more than the first: coefficients now travel **with** the kind
(`BlockB.coeffs()` is the only accessor), and the environment raises on a non-finite or absurd
reward rather than letting it propagate. Falsifier 10 checks all three models stay under 20 kW over
the observed 30–45 Hz band.

### 4.2 Block C's inversion credited free boxes with a clipped neighbour's share

Falsifier 11 asserts that `block-f-derivation`'s compensating arm and `plant/blocks.py`'s inversion
agree. They are deliberately *different* parameterisations — the derivation works in ratio space so
the scale constant `C` cancels (block-f README 3.2 makes that a feature, since `C` is Block C's
weakest number and carries the airflow unit), while the plant works in absolute target space because
that is what a simulator needs.

They disagreed by **0.36 damper points**. The plant was computing each free box's weight as
`v_target_i · W / v_requested_total`, which is right only while nothing clips: once a box hits 100%,
the achieved total falls below the request, and dividing by the request hands the free boxes the
clipped box's share. Corrected to divide by the achieved `C·W^(q−1)·SP^s`, the disagreement is
**5.7 × 10⁻¹⁴**.

This is exactly the failure mode the "one forward model, two consumers" rule exists to prevent, and
it is the third time a Block C formulation has been wrong.

---

## 5. The agent

`sac.py`, written against torch directly. Every `reproduce_*.py` in this repository is self-contained
because there is no `src/` package to import, and stable-baselines3 has no CMDP support — the dual
ascent below has to be hand-written under either choice — so a framework would have bought a black
box and little else. The hyperparameters are stated so the result stays comparable.

| | | why, on *this* data |
|---|---|---|
| Actor | 2×256 MLP, tanh-squashed Gaussian | 3: matched to ~10³ real transitions; capacity buys memorisation |
| Critics | ensemble of 5, **LayerNorm**, pessimistic min | "the single highest-value trick at this scale" — implicit pessimism against OOD actions, which is F2's failure mode |
| Entropy | auto-tuned α, target −4 | 2.1 reason 3: no held-out set large enough to sweep it |
| γ / τ | 0.99 / 0.005 | 96-step episodes |
| Replay | 1e6, batch 256, **seeded from the record** | 2.1 reason 4: start from the plant's own behaviour, because the surrogate is least trustworthy where the plant has never been |
| Constraints | λ_comfort, λ_vent by dual ascent | a CMDP, not a scalarised weight — F5 |

**Two implementation details that are easy to get wrong.** The bootstrap is kept on `truncated` and
zeroed only on `terminated`, which is *always* False here — marking the horizon terminal trains fine
and converges to a critic that believes the world ends at midnight. And the buffer stores reward and
costs **separately**, applying the Lagrangian at sample time, so a multiplier update never leaves
stored rewards stale.

### 5.1 The comfort budget is measured, not chosen

It is the as-operated policy's own comfort cost on the training window (**≈1395** per episode), so
the constraint reads *"no worse than what the plant already did"* — the brief's rule that savings
bought with discomfort do not count.

It has to be measured, because 4.3 keeps the rogue boxes in the comfort statistic and those zones
are starved whatever the agent does; `vav_8_1_28` sits at a median 28.9 °C with its damper pinned.
They contribute a large constant the agent cannot act on, and setting the budget from the baseline
cancels it. Set to a guessed 1.0, the multiplier saturated at its cap within 50 episodes.

### 5.1a Which comfort statistic — and why the one-sided one was the wrong objective

`Floor8SupervisoryEnv(comfort_mode=...)` selects what `cost_comfort` returns. Both are always
computed and both always reach `info`, so a run can be re-scored without re-simulating.

| mode | statistic | what it can see |
|---|---|---|
| `one_sided` | `max(0, T_i − T_sp − db)²` | too warm only — the original |
| `two_sided` | `max(0, \|T_i − T_sp\| − db)²` | **holds the setpoint**; overcooling is a violation |

The distinction is not academic on this floor. Measured over 29,334 scored zone-steps, the record
runs a mean **1.46 K below its own setpoints**, with **51.1%** of zone-steps more than 0.5 K *too
cold* against **12.7%** too warm. The one-sided metric is blind to that half, and it is the
expensive half — cooling past the target is energy spent to make people less comfortable, not more.
Scored two-sided, the record's comfort cost is ~31× what the one-sided number reports.

The consequence showed up directly in a comparison run: optimising against `one_sided`, SAC spent
~6% more electricity to run the floor 0.02 K *cooler* — an improvement the one-sided objective
rewarded and the occupants would not have noticed. Under `two_sided` the two goals stop fighting,
because warming an overcooled zone back toward its setpoint saves fan *and* coil energy.

### 5.1b The budget for checkpoint selection is measured on the validation days

The training constraint uses the fit-window budget; **checkpoint selection uses a budget measured
on the validation days themselves**. The two day sets are not equally hard — validation ran ~1.5×
the fit-window comfort cost — so scoring a validation rollout against the fit budget marked every
checkpoint infeasible, including ones comfortably better than the plant on those very days.
Feasibility has to be judged against what the plant achieved on the *same* days.

### 5.1c The dual ascent was unstable, and the fix was three things

A 150k-step run left `λ_comfort` slamming between 2.8 and its 1000 cap while per-episode comfort
swung 64 → 3825, right to the final episode. Three compounding causes, all now fixed:

1. **The dual step ran every environment step** — 150,000 times — on a statistic that only changes
   at the episode boundary. It now runs once per episode, from `note_episode_costs`.
2. **The violation was raw, not normalised.** The gradient on `log λ` is `λ·(J_c − budget)`; with an
   excess of ~2400 and `λ≈1`, a single step moved `log λ` by +2.4, i.e. `λ` ×11. Both the
   sample-time penalty and the dual gradient are now divided by the budget, so one budget's worth
   of violation is 1.0 and `λ` is a readable exchange rate.
3. **`lam_max = 1e3` was three orders of magnitude too loose** on that normalised scale. At
   `λ_c ≈ 1000` the constraint term buried the energy reward and the critic was fitting the
   multiplier rather than the plant. Now 20.

`lr_dual` is correspondingly a **per-episode** rate (0.05), not a per-step one.

### 5.2 The movement penalty was mis-scaled, and it mattered

`move` is in units of the rate limit and reaches 4.0 when every lever moves at full rate, while a
step's actual energy is ~1.5 kWh. At the initial `λ_move = 0.5` the smoothness term was **1.3× the
entire objective** and the agent was optimising for standing still. Now 0.02 — a full-rate move
costs ~5% of a step's energy: a nudge against actuator chatter, not a competing objective.

---

## 6. Falsifiers

`scripts/verify_env.py`, 11 checks, all passing. Each is a claim one of the documents makes, turned
into an assertion.

| | check | current |
|---|---|---|
| 1 | closure audit — every state variable has exactly one producer | 9 fields finite |
| 2 | determinism under a fixed seed | bit-identical |
| 3 | `terminated` never True; `truncated` exactly at the horizon | ok |
| 4 | action clipping to logged support, rate limit, post-horizon raises | ok |
| 5 | adding-up `Σ V_i = V_total` (4.2) | 4.2e−16 |
| 6 | Block F inverts Block C (round trip) | 4.3e−14 damper points |
| 7 | λ = 0 reproduces block-c's frozen arm | 10.79% / 14.92% |
| 8 | elasticity recovery under a −20% cut (2.4) | β 0.3816 / 0.2316 |
| 9 | reward equals the analytic energy | ok |
| 10 | all three fan models physical over the observed band | max 5.9 kW |
| 11 | derivation and plant agree | 5.7e−14 damper points |

Falsifier 1 is the one that would have caught `d_i`, `valve(t)` and `T_ra` — three variables the
1.1 step order consumed with nothing producing them.

---

## 7. What this layer does and does not establish

**Establishes.** That the blocks compose into a plant which reproduces the record under the record's
own actions (§1, finding 1); that the environment is deterministic, mask-symmetric across policies,
and bounded to the logged support; and that a controller comparison can be run at all.

**Does not establish.** Any absolute kWh claim — 10.1 forbids it while control-gap 5's excess is
unexplained, and 7.3 widened the band rather than narrowing it. Electric totals are reported as a
COP band and coil duty stays thermal. Every agent number is reported across the λ sweep, because λ
decides the ventilation cost and this record cannot locate it.

**And the rule that outranks the rest:** whatever an agent saves inside this surrogate is bounded by
the surrogate's fidelity, not the building's. The frontier assertion fires on the rule-based arms
already.
