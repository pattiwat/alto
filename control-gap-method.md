# Quantifying the Control Gap on Floor 8 — Method and Findings

> **What this is.** How to measure how far this floor's control sits from what it could do, which
> methods are sound on *this* dataset, and what they return when run. Every number below is produced
> by `python scripts/run_control_gap.py` and written to `report/` as CSV — nothing here is typed by
> hand.
>
> **It corrects two claims** in
> [from-altotech-take-home-pdf-and-alto-bui-jolly-turing.md](from-altotech-take-home-pdf-and-alto-bui-jolly-turing.md)
> (§1.5, §1.9) and [hvac-system-logical-flow.md](hvac-system-logical-flow.md) (§2, §5): the
> supervisory setpoints are **not** constants, and the static-pressure → frequency link is **not**
> unmeasurable from this data. See §1. Where those documents and this one disagree, this one is
> derived from the raw CSV by executable code and should be preferred.

---

## 0. The gap is three gaps

Conflating them is the standard error, and each needs a different method.

| | Gap | The question | Method |
|---|---|---|---|
| **G1** | Regulatory | Do the loops hold the setpoints they were *already given*? | M1 |
| **G2** | Supervisory | Are those setpoints the right ones? | M4 → M5 |
| **G3** | Energy | What do G1 and G2 cost in kWh? | M4 + M6 |

**G1 and G2 are not additive.** Fixing a mis-tuned SAT loop changes the load a pressure reset then
sees. They are evaluated cumulatively in a stated order, never summed.

---

## 1. The premise check — the setpoints move

Both prior memos state that the static-pressure and SAT setpoints are fixed for the whole period,
and conclude that the SP → Hz link must therefore be assumed rather than measured. Neither is true.

| Point | Prior memos say | Actually |
|---|---|---|
| `ahu_b8_1__static_pressure_setpoint_read` | "0.55 inWG (fixed)" | 0.55 on 7,362 / 8,650 steps, but **122 distinct values**, range 0.300–0.636 |
| `ahu_b8_2__static_pressure_setpoint_read` | "0.60 inWG (fixed)" | 146 distinct values, range 0.400–0.780 |
| `ahu_b8_1__supply_air_temperature_setpoint_read` | "18.3 °C" | on-hours range **1.27 K**, moving hour to hour |
| `ahu_b8_2__supply_air_temperature_setpoint_read` | "16.8 °C" | on-hours range **3.00 K** |

Minor: §6 of the logical-flow memo lists the outside-air damper as *inferred*.
`ahu_b8_1__fresh_air_damper_position_read` is a logged point (909 unique values, 0–100%).

**The plant ran its own experiments.** That is the opportunity this document is built on.

---

## 2. The analysis mask

Defined once in [src/masks.py](src/masks.py), used by every method, so that baseline and
counterfactual can never see different steps — the one failure that manufactures savings on its own.

| Component | Why | AHU-1 | AHU-2 |
|---|---|---:|---:|
| on-hours | fan stopped ⇒ tracking error is meaningless | −6,574 | −6,393 |
| outage | four contiguous collector failures | −0 | −0 |
| override | ~8% of steps a human was driving by hand | −49 | −44 |
| settling | first 2 steps after a setpoint change are transients | −875 | −770 |
| complete | every input a priced quantity reads is present | −0 | −187 |
| **Evaluation set** | | **1,807** | **1,911** |

Two points about this mask are load-bearing:

* **Settling is per-signal.** For the episode analysis it is applied relative to the setpoint being
  manipulated, inside the pairing, not to every setpoint at once. Applying it globally there thinned
  the control windows and cut usable AHU-1 episodes from 15 to 7 — enough to move the answer.
* **Completeness is part of the mask, not a `dropna()` at the point of use.** The fan curve, the
  setpoint cut and the airflow frontier read different columns; left alone each would silently drop
  different steps and be compared anyway. `run_control_gap.py` asserts all three price an identical
  step count and prints the confirmation.

---

## 3. M1 — the regulatory gap (G1)

Statistics chosen to survive 15-minute averaging, so each is a **lower bound** on the underlying
1-minute behaviour: averaging can hide movement, never invent it.

| | AHU-1 SAT | AHU-2 SAT | AHU-1 SP | AHU-2 SP |
|---|---|---|---|---|
| error mean | **−1.02 K** | −0.22 K | +0.00 inWG | −0.05 inWG |
| \|error\| beyond tolerance | 32.5% | 34.4% | 7.9% | 16.1% |
| actuator mean | 41.8% | 41.4% | 35.7 Hz | 35.2 Hz |
| error sign changes | 8% | 11% | 50% | 40% |

**The authority diagnostic** is the part that carries weight. Splitting large-error steps by whether
the actuator had anywhere left to go separates two failures that look identical in an error
statistic:

| Loop | large error, saturated | large error, mid-range | Verdict |
|---|---:|---:|---|
| AHU-1 SAT | 14 | **574** | **MIS-TUNED** — 2.4% saturated when missing setpoint |
| AHU-2 SAT | 214 | 508 | MIS-TUNED — but 29.6% saturated; part of AHU-2 is genuinely short of capacity |
| AHU-1 SP | 0 | 142 | MIS-TUNED |
| AHU-2 SP | 0 | 337 | MIS-TUNED |

This turns "mis-tuned, not starved" from an adjective into evidence, and it decides whether a
setpoint reset is even admissible: resetting a setpoint the plant is not holding moves a target the
loop was already missing.

> **Implementation note worth knowing.** Saturation is a fraction of each actuator's *own* span
> (valve 0–100%, VSD 0–50 Hz). A fixed ">95" threshold — the obvious first cut — can never fire on
> the fan, so the pressure loops would have silently returned "mis-tuned" regardless of the data.
> `tests/test_loop_performance.py::test_saturation_bounds_use_each_actuators_own_span` pins this.

---

## 4. M4 — measuring what a replay would otherwise assume

The affinity relation says fan pressure goes as Hz² at a fixed system curve, so
`d ln(Hz) / d ln(SP) = 0.5`, and every kWh downstream of a pressure reset rides on that 0.5.

**Design.** Each episode where the plant held its setpoint away from the modal level is compared
against its *own* adjacent steps at the normal setpoint — same day, same weather, same occupancy.
The first 2 steps of each episode and of the recovery are dropped as transients.

### 4.1 The episodes

| | AHU-1 | AHU-2 |
|---|---|---|
| Episodes found | 22 | 30 |
| Paired with a valid control window | **15** | 30 |
| Median length | 13 steps | 5 steps |
| Under operator override | 0 | 0 |
| Hour-of-day distribution | spread 07h–17h | **93% at 08h** |

> **AHU-2's episodes are start-up transients.** The runner detects and warns about this
> automatically. AHU-2's estimate is therefore precise but not trustworthy, and AHU-1's is
> trustworthy but imprecise — an awkward pairing that has to be reported rather than resolved.

### 4.2 The estimates

| | AHU-1 | AHU-2 *(transient-contaminated)* |
|---|---|---|
| Mean excitation | −17.4% | −30.4% |
| Mean frequency response | −6.7% | −7.0% |
| **`d ln(Hz)/d ln(SP)`** | **0.382**, 95% CI **[0.241, 0.529]** | **0.232**, CI [0.178, 0.283] |
| vs assumed 0.500 | **not rejected** | **rejected** |
| `d ln(kW)/d ln(SP)` | 1.130, CI [0.770, 1.553] | 1.096, CI [1.001, 1.196] |
| vs assumed 1.500 | not rejected | rejected |

**Estimator choice, stated before the answer was seen.** These episodes are not a dose-response
experiment — the plant cut its setpoint by roughly the same amount every time, so treatment
intensity is nearly constant (CV 0.33 / 0.36). In that regime least squares through the origin is
identified almost entirely by the small residual spread in the treatment, which here is mostly
noise, while the ratio of total response to total excitation is identified by the level. So the
ratio of sums is the headline. Both are reported, and on this data they agree within 12%, so the
choice does not move the conclusion.

That also explains a centred R² that comes out negative on AHU-1: with almost no spread in the
treatment there is nothing for a slope to explain *across* episodes. The R² is a diagnostic of the
design, not of the effect.

### 4.3 The specification sweep — the honest part

Three judgement calls define an episode, and none is forced by the data. All 36 combinations are run:

| | Estimate spans | Below 0.50 | Excludes 0.50 at 95% |
|---|---|---|---|
| AHU-1 | 0.208 → 0.785 | 28 / 36 | **15 / 36** |
| AHU-2 | 0.126 → 0.337 | 24 / 24 | 23 / 24 |

**This is the finding.** On the clean unit the answer moves more than the effect does, so the data
**bounds** the exponent rather than settling it. Reporting only the specification that rejects would
have been indefensible, and the sweep is what makes that visible.

### 4.4 What it costs

A blunt 20% cut to the static-pressure setpoint — the size the plant itself demonstrated — priced
twice, over the identical evaluation step set:

| | measured exponent | assumed 0.5 | Inflation |
|---|---|---|---|
| AHU-1 | 190 kWh (14.2%) | 240 kWh (17.9%) | **1.26×** |
| AHU-2 | 175 kWh (12.0%) | 346 kWh (23.8%) | **1.98×** |

**Conclusion: the affinity assumption inflates the fan-side prize by roughly 1.3–2×** on this floor.
Not the 2× a single loose specification first suggested, and not nothing.

---

## 5. M6 — the model-free cross-check

Within airflow bins, the p10 of fan power per unit airflow is a level this plant actually reached
with the same fan and the same duct. No fan law, no exponent, no COP.

| | Actual | Demonstrated-achievable | Gap |
|---|---:|---:|---:|
| AHU-1 | 1,339 kWh | 1,287 kWh | **52 kWh (3.9%)** |
| AHU-2 | 1,452 kWh | 1,389 kWh | **64 kWh (4.4%)** |

The modelled pressure-lever saving (190 / 175 kWh) is **larger** than the frontier gap. The stated-in-
advance agreement factor of 2× is exceeded, and that is informative rather than a failure:

> The frontier is taken **within airflow bins**, so it measures only the efficiency gap at *equal
> delivered air*. A pressure cut on this floor also delivers **less** air — at least one box sits at
> 100% damper at every occupied step (§1.6 of the source memo), so those zones cannot hold their
> flow setpoint when duct pressure falls.
>
> **So 73% (AHU-1) and 64% (AHU-2) of the modelled saving is bought by moving less air, not by
> moving the same air more efficiently.** That portion has to be defended on comfort, not on energy.

This is the most consequential number in the document. It says the pressure-reset prize is mostly a
ventilation reduction wearing an efficiency costume, and the brief is explicit that savings bought
with discomfort do not count.

---

## 6. Methods rejected, with the reason

Rejecting a plausible method with a number is worth more than silently ignoring it.

| Method | Why not |
|---|---|
| **Harris / minimum-variance index** | The textbook loop-performance answer. These loops have 1–5 minute time constants and the data is 15-minute **averages** — below Nyquist for the dynamics the index is built on, and averaging shrinks apparent variance, so any index would be optimistic by an unknown factor. Rejected on the sampling rate, not on principle. |
| **IPMVP Option C baseline regression** | The industry M&V standard, and needed for the Task C M&V plan — but there is no intervention in this record to difference against. It measures change, not gap. |
| **Cross-AHU paired comparison** | The "identical twins" assumption is already falsified: AHU-2 serves 5.8 zones at ≥90% damper after rogue exclusion and falls below 0.8× SP on 12% of steps. Report the asymmetry as a finding; do not use it as a control group. |

---

## 7. What this does and does not establish

**Established.**

1. The setpoints move, so the SP → Hz link is measurable on this floor, not merely assumable.
2. The measured exponent is below 0.5 under 52 of 60 specifications across both units.
3. The affinity assumption inflates the fan-side prize by ~1.3–2×.
4. Most of that prize comes from moving less air, not from moving air more efficiently.
5. The SAT loops are mis-tuned rather than starved, with the contingency table to prove it.

**Not established, and stated as such.**

1. **The affinity exponent is not pinned down.** AHU-1's interval includes 0.5; only 15 of 36
   specifications exclude it. AHU-2 excludes it robustly but its episodes are start-up transients.
2. **The episodes were not randomised.** If the setpoint was lowered *because* load was low, part of
   the observed power drop belongs to the load. Conditioning on outdoor drybulb is a weak control on
   15 episodes, and the conditioned interval is wide by construction. The confound biases the
   estimate **upward**, so it is conservative with respect to conclusion 3 — but it is not removed.
3. **Only exogenous controls are admissible.** Airflow, damper position and cooling rate are
   *mediators* of a pressure change; conditioning on them would block the causal path being measured.
   This is why the control list in `config.yml` is deliberately one variable long.
4. **The frontier is optimistic.** A bin's best steps may be best for a reason the binning missed.

---

## 8. Reproducing it

```bash
python scripts/run_control_gap.py    # every number above; writes 11 tables to report/
python -m pytest tests/ -q           # 58 tests
```

Code: [src/masks.py](src/masks.py) · [src/excitation.py](src/excitation.py) ·
[src/loop_performance.py](src/loop_performance.py) · [src/effects.py](src/effects.py) ·
[src/benchmarks.py](src/benchmarks.py). All tuning parameters are in
[config.yml](config.yml) under `control_gap:`.

The test that matters most is
`tests/test_excitation.py::test_recovers_a_known_elasticity`: synthetic data with an exponent put in
by hand, which the estimator has to recover to 1e-9 for β ∈ {0.0, 0.183, 0.5, 1.5}. Without it, a
plausible-looking number from real data is only plausible.

---

## 9. What this changes for Task B

1. **The SP-reset prize is smaller than the memos assume, and mostly ventilation.** §2.3 Option 1 is
   rated ★★★★☆ there partly on "every link is measured". Every link *is* now measured — and the
   measurement cuts the prize by 1.3–2× and shows most of the remainder is airflow reduction. Option 1
   is weaker than it looked.
2. **That makes the SAT lever relatively more attractive**, and the AHU-1 overcooling finding
   (−2.60 K, 50.3% of steps) is untouched by any of this.
3. **The real differentiator is no longer which lever, but that the effect model was measured rather
   than assumed — and that the measurement was honest enough to return "not settled" on one unit.**
   That is a stronger submission than a larger number would have been.
