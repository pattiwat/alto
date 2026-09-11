# AltoTech Take-Home — Delivery Options & Recommendations

> **Purpose of this document.** You asked for the ways of delivering Tasks A/B/C, with pros and cons,
> and how close each sits to the gold standard. Everything below is grounded in the actual CSV — I
> profiled all 9,305 rows × 665 columns before writing this. Every number here is reproducible.
>
> **Read Part 1 first** (what the data actually says). The option menus in Part 2 only make sense
> once you know which findings are real, because several textbook-correct strategies are *ruled out
> by this floor's data* and picking one would sink the submission.
>
> Nothing has been built yet. Decide from Part 3, and I'll implement.

---

## Context

**The assignment.** Three months of 15-minute BMS telemetry from one office floor (Alto Building,
Floor 8, Bangkok). Two AHUs with VSD fans, 53 VAV boxes, 14 FCUs, 2 exhaust fans, 5 power meters,
13 wireless IAQ sensors, a weather station. Nothing has been cleaned. Deliver:

- **Task A** — load/validate, describe the operating pattern, rank problems, shortlist opportunities.
- **Task B** — pick a supervisory control strategy, implement it as `f(data ≤ t) → command at t`, replay it
  over the history, convert the control-input difference into energy with a *stated* effect model.
- **Task C** — quantify, qualify, recommend deploy/don't-deploy.

**Deliverables:** private GitHub repo (script that runs end-to-end beats a notebook; do not commit the
dataset) + a ≤10-page PDF for an engineering manager who has not seen the data. Every number must
trace to a place in the repo. 8–12 hours of focused work; 7 calendar days.

**What is actually being graded** (from §1 of the brief): data handling, engineering judgement, choice
*and justification* of strategy, rigour and honesty of the backtest, code quality, communication.
Explicitly: *"A modest, well-defended saving with clear assumptions scores higher than a large one that
cannot be traced back to the data."* This single sentence should drive every choice below.

---

# Part 1 — What the data actually says

This section exists because four of the five strategies on the brief's menu are either **ruled out** or
**booby-trapped** by this specific dataset. You cannot choose well without it.

## 1.1 Shape and coverage

| Property | Value |
|---|---|
| Rows × columns | 9,305 × 665 (663 data points) |
| Period | 2026-05-18 12:00 → 2026-08-23 10:00, Asia/Bangkok (UTC+7) |
| Grid | Complete 15-min grid, no duplicate or missing timestamps |
| Devices | 93 |
| Coverage loss | 655 steps (7.0%) where >50% of columns are null |

**The 7% is not scattered noise — it is four contiguous collector outages:**

| Start | End | Duration |
|---|---|---|
| 2026-07-30 11:30 | 2026-08-03 13:00 | 4 d 01:30 |
| 2026-06-14 00:00 | 2026-06-16 10:00 | 2 d 10:00 |
| 2026-05-20 01:15 | 2026-05-20 09:15 | 8 h |
| 2026-06-08 19:15 | 2026-06-08 19:30 | 15 min |

**Why this matters:** a global `ffill()` would invent four days of plant operation in the busiest month.
Any strategy comparison must apply *the same* mask to baseline and counterfactual, or the gap treatment
alone manufactures savings.

## 1.2 Units are undocumented — and mixed within the same device

The brief says *"Units: Not documented. Infer them from the values and the physics."* This is a graded
trap. The dataset mixes SI and imperial **on the same AHU**:

| Signal | Inferred unit | Evidence |
|---|---|---|
| Room / supply air temperature | **°C** | rooms 20–29, SAT 9–28 |
| Outdoor drybulb | **°F** | p1–p99 = 77.0 → 96.4 (= 25.0 → 35.8 °C) — correct for Bangkok May–Aug |
| Chilled water supply / return / ΔT | **°F** | **coil-on** SWT median 44.8 °F = **7.1 °C** — textbook CHW supply. Must be read coil-on: the all-steps median is 54.8 °F because the water drifts toward ambient when the plant is off, which makes the unit call look ambiguous. |
| Duct static pressure | **inWG** | setpoints 0.55 / 0.60 → 137 / 149 Pa — typical VAV |
| Cooling valve flow rate | **L/min** | see balance check below |
| Cooling rate | **kW (thermal)** | see balance check below |
| Fan frequency | Hz (0–50) | |
| Damper / valve position | % | |
| Power / cumulative_energy | kW / kWh | |

**Energy-balance check that confirms flow and cooling_rate.** Built in the repo as
`src/energy_balance.py`, run by `scripts/run_task_a.py`, asserted in `tests/test_energy_balance.py`.
It is structured as a *falsification*, not a fit — the constant is derived from each candidate unit
hypothesis **before** `cooling_rate` is looked at, and the observed slope then selects among them.

**Layer 1 — the predicted constant.** Sensible heat transfer to a single-phase liquid stream:

```
Q [kW]  =  ṁ [kg/s] × cp [kJ/kg·K] × ΔT [K]
        =  (flow_L_per_min / 60 × ρ) × 4.186 × (ΔT_°F × 5/9)

k = 4.186 × (5/9) / 60 = 0.0387593  kW per (L/min·°F)      ρ = 0.9997 → 1.000 kg/L
```

ΔT is a **difference**, so °F → K is the scale factor 5/9 with **no 32° offset**. Applying the
absolute conversion instead puts every coil number out by 1.8×.

**Layer 2 — the mask, stated.** `flow > 5 L/min ∧ ΔT > 1 °F ∧ cooling_rate > 0.5 kW` → 2,679 (AHU-1)
/ 2,677 (AHU-2) steps, 31% of the record. Off-coil steps make `Q/(flow·ΔT)` a 0/0 form; that, not
the plant, is what destabilises a naive ratio.

**Layer 3 — the estimator, named.** This matters more than it looks: the estimator choice moves the
answer across ~13 points, **wider than the residual it is measuring.**

| Estimator | AHU-1 | AHU-2 |
|---|---|---|
| **Slope through origin** `Σxy/Σx²` ← headline | **0.03883 (+0.2%)**, CI [0.997, 1.007] | **0.03992 (+3.0%)**, CI [1.026, 1.033] |
| Ratio of sums (energy-weighted; the one that maps onto kWh) | 0.03935 (+1.5%) | 0.04019 (+3.7%) |
| Median pointwise ratio — *biased high by the low-flow tail* | 0.04161 (+7.3%) | 0.04160 (+7.3%) |
| Free-intercept OLS (diagnostic only) | slope 0.03628, intercept **+1.06 kW** | slope 0.03874, intercept **+0.52 kW** |

R² = 0.936 / 0.967. Through the origin because the physics says zero flow → zero heat: an intercept
is not a free parameter, it is a defect indicator — and the ~+1 kW it finds is the honest reading of
the residual (flow-meter zero offset or low-flow cut-in), not a unit error.

**Layer 4 — discrimination. This is the layer that gives the test its power** (pooled k̂ = 0.03939):

| Hypothesis | predicted k | observed / predicted | out by |
|---|---:|---:|---:|
| **flow L/min, ΔT °F → kW** | 0.03876 | **1.02** | **1.0×** |
| flow L/min, ΔT K/°C → kW | 0.06977 | 0.57 | 1.8× |
| flow L/min, ΔT °F → tons | 0.01102 | 3.57 | 3.6× |
| flow gpm, ΔT °F → kW | 0.14672 | 0.27 | 3.7× |
| flow m³/h, ΔT °F → kW | 0.64599 | 0.06 | 16.4× |
| flow L/s, ΔT °F → kW | 2.32556 | 0.02 | 59.0× |

**The nearest rival hypothesis is 1.8× out.** That is the claim to make in the report — "one
hypothesis is consistent and every alternative is off by 1.8× to 59×" — rather than "the balance
closes to n%", which only invites the question of whether n is small enough. A ±10% tolerance band
is the weaker test: it would pass even if two unit systems were indistinguishable.

## 1.3 The meter hierarchy — the brief's explicit warning

> *"Which loads sit behind each meter is not documented… Getting this wrong quietly invalidates
> everything in Task B."*

| Meter | ∑ P·0.25 (kWh) | `cumulative_energy` Δ (kWh) |
|---|---:|---:|
| `db_b8` (main board) | 22,136 | 23,555 |
| `pp_b8_1` | 10,290 | 10,944 |
| `pp_b8_2` | 9,869 | 10,510 |
| `power_meter_1` | 2,578 | 2,736.6 |
| `power_meter_2` | 2,578 | **2,736.6 — identical** |
| `ahu_b8_1` | 2,087 | 0 (dead counter) |
| `ahu_b8_2` | 2,210 | 0 (dead counter) |

Findings:

1. **`pp_b8_1 + pp_b8_2` = 20,158 kWh = 91.1% of `db_b8`.** Residual mean +0.91 kW, negative on only
   1.3% of steps. The documented hierarchy `db_b8 ⊃ {pp_b8_1, pp_b8_2}` is **verified**.
2. **`power_meter_1` and `power_meter_2` are the same physical meter mapped twice.** Their
   `cumulative_energy` series are identical (4,883.1 → 7,619.7 on both). Correlation of instantaneous
   power = 1.000. *Summing them double-counts 2.7 MWh.* The brief lists them as "two further meters" —
   the documentation is wrong and the data proves it.
3. That meter is **AHU-1's feeder**: corr 0.996 with `ahu_b8_1__power`, ratio 1.13× when the AHU runs,
   and a 0.14 kW floor when it is off (controls standby). It is not an independent load.
4. **`ahu_b8_*__cumulative_energy` are stuck at 0** for the whole period. Use `power × 0.25` for AHUs,
   the counter for everything else — and say so.
5. Integrated power under-reads the counters by ~6%, which is exactly the outage fraction. **The counters
   are the honest period total; integrated power is the honest on-hours total.** State which you use.

**The number that anchors Task C:** fan energy = 4,297 kWh = **19.4% of floor electricity**. Every fan
saving must be reported against this, not against the floor total, or it will look dishonest.

## 1.4 Operating pattern

- **Weekdays ~07:00–18:00.** On-fraction 0.87–0.90 from 08:00–17:00; 0.70 at 07:00, 0.05 at 06:00.
- Both AHUs on together 29.3% of steps; AHU-2 alone 2.0%; neither 68.5%. **No staging logic** — they
  run as a pair.
- AHU-2 carries a weekend/overnight baseline (~7% of weekend hours); AHU-1 essentially none.
- **Out-of-hours running is only 3.8% (AHU-1) / 3.5% (AHU-2) of fan kWh.** → *Scheduling / optimal
  start-stop is not the prize on this floor.* This is a valuable negative result; state it, with the
  number, and move on.
- PIR occupancy (4 sensors): 0.32 at 07:00, ~0.28 plateau, 0.20 at 17:00, 0.08 at 18:00, 0.02 overnight.
  The schedule broadly matches occupancy. The 18:00 hour is marginal.
- Floor load `db_b8` (weekday median): 1.1 kW overnight → **12.2 kW at 05:00, two hours before any AHU
  starts** → 25–26 kW plateau → dips to 15.4 kW at 12:00 (lunch). The pre-07:00 ramp is non-AHU
  (FCUs/lighting) and worth one sentence in Task A.

## 1.5 Control loops — what is working and what is not

On-hours only (fan > 5 Hz):

| | AHU-1 | AHU-2 |
|---|---|---|
| Fan frequency, median (p5–p95) | 35.7 Hz (33.3–40.7) | 35.5 Hz (31.4–44.5) |
| Fan power, mean | 3.05 kW | 3.03 kW |
| Static pressure setpoint | 0.55 inWG **modal, not fixed** — 122 distinct values, range 0.300–0.636 | 0.60 inWG **modal, not fixed** — 146 distinct values, range 0.400–0.780 |
| SP tracking error, median | 0.000 — **loop is healthy** | +0.001, but **12% of steps below 0.8×SP** |
| SAT setpoint | 18.3 °C | 16.8 °C |
| **SAT error (actual − setpoint), mean** | **−1.13 K (overcooling)** | −0.37 K |
| SAT error p5 / p95 | −4.92 / +0.43 | −4.70 / +6.15 |
| \|SAT error\| > 1 K | 43% of steps | 39% of steps |
| Cooling valve, mean | 43.5% (saturated >95% only 2.4%) | 43.7% (saturated 3.9%) |
| Valve movement, mean \|Δ\| per 15 min | 6.1 % (p95 27.9) | 6.9 % (p95 30.8) |
| Water ΔT | 20.4 °F = **11.3 K** | 18.4 °F = **10.2 K** |

**Three engineering conclusions:**

1. **The SAT loop is mis-tuned, not starved.** It overcools by 1.1 K on average while the valve sits at
   43% — there is plenty of authority, the loop just is not holding setpoint. Any SAT reset must be
   presented as *fix the loop, then reset it*, not as reset alone.
2. **Valve hunting.** 6–7% valve travel per 15-minute *average* (p95 ≈ 28–31%) means the underlying
   1-minute signal is oscillating hard. Rankable problem, with a figure.
3. **Water ΔT is 10–11 K, not low.** Design is typically 5.5 K. **"Chilled water delta-T management"
   from the brief's menu targets over-pumping / low ΔT — this floor has the opposite condition.**
   Choosing that strategy would be choosing a solution to a problem this floor does not have. Rule it
   out explicitly, with the number, and you gain credit rather than lose it.

## 1.6 The VAV side — and the trap that kills a naive trim-and-respond

53 boxes: `vav_8_1_{1..22,25..28}` (26) + `vav_8_2_{1..23,25..28}` (27). Missing numbers are naming gaps,
not missing data.

| | AHU-1 | AHU-2 |
|---|---|---|
| Damper on-hours, mean / median | 52.6% / 43.5% | 59.7% / 47.2% |
| Box-timesteps at ≥90% damper | 11.1% | 21.5% |
| **p10 of per-step max damper** | **100.0** | **100.0** |

> **At literally every on-hours timestep, at least one box is wide open on both AHUs.**
> A textbook trim-and-respond that trims only when zero zones request will **never trim.** Simulation
> confirms it: with no rogue exclusion, mean setpoint reduction = 0.0%.

**The cause is identifiable, per box:**

| Box | Symptom | Reading |
|---|---|---|
| `vav_8_1_28` | damper has **exactly 1 unique value** (pinned 100%) | Room T median **28.9 °C**, max 30.8 — hottest zone on the floor. Genuinely starved, or a failed actuator. |
| `vav_8_2_28` | damper pinned 100%, **flow max 50.5** against a 1,000 max-flow setpoint | Box is disconnected / failed. Commands 100%, delivers nothing. |
| `vav_8_2_15` | mean damper 96.7%, 715 unique values | Near-rogue |
| `vav_8_2_23` | mean damper 92.5%, 908 unique values | Near-rogue |
| `vav_8_1_26` | mean damper 92.2%, 364 unique values | Near-rogue |
| `vav_8_1_16` | mean damper 81.3% | Near-rogue |

**The remedy is standard and it is exactly what ASHRAE Guideline 36 prescribes:** ignore the top *I*
requests. Results of simulating that:

| Ignore top *I* | AHU-1: steps with 0 requests | AHU-2: steps with 0 requests |
|---|---|---|
| 0 | 0.0% | 0.0% |
| 1 | 2.4% | 6.8% |
| **2** | **68.0%** | **11.2%** |
| 3 | 87.7% | 57.3% |

**Trim-and-respond simulation** (ignore top 2, trim −0.02, respond +0.03/request):

| | Mean SP achieved | vs current |
|---|---|---|
| AHU-1, floor 0.25 inWG | 0.376 | **−31.6%** |
| AHU-1, floor 0.35 inWG | 0.432 | **−21.4%** |
| AHU-2, floor 0.25 inWG | 0.589 | **−1.8%** |

> **This asymmetry is the single best finding in the dataset.** AHU-1 has large, real static-pressure
> headroom. AHU-2 has almost none — it genuinely serves 5.8 zones at ≥90% damper even after excluding
> rogues, i.e. its duct is constrained or its boxes are undersized. A submission that reports one blended
> number hides this; a submission that reports it per-AHU and explains *why* demonstrates exactly the
> judgement the brief says it is grading.

**Zone setpoints — twelve zones are effectively switched off:**

- Parked at **27.0 °C**: `vav_8_1_{1,2,3,4,5,10,11,14}`, `vav_8_2_{19,20,21,22}` — 12 zones that will
  essentially never call for cooling.
- At 24.4 °C: `vav_8_1_{6,8}`, `vav_8_2_{6,8}`. At 24.0: `vav_8_2_26`. At 23.5: `vav_8_1_{15,17,18}`.
- The rest at 23.0 °C — these are the only zones actually driving demand.

**AHU-1's zones are systematically overcooled:** room T − setpoint averages **−2.60 K**, and is more than
1 K *below* setpoint on **50.3%** of on-hours steps (p5 of zone temperature = 15.4 °C). AHU-2: −0.62 K.
That is comfort being *over*-bought, and it is the strongest argument for the SAT-reset lever.

## 1.7 Points that are present but must not be trusted

> *"Do not assume a value is trustworthy because it is present."* — the brief. Here is the list.

| Point | What it does | Verdict |
|---|---|---|
| **`0` in `vav_8_1_*__room_temperature` and `..._setpoint_read`** | min = 0.00 on every AHU-1 box; min non-zero = 6.49 / 7.20 | **`0` is a bad-read sentinel, not a value.** Mask it, don't average it. AHU-2 boxes are mostly clean (min 20.6). |
| `modbus_gateway_*` health block | connection 1, devices 4/4, error_count 0, in_backoff 0, response ~239 ms — **frozen straight through the 4-day July outage** | Health telemetry did not detect the outage. Useless. |
| All 13 `floor_8_zone_1_iaq_*__online_status` | constant **0.000**, including while the sensor is clearly reporting | Inverted or dead. |
| `co2_duct_sensor_1/2__co2_concentration` | pinned **1,800–2,900 ppm** while AHU return CO2 reads 550–680 and zone medians are 526–644 | Mis-scaled / failed. **Do not use for demand-controlled ventilation.** |
| `ahu_b8_2__return_air_humidity` | mean 90.6 %RH, **max 111.6 %RH** | Physically impossible. Use IAQ-sensor RH instead — this matters for PMV. |
| `floor_8_zone_1_iaq_{1,2,3}__co2` | spikes to 11,342 / 13,582 / 15,052 ppm | Spike artefacts. Medians (526–644) are fine — clip, don't discard. |
| `ahu_b8_*__cumulative_energy` | constant 0 | Dead counter. |
| `ahu_b8_*__water_delta_temperature` | identically `return_water_temperature − supply_water_temperature` (mean residual −0.0009 / +0.0001 °F, max \|resid\| 1.13 / 0.86) | **Derived, not measured.** Trustworthy, but it carries no independent information — it must not be cited as separate corroboration of the temperature unit inference in §1.2. Asserted by `qc.check_delta_t_is_derived`. |
| `ahu_b8_1__water_delta_temperature_setpoint` | constant 0.00 (AHU-2: constant 1.00) | Unconfigured. |
| `fcu_b8_1__status_read` | constant 0, while `fcu_b8_1__UNMAPPED_runtime` has mean 0.69 | Real signal is in the UNMAPPED column; the mapped point is dead. |
| `ex_b8_{1,2}__status_write`, `ahu_b8_1__status_write` | constant 0 while the matching `status_read` is 0.116 / 0.355 / 0.31 | Write points are not commanding. **Relevant to Task C: your controller cannot use these.** |
| IAQ radio health | RSSI −96 to −108 dBm on 10 of 13; **battery 31–35% on sensors 1–4** | The only four sensors with CO2/PM/PIR are the ones about to die. Any IAQ-dependent strategy inherits this risk. |

## 1.8 Operator interventions

> *"Awareness that a building dataset reflects how the plant was actually run, including operator
> interventions, not just how it was designed to run."* — the brief's "what good looks like".

| Point | Fraction of steps ≠ normal |
|---|---|
| `ahu_b8_1__override_control` ≠ 1 | **7.85%** |
| `ahu_b8_2__override_control` ≠ 1 | **8.36%** |
| `ahu_b8_1__auto_manual_control_mode_read` < 1 | **7.64%** |

The values are fractional (1.067, 1.2, 1.867, 2.0 / 0.133, 0.333, 0.4, 0.6) precisely because the brief
says on/off points are given as *fraction of the slot the point was on* — the override was engaged for
part of a 15-minute bucket. **These are humans driving the plant by hand.** They must be excluded from
the baseline, and your controller must hand back control during them. This is a direct hit on a stated
grading criterion.

## 1.9 The effect model — where an honest submission separates from a naive one

This is the most important technical decision in Task B, because it converts a control-input difference
into the headline kWh number.

**The textbook cubic fan law does not fit this data.**

| Model | AHU-1 | AHU-2 |
|---|---|---|
| Naive power law `P ∝ f^n` | n = **1.42**, R² = 0.655 | n = **1.73**, R² = 0.793 |
| **Offset cubic `P = a + b·(f/50)³`** | **P = 0.719 + 6.199·(f/50)³, R² = 0.903** | **P = 0.537 + 6.806·(f/50)³, R² = 0.960** |

Why the naive exponent comes out at ~1.5: motor and VSD losses are roughly fixed, and the operating band
is narrow (p5–p95 = 33.3–40.7 Hz on AHU-1), so a pure power law is poorly identified over it. The offset
form is both better-fitting and physically motivated — a fixed loss plus a cube-law aerodynamic term.

**Reporting the ideal cubic alone would overstate savings by roughly 2×.** Reporting all three as a band
is the honest move and costs nothing:

```
pessimistic   empirical power law, n = 1.42 / 1.73
central       offset cubic, fitted per AHU   ← headline
optimistic    ideal cubic
```

**The one link that is modelled rather than measured:** static pressure → frequency. At a fixed system
curve, fan pressure ∝ Hz², so `Hz_new = Hz_old · √(SP_new / SP_old)`. This assumption must be stated in
the report and sensitivity-tested, because it sits directly under the headline number.

> **CORRECTED — see [control-gap-method.md](control-gap-method.md) §4.** This link *is* measurable
> from this dataset. The setpoints are not fixed (see §1.5 above), so the plant's own setpoint
> episodes identify the exponent: **0.382, 95% CI [0.241, 0.529]** on AHU-1 and 0.232 [0.178, 0.283]
> on AHU-2, against the 0.500 assumed here. The point estimate is below 0.5 under 52 of 60
> specifications, but only AHU-2's interval excludes it — and AHU-2's episodes are start-up
> transients. Net effect: the affinity assumption **inflates the fan-side prize by ~1.3–2×**, and
> the sensitivity axis below becomes a measured interval rather than an assumed range.

**Coil side:** total `cooling_rate` = 18,732 kWh-thermal (AHU-1 9,131 + AHU-2 9,601) against 4,297 kWh
fan-electric — the coil is **4.4× the fan thermally**. But converting thermal to electric needs a chiller
COP that **is not in this dataset**. That constraint drives the strategy choice below.

---

# Part 2 — Delivery options

Legend: ★ = closest to gold standard · ✓ = solid · ~ = acceptable · ✗ = avoid

## 2.1 Task A — code architecture

| Option | Pros | Cons | Gold |
|---|---|---|---|
| **A1. Notebook only** | Fastest to write; narrative and figures inline | Brief says *"a script that runs end to end is better"*; hidden execution-order state; numbers not traceable to a function; painful to re-run after a fix | ~ |
| **A2. `src/` modules + thin `run_task_a.py`, notebook optional as a display layer** | Every number traces to a named function; re-runnable from a clean checkout; unit-testable; directly satisfies the stated preference | ~1–2 h more setup | ★ |
| **A3. A2 + full data-contract layer (`pandera` / Great Expectations, `pint` unit registry)** | Highest formal rigour; unit bugs become impossible | Gold-plating for a 8–12 h budget; a reviewer may read it as effort spent in the wrong place | ✓ |

**Recommendation: A2, plus a lightweight hand-rolled unit registry** — a `units.yml` mapping column
pattern → unit + physically plausible range, checked at load. That captures ~80% of A3's value (it is
what catches the °F/°C mix in §1.2 and the 111.6 %RH in §1.7) for ~10% of the cost, and it makes the
data-quality section of the report fall out of the code for free.

## 2.2 Missing-data treatment

The brief specifically says: *"Decide, and state, how you treat missing data."*

| Option | Pros | Cons | Gold |
|---|---|---|---|
| Drop rows with any NaN | Trivial | Discards far more than 7%; throws away rows that are fine for the columns you need | ✗ |
| **Global forward-fill** | Keeps the grid full | **Invents four days of plant operation** in the busiest month; poisons every aggregate | ✗ never |
| Model-based imputation | Sophisticated-looking | Unjustifiable in a control backtest — you would be scoring a controller against fabricated inputs | ✗ |
| **Tiered: hard-gap mask + short-gap ffill (≤2 steps / 30 min) + a per-point `valid` flag** | Honest; preserves partial rows; the mask is explicit and auditable | More code | ★ |

**The rule that must be stated in the report:** *the baseline and the counterfactual see exactly the same
mask.* If the controller is evaluated on more steps than the baseline, the gap treatment alone
manufactures savings. This one sentence is worth more than a page of imputation theory.

## 2.3 Task B — which strategy

This is the decision you asked to be educated on. All five rows of the brief's menu are assessed against
what §1 actually found.

---

### Option 1 — Static pressure reset only (trim-and-respond, G36)

**What it does.** Replace the fixed 0.55 / 0.60 inWG duct static pressure setpoint with one derived from
VAV damper demand, ignoring the top *I* requests to defeat rogue zones.

| Pros | Cons |
|---|---|
| **Every link in the causal chain is measured in this dataset:** damper % → SP setpoint → Hz → kW. Nothing is assumed except the SP→Hz affinity step. | The prize is bounded by fan energy = 19.4% of floor load. |
| **No COP assumption anywhere.** The headline number is electrical and directly observed. | AHU-2 delivers ~nothing (−1.8%). Roughly half the equipment shows no benefit. |
| Largest single defensible saving found: AHU-1 setpoint −21% to −32%. | It is the most "expected" answer — the brief lists it first. |
| The rogue-zone discovery (§1.6) makes the implementation genuinely non-trivial and demonstrates real judgement. | |
| Comfortably fits the 8–12 h budget with time left for a proper Task C. | |
| Smallest attack surface in the follow-up interview — very hard to poke a hole in. | |

**Gold-standard rating: ★★★★☆** — the *safest* excellent submission. Loses a little ceiling because it
leaves the largest thermal prize untouched and does not engage the overcooling finding.

---

### Option 2 — Supply air temperature reset only

**What it does.** Raise the SAT setpoint when zone cooling demand is low, subject to humidity and comfort
limits.

| Pros | Cons |
|---|---|
| Targets the bigger prize: coil load is **4.4×** fan energy thermally. | **The kWh number needs a chiller COP that is not in this dataset.** The headline rests on an assumption, which is exactly what the brief warns against. |
| Directly attacks a real, quantified fault: AHU-1 zones sit **2.60 K below setpoint, 50% of the time**. | The SAT loop is currently **mis-tuned** (−1.13 K tracking error, §1.5). You are resetting a setpoint the plant is not holding — you must fix the loop first, and say so. |
| The thermal effect *is* directly measurable in `cooling_rate` (validated kW, §1.2), so you can report kWh-thermal with no assumption at all and convert separately. | Raising SAT raises airflow to meet the same zone load → **fan energy goes up**. You must model the fan penalty or the saving is overstated. |
| Humidity constraint is real and interesting in Bangkok (outdoor wetbulb median 77.3 °F). | AHU-2's RH sensor reads >100% and is unusable, so the humidity guard is asymmetric across units. |

**Gold-standard rating: ★★★☆☆ alone.** Strong physics, weaker number. The COP dependency is a genuine
vulnerability in the follow-up session.

---

### Option 3 — Static pressure reset **+** SAT reset, per AHU  ← **my recommendation**

**What it does.** Both levers, layered, with an explicit hierarchy and an explicit interaction term.

| Pros | Cons |
|---|---|
| **Primary lever (SP reset) carries the headline number with zero COP assumption.** Secondary lever (SAT reset) is reported in kWh-thermal — measured — and converted to electricity only in a clearly-labelled, sensitivity-tested band. The number never rests on an assumption you cannot defend. | Most work: ~10–12 h, the top of the stated budget. |
| **Handles the AHU-1/AHU-2 asymmetry honestly.** AHU-1 gets its saving from SP reset; AHU-2, which has no SP headroom, gets its saving from SAT reset instead. *The strategy adapts to what each unit's data supports* — this is the strongest possible answer to *"explain why it fits this floor best."* | Two levers interact (raising SAT raises airflow, which raises the SP requirement) — the interaction must be modelled, not ignored. This is real work but it is also the most impressive part. |
| Uses two **independent** effect models, so a reviewer who rejects one still has the other. | More surface area for a reviewer to question. |
| Engages *both* headline problems from Task A (rogue zones, systematic overcooling) rather than one. | |
| The interaction term is itself a finding: naively stacking both levers double-counts, and showing you caught that is a differentiator. | |

**Gold-standard rating: ★★★★★** — highest ceiling, and it is the option where Tasks A, B and C form one
argument rather than three sections. Requires the full budget.

---

### Option 4 — Zone setpoint & airflow optimisation / scheduling

| Pros | Cons |
|---|---|
| 12 zones parked at 27 °C is a real, findable anomaly worth reporting in Task A. | **Out-of-hours running is only 3.5–3.8% of fan energy** (§1.4) — the scheduling prize is small and the data says so. |
| PIR occupancy data exists. | PIR comes from **4 sensors at 31–35% battery** (§1.7). Building a controller on the floor's least reliable instruments is poor judgement and invites exactly that question. |
| | Changing zone setpoints trades comfort directly for energy — the brief says *"savings bought with discomfort do not count."* |

**Gold-standard rating: ★★☆☆☆.** Report the 27 °C zones and the schedule/occupancy match as Task A
findings with their numbers. **Do not** build Task B on this.

---

### Option 5 — Chilled water delta-T management

| Pros | Cons |
|---|---|
| Listed on the brief's menu. | **The data rules it out.** Water ΔT is 10.2–11.3 K against a typical 5.5 K design (§1.5). This floor has *high* ΔT, i.e. the opposite of the over-pumping condition this strategy addresses. |
| | Both ΔT setpoint points are dead (constant 0.00 / 1.00, §1.7) — the BMS has never configured this loop. |

**Gold-standard rating: ★☆☆☆☆ as a strategy — but ★★★★★ as a *rejection*.** Spending one paragraph in
Task B saying *"the menu offers this; here is the number that rules it out on this floor"* demonstrates
precisely the judgement being graded. Rejecting a plausible option with evidence scores better than
silently ignoring it.

---

### Strategy summary

| | Prize size | Number defensibility | Effort | Engages Task A findings | Gold |
|---|---|---|---|---|---|
| 1. SP reset only | Medium | **Highest** (no COP) | Medium | Rogue zones | ★★★★☆ |
| 2. SAT reset only | Large | Medium (COP needed) | Medium | Overcooling | ★★★☆☆ |
| **3. SP + SAT, per AHU** | **Large** | **High** (layered) | **High** | **Both** | **★★★★★** |
| 4. Setpoint / schedule | Small | Medium | Low | — | ★★☆☆☆ |
| 5. Delta-T | — | Ruled out by data | — | — | ★ (as a rejection) |

**If you have the full 10–12 hours: Option 3.**
**If time is tight or you want the lowest-risk excellent submission: Option 1**, with the overcooling
finding and the delta-T rejection reported in Task A as "further opportunities not pursued".

## 2.4 Task B — backtest architecture

| Option | Pros | Cons | Gold |
|---|---|---|---|
| **B1. Vectorised recompute** (compute the whole setpoint series at once) | Fast, few lines | **Wrong.** Trim-and-respond is stateful — SP(t) depends on SP(t−1). Vectorising it either changes the algorithm or silently leaks future information | ✗ |
| **B2. Stepwise causal replay, explicit `Controller` class: `step(t, obs≤t) → command(t)`** | Exactly the brief's wording; look-ahead leakage becomes *structurally impossible*; the controller is unit-testable in isolation against synthetic inputs; state (integrator, hold timers, fault latches) has an obvious home | Slower — irrelevant at 9,305 steps | ★ |
| **B3. Closed-loop simulation with an identified duct/zone model** | Closest to physical truth | The brief **explicitly** says a replay is what it wants and warns the building never responded. Identifying a zone model from 3 months is a separate project, and a bad model is *worse* than a transparent one | ✗ (over-reach, and against instructions) |

**Recommendation: B2**, with the controller's tuning parameters in a `config.yml` (the brief: *"make
tuning parameters explicit and configurable"*) and an explicit, tested `on_bad_data()` path (the brief:
*"make it obvious what it does when data is missing or a sensor reads nonsense"*) — that path is not
decorative here, it fires on the 7% outage, the `0`-sentinel zone temps, and the 8% operator overrides.

## 2.5 Task B — the comfort constraint

The brief: *"A zone temperature band is the minimum. Thermal comfort is more than temperature, so a
comfort index such as PMV… is the stronger test. Savings bought with discomfort do not count."*

| Option | Pros | Cons | Gold |
|---|---|---|---|
| Zone temperature band only (e.g. 22–25 °C) | Simple, defensible, fast | Explicitly described as *the minimum* | ~ |
| + airflow floor (`minimum_air_flow_rate_setpoint_read`) + CO2 ceiling | Catches ventilation loss, which SP reset can genuinely cause; uses points that exist and are trustworthy | Says nothing about humidity — a real issue in Bangkok | ✓ |
| **+ PMV (Fanger) from zone T, IAQ RH, and stated met/clo/air-speed** | The brief names it as the stronger test; RH is available from 13 IAQ sensors; makes the humidity risk of SAT reset visible | Needs stated assumptions (met 1.1, clo 0.5, v 0.15 m/s) | ★ |

**Recommendation: all three, layered as hard constraints in the controller.** The PMV layer is ~30 lines.

> **Trap to avoid:** compute PMV from `floor_8_zone_1_iaq_*__humidity`, **never** from
> `ahu_b8_2__return_air_humidity` — that sensor reads up to 111.6 %RH (§1.7). Using it would produce
> nonsense PMV and a reviewer would spot it immediately.

## 2.6 Task B — benchmarks

The brief: *"Benchmark first against what the building actually commanded. Beyond that, be creative: a
simpler alternative such as a fixed setpoint change shows what your strategy adds."*

Once the replay harness from B2 exists, each extra benchmark is ~10 lines. Run all four:

| # | Benchmark | What it proves |
|---|---|---|
| 1 | **As-operated baseline** | Mandatory. The reference. |
| 2 | **Naive fixed SP cut** (0.55 → 0.45, constant) | Whether the *smart* controller earns its complexity, or a screwdriver would have done. |
| 3 | **T&R with no rogue-zone exclusion** | Saves ~0%. Proves the §1.6 finding was the thing that made the strategy work — the single most persuasive figure in the report. |
| 4 | **Oracle bound** (lowest SP that would still have satisfied every non-rogue zone) | Upper bound on the prize; shows how much of the theoretical maximum your controller captures. |

Benchmark 3 is the one that turns a competent submission into a memorable one.

## 2.7 Task C — how to present the number

| Option | Pros | Cons | Gold |
|---|---|---|---|
| Single point estimate ("saves X kWh / Y%") | Crisp, easy to read | The brief explicitly prefers a defended modest number over a big bare one; a point estimate hides every assumption | ~ |
| **Tornado chart: savings vs the 5 driving assumptions, with a stated range and the biggest driver named** | Directly answers *"which assumptions drive the number most, and what happens if they are wrong"*; makes honesty visible in one figure | One more figure to build | ★ |
| Monte Carlo over assumption priors | Formally rigorous | Reads as false precision on a replay; the priors themselves are indefensible | ✓ |

**Recommendation: tornado, over these five axes** (all of which are real uncertainties identified in §1):

1. Fan model: empirical power law ↔ offset cubic ↔ ideal cubic (**expected to be the largest driver**)
2. SP → Hz affinity exponent (2.0 assumed; test 1.8–2.2)
3. Rogue-zone ignore count *I* (1 / 2 / 3)
4. Minimum SP floor (0.25 / 0.30 / 0.35 inWG)
5. Chiller COP, *if* the SAT lever is included (3.0 / 4.0 / 5.0)

Then add two short sections the brief asks for by name:

- **"Where data quality limited the conclusion"** — §1.7 writes itself. The `power_meter_2` duplicate,
  the frozen gateway health, the dead write points, the dying IAQ batteries.
- **"Periods where the controller would have behaved badly, and what guarded against it"** — the 4-day
  July outage, the 8% operator-override windows, the `0`-sentinel zone temperatures, and `vav_8_2_28`
  commanding 100% while delivering no flow. Show the fault-handling path firing on each.

**And an M&V plan** (the brief: *"how you would measure it once the strategy runs on the real building"*):
which points, which baseline period, what regression against outdoor drybulb + occupancy, what control
chart triggers rollback. Name the rollback trigger explicitly — that is what "the judgement of someone
who will be responsible for a controller running on a real building" means.

## 2.8 Repository layout

```
alto-floor8/
├─ README.md              # setup, how to run, how each report number is reproduced
├─ requirements.txt       # or pyproject.toml
├─ config.yml             # ALL tuning params: I, SP floor, trim/respond rates, comfort band, COP
├─ units.yml              # column pattern -> unit + plausible range  (catches the °F/°C mix)
├─ data/                  # .gitignored — dataset must NOT be committed
├─ src/
│  ├─ io.py               # load, tz-localise, assert the 15-min grid
│  ├─ qc.py               # gap detection, sentinel masks, stuck/flatline, range checks, override flags
│  ├─ energy_balance.py   # the §1.2 coil balance: derive -> mask -> estimate -> discriminate
│  ├─ meters.py           # hierarchy verification + the power_meter_2 duplicate proof
│  ├─ features.py         # on-hours mask, damper requests, zone aggregates, occupancy
│  ├─ controller.py       # Controller.step(t, obs<=t) -> command(t)   + on_bad_data()
│  ├─ effects.py          # fan-law fits, SP->Hz, coil load, COP band
│  ├─ backtest.py         # replay harness + the 4 benchmarks
│  └─ figures.py          # every figure in the report
├─ tests/                 # controller unit tests, unit-inference tests, leakage test
├─ scripts/
│  ├─ run_task_a.py       # -> figures/ + findings.json + data_quality.csv
│  ├─ run_task_b.py       # -> backtest results + sensitivity grid
│  └─ run_all.sh          # clean checkout -> every figure and number in the report
└─ report/                # figures + the ≤10-page PDF
```

**The `run_all.sh` matters more than it looks.** *"Every number should be traceable to a place in the
repository"* is a stated requirement; one command that regenerates the entire report from the raw CSV is
the strongest possible answer to it.

**One test worth writing by name:** a **leakage test** that feeds the controller a series where every
value after index *k* is `NaN`, and asserts the command at *k* is unchanged. That proves the causality
claim mechanically rather than by assertion.

## 2.9 Report structure (≤10 pages)

| Pages | Content |
|---|---|
| 1 | Executive summary — the number, the range, the recommendation, in that order |
| 2 | Data quality: what I trust, what I do not, why (§1.2, §1.3, §1.7) — including the `power_meter_2` duplicate and the units table |
| 3–4 | Task A: operating pattern + the ranked problems, each with a figure and a number |
| 5 | The rogue-zone finding and why it changes the strategy (the §1.6 table + the ignore-*I* chart) |
| 6 | Task B: the controller, its tuning parameters, and its fault behaviour |
| 7 | The effect model and its fit — including *why not the ideal cubic* (§1.9) |
| 8 | Backtest results, all four benchmarks, per AHU |
| 9 | Sensitivity tornado + where data quality limited the conclusion |
| 10 | Recommendation, pre-deployment checklist, M&V and rollback plan |

**Figure budget: 6 figures** (the brief asks for three to six, each with a one-sentence takeaway — *"not a
wall of plots"*). Proposed:

1. Coverage / outage map across the period
2. Weekday load profile with the AHU schedule and PIR occupancy overlaid
3. Damper-demand distribution with the rogue zones called out
4. Fan power vs frequency, with all three candidate laws drawn over the scatter
5. SP setpoint: as-operated vs controller, over a representative week
6. Sensitivity tornado

---

# Part 3 — What I need from you

**Decision 1 — strategy for Task B.** My recommendation is **Option 3 (SP + SAT reset, per AHU)** if you
have the full 10–12 hours, because it is the only option where the AHU-1/AHU-2 asymmetry becomes the
*point* of the submission rather than a footnote. If you want the lower-risk path, **Option 1 (SP reset
only)** is a genuinely excellent submission and finishes faster — with the overcooling and delta-T
findings reported in Task A as opportunities identified but not pursued.

**Decision 2 — build scope.** Full submission (repo + report content), Task A first for review, or
skeleton plus one worked slice.

---

## Verification (once built)

- `bash scripts/run_all.sh` from a clean checkout with only the CSV dropped into `data/` → regenerates
  every figure and every number in the report. Nothing manual.
- `pytest tests/` — controller unit tests, the leakage test (§2.8), unit-inference tests, and a
  regression test pinning the headline savings number so a refactor cannot silently move it.
- **Meter reconciliation printed by `run_task_a.py`:** `pp_b8_1 + pp_b8_2` within ~10% of `db_b8`, and the
  `power_meter_1` ≡ `power_meter_2` identity asserted, so the duplicate can never creep back into a total.
- **Energy-balance assertion** (`pytest tests/test_energy_balance.py`, 13 tests, all passing): the
  correct unit hypothesis must be the argmin of `|log(k̂ / k_candidate)|` **and beat the runner-up by
  >1.5×** (actual: 1.74×). This is the test, not a ±10% tolerance band — a band would pass even if
  two unit systems were indistinguishable, and would not reliably catch a refactor that swapped a
  unit. Supporting tests pin `derive_k` against the hand derivation, assert the °F→K difference
  carries no 32° offset, and recover a known constant from synthetic data to 1e-9.
- **Mask symmetry assertion:** baseline and counterfactual evaluated over an identical step count — the
  single check that stops the gap treatment from manufacturing savings.
- Spot-check the controller by hand on one representative week against the plotted setpoint trace.
