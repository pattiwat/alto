# Block E, derived — what leaves the coil, and what it costs

> **What this is.** The physical and mathematical derivation behind
> [grey-box-surrogate.md](../grey-box-surrogate.md) §6 — the only block in the surrogate whose
> equations have never been checked against the CSV. Block E answers two questions: what temperature
> leaves the coil, and what does the coil cost to run?
>
> ```
>     E1 :  T_sa    =  SAT_sp + b0 + b1*valve + eta        eta ~ N(0, sigma^2)
>     E2 :  T_ma    =  phi*T_oa + (1 - phi)*T_ra
>           Q_req   =  rho*c_p * V_total * (T_ma - T_sa)   [kW]
>           phi     =  phi0 + phi1*damper%                 clipped to [0, 1]
> ```
>
> The logged points it is built from — `ahu_b8_*__supply_air_temperature`,
> `ahu_b8_*__supply_air_temperature_setpoint_read`, `ahu_b8_*__cooling_valve_position_read`,
> `ahu_b8_*__cooling_valve_flow_rate`, `ahu_b8_*__water_delta_temperature`,
> `ahu_b8_*__fresh_air_damper_position_read` — and two it is built from that **do not exist**.
>
> **Run it:** `python block-e-derivation/reproduce_sat_coil.py` — derives the water-side constant
> from config's fluid properties rather than pasting it, classifies every temperature column's unit,
> fits the setpoint gain five ways, scores every candidate bias regressor for whether the surrogate
> could actually step it, discriminates the airflow unit against the energy balance, runs §6.3's
> closure on a held-out month, asserts its guards, and writes twelve tables to [data/](data/).
>
> **Four findings up front.**
>
> 1. **The record mixes °F and °C, and §6.2's mixing equation adds them together.** Outdoor and
>    water-circuit points are °F; zone and supply-air points are °C. The falsifier needs no climate
>    judgement: read as °C, the *chilled* water entering the coil is 44.8 °C while the air leaving it
>    is 17.0 °C, so the coil would be moving heat from the cold side to the hot side. That happens on
>    **0.0%** of coil-on steps under the °F reading and **100%** under the °C reading.
> 2. **E₁ cannot be stepped as written.** Its only regressor is valve position, which is absent from
>    §1's state vector and produced by no block in the §1.1 step order. §5 scores every rival the
>    surrogate *can* step, and finds one — but it is endogenous in exactly the way the valve is.
> 3. **The mixing map's fitted slope is negative on both AHUs.** §6.2 needs `φ₁ > 0`: opening the
>    outside-air damper must admit more outside air. Fitted, it is **−0.0179** and **−0.0132**, with
>    correlations of −0.48 and −0.28. The damper is not uninformative; it is informative with the
>    wrong sign, which is worse.
> 4. **The airflow unit that blocks Block C is discriminated, and it is m³/h.** §4.6 asks for exactly
>    this test and Block C could not run it — its own ratios are unit-invariant. Requiring mixed air
>    to lie between the two streams it is mixed from, m³/h wins by **25.0×** and **16.5×** over the
>    runner-up. CFM and L/s put mixed air *colder than both sources* on 97–100% of steps.

---

## 1. What this replaces

Blocks A–D each have a derivation folder. Block E has had none, and §6 has therefore been carrying
four equations that no one had put against the record. §6.1 is anchored on a measured number (the
−1.02 K SAT tracking bias, from control-gap §3) and then generalised into a functional form that was
never fitted; §6.2's mixing map was specified — "fit an affine map `φ = φ₀ + φ₁·damper%` on the
training window" — as an instruction, not a result; §6.3's falsification was written down but never
run; and §6.4's 4.4× is the one part that survives intact.

That is not a criticism of the specification. §6 says explicitly "Nothing below is fitted yet — this
is the specification the fitting code must implement." This document is the fitting code's report,
and four of the instructions turn out not to be executable as written.

**The mask, and what it leaves** ([data/identities.csv](data/identities.csv)):

| | AHU-1 | AHU-2 |
|---|---:|---:|
| On-hours, no override, no outage | 2,688 | 2,868 |
| ...and settled | 2,094 | 2,213 |
| ...and coil-on (`energy_balance.mask`) | **2,679** | **2,676** |
| Clean boxes / dropped | 23 / 3 | 24 / 3 |

The coil-on counts land within two steps of Block B's 2,678 / 2,681 anchor from a different base,
which is orientation rather than agreement — Block E masks on the water circuit and Block B does not.
Dropped boxes are config's named rogues plus whatever fails config's measured dead-box rule:
`vav_8_1_16`, `vav_8_1_26`, `vav_8_1_28` and `vav_8_2_15`, `vav_8_2_23`, `vav_8_2_28`.

---

## 2. The physics, and the one constant that is not in dispute

Both halves of Block E are the steady-flow energy equation applied to a control volume, once across
the mixing box and once across the coil. For air treated as an ideal gas at constant pressure:

```
        Q̇  =  ṁ·c_p·ΔT  =  ρ·c_p·V̇·ΔT
```

Nothing in this document models a transient. The SAT loop's own time constant is 1–5 minutes and the
record is 15-minute averages, so §0's Nyquist argument forbids it — which is why §6.1 is right to
write a quasi-steady bias rather than a lag, whatever else is wrong with it.

**`ρ·c_p` is an assumption, and a visible one.** At 20 °C and 1 atm, dry air is ρ ≈ 1.20 kg/m³ and
c_p ≈ 1.006 kJ/(kg·K), so `ρ·c_p ≈ 1.21 kJ/(m³·K)` — the value §4.6 quotes. This is **sensible heat
only**. A coil on this floor, in this climate, at a 77 °F wetbulb, is certainly doing latent work as
well, and every `Q_air` below therefore understates the coil's true duty by whatever the latent
fraction is. §9 is where that shows up, and it is not separable from the other errors there.

**The water side is the anchor, and it is derived rather than trusted.** §1.2 of the data memo
discriminated `flow L/min, ΔT °F → kW` against every rival by a factor of 1.8× or more. That constant
follows from config's own fluid properties:

```
        Q[kW] = ρ[kg/L]·flow[L/min]/60 · c_p[kJ/(kg·K)] · ΔT[°F]·5/9
        k     = ρ·c_p·5/(60·9)  =  1.000 × 4.186 × 5 / 540  =  0.0387592593
```

against the memo's 0.0387593. Two identities are asserted on top of it:
`water_delta_temperature` is `(return − supply)` water temperature to within **1.13 / 0.86 °F**
against [config.yml](../config.yml) `delta_t_derived_tolerance_deg_f` of 2.0, and the logged
`cooling_rate` reproduces `k·flow·ΔT` at a median ratio of **1.0735 / 1.0733**, inside config's
`k_ratio_bounds` of [0.95, 1.10].

Everything else in Block E is measured against this.

---

## 3. The units problem

**Two populations, and nothing between them** ([data/unit_scales.csv](data/unit_scales.csv)). Every
temperature column in the record has a median either near 24 or near 65. There is no column in
between, so the classification is not a judgement call:

| Column | Median | p1 | p99 | If °F, in °C | Verdict |
|---|---:|---:|---:|---:|---|
| `outdoor_weather_station__drybulb_temperature` | 85.00 | 77.01 | 96.43 | **29.44** | **°F** |
| `outdoor_weather_station__wetbulb_temperature` | 77.28 | 72.46 | 81.65 | **25.16** | **°F** |
| `ahu_b8_1__supply_water_temperature` | 54.82 | 43.12 | 58.14 | **12.68** | **°F** |
| `ahu_b8_1__return_water_temperature` | 66.44 | 49.97 | 72.23 | **19.13** | **°F** |
| `ahu_b8_1__supply_air_temperature` | 23.17 | 12.70 | 25.15 | −4.91 | **°C** |
| 66 zone-level points (VAV + IAQ) | 24.47 | — | — | — | **°C** |

Read as °C, outdoor drybulb would run to 96 °C. Read as °F it is 25.0–35.8 °C, which is Bangkok in
May–August. That argument is persuasive but it rests on knowing the climate.

**The argument that does not** ([data/second_law.csv](data/second_law.csv)): a cooling coil requires
the water entering it to be colder than the air leaving it. Otherwise it is not a cooling coil.

| Over coil-on steps | AHU-1 | AHU-2 |
|---|---:|---:|
| Supply air temperature, °C | 16.97 | 15.78 |
| Entering water, read as °C | 44.78 | 44.79 |
| Entering water, read as °F → °C | **7.10** | **7.11** |
| Share of steps with T_water < T_air, °C reading | **0.0000** | **0.0000** |
| Share of steps with T_water < T_air, °F reading | **1.0000** | **1.0000** |

Not "mostly" — every step, both ways, on both AHUs. **This is asserted.**

**What it costs §6.2.** The mixing equation `T_ma = φ·T_oa + (1 − φ)·T_ra` takes `T_oa` from a °F
column and `T_ra` from °C columns. Evaluated as written at a typical step it returns
`φ·86.0 + (1 − φ)·22.3`, which for any φ above about 0.05 exceeds the highest temperature anywhere
in the air path. Every `Q_req` computed from it is wrong by a multiple, not a margin.

**This reaches further than Block E.** Surrogate §1 lists `T_oa` in `w_t` with no unit; §5.1's
`γ_i·(T_oa − T_i)` and [grey-box-technique.md](../grey-box-technique.md)'s `g_i·(T_oa − T_i)`
difference a °F quantity against a °C one, which is an error in *offset* and not merely in scale —
the term cannot change sign at the right place.
[block-d-derivation/README.md](../block-d-derivation/README.md) reports `g` fitting to zero on 40 of
50 boxes and reaches that conclusion by a route
that does not depend on the unit (its within-night slope is scale-free), so its **conclusion** stands;
the coefficient's units do not. §14 states what this forces.

---

## 4. E₁: the setpoint gain nobody measured

§6.1 writes `T_sa = SAT_sp + b₀ + b₁·valve + η`. The unit coefficient on `SAT_sp` is not an estimate
— it is an assumption, and it is the assumption that matters most, because `SAT_sp` is one of only
two actions the agent has. Block A measured the analogous quantity for the pressure channel (β, with
an interval, carried into domain randomisation). **The SAT channel — the one §6.4 says is 4.4× the
fan thermally — has no equivalent number anywhere in the repo.**

Five estimators, one record ([data/sat_gain.csv](data/sat_gain.csv)):

| Estimator | AHU-1 slope | R² | n | AHU-2 slope | R² | n |
|---|---:|---:|---:|---:|---:|---:|
| Levels | **0.999** | **0.041** | 2,688 | 1.378 | 0.082 | 2,868 |
| First differences | 2.575 | 0.172 | 2,620 | 2.524 | 0.153 | 2,803 |
| Differences, \|Δ\| ≥ 0.15 K | **3.019** | 0.476 | 159 | **2.900** | 0.443 | 174 |
| Episodes (ratio of sums) | 2.793 | — | 80 | **0.913** | — | 87 |
| Episodes, ex-startup | 2.793 | — | 80 | 0.913 | — | 87 |

**AHU-1's level slope of 0.999 reads as vindication until you notice what it explains.** R² = 0.041:
the setpoint accounts for four per cent of the variance in the thing it is supposed to command. The
same channel on the same AHU returns 3.02 in differences. AHU-2 inverts the pattern — 1.38 in levels,
0.91 from episodes, 2.90 in restricted differences. **The five estimates span 1.00–3.02 on AHU-1 and
0.91–2.90 on AHU-2, and nothing in this record adjudicates between them.**

**Why the channel is so badly identified, and this part is asserted.** The setpoint barely moves:

| | AHU-1 | AHU-2 |
|---|---:|---:|
| sd(`SAT_sp`) over on-hours | 0.393 K | 0.517 K |
| sd(`T_sa`) over the same steps | 1.942 K | 2.495 K |
| **Ratio** | **0.202** | **0.207** |

The commanded variable moves a fifth as much as what it commands. AHU-1's entire on-hours setpoint
range is 17.50–18.77 °C. A gain estimated across 1.3 K of excitation, against a response with five
times the spread, is not going to be settled by choosing a better estimator.

**A note on the episodes.** 167 survive across both AHUs, and the earliest is at **09h**. This was
not imposed: requiring an on-hours *control* window before the change removes every 07h start-up move
on its own, because the fan has only just started and the before-window is off-hours. Block A had to
report that 93% of AHU-2's β episodes sat at 08h and were contaminated by start-up transients
(§2.2); the same contamination is present here and the episode construction excludes it mechanically.
That is worth stating because it means the `ex-startup` row is identical to the `episodes` row by
construction, not by accident.

**The gain is reported, never asserted.** What is asserted is the excitation ratio above — if the
setpoint ever starts moving comparably to `T_sa`, this section becomes fittable and must be refitted
rather than merely re-run.

---

## 5. E₁: the bias term, and why nothing can carry it

§6.1's `b₁·valve` term is well motivated. The tracking error is not a constant offset — its p5/p95
span is −4.74 / +0.27 K on AHU-1 — so a single mean would misplace the bias exactly at high load,
and the linear-in-valve term is the obvious fix. Fitted, it works:

| Regressor | AHU-1 b₀, b₁ | R² | AHU-2 b₀, b₁ | R² | Simulable? | Admissible? |
|---|---:|---:|---:|---:|:---:|:---:|
| **`valve`** — the spec's own | 3.161, −0.1033 | **0.341** | 4.046, −0.1052 | **0.565** | **no** | yes |
| `T_ra` (zone mean) | −34.08, 1.4742 | **0.399** | −47.95, 2.0638 | **0.654** | yes | yes |
| `V̇_total` | −6.699, 0.0009 | 0.086 | 3.278, −0.0007 | 0.296 | yes | yes |
| `fan Hz` | −3.156, 0.0541 | 0.006 | 6.603, −0.1966 | 0.076 | yes | yes |
| `T_oa` | −4.295, 0.0940 | 0.017 | 0.901, −0.0380 | 0.002 | yes | yes |
| `T_oa − T_sa` | 3.287, −0.3035 | 0.247 | 6.663, −0.4463 | 0.474 | yes | **no** |
| `T_ra − T_sa` | 4.416, −1.0926 | **0.792** | 7.649, −1.2129 | **0.846** | yes | **no** |
| none (constant bias) | −1.241, — | 0.000 | −0.320, — | 0.000 | yes | yes |

Three things to read off it, in order of how much they cost.

**1. The spec's own regressor is not simulable.** Surrogate §1's state vector `x_t` contains zone
temperatures, flow setpoints, damper positions, `T_sa`, `SP_act`, `f`, and the two supervisory
setpoints. **It does not contain valve position, and no block in the §1.1 step order produces it.**
So `T_sa(t) = SAT_sp'(t) + b₀ + b₁·valve(t) + η` has an undefined right-hand side at simulation time.
§1.1's own sketch quietly substitutes `b(load)` for `b₁·valve`, which is not the same equation and is
not defined anywhere. **As written, E₁ cannot be stepped.**

**2. The two rows that beat everything are artifacts, and they are exactly the rows a careless fit
would pick.** The dependent variable is `T_sa − SAT_sp`. Any regressor containing `T_sa` shares a
term with it and will correlate mechanically. `T_ra − T_sa` returns R² 0.79 / 0.85 — by far the best
in the table — with a slope of **−1.09 / −1.21**. That slope near −1 *is* the shared term and nothing
else. Both are marked inadmissible in [data/sat_bias_fit.csv](data/sat_bias_fit.csv) rather than
quietly dropped, because the trap is worth leaving visible.

**3. There is a candidate replacement, and it is not a clean one.** `T_ra` — the flow-weighted zone
mean — is simulable (Block D produces zone temperatures), is admissible (no shared term), and scores
**0.399 / 0.654** against the spec's own unsimulable **0.341 / 0.565**. It beats the valve term on
both AHUs. It also has a physical reading: warmer return air is a heavier coil load, and a mis-tuned
loop falls further behind at load.

> **But it is endogenous in exactly the way the valve is, and this document will not pretend
> otherwise.** §1.1 places E₁ before Block D updates the zones, so `T_ra(t)` is causally prior
> *within* a step and is legitimately available. Across steps it is a closed loop: colder supply air
> cools the zones, which lowers `T_ra`, which the regression then reads as a cause. **This record
> cannot separate "warm zones cause SAT error" from "SAT error causes warm zones."** Block C escaped
> the analogous problem by differencing pressure out across rooms at a single instant (§4). There is
> no equivalent trick here, because there is one AHU and one `T_sa` per subsystem — nothing to
> difference against.

So the honest recommendation is the last row: **a constant bias, per AHU, carried with its residual
sd** (1.79 / 2.38 K), with `T_ra` offered as a DR arm rather than as the headline. That is weaker
than §6.1 and it is what the record supports.

**One number to reconcile.** [control-gap-method.md](../control-gap-method.md) §3 reports mean SAT
tracking error of −1.02 / −0.22 K. On
the settled mask this document uses, it is **−1.24 / −0.32 K**. The difference is the settling filter
— dropping the two steps after each setpoint change removes transients in which `T_sa` has not yet
moved, and those transients sit on the *shallow* side of the error distribution. Neither number is
wrong; §8.1 of the surrogate anchors E₁ on the control-gap value, and should say which mask it means.

---

## 6. E₂: return air is not a logged point

§6.2 notes that mixed-air temperature is unlogged and identifies `φ` through the energy balance
because of it. That is the right move. But it then uses `T_ra` on the other side of the same
equation, and **`T_ra` is not logged either.** Neither AHU has a return-air temperature point — the
only return-air measurement on the floor is `ahu_b8_*__return_air_humidity`. This is asserted: if the
point ever appears, the proxy table below should be replaced rather than kept.

Three constructions are available, and they do not agree
([data/return_air_proxies.csv](data/return_air_proxies.csv)):

| Proxy | AHU-1 mean | φ median | φ in [0,1] | AHU-2 mean | φ median | φ in [0,1] |
|---|---:|---:|---:|---:|---:|---:|
| **Flow-weighted room temp** | 22.30 °C | 0.056 | **74.4%** | 22.91 °C | −0.048 | 33.3% |
| Unweighted room temp | 22.16 °C | 0.076 | **78.2%** | 22.84 °C | −0.042 | 33.9% |
| IAQ sensor mean (13 sensors) | 23.53 °C | −0.082 | 27.3% | 23.53 °C | −0.123 | 23.9% |

**A 1.2 K spread between proxies, and it lands undivided on φ.** The IAQ sensors read 1.2 K warmer
than the flow-weighted VAV mean and are shared between both AHUs, so they cannot distinguish the two
subsystems at all. The flow-weighted mean is the one this document uses, on the argument that return
air is drawn in proportion to supply — but the unweighted mean scores marginally *better* on the
in-band test, which is a reminder that the choice is not settled by fit.

The deeper problem is that neither proxy is the return air. Return air on this floor travels through
a ceiling plenum, picking up lighting and equipment heat before it reaches the AHU, so the true
`T_ra` is **above** any of these — which pushes implied φ further negative and makes §7 worse, not
better. That is stated as a direction, not a correction: the plenum gain is not measured anywhere in
this record.

---

## 7. E₂: the mixing map has the wrong sign

§6.2's instruction is to fit `φ = φ₀ + φ₁·damper%`, constrained to [0, 1], and treat `φ₀, φ₁` as
fitted parameters. Backing φ out of the energy balance step by step and running exactly that
regression ([data/mixing_map.csv](data/mixing_map.csv)):

| | AHU-1 | AHU-2 |
|---|---:|---:|
| `φ₀` | 1.7294 | 1.2124 |
| **`φ₁`** | **−0.0179** | **−0.0132** |
| R² | 0.227 | 0.080 |
| corr(damper, φ) | **−0.477** | **−0.283** |
| Damper mean / sd | 88.3% / 8.9 | 86.6% / 7.5 |
| Damper min / max | 22.7 / 100.0 | 22.7 / 100.0 |
| φ median | 0.056 | −0.048 |

**The slope is negative on both AHUs.** The map requires it to be positive — opening the outside-air
damper must admit more outside air, or the parameter has no meaning. Fitted on two independent units
of the same design, on the same floor, over the same season, it comes back negative both times, with
a correlation of −0.48 on AHU-1 that is far too strong to dismiss as noise.

**This is worse than an uninformative damper, and the distinction matters.** If `φ₁` fitted to zero
at R² 0.00, the reading would be "the damper position tells us nothing about outside-air fraction,
so φ must be identified some other way" — inconvenient but harmless. What the record actually shows
is a *real* relationship pointing the wrong way, which means the fitted map does not merely fail to
help: **applied inside the surrogate, it would move φ the wrong way in response to the damper.** §9
confirms that it does, out of sample.

**And the levels are as damning as the slope.** The damper reads a mean of 88.3% / 86.6% open while
the implied outside-air fraction sits at **0.056 / −0.048**. A floor drawing ~87% outside air in
Bangkok would carry a coil load several times what is measured. Whatever
`fresh_air_damper_position_read` is reporting — a commanded position against a stuck actuator, a
damper in a bypass path, a scaling convention, a point mapped to the wrong device — **it is not
outside-air fraction**, and §6.2's identification of φ through the energy balance is left with no
regressor at all.

**The sign is the assertion**, because it is what makes this a falsification rather than a weak fit.
R² is reported alongside it.

---

## 8. The airflow unit, discriminated

Surrogate §4.6 flags the VAV airflow unit as **UNRESOLVED**, blocking Block C's absolute totals and
the ventilation constraint, and names the air-side energy balance as the discriminating test:

```
        Q_air  =  ρ·c_p·V̇_total·(T_ma − T_sa)   ≡   Q_water  =  k·ṁ_w·ΔT_w
```

Block C could not run it. Its own quantities are unit-invariant twice over — the shares are
normalised and the unit lives entirely in the scalar `C` (§4.2 of the surrogate) — so Block C has no
purchase on the question at all. Block E does, because the water side is measured.

**The discriminating criterion is a physical bound, not a fit.** Mixed air is a blend of return air
and outdoor air, so `T_ma` must lie between them, i.e. `φ ∈ [0, 1]`. Each unit hypothesis implies a
different coil ΔT and therefore a different φ ([data/airflow_unit.csv](data/airflow_unit.csv)):

| AHU | Unit | V̇ median, m³/s | Implied coil ΔT | φ median | **φ ∈ [0,1]** |
|---:|---|---:|---:|---:|---:|
| 1 | CFM | 2.950 | 3.03 K | −0.178 | 3.0% |
| 1 | **m³/h** | **1.736** | **5.15 K** | **0.056** | **74.4%** |
| 1 | L/s | 6.251 | 1.43 K | −0.356 | 0.2% |
| 2 | CFM | 2.574 | 3.70 K | −0.321 | 2.0% |
| 2 | **m³/h** | **1.515** | **6.29 K** | **−0.048** | **33.3%** |
| 2 | L/s | 5.454 | 1.75 K | −0.552 | 0.0% |

**m³/h wins by 25.0× on AHU-1 and 16.5× on AHU-2**, against [config.yml](../config.yml)
`min_discrimination_margin` of 1.5 — the same argmin-plus-margin form §1.2 used for the water side. CFM and L/s do not merely fit
worse: they place mixed air *colder than both streams it is mixed from* on 97–100% of steps, which
is not a bad estimate but an impossible one. **This is asserted.**

**What it does and does not unblock.** A 1,000 max-flow setpoint is 1,000 m³/h ≈ 278 L/s per box,
which is a plausible large VAV box. Surrogate §4.6's "run this before Block C's absolute numbers are
quoted anywhere" has now been run. But two cautions travel with it:

**Caution 1 — AHU-2's in-band share is 33%, not 74%.** The *ranking* is unambiguous on both AHUs; the
*level* is not. AHU-2's median φ is slightly negative, meaning the balance does not quite close there
even under the winning unit. AHU-2 is the subsystem control-gap §3 finds saturated on 29.6% of its
large-error steps and §4.4 finds below 0.8× its pressure setpoint on 12% of steps. It is the sicker
unit under every diagnostic in the repo, and it is the sicker unit here too.

**Caution 2 — fan heat moves φ and is not measured**
([data/fan_heat.csv](data/fan_heat.csv)). If the fan sits downstream of the coil, its shaft power
raises `T_sa` above the coil-leaving temperature, and the balance must be corrected. How much of
~2.9 kW reaches the air is not logged:

| Fraction of fan power into the air | ΔT_fan, AHU-1 | φ median | φ ∈ [0,1] |
|---|---:|---:|---:|
| 0.0 | 0.00 K | 0.056 | 74.4% |
| 0.5 | 0.70 K | −0.016 | 42.7% |
| 1.0 | 1.39 K | −0.086 | 27.3% |

At full fan heat the m³/h reading also drives φ negative. **It does not change the ranking** — CFM
and L/s are worse at every fraction — which is why the unit verdict stands while the φ level does
not. The logged fan power (2.90 / 2.94 kW median) agrees with Block B's offset cubic (2.97 / 3.01 kW)
to within 2%, so the ΔT_fan scale is right even though the fraction is unknown.

---

## 9. Coil closure, on a held-out month

§6.3 specifies the falsification: fix parameters on the training window, then compare `Q_req` against
the measured `Q_water` on the test window. Run as specified, with the training window ending
**29 Jul 2026** to match Block D ([data/coil_closure.csv](data/coil_closure.csv)):

| AHU | Mixing model | Window | n | Ratio median | **Median abs % error** |
|---:|---|---|---:|---:|---:|
| 1 | Constant φ (train median) | train | 1,990 | 1.000 | 11.65% |
| 1 | Constant φ (train median) | **test** | 596 | 0.965 | **13.19%** |
| 1 | **Affine map on damper** | **test** | 596 | 1.036 | **15.12%** |
| 1 | φ = 0, all return air | **test** | 596 | 0.890 | **11.52%** |
| 2 | Constant φ (train median) | **test** | 583 | 1.110 | **15.74%** |
| 2 | **Affine map on damper** | **test** | 583 | 1.162 | **22.08%** |
| 2 | φ = 0, all return air | **test** | 583 | 1.110 | **15.74%** |

**§6.2's affine map is the worst of the three on both AHUs, in and out of sample.** That is §7's
sign result showing up as a cost rather than as a coefficient. On AHU-1 the best held-out model is
the one that says there is *no outside air at all* — 11.52% against the map's 15.12%. On AHU-2 the
constant-φ and φ=0 rows are identical, because the training median φ was negative and clipped to
zero: the fitted constant *is* zero there.

**Reported, never asserted, and the reason is important.** A 12–16% residual on a held-out month is
not bad for a model with two fitted parameters. But it inherits Block C's total (R² 0.42 / 0.64 by
its own §4.3), the mixing map, the E₁ bias, the return-air proxy, the airflow unit and the missing
latent load, **all at once**. §6.3 claims this test "simultaneously tests Block C's total, Block E₁'s
bias model, and the mixing map" — it does, and that is precisely why a number coming out of it
**cannot localise the fault**. It is a smoke test, not a falsification of any single block. Treating
a 13% closure as evidence that any one of those six is right would be reading the test backwards.

---

## 10. COP, and why the coil stays outside the reward

The one part of §6 that survives untouched ([data/energy_context.csv](data/energy_context.csv)):

| | AHU-1 | AHU-2 | Floor |
|---|---:|---:|---:|
| Coil, kWh-thermal | 9,127.8 | 9,599.5 | **18,727.4** |
| Fan, kWh-electric | 2,016.3 | 2,140.7 | **4,157.0** |
| **Ratio** | 4.53 | 4.48 | **4.51** |
| Implied kWh-e at COP 3 / COP 5 | — | — | 6,242 / 3,745 |

§6.4 quotes 18,732 kWh-th and 4,297 kWh-e for a ratio of 4.4×. The thermal total reproduces to
**0.03%**; the electric total is 3.3% lower here on a mask that also requires no-override and
no-outage, giving 4.51× rather than 4.4×. §6.4's conclusion is unaffected and its arithmetic is
confirmed.

The point of the table is the last row. The coil is 4.5× the fan thermally, and converting it to
electricity requires a COP that **does not exist in this dataset** — the plant is off-floor and
unmetered. At COP 3 the coil is 6,242 kWh-e; at COP 5 it is 3,745. The band is wider than the entire
fan consumption. §6.4's rule — report it in `info` as kWh-thermal, let it into the electric objective
only through domain randomisation, never fold a COP into a scalar reward — is correct and this
document has nothing to add to it except confirmation.

---

## 11. Guards

Asserted — a reversal fails the run:

| Guard | Rule | Status |
|---|---|---|
| Water constant | `k` derives from config's fluid properties | 0.0387592593 |
| ΔT identity | `water_delta_temperature` = return − supply | 1.13 / 0.86 °F, tol 2.0 |
| Coil anchor | logged `cooling_rate` ÷ `k·flow·ΔT` inside `k_ratio_bounds` | 1.0735 / 1.0733 |
| **Second law** | °F reading puts entering water below leaving air; °C reading does not | **1.0000 / 0.0000** |
| Missing point | no `return_air_temperature` column exists | passes |
| **Airflow unit** | winner beats runner-up by `min_discrimination_margin` | **24.97× / 16.48×** |
| SAT excitation | sd(`SAT_sp`) / sd(`T_sa`) below 0.50 | 0.202 / 0.207 |
| **Mixing sign** | fitted `φ₁ < 0` on both AHUs, where §6.2 needs `φ₁ > 0` | **−0.0179 / −0.0132** |

Reported, never asserted: the setpoint gain (§4), every candidate bias regressor (§5), the φ implied
by each return-air proxy (§6), the fan-heat band (§8), and the coil-closure residual (§9). Each is a
place the model is weak or the record is thin, and a guard that passed by not looking would be worse
than no guard.

---

## 12. Tables shipped

| File | One row per | Contents |
|---|---|---|
| [data/unit_scales.csv](data/unit_scales.csv) | temperature column | median, p1/p99, the °C reading, the verdict — §3 |
| [data/second_law.csv](data/second_law.csv) | AHU | the coil-direction falsifier, both readings — §3 |
| [data/sat_gain.csv](data/sat_gain.csv) | AHU × estimator | slope, R², n, and both sds — §4 |
| [data/sat_episodes.csv](data/sat_episodes.csv) | surviving episode | hour, direction, ΔSAT_sp, ΔT_sa, ratio — §4 |
| [data/sat_bias_fit.csv](data/sat_bias_fit.csv) | AHU × regressor | b₀, b₁, R², σ, and the simulable/admissible flags — §5 |
| [data/return_air_proxies.csv](data/return_air_proxies.csv) | AHU × proxy | mean, sd, and the φ each implies — §6 |
| [data/mixing_map.csv](data/mixing_map.csv) | AHU | φ₀, φ₁, R², correlation, damper distribution — §7 |
| [data/airflow_unit.csv](data/airflow_unit.csv) | AHU × unit | implied ΔT and φ, in-band share, margin — §8 |
| [data/fan_heat.csv](data/fan_heat.csv) | AHU × fan fraction | ΔT_fan and its effect on φ; Block B cross-check — §8 |
| [data/coil_closure.csv](data/coil_closure.csv) | AHU × model × window | `Q_req` vs `Q_measured`, train and held-out — §9 |
| [data/energy_context.csv](data/energy_context.csv) | AHU (+ floor) | kWh-thermal, kWh-electric, the ratio, the COP band — §10 |
| [data/identities.csv](data/identities.csv) | AHU | `k`, both anchor identities, step counts — §2 |

---

## 13. What this block does and does not settle

**Settles:**

1. **The airflow unit is m³/h**, by a 25×/16× margin on a physical bound rather than a fit. Surrogate
   §4.6 and §10's item 5 can be closed — with §8's two cautions attached.
2. **The record mixes °F and °C**, provably and without appeal to climate. Every equation that
   differences `T_oa` against an air-side temperature is currently wrong.
3. **§6.2's affine mixing map is falsified**, on the sign of its own fitted coefficient, on both
   AHUs, and it costs held-out accuracy when applied (§9).
4. **E₁ as specified cannot be stepped**, because its only regressor is not in the state vector.
5. **§6.4's 4.4× is confirmed** at 4.51× on a stricter mask.

**Does not settle:**

1. **The SAT setpoint gain.** Five estimators, 0.91–3.02, on a channel excited at a fifth of the
   amplitude of what it commands. This is the largest unquantified parameter in the surrogate and
   the record cannot fix it.
2. **What carries the E₁ bias.** `T_ra` is the best simulable, admissible candidate and it is
   endogenous. A constant bias is what the record supports.
3. **The true outside-air fraction.** φ is ≈ 0 under every proxy and every unit, which is either
   true (the floor runs on recirculated air and the damper point is misreporting) or an artifact of
   the return-air proxy and the missing latent load. **These are not distinguishable here.**
4. **What `fresh_air_damper_position_read` actually reports.** It is a live, finely-resolved point —
   909 distinct values on AHU-1 and 406 on AHU-2 over the record (904 / 400 on-hours), which is why
   [control-gap-method.md](../control-gap-method.md) §1 lists it as a usable signal — and it behaves
   like nothing in the mixing box.
5. **The latent load.** Every `Q_air` here is sensible-only, against a `Q_water` that is not.
   §9's residual contains this and cannot separate it.

---

## 14. Corrections this forces upstream

Stated, **not applied** — this document changes nothing outside its own folder.

| Document | Section | What it now needs |
|---|---|---|
| grey-box-surrogate.md | §1 | `w_t` must state that `T_oa`, `T_wb` are **°F** and require conversion |
| grey-box-surrogate.md | §6.1 | the unit setpoint gain is an assumption, not a measurement — §4's spread belongs in §8.2 as a DR axis alongside β |
| grey-box-surrogate.md | §6.1 | `b₁·valve` is not simulable; §1.1's `b(load)` is undefined. Replace with a constant bias, `T_ra` as a DR arm |
| grey-box-surrogate.md | §6.2 | the mixing map is falsified on its sign (§7); `T_ra` is not a logged point (§6); the equation mixes °F and °C (§3) |
| grey-box-surrogate.md | §6.3 | the closure cannot localise a fault across six inherited error sources (§9) |
| grey-box-surrogate.md | §4.6, §10.1 item 5 | the airflow unit is resolved to **m³/h** (§8) |
| grey-box-surrogate.md | §8.1 | E₁'s row should name the mask its −1.02 / −0.22 K anchor comes from; the settled mask gives −1.24 / −0.32 |
| [grey-box-technique.md](../grey-box-technique.md), [block-d](../block-d-derivation/README.md) | §2.3, §5 | `g_i·(T_oa − T_i)` differences °F against °C. The **conclusion** survives (the within-night slope is scale-free); the coefficient's units do not |

A `CORRECTED` banner on §6, in the form §4 and §5 already carry, is the natural next step. It is not
written here because it belongs in that file.
