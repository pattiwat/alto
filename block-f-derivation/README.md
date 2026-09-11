# Block F, derived — the loops the agent does not command

> **What this is.** The audit trail behind [grey-box-surrogate.md](../grey-box-surrogate.md) §7, and
> the block that closes the surrogate. Blocks A–E describe the plant; without F nothing produces
> `d_i`, Block C's split has no input, and the environment can replay a day but cannot answer a
> counterfactual.
>
> **Run it:** `python block-f-derivation/reproduce_zone_control.py` — fits stage 1 from the raw CSV,
> inverts Block C for stage 2, sweeps the compensation axis, and writes five tables to `data/`.
>
> **Three findings up front.**
>
> 1. **§7's identification recipe is not executable, and its specification has the wrong form.** It
>    regresses `ΔV̇sp_i` on the zone temperature error — but `V̇sp_i` is not a logged point, and the
>    difference form it implies is a sub-Nyquist dynamic specification that §0 forbids. Fitted anyway,
>    it returns the **wrong sign on 18 of 19 AHU-1 zones**. §2 replaces it with a level form that
>    returns the right sign on **19 of 19** and **21 of 22**.
> 2. **The stage-1/stage-2 gap was real and it closes analytically.** Block C's split inverts in
>    closed form, and in ratio form the scale constant `C` cancels — so stage 2 needs **no new
>    parameters** and touches neither the weak total nor the unresolved airflow unit of §4.6.
> 3. **"Compensating" has two readings and they are not the same number.** §4.3's 0.11% / 0.15%
>    assumes the loops restore the *floor total*, which requires boxes with authority left to open
>    further and cover boxes that have run out. No VAV controller does that. Under the reading where
>    each loop restores **its own** flow, the same 20% cut costs **0.45% / 1.11%** — 4× and 7× wider.
>    §4.2 is that subject.

---

## 1. What was missing

§7 of the surrogate specifies one stage. The plant runs two
([hvac-system-logical-flow.md](../hvac-system-logical-flow.md) §61–62 — "modulates the damper until
measured flow matches"):

```
   stage 1    room temperature error   ->  flow setpoint      §7 has this
   stage 2    damper chases flow                              §7 does not
```

§7 ends at `V̇sp_i`. Block C §4.2 begins at `d_i`. Nothing in the §1.1 step order joined them, so the
step order walked straight past a variable with no producer. It is not the only one — `valve(t)` and
`T_ra` have the same problem in Block E ([block-e-derivation](../block-e-derivation/README.md) §5–6)
— but it is the one that stops the loop closing, because `d_i` is the sole input to Block C's split.

**Why this block matters more than its position suggests.** §4.3 states that "the choice of Block C
formulation is not what determines the ventilation cost of a pressure reset — the behaviour of the
zone flow loops is." That behaviour is this block. The headline uncertainty in the whole document —
a 20% cut costing 0.1% or 10.8% of delivered air — is a Block F parameter.

---

## 2. Stage 1, and why the difference form had to go

### 2.1 The variable §7 regresses on is not logged

Each VAV box logs exactly seven points:

```
   air_flow_rate                          room_temperature
   damper_position                        room_temperature_setpoint_read
   maximum_air_flow_rate_setpoint_read    room_temperature_setpoint_write
   minimum_air_flow_rate_setpoint_read
```

There is no commanded flow setpoint. §7's "`K_i` identified per zone by regressing `ΔV̇sp_i` on the
zone temperature error" is therefore a regression on an unobservable — **verbatim** the objection
§4.7.2(3) used to retire Block C v0 ("`V̇sp_i` **is not logged.** The PI branch is an equation in an
unobservable"). The objection was raised against one block and not applied to another.

### 2.2 §0's Nyquist argument, used forward

The way out is the constraint that usually takes things away. The damper loop runs at 1–5 minutes
against a 15-minute sample, so **it has settled within a step**: a settled box sits *at* its flow
setpoint, and `V̇_i(t) ≈ V̇sp_i(t)`. The unlogged setpoint is observable through logged flow.

The same argument fixes the *form*. If the loop has settled, what the record can see is the **static
map** from error to flow — not the integrating action that produced it. So stage 1 is a level:

```
        V̇_i(t)  =  A_i  +  G_i · ( T_i(t−1) − Tsp_i(t−1) )
```

**and not** the increment §7 writes. That increment asks a 15-minute record to resolve a 1–5 minute
integrator, which is precisely what §0 rules out for every other block in the document.

### 2.3 Both were fitted. The difference form fails on its own terms

Refitted on identical rows ([data/zone_gain_fit.csv](data/zone_gain_fit.csv)):

| | AHU-1 | AHU-2 |
|---|---:|---:|
| zones identified | 19 / 26 | 22 / 27 |
| **level form** `V̇ = A + G·e` — zones with `G > 0` | **19 / 19** | **21 / 22** |
| median `G` (flow units per K) | 12.87 | 32.17 |
| median `A` (parked flow) | 235.1 | 237.7 |
| median R² | 0.050 | 0.094 |
| **difference form** `ΔV̇ = K·e` — zones with `K > 0` | **1 / 19** | **5 / 22** |
| median `K` | −16.84 | −18.56 |

The sign reversal is not noise, it is mechanical. The controller has already put `e(t−1)` into
`V̇(t−1)`, so `ΔV̇ = V̇(t) − V̇(t−1)` inherits that level's negative — the standard artefact of
regressing a difference on a level two cointegrated series share. Adding `V̇(t−1)` back as a
regressor repairs the sign (79% / 59% positive) at R² 0.65 / 0.48, but that specification is
algebraically the level form with extra steps, so the level form is what ships.

**Kept visible, not deleted.** `K_difference_form` and `r2_difference_form` are columns in the
shipped table, the same discipline §4.7 applies to Block C's v0 and v1.

### 2.4 The authority exclusions are identification, not hygiene

`V̇_i ≈ V̇sp_i` holds only while the box is off its stops. Against a stop the damper cannot chase
flow and the proxy fails exactly where the interesting behaviour is. So the gain fit drops steps
where `d_i ≥ 99%`, or flow is within 2% of the box's own min or max setpoint, **at both `t` and
`t−1`** — a box released from a stop is not sitting on its static map either.

Those exclusions apply to **the gain fit alone**. Never to the comfort constraint (§4.3: "a starved
zone is still a zone") and never to the counterfactual of §3, where clipping is the effect being
measured.

### 2.5 What the fit does not fix

The error is lagged one step so the controller cannot respond to a temperature it has not yet
measured — the same one-step-lead protection §5.4(1) uses on Block D. **It is not a cure.** The
reverse path still runs: more air cools the room, which lowers the error, which the regression reads
as a cause. `G` is biased toward zero, in the same direction and for the same reason as Block D's
`α`. Stated, not solved.

### 2.6 Two things the fit recovers on its own

**The named exclusions, mechanically.** Twelve boxes fail the step-count threshold and are reported
unidentified rather than fitted. They include `vav_8_1_28` and `vav_8_2_28` — config's dead box and
its hottest rogue. The model rediscovered the hand-named list without being shown it, which is the
same evidence Block C v1 offered for itself (§4.7.1).

**The one negative gain is a parked zone.** `vav_8_2_19` is the sole identified box with `G < 0`, and
it is one of the twelve zones §5.3(4) names as parked at 27.0 °C. Those zones never call for cooling,
so their flow barely varies and the gain is unidentified rather than wrong — §5.3(4) already says to
pool them.

### 2.7 R² of 0.05–0.09 is the finding

It is tempting to read a 5% R² as a failed fit. It is a measurement: §5.5's floor sits **2.60 K below
setpoint**, so at a typical error the term `G·e` contributes about −33 (AHU-1) and −84 (AHU-2)
against a parked level `A` of ~236. **Most of each box's flow is its floor, not its controller.**
These loops are barely modulating, which is the same fact §5.5 reports from the temperature side.

---

## 3. Stage 2 — F-a, and why it costs nothing

### 3.1 The inversion is closed form

Block C forward is `V̇_i = C·W^(q−1)·SP^s·a_i·d_i` with `W = Σ_j a_j d_j`. Summing over boxes,

```
        W^q  =  V̇_target,total / ( C · SP^s )        ⇒        d_i  =  V̇_i · W / ( a_i · V̇_total )
```

No search. The iteration in `dampers_for_targets` exists only to redistribute after boxes hit the
0/100 stops, and its inner solve is a scalar bisection on a monotone function.

### 3.2 In ratio form, `C` cancels — and that matters

Working relative to the current step rather than in absolute flow, the fixed point becomes

```
        W^(q−1) · ( W − W_stopped )  =  W_free · W_now^(q−1) · (SP'/SP)^(−s)
```

`C` is gone, and so is `SP` — only the *ratio* of new to old pressure enters. That is not tidiness.
`C` is the weakest number in Block C (the total's R² is 0.42 / 0.64 against 0.94 / 0.90 for the
split), and it carries the **unresolved airflow unit** of §4.6. **F-a touches neither.**

### 3.3 The round trip is exact

Asked for the flows the plant already had, at unchanged pressure, F-a must return the dampers the
plant was already sitting at ([data/inversion_round_trip.csv](data/inversion_round_trip.csv)):

| | AHU-1 | AHU-2 |
|---|---:|---:|
| max abs damper error | **0.0** | 2.8e−14 |
| max abs share error | **0.0** | 4.2e−17 |
| max share-sum error | 4.4e−16 | 4.4e−16 |

Asserted. This is the concrete form of "one forward model, two consumers": F-a inverts the same
`block_c.fit_split` the surrogate forward-steps, imported rather than restated, so a third correction
to Block C cannot land in only one of them.

---

## 4. The compensation band

### 4.1 λ, swept

A 20% pressure cut, dampers interpolated `d(λ) = d + λ·(d_compensating − d)` and stepped forward
through Block C ([data/compensation_band.csv](data/compensation_band.csv)):

| λ | 0 (frozen) | 0.25 | 0.5 | 0.75 | 1 (compensating) |
|---|---:|---:|---:|---:|---:|
| **AHU-1** loss % | **10.79** | 8.19 | 5.60 | 3.02 | **0.45** |
| **AHU-2** loss % | **14.92** | 11.53 | 8.10 | 4.63 | **1.11** |

λ = 0 reproduces Block C's frozen arm to four decimals, as it must — with the dampers held still both
routes reduce to the same `(1−cut)^s` identity. That agreement is asserted.

**λ is not fitted, and this block does not pick a point inside the band.** §4.3's reason stands: no
event in this record separates loop compensation from a demand move, because pressure and demand
moved together in every episode it contains. λ ships as a domain-randomisation axis over
`lambda_range`, the same treatment §8.2 gives β and `s`.

### 4.2 "Compensating" means two different things, and §4.3 chose the generous one

This is the finding that changes a number upstream.

| | AHU-1 | AHU-2 |
|---|---:|---:|
| **own-flow** — each loop restores *its own* flow | **0.45%** | **1.11%** |
| **restore-total** — one common `k`, solved after clipping | 0.11% | 0.15% |
| steps where nothing clips | 1,888 / 2,681 | 1,686 / 2,681 |
| max gap between the arms on those steps | **8.0e−16** | 1.2e−15 |

The two arms are **provably the same arm** wherever no box clips, and the shipped table asserts it to
floating point. The entire divergence is the clipping steps, and it turns on one question: when a box
runs out of damper, do its neighbours open further to make up the floor total?

**Under `restore-total` they do.** That is Block C's `required_scaling`, which searches for the common
`k` that brings the *post-clip* sum back to target — so boxes with authority left are scaled by more
than the split-preserving factor to cover boxes that have none. It is where §4.3's 0.11% comes from.

**No VAV controller does this.** Each box has its own flow controller tracking its own setpoint; a
satisfied box does not open because a neighbour is starved. That is §4.1's pressure-independent
premise taken literally, and under it the shortfall is simply not made up.

**So the band's compensating end is 4× (AHU-1) and 7× (AHU-2) wider than §4.3 states.** The direction
matters: it widens the band, which weakens the attribution in control-gap §5 further rather than
rescuing it. §10.1's "unexplained" verdict stands with more force, not less.

Reported, not resolved — the record cannot show which reading is right either, because it contains no
step where a box demonstrably ran out of authority (§4.3: saturated share 0.0000 in both windows).

### 4.3 The plant's own cuts point away from compensation

Seven AHU-1 episodes survive the filter — a setpoint change is not a pressure cut unless the achieved
pressure actually fell ([data/episode_replay.csv](data/episode_replay.csv)):

| median over 7 episodes | |
|---|---:|
| achieved pressure ratio | 0.896 |
| **observed** mean-damper change | **−1.08 pts** |
| **F-a predicted** mean-damper change | **+2.98 pts** |

F-a says dampers must **open** to hold flow at lower pressure. They **closed**. Reported, never
asserted — and the right reading is not that F-a is wrong but that these episodes are demand moves:
the floor was 0.89 K cooler in the treated windows (§4.3), so the loops were closing dampers on
falling load, not opening them against falling pressure. **Which is exactly why this record cannot
identify λ.**

**Every AHU-2 episode was dropped**, and that is a result too. §2.2 records that 93% of them sit at
08h inside the start-up ramp; duct static is *climbing* there while the setpoint steps down, so they
are not pressure cuts. The filter finds this without being told.

---

## 5. Guards

Asserted — a reversal fails the run:

1. **Round trip.** Inverting Block C and stepping forward returns the shares asked for, to
   `inversion_tolerance`. Shares still sum to 1.
2. **Frozen endpoint.** λ = 0 matches `block-c-derivation/data/pressure_cut_counterfactual.csv`
   within `band_endpoint_tolerance_pct`. Both routes reduce to the same identity there.
3. **Restore-total endpoint.** The `tot` arm reproduces Block C's compensating number — it delegates
   to `block_c.required_scaling`, so a disagreement means the surrounding bookkeeping drifted.
4. **The arms coincide where nothing clips**, to floating point. This is what makes their divergence
   elsewhere a finding rather than a bug.
5. **Declining to cover a neighbour cannot deliver more air**: `loss_own ≥ loss_tot`.
6. **Compensation opens dampers.** Median scaling ≥ 1. A sign check on the algebra.
7. **The gain sign.** `G > 0` on at least `min_positive_gain_fraction` of identified zones.

Reported, never asserted: λ itself, the episode replay, the difference-form gains, and R².

---

## 6. Tables shipped

| File | Contents |
|---|---|
| [data/zone_gain_fit.csv](data/zone_gain_fit.csv) | per box: `G`, `A`, R², n, `identified`, and the rejected difference-form `K` — §2 |
| [data/inversion_round_trip.csv](data/inversion_round_trip.csv) | F-a against Block C forward, per AHU — §3.3 |
| [data/compensation_band.csv](data/compensation_band.csv) | the 20% cut over the λ grid, both arms — §4.1 |
| [data/arm_comparison.csv](data/arm_comparison.csv) | own-flow vs restore-total, and where they must agree — §4.2 |
| [data/episode_replay.csv](data/episode_replay.csv) | the plant's own cuts, observed vs F-a predicted — §4.3 |

---

## 7. What this block does and does not settle

**Settles.**

- The surrogate closes. `d_i` has a producer, so Block C's split has an input and the environment can
  be stepped against a counterfactual setpoint.
- Stage 1 is identifiable from logged points, in a form §0 permits, with the sign physics predicts.
- Stage 2 costs no parameters and is independent of `C`, of the total's weak R², and of §4.6's
  unresolved airflow unit.

**Does not settle.**

- **λ.** The band is measured; the point inside it is not, and this record cannot locate it.
- **Which reading of "compensating" is right** (§4.2). The physical argument favours own-flow; the
  record is silent because no box in it demonstrably ran out of authority.
- **`G`'s endogeneity** (§2.5). Biased toward zero by the same reverse path that biases Block D's `α`,
  and the one-step lag is mitigation rather than a fix.
- **The absolute value of `G`**, which carries §4.6's unresolved airflow unit — as block-d's `a_i`
  does, and harmlessly, provided the fit and the rollout agree on the unit.

---

## 8. Corrections this forces upstream

1. **§7** — the specification becomes a level, not an increment; the identification recipe stops
   citing an unlogged variable; stage 2 is added. A correction banner in the style §4 and §5 already
   carry.
2. **§1.1** — Block F must produce `d_i`, or a stage-2 line must appear between F and C. As written
   the step order consumes a variable nothing produces.
3. **§4.3 and §10.1** — the compensating end of the band is 0.45% / 1.11%, not 0.11% / 0.15%, unless
   the intended claim is that zone loops cover each other's shortfall. Either way the sentence needs
   to say which reading it means.
4. **§8.1 / §8.2** — rows for `G`, `A` and λ; λ joins the tornado axes.
5. **§10** — the zone-loop inference moves from a listed caveat to the stated determinant of the
   ventilation attribution.
