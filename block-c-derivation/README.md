# Block C, derived — each room's share of the air

> **What this is.** The audit trail behind [grey-box-surrogate.md](../grey-box-surrogate.md) §4.
> Block C answers one question: given duct static pressure and where every damper is sitting, how
> much air reaches each room? The answer here is a **share**:
>
> ```
>     split :   V̇_i      =  V̇_total  ·  ( a_i · d_i ) / Σ_j ( a_j · d_j )
>     total :   V̇_total  =  C · ( Σ_j a_j·d_j )^q  ·  SP_act^s
> ```
>
> Three logged points and nothing else — `vav_*__air_flow_rate`, `vav_*__damper_position`,
> `ahu_b8_*__static_pressure`.
>
> **Run it:** `python block-c-derivation/reproduce_zone_flow.py` — fits both halves from the raw
> CSV, runs the head-to-head against the model this replaces, asserts the guards in
> [config.yml](../config.yml) `zone_flow`, and writes eight tables to [data/](data/).
>
> **Three findings up front.**
>
> 1. **Static pressure cancels out of the split.** It is common to every box at a timestep, so it
>    divides away. Which room gets what air is *purely* relative damper opening; pressure only sets
>    how much air there is to divide. This is not a simplification imposed on the physics — it is a
>    property of it, and §5 shows it is what makes the model identifiable.
> 2. **The share is linear in damper opening.** Fitted free, the exponent comes back at 0.899 and
>    1.091. Imposing exactly 1 costs **−0.000 to +0.002** of held-out R². The exponent leaves the
>    specification.
> 3. **This is smaller *and* better — on one metric, and worse on another.** Against the per-box
>    valve law it replaces, it wins R² at every holdout origin on both AHUs with half the
>    parameters, and **loses** on median per-room percentage error. §5.2 states both.

---

## 1. What this replaces, and why twice

Block C has now been wrong in two different ways, and both corrections are kept visible.

**v0 — the two-regime switch.** A box either held its flow setpoint (damper < 100%, elasticity
exactly zero) or followed `√SP` (damper = 100%). The entire modelled response to a pressure cut lived
in the second branch, which is occupied on a median of 0.56% (AHU-1) and 2.63% (AHU-2) of a box's
on-steps, with 10 of 25 clean AHU-1 boxes never entering it. §2 of this document's previous revision
listed six objections; [grey-box-surrogate.md](../grey-box-surrogate.md) §4.7 keeps the text.

**v1 — the per-box valve law.** `V̇_i = A_i·R_i^(d_i/100)·SP^½`, two fitted parameters per box, ~103
for the floor. It fixed v0's bias and is still the reference this document scores against. Its
problem is not correctness, it is weight: 103 coefficients, none of which can be read off a drawing,
and an aggregation that summed to nothing in particular.

**v2 — the share form**, below.

---

## 2. The model

```
        V̇_i  =  V̇_total  ·  w_i / Σ_j w_j          w_i  =  a_i · d_i
```

| Symbol | Is | Count |
|---|---|---:|
| `a_i` | the room's **relative** authority — how much air it draws per point of damper opening. Normalised so `Σ a_i = 1`. | 23 / 25 |
| `d_i` | damper opening %, logged | — |
| `C, q, s` | the total: scale, conductance exponent, pressure exponent | 3 per AHU |

Two properties v1 never had:

1. **Adding-up holds by construction.** `Σ_i V̇_i = V̇_total`, to floating point, asserted. The
   per-box fits had no such constraint and summed to whatever they summed to.
2. **Saturation is relative and automatic.** A box pinned at 100% loses share when its neighbours
   open. There is no regime, no threshold, no crossover rule, and no discontinuity anywhere.

### 2.1 Why the exponent on `d` is 1

Fitting `w_i = a_i·d_i^p` and leaving `p` free ([data/split_fit.csv](data/split_fit.csv)):

| | AHU-1 | AHU-2 |
|---|---:|---:|
| `p`, free | 0.899 | 1.091 |
| Cost of imposing `p = 1`, **held-out R²** | **−0.000 … −0.003** | **−0.001 … +0.002** |
| Cost of imposing `p = 1`, within-R² of the fitting regression | 0.033 | 0.101 |

The two cost columns disagree and the distinction matters. Within-R² is computed on the demeaned
fitting regression, where the damper term is the *only* regressor and every unit of misfit lands on
it; held-out R² is computed on the quantity the surrogate actually needs. **The guard is on the
held-out number** ([config.yml](../config.yml) `max_r2_cost_of_linear_share`), and both are shipped
so the choice can be argued with.

---

## 3. The total, and why it is the weak half

```
        ln V̇_total  =  ln C  +  q·ln( Σ_j a_j·d_j )  +  s·ln SP_act
```

[data/total_fit.csv](data/total_fit.csv):

| | AHU-1 | AHU-2 |
|---|---:|---:|
| `q` (aggregate conductance) | 0.886 | 1.501 |
| `s` (pressure) | **0.512** | 0.724 |
| R² | **0.424** | **0.639** |
| R², aggregate conductance alone | 0.000 | 0.340 |
| R², static pressure alone | 0.154 | 0.002 |

**Say this plainly rather than let it be discovered later: the split is the trustworthy half of this
model and the total is not.** R² 0.42 / 0.64, against 0.94 / 0.90 for the split.

Two things are worth reading off the table anyway. AHU-1's `s = 0.512` lands almost exactly on the
orifice ½ — arrived at from a completely different direction than v1's imposed value. And **static
pressure alone explains essentially nothing** (R² 0.154 / 0.002); it only signs once aggregate damper
opening is held fixed, because pressure and demand move together on this floor. That is the same
confound that runs through every other Block C finding, and it is why §9's counterfactual is a band
rather than a number.

`q ≠ 1` on either AHU, and AHU-2's 1.501 has no physical reading — a parallel-conductance network
gives exactly 1. It is carried as fitted and flagged as unexplained.

---

## 4. Identification — the part that got easier

`a_i` and `p` come from a **within-timestep** regression:

```
        ln V̇_i,t  =  ln a_i  +  p·ln d_i,t  +  τ_t
```

`τ_t` is a time fixed effect. Because `SP_act` is common to every box at time *t*, **the time effect
absorbs it completely**, along with the total, the weather, the hour and the occupancy. So `a_i` and
`p` are identified purely from *cross-sectional* variation at a single instant.

> **This is why the model got simpler without getting worse.** v1 had to *impose* its pressure
> exponent, because the damper moves in response to pressure: fitted free alongside the damper term,
> the exponent absorbed the controller and drove implied rangeability **negative on 10 of 25 (AHU-1)
> and 8 of 26 (AHU-2)** boxes — "opening the damper reduces flow." That endogeneity **cannot reach
> the split.** Pressure is differenced out across rooms at a single instant, so the problem that
> forced an imposed exponent in v1 does not arise in the half of the model that carries the room
> allocation.
>
> It has not gone away — it has been *confined to the total*, which is exactly where §3 reports the
> model is weak. That is an honest relocation of a difficulty, not a solution to it.

Implementation: within-time demeaning rather than 2,681 dummy columns, one box held as reference to
break the rank deficiency, then weights normalised to sum to 1
([reproduce_zone_flow.py](reproduce_zone_flow.py) `fit_split`).

---

## 5. Head to head against v1

Both models fitted on the same training rows and scored on the same held-out rows, at each of
config's three holdout origins. The share form is given the observed total and asked only to
*allocate* it — which is how the surrogate uses it, the total coming from the fan side
([data/head_to_head.csv](data/head_to_head.csv)):

| AHU | origin | params: share / v1 | R² share | R² v1 | median abs % share | median abs % v1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | back21 | **23** / 46 | **0.945** | 0.935 | 5.36 | **3.59** |
| 1 | back42 | **23** / 46 | **0.934** | 0.909 | 5.62 | **3.25** |
| 1 | back63 | **23** / 46 | **0.959** | 0.949 | 5.23 | **3.55** |
| 2 | back21 | **25** / 50 | **0.903** | 0.853 | 9.16 | **5.75** |
| 2 | back42 | **25** / 50 | **0.901** | 0.869 | 10.04 | **6.49** |
| 2 | back63 | **25** / 50 | **0.905** | 0.814 | 7.71 | **5.59** |

### 5.1 What the share form wins

R² on ln V̇ at **every origin on both AHUs**, with half the parameters, and by the widest margin
exactly where v1 was weakest (AHU-2 back63: 0.905 against 0.814). This is the asserted guard — if it
ever reverses, the run fails, because it is the entire case for the change.

### 5.2 What it loses, stated in the same breath

**Median per-room percentage error is worse — 5.4% against 3.6% on AHU-1, 9.2% against 5.7% on
AHU-2.** v1 spends a second parameter per room and buys per-observation accuracy with it; the share
form spends its single parameter on getting the *allocation* right and takes the level from the
total.

Which matters depends on the consumer. Block D advances zone temperature from delivered flow per
room, so per-room error propagates — on that reading v1's 3.6% is worth something real. The share
form's compensating advantages are the adding-up identity, the clean identification of §4, half the
coefficients, and the fact that **its errors cannot all point the same way**: a room over-predicted
is another room under-predicted, because the shares sum to 1. v1's errors have no such constraint
and can bias the floor total.

**Neither of those arguments is decisive, and the table above is the honest summary.** The
recommendation to ship the share form rests on identification and structure, not on a clean sweep of
the fit statistics.

---

## 6. The room weights

[data/room_weights.csv](data/room_weights.csv) ships `a_i` per room alongside its mean damper, mean
flow and logged max-flow setpoint, so each weight can be sanity-checked against the room it belongs
to.

**The logged max-flow setpoint will not stand in for `a_i`.** It varies 5× across rooms (200–1000)
and is tempting as a free size weight requiring no fitting at all. Substituted for the fitted
weights, its exponent comes back at **0.665** (AHU-1) and **0.127** (AHU-2) rather than 1, and
within-R² collapses to **0.460 / 0.013** from 0.919 / 0.787. On AHU-2 it carries essentially no
information about how much air a room draws. One fitted number per room is what buys the accuracy,
and it cannot be read off a logged point.

---

## 7. A correction to this document's previous revision

The previous revision listed as flaw 4 that v0's damper rule `d' = d·(SP/SP')^½` "implies `C_v ∝ d`
(linear), contradicting §4.1's own orifice law with a real (equal-percentage) damper."

**That was overstated.** Holding structure fixed at two parameters per box and testing the two
damper characteristics head to head out of sample, they are indistinguishable — median R² **0.557
(equal-percentage) vs 0.545 (power)** on AHU-1, and **0.633 vs 0.650** on AHU-2, with the per-box win
split 8/23 and 13/24. **This record cannot identify the damper characteristic** over the observed
opening range. Linear is a legitimate choice — which is precisely what makes the share form
available.

The part of flaw 4 that stands: v0 used that rule to *assign regimes* while disclaiming it as
"saturation detection only". The objection was to the load-bearing disclaimer, not to linearity.

---

## 8. What a 20% pressure cut does — as a band, not a number

[data/pressure_cut_counterfactual.csv](data/pressure_cut_counterfactual.csv), at config's
`naive_cut_fraction`:

| | AHU-1 | AHU-2 |
|---|---:|---:|
| **Frozen dampers** — no compensation | **10.79%** | **14.92%** |
| **Compensating** — loops restore the total, clipping at 100% | **0.11%** | **0.15%** |
| Damper scaling the loops would need (median) | ×1.137 | ×1.114 |
| Steps where at least one box clips | 793 / 2,681 | 995 / 2,681 |
| Largest share shift under compensation | 0.017 | 0.017 |

The two arms bound the answer and **the record cannot say where in the band the truth sits.** Under
compensation the loops scale every damper by a common factor, which leaves the split untouched and
restores the total almost exactly — the residual is entirely the boxes that clip at 100%. Under
frozen dampers everything falls as `SP^s`.

This is the same bound the previous revision reached from the per-box side (0.3% to 10.6%), by a
completely different route, which is worth something: **the choice of Block C formulation is not what
determines the ventilation cost of a pressure reset. The behaviour of the zone flow loops is.**

---

## 9. The plant's own pressure cuts

[rl-sac-feasibility.md](../rl-sac-feasibility.md) §2.3 pairs 8 AHU-1 pressure-cut episodes against
their own 8-step pre-window, finds delivered airflow at 0.939 of control, and concludes "Block C's
pressure-to-airflow mechanism is real." Reproduced here
([data/episode_summary.csv](data/episode_summary.csv)):

| Median over 8 AHU-1 cut episodes | treated / control |
|---|---:|
| Static pressure, achieved | 0.905 |
| Delivered flow, observed | 0.951 |
| **Share-model allocation error** | **5.3% of total flow** |
| Mean damper position | **45.51 vs 47.16 → −1.65 points** |
| Share of boxes saturated | 0.0000 vs 0.0000 |

**The episodes rule the saturation mechanism out rather than confirming it.** If flow had fallen
because boxes ran out of damper authority, the dampers would have **opened** and the saturated share
would have **risen**. Neither happened: dampers *closed* by 1.65 points and no box was saturated in
either window. The floor was 0.89 K cooler during the cuts (rl-sac-feasibility §2.3 reports this
itself, as its identification confound), so the zone loops were shedding demand and closing dampers
to do it.

So §2.3's conclusion over-reads its own table. The flow ratio is real; the mechanism attributed to it
is not the one operating. And what the episodes *cannot* do — because pressure and demand moved
together in every one of them — is measure the compensating case. **This record contains no pressure
cut with demand held fixed**, which is exactly why §8 is a band.

*(AHU-2 contributes no episodes: none of its downward setpoint moves has both a treated and a control
window surviving the on-hours mask.)*

### 9.1 The consequence for control-gap §5

[control-gap-method.md](../control-gap-method.md) §5 concludes that **73% (AHU-1) and 64% (AHU-2) of
the modelled saving is bought by moving less air**. That number is a **residual** — modelled saving
minus the model-free frontier gap, `(190−52)/190 = 72.6%` and `(175−64)/175 = 63.4%` — attributed to
ventilation on the strength of "at least one box sits at 100% damper at every occupied step."

The subtraction is arithmetic and stands. The **mechanism** does not: §9 shows the saturation
signature is absent from the plant's own cuts, and §8 shows the ventilation loss is 0.1% under
compensation. The attribution needs the frozen-damper end of the band (10.8% / 14.9%) to survive —
which is a coherent position, but it is not the one control-gap §5 argues, and it contradicts the
premise that these are pressure-independent boxes with working flow control.

**Until the band is narrowed, the gap between the modelled saving and the frontier is unexplained
rather than explained.** Candidates: the fan model (§3 of the surrogate carries a 2× band), β (itself
[0.241, 0.529] on AHU-1), or the frontier's within-bin construction.

---

## 10. Guards

Asserted — a reversal fails the run:

| Guard | Rule | Status |
|---|---|---|
| Pressure cancels | scaling `SP_act` by an arbitrary constant leaves every share identical | < 1e-9 |
| Adding-up | shares sum to 1 at every timestep | < 1e-9 |
| Linear share | imposing `p = 1` costs < `max_r2_cost_of_linear_share` of held-out R² | −0.003 … +0.002 |
| **Head to head** | share form ≥ v1 on R², at every origin, both AHUs | passes 6/6 |
| Weights | `a_i > 0` on every clean room | passes |

Reported, never asserted: the total (§3), the per-room percentage-error loss (§5.2), the logged
max-flow contrast (§6), and both counterfactual arms (§8). Each is a place the model is weak or the
evidence is thin, and a guard that passed by not looking would be worse than no guard.

---

## 11. Tables shipped

| File | One row per | Contents |
|---|---|---|
| [data/room_weights.csv](data/room_weights.csv) | room | `a`, mean damper, mean flow, logged max-flow setpoint — §6 |
| [data/split_fit.csv](data/split_fit.csv) | AHU | `p` free and imposed, within-R², the logged-max-flow contrast — §2.1, §6 |
| [data/total_fit.csv](data/total_fit.csv) | AHU | `C, q, s`, R² and the two single-regressor R²s — §3 |
| [data/head_to_head.csv](data/head_to_head.csv) | AHU × origin | share vs v1, R² and median abs %, parameter counts — §5 |
| [data/pressure_cut_counterfactual.csv](data/pressure_cut_counterfactual.csv) | AHU | both arms of the 20% cut — §8 |
| [data/share_invariance.csv](data/share_invariance.csv) | AHU | the two structural identities — §10 |
| [data/episode_decomposition.csv](data/episode_decomposition.csv) | real cut episode | observed ratio, allocation error, damper and saturation levels — §9 |
| [data/episode_summary.csv](data/episode_summary.csv) | AHU | the §9 table |

---

## 12. What this block does and does not settle

**Settles:**

1. Room allocation is relative damper opening, linearly, with one weight per room. Pressure plays no
   part in the split — structurally, not approximately.
2. The endogeneity that forced v1's imposed exponent does not touch the split. It is confined to the
   total.
3. The saturation mechanism did not drive the flow drop in the plant's own pressure cuts.
4. The choice of Block C formulation does not determine the ventilation cost of a pressure reset —
   v1 and v2 give the same band from different directions.

**Does not settle:**

1. **The total.** R² 0.42 / 0.64, `q` unexplained on AHU-2, and static pressure carrying almost none
   of it on its own.
2. **Whether the flow loops compensate**, which is what the 0.1%–10.8% band turns on. No event in
   this record separates a pressure move from a demand move.
3. **Per-room accuracy against v1** (§5.2). The share form is chosen on structure, not on beating v1
   everywhere.
4. **The airflow unit** — untouched, still blocking absolute totals, and now living in `C`.
5. **Where control-gap §5's saving actually comes from** (§9.1).
