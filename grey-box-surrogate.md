# The Grey-Box Surrogate — Mathematical Specification

> **What this is.** Every equation in the Floor 8 simulator, its physical derivation, how its
> parameters are identified from *this* CSV, and the test that would falsify it. It exists because
> an RL agent's number is only as good as the environment it was trained in, and an environment
> whose equations are not written down cannot be audited.
>
> **Companion document:** [rl-environment-design.md](rl-environment-design.md) — why this surrogate
> is needed, which algorithm runs on top of it, and how the result must be reported.
>
> **Authorities.** Numbers here trace to [control-gap-method.md](control-gap-method.md) (derived by
> executable code from the raw CSV — preferred wherever documents disagree),
> [from-altotech-take-home-pdf-and-alto-bui-jolly-turing.md](from-altotech-take-home-pdf-and-alto-bui-jolly-turing.md)
> (the data profile), and [hvac-system-logical-flow.md](hvac-system-logical-flow.md) (the plant
> description). Section references are given inline. Nothing below is fitted yet — this is the
> specification the fitting code must implement.

---

## 0. The one thing to understand first

**The sample-starvation problem and the system-identification problem are not the same problem.**

| | Observations | Parameters | Ratio |
|---|---:|---:|---:|
| **Policy learning** (RL) | **62 episodes** | a 2×256 MLP actor-critic | hopeless |
| **System identification** (this document) | **≈ 95,800 zone-steps** (1,807 masked steps × 53 boxes) + 3,718 AHU-steps | **≈ 280** | ~340:1 |

The record is far too small to *learn a policy from* and entirely adequate to *identify a plant
model from*, because every zone contributes its own history rather than sharing one. That asymmetry
is the whole reason a surrogate is the right move: it converts a problem the data cannot answer
into one it can.

What the data still cannot give the surrogate is **dynamics faster than 30 minutes**. The loops have
1–5 minute time constants and the record is 15-minute *averages* — below Nyquist
(control-gap §6). Every block below is therefore either quasi-steady-state or has a time constant
comfortably above the sample interval. Nothing in this document models a PID.

---

## 1. State, action, timebase

Δt = 0.25 h. One AHU subsystem *a* ∈ {1, 2}; 53 VAV boxes indexed *i*, partitioned by AHU
(26 on AHU-1, 27 on AHU-2).

| | Symbol | Contents |
|---|---|---|
| **Endogenous state** | `x_t` | zone temperatures `T_i`, zone flow setpoints `V̇sp_i`, damper positions `d_i`, supply air temperature `T_sa`, achieved static pressure `SP_act`, fan frequency `f`, current supervisory setpoints `SP_sp`, `SAT_sp` |
| **Exogenous input** | `w_t` | outdoor drybulb `T_oa`, wetbulb `T_wb`, PIR occupancy fraction `occ`, non-AHU floor load |
| **Action** | `u_t` | `ΔSP_sp`, `ΔSAT_sp` |

**`w_t` is replayed from the record, never simulated.** This is the single most important structural
choice in the surrogate: weather and occupancy are the dominant drivers of everything, and replaying
them means the environment carries zero modelling risk on its largest term. An episode *is* a real
day. What the surrogate models is only the plant's response to a counterfactual setpoint.

### 1.1 Step order

The blocks are not independent equations to be solved simultaneously; they execute in this order,
and the order is part of the specification because it is what stops the model from being circular.

```
 u_t ──► [clip to logged support, rate-limit]  ──►  SP_sp', SAT_sp'
   │
   ├─► §6  Block E₁ : T_sa   = SAT_sp' + b(load) + η
   │
   ├─► §7  Block F₁ : zone controllers set V̇sp_i from (T_i − Tsp_i)      [stage 1, a level]
   ├─► §7  Block F₂ : dampers chase flow      d_i = C⁻¹(V̇sp_i, SP_act)   [stage 2]
   │
   ├─► §4  Block C  : total V̇ from (Σa_j d_j, SP_act) , then split by share  a_i d_i / Σ a_j d_j
   │        §4.4     : if the fan cannot make SP_sp', SP_act < SP_sp'   (AHU-2's failure mode)
   │
   ├─► §2  Block A  : f = f_ref·(SP_act/SP_ref)^β
   ├─► §3  Block B  : P_fan = a + b(f/50)³
   │
   ├─► §5  Block D  : T_i(t+1) advanced from delivered V̇_i and T_sa
   │
   └─► §6  Block E₂ : Q_coil from V̇, (T_ma − T_sa)   →  info only, never the reward
```

**Blocks A and C are two projections of one physical event, not a chain.** A gives fan speed, C
gives delivered air. Neither feeds the other, because β is a *reduced-form* elasticity that already
contains the damper response (§2.2). Wiring C into A would double-count it. §4.5 gives the
consistency test that keeps them honest.

---

## 2. Block A — static pressure setpoint → fan frequency

### 2.1 The physics, and why it is not used directly

At a **fixed system curve**, fan pressure rise scales with the square of speed and volume flow
scales linearly with it:

```
Δp ∝ Q²  ,   Q ∝ N        ⇒   N ∝ √Δp        ⇒   d ln f / d ln SP = ½
```

This is the affinity relation, and it is what almost every published fan-energy saving rides on.

### 2.2 Why ½ is wrong here, and what replaces it

The system curve on this floor is **not** fixed, because the dampers move. The p10 of the per-step
maximum damper position is **100.0 on both AHUs** (§1.6) — at every occupied step at least one box
is wide open. When duct pressure falls, those boxes cannot open further and the partly-open ones do,
which flattens the system curve and means the fan slows *less* than affinity predicts.

The surrogate therefore uses the reduced-form elasticity **measured from the plant's own setpoint
episodes** rather than the textbook value:

```
        f(t) = f_ref · ( SP_act(t) / SP_ref )^β
```

| | AHU-1 | AHU-2 |
|---|---|---|
| β̂ (ratio of sums, headline) | **0.382** | 0.232 |
| 95% CI (episode block bootstrap) | [0.241, 0.529] | [0.178, 0.283] |
| vs affinity β = 0.5 | not rejected | rejected |
| Specification sweep span (36 / 24 combinations) | 0.208 → 0.785 | 0.126 → 0.337 |
| Episodes used | 15 | 30, but **93% at 08h — start-up transients** |

Source: control-gap §4.2–4.3. Implemented by [src/effects.py](src/effects.py) `sp_to_hz`, which
already takes the exponent as an argument rather than hard-coding 0.5.

### 2.3 Identification status

**Measured, but not settled.** AHU-1's interval contains 0.5; only 15 of 36 specifications exclude
it. AHU-2 excludes it robustly but its episodes are contaminated. This is not a defect to be
resolved before proceeding — it is the parameter uncertainty the surrogate must *carry*, and §8
turns it into the domain-randomisation distribution.

Two confounds are stated rather than removed (control-gap §7): the episodes were not randomised, so
if the setpoint was lowered *because* load was low, part of the observed drop belongs to the load —
biasing β upward, i.e. conservatively with respect to the conclusion that the affinity assumption
inflates the prize. And only exogenous controls are admissible: airflow and damper position are
*mediators* of a pressure change, so conditioning on them would block the path being measured.

### 2.4 Falsification

Perturb `SP_sp` by −20% inside the finished surrogate and re-estimate `d ln f / d ln SP` by the same
episode method. It must return β̂ ± its CI. Same shape as
`tests/test_excitation.py::test_recovers_a_known_elasticity`, which already pins the estimator
against synthetic data with a known exponent to 1e-9.

---

## 3. Block B — frequency → fan power

```
        P_fan(f) = a + b·(f/50)³           [kW]        ← headline
        E_fan    = P_fan · Δt              [kWh]
```

A fixed motor/VSD loss plus a cube-law aerodynamic term.

| Model | AHU-1 | AHU-2 | Role |
|---|---|---|---|
| empirical power law `P = a·f^n` | n = 1.42, R² 0.655 | n = 1.73, R² 0.793 | pessimistic DR arm |
| **offset cubic** | **P = 0.719 + 6.199·(f/50)³**, R² **0.903** | **P = 0.537 + 6.806·(f/50)³**, R² **0.960** | **central, headline** |
| ideal cubic `P = b(f/50)³` | — | — | optimistic DR arm |

Source: §1.9 of the data memo; fitted by [src/effects.py](src/effects.py) `fit_fan_models`, written
to `report/fan_models.csv`.

**Why the naive power law comes out near 1.5 rather than 3:** motor and VSD losses are roughly fixed
and the operating band is narrow (p5–p95 = 33.3–40.7 Hz on AHU-1), so a pure power law is poorly
identified over it. **Reporting the ideal cubic alone would overstate savings by roughly 2×** — which
is why all three are carried, and why the fan model is the first axis of the sensitivity tornado.

**Identification status: fitted, well-identified**, on 2,679 / 2,677 coil-on steps. This is the
best-determined block in the surrogate. Its residual is the *smallest* uncertainty the agent faces.

**Extrapolation guard.** The fit is valid over the observed band. The action clipping in §9 exists
partly so the agent cannot drive `f` outside it and collect savings from a polynomial's tail.

---

## 4. Block C — static pressure → delivered airflow

> **CORRECTED TWICE — see [block-c-derivation/README.md](block-c-derivation/README.md).** This block
> has been wrong in two different ways and both corrections are kept visible in §4.7.
>
> **v0, the two-regime switch**, put the block's entire response to a pressure cut in a branch
> occupied on a median of 0.56% (AHU-1) and 2.63% (AHU-2) of a box's on-steps, and asserted an
> elasticity of exactly zero for the other ~97%. The flow setpoint its main branch held is **not a
> logged point**.
>
> **v1, a per-box valve law** `A_i·R_i^(d_i/100)·SP^½`, fixed that bias at the cost of ~103 fitted
> coefficients and an aggregation with no adding-up constraint.
>
> **v2, below, is a share.** Static pressure cancels out of the room allocation entirely, one number
> per room replaces two, and `Σ_i V̇_i = V̇_total` holds by construction.

This block exists because control-gap §5 attributes **64–73% of the modelled pressure-reset saving**
to moving less air rather than to moving the same air more efficiently. If that is real, the
surrogate must model it or the agent will find and monetise it. **What the block now says is that the
stated mechanism for it does not hold** (§4.3): the ventilation cost of a 20% cut is a *band*,
0.1%–10.8%, and which end applies turns on whether the zone flow loops compensate — not on anything
Block C chooses. In the plant's own pressure cuts the dampers *closed* rather than opening, the
opposite of boxes running out of authority.

### 4.1 The box as a variable orifice, and what the record can identify

A pressure-independent VAV box is a damper in series with a flow sensor, under a local controller.
Its hydraulics are an orifice:

```
        V̇_i = C_v(d_i) · √( Δp_box,i )  ,     Δp_box,i ≈ κ_i · SP_act
```

`C_v(d)` is the damper flow coefficient, increasing in `d`; `κ_i` is the fraction of duct static
across box *i*, absorbed into the box's own scale.

**The record cannot identify the shape of `C_v(d)`.** Tested head to head at fixed structure, an
equal-percentage characteristic (`C_v ∝ R^{d/100}`) and a power one (`C_v ∝ d^p`) are
indistinguishable out of sample — median R² 0.557 vs 0.545 on AHU-1, 0.633 vs 0.650 on AHU-2, with
the per-box win split 8/23 and 13/24. Over the observed opening range there is nothing to choose
between them, so the block chooses the one that makes everything else simple: **`C_v` linear in
`d`**.

### 4.2 The share form

```
        split :   V̇_i      =  V̇_total  ·  ( a_i · d_i ) / Σ_j ( a_j · d_j )

        total :   V̇_total  =  C · ( Σ_j a_j·d_j )^q  ·  SP_act^s
```

Three logged points and nothing else: `vav_*__air_flow_rate`, `vav_*__damper_position`,
`ahu_b8_*__static_pressure`.

| Symbol | Is | Count |
|---|---|---:|
| `a_i` | the room's **relative** authority — air drawn per point of damper opening, `Σ a_i = 1` | 23 / 25 |
| `C, q, s` | the total: scale, conductance exponent, pressure exponent | 3 per AHU |

**Static pressure cancels out of the split.** It is common to every box at a given step, so it
divides away. Which room gets what air is *purely* relative damper opening; pressure only sets how
much air there is to divide. Two consequences the per-box law never had:

1. **Adding-up by construction.** `Σ_i V̇_i = V̇_total`, asserted to floating point.
2. **Saturation is relative and automatic.** A box pinned at 100% loses share when its neighbours
   open. No regime, no threshold, no crossover rule, no discontinuity — and no appeal to the
   unlogged flow setpoint.

**The exponent on `d` is 1.** Fitted free it returns 0.899 / 1.091; imposing exactly 1 costs
−0.000…+0.002 of held-out R², so it leaves the specification entirely.

**Identification — the part that got easier.** `a_i` comes from a within-timestep regression,
`ln V̇_i,t = ln a_i + p·ln d_i,t + τ_t`, where the time effect `τ_t` absorbs `SP_act` completely
along with weather, hour and the total. v1 had to *impose* its pressure exponent because the damper
moves in response to pressure — fitted free it absorbed the controller and drove implied rangeability
negative on 10 of 25 and 8 of 26 boxes. **That endogeneity cannot reach the split**, because pressure
is differenced out across rooms at a single instant. It has not gone away; it is confined to the
total, which is where §4.3 reports the model is weak.

**Head to head against v1** ([block-c-derivation/data/head_to_head.csv](block-c-derivation/data/head_to_head.csv)),
both fitted and scored on the same rows at each of three holdout origins: the share form wins R²(ln V̇)
at **every origin on both AHUs** — 0.945/0.934/0.959 against 0.935/0.909/0.949 (AHU-1) and
0.903/0.901/0.905 against 0.853/0.869/0.814 (AHU-2) — with half the parameters. It **loses** on median
per-room percentage error, 5.4% against 3.6% (AHU-1) and 9.2% against 5.7% (AHU-2), because v1 spends
a second parameter per room on exactly that. Both are stated; the choice rests on identification and
structure, not a clean sweep.

### 4.3 Aggregation, exclusions, and the ventilation constraint

Aggregation is no longer a modelling step — it is the definition. `V̇_total` is the model's own
input to the split, and the split's shares sum to 1, so the floor total is whatever the total
equation says it is.

**The total is the weak half of this block, and it should be read as such**
([block-c-derivation/data/total_fit.csv](block-c-derivation/data/total_fit.csv)):

| | AHU-1 | AHU-2 |
|---|---:|---:|
| `q` (aggregate conductance) | 0.886 | 1.501 |
| `s` (pressure) | **0.512** | 0.724 |
| R² | **0.424** | **0.639** |
| R², static pressure alone | 0.154 | 0.002 |

Against R² 0.94 / 0.90 for the split. AHU-1's `s = 0.512` landing on the orifice ½ from a completely
different direction than v1's imposed value is worth noting. **Static pressure alone explains
essentially nothing** — it only signs once aggregate damper opening is held fixed, because pressure
and demand move together on this floor. `q ≠ 1` on either AHU and AHU-2's 1.501 has no physical
reading.

**What a 20% cut costs in air — a band, not a number**
([block-c-derivation/data/pressure_cut_counterfactual.csv](block-c-derivation/data/pressure_cut_counterfactual.csv)):

| | AHU-1 | AHU-2 |
|---|---:|---:|
| **Frozen dampers** — no compensation | **10.79%** | **14.92%** |
| **Compensating** — loops restore the total, clipping at 100% | **0.11%** | **0.15%** |
| **Compensating** — each loop restores *its own* flow (§7.3) | **0.45%** | **1.11%** |
| Damper scaling the loops would need (median) | ×1.137 | ×1.114 |
| Steps where at least one box clips | 793 / 2,681 | 995 / 2,681 |

> **The two compensating rows are the same calculation wherever nothing clips** — verified to 8e−16
> — and differ only in what a box does when a *neighbour* runs out of damper. Restoring the total
> requires boxes with authority left to open further and cover it; restoring your own flow does not,
> and no VAV controller does the former. §7.3 and
> [block-f-derivation/README.md](block-f-derivation/README.md) §4.2 carry the argument. **The band is
> therefore wider than the two rows above it suggested**, which weakens the attribution below rather
> than rescuing it.

Under compensation the loops scale every damper by a common factor, which **leaves the split
untouched** and restores the total almost exactly; the residual is entirely the boxes that clip at
100%. Under frozen dampers everything falls as `SP^s`.

v1 reached the same band (0.3%–10.6%) from the per-box side by a completely different route. **The
choice of Block C formulation is not what determines the ventilation cost of a pressure reset — the
behaviour of the zone flow loops is**, and this record cannot observe it, because pressure and demand
moved together in every episode it contains.

**What the plant's own cuts show.** Over the 8 AHU-1 pressure-cut episodes of rl-sac-feasibility
§2.3 ([block-c-derivation/data/episode_summary.csv](block-c-derivation/data/episode_summary.csv)):

| Median, treated / control | |
|---|---:|
| Static pressure achieved | 0.905 |
| Delivered flow, observed | 0.951 |
| **Share-model allocation error** | **5.3% of total flow** |
| Mean damper position | **45.51 vs 47.16 → −1.65 pts** |
| Share of boxes saturated | **0.0000 vs 0.0000** |

The dampers *closed* and no box was saturated in either window, with the floor 0.89 K cooler — so the
observed drop is largely demand, and **the saturation mechanism is ruled out rather than confirmed.**
rl-sac-feasibility §2.3 concludes from the same episodes that "Block C's pressure-to-airflow
mechanism is real"; the flow ratio it cites is real, the mechanism it attributes it to is not the one
operating.

> **This falsifies the mechanism behind control-gap §5, not its arithmetic.** That document's
> 73% / 64% is a **residual**: modelled saving minus the model-free frontier gap,
> `(190 − 52)/190 = 72.6%` and `(175 − 64)/175 = 63.4%`, attributed to ventilation on the strength
> of "at least one box sits at 100% damper at every occupied step." The subtraction is sound; the
> saturation signature it invokes is absent from the plant's own cuts.
>
> The attribution needs the **frozen-damper end** of §4.3's band (10.8% / 14.9%) to survive. That is
> a coherent position — but it is not the one control-gap §5 argues, and it contradicts §4.1's
> premise that these are pressure-independent boxes with working flow control. Until the band is
> narrowed the gap is **unexplained rather than explained**. Candidates: the fan model (§3 carries a
> 2× band), β (itself [0.241, 0.529] on AHU-1), or the frontier's within-bin construction.
> **§10.1's warning stands with more force, not less** — a large number here is still evidence the
> model is wrong.

Two exclusions, both already in [config.yml](config.yml). They are applied to the panel before the
split is fitted, so no rogue box contributes a room weight:

- **Dead boxes** — flow ratio < 0.10 of the max-flow setpoint (`max_flow_ratio_dead_box`).
  `vav_8_2_28` commands 100% damper and delivers a maximum of 50.5 against a 1,000 setpoint. It
  cannot be summed into a delivered-air total, and it must not be counted as a cooling request.
- **Rogue boxes** — `vav_8_1_28` (one unique damper value, room median 28.9 °C), `vav_8_2_15`,
  `vav_8_2_23`, `vav_8_1_26`, `vav_8_1_16`. Excluded from the *demand* statistic that the agent
  observes (the G36 ignore-top-*I* rule), but **not** from the comfort constraint — a starved zone
  is still a zone.

Hard constraint, per box:

```
        V̇_i  ≥  minimum_air_flow_rate_setpoint_read_i
```

### 4.4 When the fan cannot make pressure

Block A inverts to a required frequency. If it exceeds the VSD limit, the setpoint is not achieved:

```
        f_req = f_ref·(SP_sp'/SP_ref)^β  ;   f = min(f_req, 50)
        SP_act = SP_ref · ( f / f_ref )^{1/β}      when f_req > 50
```

This is not a hypothetical: **AHU-2 falls below 0.8× its setpoint on 12% of steps** (§1.5) and its
SAT loop is saturated on 29.6% of its large-error steps (control-gap §3) — part of AHU-2 is
genuinely short of capacity. A surrogate that always grants the requested pressure would let the
agent write setpoints AHU-2 cannot deliver and then bank the comfort that never arrived.

### 4.5 Falsification — the specific-fan-power consistency test

Blocks A and C are independent projections (§1.1), so they can be asked to agree. Their implied
specific fan power

```
        SFP  =  P_fan / V̇_total          [kW per unit flow]
```

must land inside the observed SFP distribution within each airflow bin — the same binned quantity
[src/benchmarks.py](src/benchmarks.py) already computes for the demonstrated-achievable frontier
(control-gap §5). If a simulated pressure cut produces an SFP the plant has never exhibited, the two
blocks disagree and the surrogate is wrong. This is the sharpest single check in the document,
because it tests the *interaction* rather than either block alone.

### 4.6 The prerequisite this block is blocked on

> **The VAV airflow unit is not established.** The water side was discriminated in §1.2 of the data
> memo (`flow` L/min, ΔT °F → kW, with every rival hypothesis 1.8× to 59× out). **No equivalent test
> has been run on the air side.** A 1,000 max-flow setpoint is plausible as CFM (≈ 472 L/s), as
> m³/h (≈ 278 L/s), or as L/s (implausibly large for one box, but it must be *shown*, not assumed).

The discriminating equation is the air-side energy balance against the already-validated water side:

```
        Q_air  =  ρ·c_p · V̇_total · (T_ma − T_sa)   ≡   Q_water  =  k · ṁ_w · ΔT_w
        ρ·c_p ≈ 1.21 kJ/(m³·K)  at 20 °C            k = 0.0387593 kW/(L/min·°F)
```

The candidate units differ by 1.7× and 3.6×, so the same argmin-plus-margin test used for the water
side has power here. **Run this before Block C's absolute numbers are quoted anywhere.** Ratios
within Block C (§4.2) are unit-invariant and are safe in the meantime; totals and the ventilation
constraint are not.

**§4.2 narrows this further.** The share form is unit-invariant twice over: the `a_i` are normalised
to sum to 1, so **shares are dimensionless**, and multiplying every flow reading by a constant moves
only `C` in the total. The unresolved unit now touches **one scalar per AHU** and nothing in the room
allocation at all. It still blocks absolute totals.

### 4.7 Superseded, kept visible

Both earlier formulations are retained so the corrections can be audited. **Neither is the
specification; §4.2 is.**

#### 4.7.1 v1 — the per-box valve law

```
        V̇_i(t)  =  A_i · R_i^( d_i(t)/100 ) · SP_act(t)^s              s ≡ ½ , imposed
```

Two fitted parameters per box, ~103 for the floor: `A_i` a flow scale, `R_i = e^{γ_i}` an installed
rangeability (median 3.09 / 6.11, inside the engineering band [2, 12]). `s` had to be **imposed**,
because fitting it alongside the damper term let the controller into the pressure exponent and drove
γ negative on 10 of 25 and 8 of 26 boxes. Its two survivors with γ < 0 under the imposed exponent
were `vav_8_1_26` and `vav_8_2_23`, both already on config's rogue list — the model recovered the
hand-named exclusions mechanically, which was the strongest evidence for it.

**Why it was replaced:** ~103 coefficients where 26 do better on R²; no adding-up constraint, so the
per-box fits summed to nothing in particular; and an imposed exponent that the share form does not
need, because pressure differences out across rooms at a single instant (§4.2). It remains the
reference the share form is scored against at every holdout origin, and it **still wins on median
per-room percentage error** — 3.6% against 5.4% (AHU-1), 5.7% against 9.2% (AHU-2).

#### 4.7.2 v0 — the two-regime switch

> **Regime PI — pressure-independent** (`d_i < 100%`). The box holds its flow setpoint; the damper
> opens to absorb the pressure cut:
> ```
>         V̇_i'  =  V̇sp_i
>         d_i'   ≈  d_i · ( SP_act / SP_act' )^{1/2}
> ```
> *Stated approximation:* the damper reposition uses `C_v` linear in `d` over the mid-range. Real
> damper characteristics are closer to equal-percentage, so this is first-order. It is acceptable
> because `d_i'` is used **only to detect saturation**, never to compute a flow.
>
> **Regime PD — pressure-dependent** (`d_i = 100%`). The box is a fixed orifice:
> ```
>         V̇_i'  =  V̇_i · ( SP_act' / SP_act )^{1/2}
> ```
>
> **The crossover, in closed form:**
> ```
>         d_i · ( SP_act / SP_act' )^{1/2} ≥ 100      ⇔      SP_act'  ≤  SP_act · ( d_i / 100 )²
> ```
> So a box sitting at 80% damper joins the pressure-dependent set once the setpoint drops below
> 0.64× its current value; one at 95% joins at 0.90×. With AHU-1's mean damper at 52.6% and AHU-2's
> at 59.7% but 11.1% / **21.5%** of box-timesteps already at ≥90%, a 20% cut pulls a substantial and
> *computable* population across the line. That population is the honest cost of the saving.

**Why it was replaced**, in the order the objections bite
([block-c-derivation/README.md](block-c-derivation/README.md) §2):

1. The PD branch carries the block's entire pressure response and is occupied on a median of
   **0.56% / 2.63%** of a box's on-steps; **10 of 25** clean AHU-1 boxes never enter it. §2.2's
   supporting statistic is the p10 of the per-step **maximum** damper — a claim that *some* box is
   open, silently read as a per-box regime assignment.
2. The PI branch asserts an elasticity of **exactly zero** — an infinitely stiff controller — on a
   floor whose loops demonstrably are not stiff (§6.1's −1.02 K SAT bias; §4.4's 12% of steps below
   0.8× setpoint).
3. `V̇sp_i` **is not logged.** The PI branch is an equation in an unobservable.
4. `d' = d·(SP/SP')^½` is disclaimed as "saturation detection only" — but detection *is* regime
   assignment, and regime assignment *is* the flow. The approximation is load-bearing exactly where
   it is waved off. **(An earlier revision of this list also called the rule's implied linear `C_v`
   a contradiction of §4.1's orifice law. That was overstated and has been withdrawn: §4.1 now
   records that this record cannot distinguish a linear damper characteristic from an
   equal-percentage one, and v2 uses the linear one.)**
5. A 15-minute average cannot resolve a hard threshold on damper position. §0's Nyquist argument
   applies here and was not applied.
6. The crossover table is a restatement of (4), presented as measured population.

**One caveat survives both replacements and is not resolved.** Every way of asking this record for a
pressure exponent gives something above ½ on AHU-2 and near ½ on AHU-1: v0's saturated-only branch
returned **0.657 / 0.886**, and v2's total (§4.3) returns **0.512 / 0.724** from an entirely
independent regression. Either the box is not a fixed orifice, or `κ_i` is not constant as the fan
slows — AHU-2, which is short of capacity, is worse under both readings, as that explanation
predicts. **`s` is therefore carried as a DR axis over [0.5, 0.9]** — see §8.2.

---

## 5. Block D — zone thermal response

> **CORRECTED — see [grey-box-technique.md](grey-box-technique.md).** The single-node form below,
> coupled to outdoor air, is mis-specified on this floor. Checked against the CSV on 8 Sep 2026:
> the zones settle overnight to an internal mass at ≈ 24.7 °C, 4 K below outdoor, with τ ≈ 2.5 h;
> the outdoor coupling `γ_i` fits to zero on 40 of 50 boxes; and the one-step fit explains a third
> of the variance that a two-node air-plus-mass model does. Replace §5.1–5.4 with the two-node form
> and the two-stage identification in grey-box-technique §4. The endogeneity mitigations in §5.4
> are superseded by fitting the air-to-mass coupling from the fan-off window, where no controller
> runs. The section is kept as written below so the correction stays visible.

### 5.1 Lumped first-order RC

Each zone is one capacitance, cooled by supply air and coupled to outdoors:

```
        C_i · dT_i/dt  =  ρ·c_p·V̇_i · (T_sa − T_i)  +  (T_oa − T_i)/R_i  +  q_i(t)
```

### 5.2 The identifiable discrete form

Absolute `C_i` is not needed and not identifiable; divide through and discretise at Δt:

```
   T_i(t+1) − T_i(t)  =  α_i · V̇_i(t)·(T_sa(t) − T_i(t))
                       +  γ_i · (T_oa(t) − T_i(t))
                       +  δ_i · occ(t)
                       +  ε_i
```

Four parameters per zone, **linear in the parameters**, so it is ordinary least squares — with
`α_i, γ_i ≥ 0` imposed because they are conductances divided by a capacitance and a negative value
is not a fit, it is a defect indicator. **Non-negative least squares**, therefore, not OLS.

Implied time constant:

```
        τ_i  =  1 / ( α_i·V̇_i + γ_i )
```

Expected 1–3 h. **τ_i ≫ Δt = 15 min is exactly why this block is identifiable at this sampling rate
while the inner loops of §0 are not.** If a fitted τ_i comes out below ~30 min, that zone's fit is
reporting noise and must be pooled rather than used.

### 5.3 Identification procedure

1. Training window only (18 May → 29 Jul; the test month, 3 Aug → 23 Aug, is held back — the split
   falls on the 4-day collector outage, [rl-environment-design.md](rl-environment-design.md) §5.1).
2. Mask via [src/masks.py](src/masks.py) `analysis_mask`, then
   `mask_room_temp_sentinels` — the `0.00` reading on every AHU-1 box is a bad-read sentinel, not a
   room at freezing point (minimum non-zero is 6.49 °C). Averaging it drags every zone statistic
   down (§1.7).
3. NNLS per zone. Report `α_i, γ_i, δ_i, ε_i`, τ_i, R², and the design matrix condition number.
4. **Pool the dead zones.** The twelve boxes parked at 27.0 °C — `vav_8_1_{1,2,3,4,5,10,11,14}`,
   `vav_8_2_{19,20,21,22}` — essentially never call for cooling, so `V̇_i` barely varies and `α_i`
   is unidentified. Fit them as one shared "inactive zone" and say so, rather than reporting 12
   confident numbers derived from no excitation.

### 5.4 The endogeneity problem, stated not solved

> `V̇_i(t)` is chosen by the zone controller *in response to* `T_i(t)`. The regressor is not
> exogenous, and `α_i` is therefore biased.

The direction is knowable: a warm zone receives more air, so high `V̇_i` co-occurs with high `T_i`,
which pushes `α̂_i` **toward zero** — the surrogate will understate how much a zone cools per unit
of air. Consequences, both conservative in the right direction:

- an agent that cuts airflow will see zone temperatures rise *less* than reality → the comfort
  constraint binds later than it should → **savings are over-stated, comfort risk under-stated.**

That is the wrong direction for safety, so it is not enough to note it. Two mitigations, both cheap:

1. Using `V̇_i(t)` to predict `T_i(t+1)` (a one-step lead) gives partial protection, since the
   controller cannot react to a temperature it has not yet measured.
2. **The open-loop test in §8.2 is the arbiter, not the fit statistic.** A biased α shows up as
   drift in a multi-step rollout even when the one-step R² looks fine.

A stronger fix — instrumenting `V̇_i` with the *duct pressure* variation, which moves flow without
being caused by zone temperature — is available precisely because the setpoint episodes of §2.2
exist. It is the right upgrade if §8.2 fails.

### 5.5 What the fit must reproduce

The record's headline comfort finding is that **AHU-1 zones average 2.60 K below setpoint and sit
more than 1 K below on 50.3% of on-hours steps** (§1.6). A fitted zone model that cannot reproduce
this under replayed inputs has not captured the floor's thermal behaviour, whatever its R².

---

## 6. Block E — supply air temperature and coil load

### 6.1 E₁ — the SAT loop is modelled as biased, not as perfect

The obvious modelling choice, `T_sa = SAT_sp`, is wrong and expensive. The loop does not hold its
setpoint: mean tracking error is **−1.02 K on AHU-1** and −0.22 K on AHU-2 over the masked
evaluation set, with the authority diagnostic showing 574 of 588 large-error steps at a *mid-range*
valve — **mis-tuned, not starved** (control-gap §3). Assuming perfect tracking silently hands the
agent credit for a commissioning fix it never performed.

Because the loop's own time constant is below Nyquist (§0), the honest form is quasi-steady-state
with a load-dependent bias rather than a lag:

```
        T_sa(t)  =  SAT_sp(t)  +  b₀ + b₁·valve(t)  +  η_t   ,    η ~ N(0, σ²)
```

`b₀, b₁, σ` fitted by OLS on masked on-hours steps. The linear-in-valve term is there because the
error is not a constant offset — its p5/p95 spans −4.92 / +0.43 K on AHU-1 — and a single mean would
misplace the bias exactly where it matters, at high load.

**A second DR arm sets `b₀ = b₁ = 0`, labelled "SAT loop retuned first."** Keeping the two arms
separate is what lets a result distinguish *"reset the setpoint"* from *"fix the loop, then reset the
setpoint"* — the distinction §1.5 of the data memo insists on.

### 6.2 E₂ — mixing and coil duty

```
        T_ma   =  φ·T_oa + (1 − φ)·T_ra
        Q_req  =  ρ·c_p · V̇_total · (T_ma − T_sa)          [kW]
```

`φ` is the outside-air fraction. `ahu_b8_*__fresh_air_damper_position_read` is a logged point (909
unique values, 0–100% on AHU-1 — control-gap §1), but **damper position is not linear in flow
fraction**, so φ is not read off it directly. Fit an affine map `φ = φ₀ + φ₁·damper%` on the training
window, constrained to [0, 1], and treat `φ₀, φ₁` as fitted parameters.

Mixed-air temperature is not logged, so this map is identified *through* the energy balance rather
than against a direct measurement — which makes §6.3 a genuine out-of-sample test rather than a
tautology, provided the fit uses only the training window.

### 6.3 Falsification — coil closure against the validated water side

`Q_req` must reconcile with the **measured** `cooling_rate`, whose units are already established
beyond reasonable doubt (§1.2: `k = 0.0387593` kW per L/min·°F, with the nearest rival hypothesis
1.8× out and the rest 3.6× to 59× out):

```
        Q_req  ≈  Q_measured  =  k · ṁ_w · ΔT_w
```

evaluated on **test-window** replayed steps. This is a falsification, not a calibration: the
parameters were fixed on the training window before this comparison is made. It simultaneously
tests Block C's total (§4.6), Block E₁'s bias model, and the mixing map.

### 6.4 Thermal → electric, and why it stays outside the reward

```
        E_chiller  =  Q · Δt / COP        COP ∈ [3, 5], domain-randomised only
```

**No chiller COP exists in this dataset** — the plant is off-floor and unmetered (§1.9). The coil is
4.4× the fan thermally (18,732 kWh-th vs 4,297 kWh-e), so this is the largest prize and the least
knowable conversion. It is reported in `info` as kWh-thermal and appears in the electric objective
only through DR, never as a constant folded into a scalar reward. Any headline that depends on a
particular COP is an assumption wearing a measurement's clothes.

---

## 7. Block F — the zone controllers (the inner loop the agent does not command)

> **CORRECTED — see [block-f-derivation/README.md](block-f-derivation/README.md).** The form below
> was wrong in two ways and the block was also half-missing. Checked against the CSV on
> 10 Sep 2026:
>
> 1. **It stopped at stage 1.** The plant runs two stages; this section specified one. It ended at
>    `V̇sp_i`, §4.2 begins at `d_i`, and nothing joined them — so `d_i` was consumed by Block C with
>    no producer anywhere in the §1.1 step order, and the loop never closed.
> 2. **`V̇sp_i` is not a logged point.** Each box logs seven, and a commanded flow setpoint is not
>    among them. The identification recipe was a regression on an unobservable — verbatim the
>    objection §4.7.2(3) used to retire Block C v0.
> 3. **The increment form is sub-Nyquist**, which §0 forbids, and it fails on its own terms: fitted
>    on this record it returns the wrong sign on **18 of 19** AHU-1 zones (median K = −16.8).
>
> §7.1–7.3 below are the replacement. The superseded form is kept at §7.4 so the correction stays
> visible.

The agent writes AHU setpoints. The 53 zone loops keep running underneath, and they must be
simulated or the zones never respond to anything.

### 7.1 Stage 1 — error to flow setpoint, as a level

```
        V̇sp_i(t) = clip( A_i + G_i·(T_i(t−1) − Tsp_i(t−1)) ,  V̇min_i ,  V̇max_i )
```

A **level**, not an increment, and the word carries the specification. §0's Nyquist argument says the
damper loop settles inside a 15-minute step, so what the record can see is the static map from error
to flow — not the integration that produced it. The same argument makes the block identifiable:
a settled box sits *at* its setpoint, so `V̇_i ≈ V̇sp_i` and the unlogged variable is observable
through logged flow.

`A_i`, `G_i` fitted per zone by OLS on masked steps, lagged one step so the controller cannot respond
to a temperature it has not yet measured (the §5.4(1) protection). `V̇min_i`, `V̇max_i`, `Tsp_i` are
logged points, and `Tsp_i` is replayed rather than modelled — including the twelve zones parked at
27.0 °C, which stay parked.

| | AHU-1 | AHU-2 |
|---|---:|---:|
| zones identified | 19 / 26 | 22 / 27 |
| **`G_i` > 0** (a warm zone asks for more air) | **19 / 19** | **21 / 22** |
| median `G_i` | 12.87 | 32.17 |
| median `A_i` — the parked level | 235.1 | 237.7 |
| median R² | 0.050 | 0.094 |

**The low R² is a measurement, not a failed fit.** §5.5's floor sits 2.60 K *below* setpoint, so
`G·e` contributes about −33 / −84 against a parked `A` of ~236: **most of each box's flow is its
floor, not its controller.** These loops are barely modulating, which is §5.5 seen from the flow side.

**The authority exclusions are identification, not hygiene.** `V̇_i ≈ V̇sp_i` holds only off the
stops, so the fit drops steps with `d_i ≥ 99%` or flow within 2% of the box's own min or max, at both
`t` and `t−1`. They apply to the gain fit alone — never to the comfort constraint (§4.3, "a starved
zone is still a zone") and never to the counterfactual, where clipping is the effect being measured.

### 7.2 Stage 2 — flow setpoint to damper, by inverting Block C

```
        d_i  =  V̇sp_i · W / ( a_i · V̇sp,total )   ,     W = ( V̇sp,total / (C·SP_act^s) )^(1/q)
```

Closed form: Block C's split inverts analytically, and in ratio form the scale constant `C` cancels
along with `SP_act` itself — only the *ratio* of new to old pressure enters. **Stage 2 therefore
costs no new parameters and touches neither the total's weak R² (0.42 / 0.64) nor §4.6's unresolved
airflow unit.** Boxes that would need to pass 100% clip there; the iteration exists only to
redistribute after a clip.

Verified by round trip: asked for the flows the plant already had at unchanged pressure, the
inversion returns the dampers the plant was sitting at, to 0.0 and 2.8e−14 (§3.3 of the derivation).

### 7.3 λ — how much the loops actually compensate

The two stages above describe loops with full authority. Whether the real ones behave that way is
what §4.3 says this record cannot show, so it is carried as a **domain-randomisation axis**:

```
        d_i(λ)  =  d_i  +  λ·( d_i^compensating − d_i )        λ ~ U(0, 1)
```

λ = 0 freezes the dampers (§4.3's 10.79% / 14.92%), λ = 1 restores the requested flows. This is the
same treatment §8.2 already gives β and `s`, and it is what turns §4.3's band from an unstated
modelling choice into a measured training distribution.

> **"Compensating" has two readings, and §4.3 quoted the generous one.** Its 0.11% / 0.15% assumes
> the loops restore the *floor total*, which requires boxes with authority left to open further and
> cover boxes that have run out. **No VAV controller does that** — each box tracks its own flow
> setpoint, which is §4.1's pressure-independent premise taken literally. Under that reading the same
> 20% cut costs **0.45% (AHU-1) and 1.11% (AHU-2)**. The two are provably identical wherever nothing
> clips (checked to 8e−16); the whole divergence is the clipping steps.
>
> This **widens** the band, so it weakens control-gap §5's attribution further rather than rescuing
> it — §10.1's "unexplained" verdict stands with more force. Neither reading is settled, because the
> record contains no step where a box demonstrably ran out of authority (§4.3: saturated share
> 0.0000 in both windows).

**What the plant's own cuts say.** Over the seven AHU-1 episodes where achieved pressure actually
fell (median ratio 0.896), the dampers moved **−1.08 pts** while stage 2 predicts **+2.98**. That
does not falsify the block — it confirms those episodes are demand moves, the floor being 0.89 K
cooler in the treated windows, which is precisely why they cannot identify λ. Every AHU-2 episode
fails the filter: §2.2's note that 93% of them sit at 08h means duct static is *climbing* there while
the setpoint steps down.

### 7.4 Superseded, kept visible

```
        V̇sp_i(t+1) = clip( V̇sp_i(t) + K_i·(T_i(t) − Tsp_i) ,  V̇min_i ,  V̇max_i )
```

"`K_i` identified per zone by regressing `ΔV̇sp_i` on the zone temperature error over masked steps."
Retired for the three reasons in the banner. The sign reversal is mechanical rather than noise: the
controller has already put `e(t−1)` into `V̇(t−1)`, so `ΔV̇ = V̇(t) − V̇(t−1)` inherits that level's
negative. Both forms are fitted on identical rows and both ship in
[block-f-derivation/data/zone_gain_fit.csv](block-f-derivation/data/zone_gain_fit.csv).

> **This block is still inferred, not observed.** §6 of [hvac-system-logical-flow.md](hvac-system-logical-flow.md)
> flags that the internal algorithm of the box controllers is not stated anywhere in the data;
> "PI on room temperature producing a flow setpoint, damper chasing flow" is the standard
> pressure-independent arrangement and is consistent with the logged points, but it is an inference.
> What has changed is that both stages are now written down, identified from logged points, and
> falsifiable. It remains listed in §10 as inferred.

---

## 8. Parameter inventory and uncertainty

### 8.1 Every parameter, with its provenance

| Block | Parameter | Provenance | Value / range |
|---|---|---|---|
| A | β (SP→Hz elasticity) | **measured** (control-gap §4.2) | 0.382 [0.241, 0.529] · 0.232 [0.178, 0.283] |
| B | a, b (offset cubic) | **fitted**, R² 0.903 / 0.960 | 0.719, 6.199 · 0.537, 6.806 |
| B | n (power law arm) | fitted, R² 0.655 / 0.793 | 1.42 · 1.73 |
| C | `a_i` (room share weight) | **fitted**, within-timestep; pressure absent by construction (§4.2) | 23 + 25, normalised `Σ a_i = 1` |
| C | `p` (damper exponent in the share) | free fit 0.899 / 1.091; **imposed at 1**, cost < 0.003 held-out R² | 1, not fitted |
| C | `C, q, s` (the total) | **fitted**; the weak half — R² 0.42 / 0.64 (§4.3) | 3 per AHU; `s` = 0.512 / 0.724 |
| C | `C_v` linear in `d` | **chosen, not measured** — the record cannot tell linear from equal-percentage (§4.1) | — |
| C | dead / rogue box sets | **measured** (§1.6), applied to the panel before fitting | 1 dead + 5 rogue, named |
| C | airflow unit | **UNRESOLVED** (§4.6) | confined to `C`; shares are dimensionless |
| D | α_i, γ_i, δ_i, ε_i | **fitted** per zone, NNLS | 53 × 4, 12 pooled |
| E₁ | b₀, b₁, σ | **fitted** | anchored on −1.02 K / −0.22 K mean error |
| E₂ | φ₀, φ₁ | **fitted** through the balance | φ ∈ [0, 1] |
| E₂ | k (water-side constant) | **measured, discriminated** (§1.2) | 0.0387593 kW/(L/min·°F) |
| E | COP | **absent from the dataset** | DR only, U(3, 5) |
| F₁ | A_i, G_i (stage 1, a **level**) | **fitted** per zone, OLS; structure **inferred** | 2 × 41 identified, 12 unidentified → pooled |
| F₂ | stage 2 (damper chases flow) | **inversion of Block C** — no free parameters, `C` cancels (§7.2) | — |
| F | λ (loop compensation) | **not identified** — the record cannot separate it from a demand move | DR only, U(0, 1) |

### 8.2 Domain randomisation — the sensitivity axes *are* the training distribution

Sampled per episode:

| Axis | Distribution | Source |
|---|---|---|
| β | measured interval, or the 36-specification empirical spread | control-gap §4.3 |
| **`s` (Block C pressure exponent)** | **U(0.5, 0.9)** — the total fits 0.512 / 0.724, v0's saturated branch 0.66 / 0.89 | §4.3, §4.7 |
| **λ (zone-loop compensation)** | **U(0, 1)** — λ=0 costs 10.79/14.92%, λ=1 costs 0.45/1.11% of delivered air | §7.3, block-f §4 |
| fan model | {power law, offset cubic, ideal cubic} | §1.9 |
| COP | U(3, 5) | not in data |
| SAT bias | {fitted b₀,b₁} or {0, 0} = "loop retuned first" | control-gap §3 |
| ignore-*I* | {1, 2, 3} | §1.6 |
| rogue set | perturbed by ±1 box | §1.6 |

These are the five tornado axes proposed in §2.7 of the data memo, plus Block C's `s` (§4.7) and
Block F's λ (§7.3). **The sensitivity
analysis and the training distribution become the same object** — a policy trained across them is
robust *to the uncertainty that was actually measured*, and the tornado chart is then a description
of the environment the agent was trained in rather than a post-hoc afterthought.

---

## 9. Action support — the clipping rule and why it is not a detail

```
        SP_sp'  = clip( SP_sp + ΔSP ,  0.30 , 0.64 )    AHU-1        rate |ΔSP| ≤ 0.02 inWG / step
                  clip( SP_sp + ΔSP ,  0.40 , 0.78 )    AHU-2
        SAT_sp' = clip( SAT_sp + ΔSAT , 17.50 , 18.77 )    AHU-1     rate |ΔSAT| ≤ 0.3 K / step
                  clip( SAT_sp + ΔSAT , 16.00 , 19.00 )    AHU-2
```

The bounds are **the observed ranges of the logged setpoints**, not engineering limits: 122 distinct
values spanning 0.300–0.636 on AHU-1 and 146 spanning 0.400–0.780 on AHU-2 (control-gap §1); for SAT,
141 distinct values spanning 17.50–18.77 on AHU-1 and 198 spanning 16.00–19.00 on AHU-2.

> **AHU-1's SAT lever is nearly degenerate, and the number has to be stated rather than left as
> "observed span."** That span is **1.27 K**. At the 0.3 K/step rate limit an agent crosses the entire
> admissible range in four steps, so any AHU-1 SAT-reset saving rests on 1.27 K of support and should
> be reported with that width attached. Worth deciding deliberately: keep the lever and expect it to
> sit at a bound, or drop it and say why.
>
> Note the contrast that makes §6.1 vivid: AHU-1's *achieved* supply air spans 9.50–27.61 °C against
> that 1.27 K setpoint band. The loop is not tracking a narrow setpoint — it is wandering an order of
> magnitude wider than the thing it is supposed to hold, which is "mis-tuned, not starved" visible as
> raw range, and confirmation that `T_sa = SAT_sp` would be a catastrophic simplification.

Outside that support every block above is extrapolating — β was identified from cuts of ~17–30%, the
fan curve from 33.3–40.7 Hz, the zone models from the flows those setpoints produced. An agent
allowed outside it will find savings in the tails of fitted polynomials and report them as building
physics. The clip is the boundary of what this record can defend.

---

## 10. What this surrogate does and does not establish

**It can support:**

1. **Relative comparison of supervisory policies** under stated, measured uncertainty — is a learned
   reset better than G36 trim-and-respond with ignore-top-2, and by how much.
2. **A ceiling.** What a near-optimal policy captures of the demonstrated-achievable frontier.
3. **Attribution.** How much of any saving is efficiency and how much is ventilation reduction,
   because §4 separates them by construction instead of blending them.

**It cannot support:**

1. **An absolute kWh claim.** The modelled pressure saving (190 / 175 kWh) already exceeds the
   model-free frontier gap (52 / 64 kWh) by more than the stated 2× agreement factor. control-gap §5
   attributes the excess to moving less air (73% / 64%), but **§4.3 shows that number is not
   determined by this record**: the ventilation loss under the same 20% cut is a band — **0.45% to
   10.8% on AHU-1 and 1.1% to 14.9% on AHU-2** (§7.3 widens the compensating end; §4.3 quoted 0.11%
   / 0.15% under a reading in which zone loops cover each other's shortfall, which no VAV controller
   does) — and which end applies turns on **Block F's λ**, something no event in the record separates
   from a demand move. The attribution needs the far end of that band, and §4.3's episode evidence
   points the other way. The excess is therefore **unexplained**, which is a weaker position than the
   original claim, not a stronger one — an unexplained gap between two estimates is exactly the
   condition under which an absolute number must not be quoted.
2. **Anything outside the logged setpoint support** (§9).
3. **Anything relying on sub-15-minute dynamics** — valve hunting (6–7% travel per 15-min average,
   p95 28–31%), loop retuning, or PID gains. The sampling rate forbids it (§0).
4. **A COP-dependent number** as anything but a labelled band (§6.4).
5. **The absolute airflow totals**, until §4.6 is resolved.
6. **Any ventilation-cost number that does not carry λ.** Block F's structure is *inferred*, not
   observed — [hvac-system-logical-flow.md](hvac-system-logical-flow.md) §6 records that the box
   controllers' internal algorithm appears nowhere in the data. Both stages are now written down,
   identified from logged points and falsifiable (§7), but **how far the loops compensate is not
   identified at all**, and it is the single parameter that decides the attribution this document
   cannot otherwise close. It is a DR axis for that reason, not a fitted number.

**And one rule that outranks the rest:** whatever an agent saves inside this surrogate is bounded by
the surrogate's fidelity, not by the building's. The frontier assertion in
[rl-environment-design.md](rl-environment-design.md) §5 exists because a large number here is
evidence the model is wrong, not evidence the saving is real.
