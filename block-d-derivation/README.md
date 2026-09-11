# Block D, derived — where a room's temperature goes next

> **What this is.** The physical and mathematical derivation behind
> [grey-box-surrogate.md](../grey-box-surrogate.md) §5, which currently carries a **CORRECTED**
> banner pointing at [grey-box-technique.md](../grey-box-technique.md) §4 for its replacement. That
> replacement was stated in two documents and *derived* in neither. Block D answers one question:
> given the air a room actually received and the temperature it was delivered at, where does that
> room's temperature go next?
>
> ```
>     T_i(t+1) − T_i(t) = k_i · (T_m,i − T_i(t))              air ↔ internal mass
>                       + a_i · V̇_i(t) · (T_sa(t) − T_i(t))    supply air
>                       + g_i · (T_oa(t) − T_i(t))             outdoor   (fits to zero)
>                       + δ_i · occ(t) + c_i                   gains
> ```
>
> **Run it:** `python block-d-derivation/reproduce_zone_thermal.py` — derives the mass node from the
> night plateau, fits stage 1 three ways, fits stage 2 against the one-node form it replaces,
> verifies two discretisation identities and a synthetic recovery, asserts its guards, and writes
> eleven tables to [data/](data/).
>
> **Four findings up front.**
>
> 1. **The outdoor coupling can be ruled out without any estimator.** The night plateau is a
>    quasi-equilibrium, so the within-night slope of zone temperature on outdoor temperature *is*
>    `g/(k+g)`. It is **0.0057**. grey-box-technique reaches "g is zero" from where a constrained
>    estimator parks; §2.3 below reaches it from one regression with no structure in it at all.
> 2. **grey-box-technique §2.3's τ of 2.5 h is an artifact, and §5.1 reproduces it on demand.** It
>    comes from taking `T_m` as the *previous* night's plateau in a window where the zone is settling
>    into *tonight's*. That attenuates `k` by ~2.2×. Corrected, **τ = 0.94 / 1.00 h**, not 2.5 h.
> 3. **The 15-minute average — the record's worst defect — provably does not bite here.** Block
>    averaging a first-order system preserves its pole *exactly* when the input is held across the
>    step (§3.3). That is a stronger reason to identify `k` in the fan-off window than "no controller
>    is running", and it is why this block is identifiable at a sampling rate that forbids modelling
>    any PID.
> 4. **The two-node form beats the one-node form pooled, and loses on AHU-2.** 3.01× on the floor
>    median — close to technique's 3.3× — decomposing into **4.82× on AHU-1 and 0.83× on AHU-2**.
>    §5.2 states this rather than shipping the pooled number alone.

---

## 1. What this replaces

[grey-box-surrogate.md](../grey-box-surrogate.md) §5.1 models each zone as a single capacitance
coupled to supply air and to **outdoors**:

```
        C_i · dT_i/dt  =  ρ·c_p·V̇_i · (T_sa − T_i)  +  (T_oa − T_i)/R_i  +  q_i(t)
```

That structure was called wrong in [grey-box-technique.md](../grey-box-technique.md) §2.3 on the
strength of three numbers — a night plateau 4 K below outdoor, `γ_i` fitting to zero on 40 of 50
boxes, and a one-step R² one third of the two-node form's. The correction is right. What has been
missing is the derivation: where the two-node form comes from, which of its terms the record can
identify and which it cannot, and what a 15-minute average does to a 1-hour time constant. This
document is that, and in the course of writing it two of technique's own numbers turned out to need
correcting in turn (§5.1, §4.7).

---

## 2. The physics

### 2.1 The advection term is exact, and the assumption is elsewhere

Take the zone air as a control volume. Supply air enters at `T_sa` at mass rate `ṁ = ρ·V̇_i`; the same
mass leaves as return. The net enthalpy flux is therefore

```
        Q̇_air  =  ṁ·c_p·(T_sa − T_return)  =  ρ·c_p·V̇_i·(T_sa − T_i)
```

The second equality is the only assumption, and it is **`T_return = T_i`** — the zone is well mixed,
so air leaves at the temperature the sensor reads. There is no approximation in the `(T_sa − T_i)`
*form*; a badly mixed zone with short-circuiting supply air breaks the identification of `a_i`, not
the algebra. Since the room sensor is one point in the space, `a_i` absorbs any systematic difference
between what the sensor reads and the mixed-mean temperature. That is a real limitation and it is
invisible in every fit statistic in this document.

### 2.2 The fast node is not the air

The two-node structure needs a reason beyond "it fits better". Here it is: **the observed time
constant is far too long for zone air.**

Take a 50 m² zone at 3 m — 150 m³, about **181 kJ/K** — coupled to the slab through a combined
surface coefficient of order 4–12 W/m²K over 50 m² ([data/capacitance_check.csv](data/capacitance_check.csv)):

| Assumed surface coefficient | 4 W/m²K | 8 W/m²K | 12 W/m²K |
|---|---:|---:|---:|
| Conductance | 200 W/K | 400 W/K | 600 W/K |
| τ of an air-only node | **15.1 min** | **7.5 min** | **5.0 min** |
| Capacitance ratio needed to reach the measured τ (0.98 h, floor median) | ×3.9 | ×7.8 | ×11.7 |

Every input is a round number and is labelled as one; the conclusion is a **factor, not a value**. An
air-only node settles in minutes. The record settles in about an hour (§5.1). So the node the fit
calls "the zone" carries roughly 4–12× the air's capacitance, which is furniture, partitions, screed
and the interior surface layer — the mass that responds within the hour. The slab and structure
respond over days, which is the *second* node, and §2.4 shows it moves slowly enough to be treated as
an input rather than a state.

This is what makes the split physical rather than a curve-fitting convenience: the two nodes are two
measured time scales, an hour and a week, separated by more than two orders of magnitude.

### 2.3 There is no outdoor path, and it takes one regression to show it

With the fan off and gains near zero, the zone relaxes to an equilibrium that is a
conductance-weighted average of the two temperatures it is coupled to:

```
        T_eq  =  (k·T_m + g·T_oa) / (k + g)        ⇒        dT_eq/dT_oa  =  g / (k + g)
```

**The night plateau is that equilibrium, and outdoor temperature keeps falling through it** — 29.0 →
27.9 °C across the night while the zones sit flat at 24.7 °C. So regressing plateau zone temperature
on outdoor temperature *within* a night — night means removed from both sides, which sweeps out the
mass node's own night-to-night movement and the seasonal drift that would otherwise dominate — measures
`g/(k+g)` **directly**. No NNLS, no imposed structure, no fitted model
([data/outdoor_share.csv](data/outdoor_share.csv)):

| | AHU-1 | AHU-2 | Floor |
|---|---:|---:|---:|
| Within-night slope `dT_eq/dT_oa` = **`g/(k+g)`** | **0.0060** | **0.0055** | **0.0057** |
| Correlation | 0.059 | 0.095 | 0.080 |
| Within-night sd of outdoor temperature | 1.45 K | 1.45 K | 1.45 K |
| Steps · nights | 3,133 · 88 | 3,060 · 86 | 3,060 · 86 |
| Implied `g`, given `k` from §5.1 | 0.0014 | 0.0012 | 0.0013 |
| **Implied outdoor time constant** | **7.4 days** | **8.5 days** | **8.0 days** |

Outdoor air sets **0.6%** of the zone's equilibrium. There is 1.45 K of within-night outdoor
variation to drive the regression with, so this is a measurement rather than a null result for want
of signal.

**Carried through to a conductance, it implies an outdoor time constant in *days*** — and that is the
physical reading of the whole finding. No real façade has a multi-day time constant. What has one is a
zone with **no façade at all**: Floor 8 sits between conditioned floors, and most of these 53 boxes
are interior. The specification's `(T_oa − T_i)/R_i` was not a mis-estimated term. It was a term for a
heat path most of these zones do not have.

### 2.4 The mass node is slow enough to replay rather than estimate

[data/mass_node.csv](data/mass_node.csv), 88 nights, reproducing grey-box-technique §2.2 closely
enough to confirm the two documents mask the record the same way:

| | This document | technique §2.2 |
|---|---:|---:|
| Plateau mean, sd | **24.76 °C, 0.54 K** | 24.75 °C, 0.54 K |
| Range | 23.61 → 26.11 °C | 23.6 → 26.1 °C |
| Correlation with night outdoor temperature | 0.345 | 0.34 |
| Lag-1 autocorrelation | 0.496 | 0.48 |
| Weekend drift, Fri night → Sun night | **+1.31 K** | +1.4 K |
| Weekday cooling | −0.17 K/day | −0.15 K/day |

That agreement matters for a reason beyond bookkeeping: it means the τ disagreement of §5.1 is **not**
a masking difference. The two documents see the same nights and the same plateau, and still disagree
about `k` by 2.4×.

The mass moves **0.65 K per uncooled day** and under 0.3 K within one 44-step occupied episode. So
within an episode it is a constant, and `T_m,i` — the mean of zone *i*'s own reading over 21:00–05:00
— can be **replayed as an exogenous input** alongside weather and occupancy. This is the structural
choice that makes the two-node model cost almost nothing: the slow state is *measured once a day*, so
it never has to be estimated, and the model stays linear in its parameters.

**τ_m is reported as a rate, not a time constant.** Turning 0.65 K/day into `τ_m` needs the gradient
driving it — the air temperature the mass is relaxing toward — and this record never lets the floor
free-run long enough to observe it. grey-box-technique §4.1 asks for `τ_m` fitted from the weekend
sawtooth; the sawtooth gives the numerator and not the denominator.

---

## 3. From the continuous equation to the fitted one

### 3.1 What dividing by capacitance costs, and what it hides

The two-node energy balance on the fast node is

```
        C_i · dT_i/dt  =  (T_m − T_i)/R_m,i  +  ρ·c_p·V̇_i·(T_sa − T_i)  +  (T_oa − T_i)/R_oa,i  +  q_i(t)
```

`C_i` is not identifiable and is not needed. Divide through by it and every parameter becomes a
*ratio*:

```
        k_i = Δt/(R_m,i·C_i)      a_i = Δt·ρ·c_p/C_i      g_i = Δt/(R_oa,i·C_i)      δ_i, c_i = Δt·q/C_i
```

Two consequences worth stating plainly:

1. **`a_i` absorbs the unresolved airflow unit.** Surrogate §4.6 records that the VAV airflow unit is
   still unestablished — CFM, m³/h and L/s differ by 1.7× and 3.6×. Multiplying every flow reading by
   a constant divides every `a_i` by it and changes nothing else, exactly as it moves only `C` in
   Block C's total. **Block D is unit-invariant provided the same unit is used to fit and to roll
   out.** What is *not* invariant is any physical reading of `a_i` — it cannot be compared against a
   design airflow until §4.6 is resolved.
2. **`ρ·c_p` never appears separately from `C_i`.** So no fitted number here can be checked against a
   handbook value, which is why §2.2's capacitance argument had to be made on assumed geometry
   instead. That is a real weakness of the parameterisation and it is the price of not needing `C_i`.

### 3.2 Euler or zero-order hold — a 14% correction at the fitted `k`

The fitted equation is a forward difference, which invites reading `k = Δt/τ`. The exact solution over
a step with inputs held constant is not that:

```
        T(t+Δt)  =  T_∞ + (T(t) − T_∞)·e^(−Δt/τ)        ⇒        k  =  1 − e^(−Δt/τ)
```

so the correct inversion is `τ = −Δt / ln(1 − k)`, not `Δt/k`. The two agree only as `k → 0`
([data/discretisation.csv](data/discretisation.csv)):

| True τ | 0.5 h | 1.0 h | 2.5 h | 5.0 h |
|---|---:|---:|---:|---:|
| `Δt/k` overstates τ by | **+27.1%** | **+13.0%** | +5.1% | +2.5% |

**The correction grows as τ shrinks, so it matters most exactly where the disagreement of §5.1 puts
us.** At the fitted `k ≈ 0.23`, `Δt/k` gives 1.07 h and the exact inversion gives 0.94 h — a 14%
overstatement. grey-box-technique §2.3 defines τ as `Δt/k`, so its 2.5 h is an Euler reading of
`k = 0.1`; the exact reading of the same `k` is 2.37 h. That is a second-order correction sitting on
top of a 2.4× first-order one, and it is reported here so the two are not confused.

### 3.3 The 15-minute value is an average, not a sample — and here that is harmless

Surrogate §0 rules out modelling any inner loop, because the record is 15-minute *averages* and the
loops have 1–5 minute time constants — below Nyquist. Block D claims to identify a ~1 hour time
constant from the same averages. That needs an argument, not an assurance, because averaging is a
different operation from sampling.

Let `x̄_n` be the average of `x(t)` over step *n*, with the input held constant across the step. From
the exact solution,

```
        x̄_n  =  x_∞ + (x_n − x_∞)·(τ/Δt)·(1 − e^(−Δt/τ))
```

The factor multiplying `(x_n − x_∞)` **does not depend on n**. So if the input is also constant across
step *n+1*, substituting `x_{n+1} − x_∞ = e^(−Δt/τ)·(x_n − x_∞)` gives

```
        x̄_{n+1} − x_∞  =  e^(−Δt/τ) · ( x̄_n − x_∞ )
```

**The averages obey the same recursion as the samples, with the same pole, exactly.** Averaging
rescales the *amplitude* of the response and leaves the *rate* alone. Verified numerically at four
time constants, pole recovered to better than 1e-10 from block averages — an asserted guard, because
this identity is the entire licence for fitting this block at this sampling rate.

Two things follow. First, the qualifier is load-bearing: **the input must be constant across the
step.** For a control loop hunting inside the 15 minutes it is not, which is precisely why §0's
prohibition stands for loops and not for this block. Second, this gives stage 1 a **second and
independent** justification for using the fan-off window: not only is no controller running there, but
"the fan is off" *is* the constant-input condition, so it is the one window where the pole comes out
unbiased by the averaging.

---

## 4. Identifiability, term by term

### 4.1 Why `k` cannot be fitted alongside `a`

Both regressors are driven by `T_i(t)`: the mass term as `(T_m − T_i)`, the air term as
`V̇_i·(T_sa − T_i)`. Both fall as `T_i` rises, so they are collinear by construction, and the zone
controller tightens the knot by raising `V̇_i` when `T_i` is high. Least squares then splits the shared
variation arbitrarily. grey-box-technique §2.3 measures the consequence: fitting `k` jointly with the
daytime data returns **τ = 7.1 h**, three times the free-response value.

Hence two stages, and the ordering is not a convenience — it is what makes the estimator identified.
Stage 1 takes `k` from a window where `V̇_i = 0`, so the air regressor is *absent* rather than
controlled for. Stage 2 fixes `k` and fits the rest. The bias that remains in `a` is the endogeneity
surrogate §5.4 describes, and technique §4.2 step 4 addresses it with an 8-step output-error
refinement.

**That refinement is not a nicety, and this is worth saying because the specification treats it as
one.** A one-step fit puts the noisy `T_i(t)` on the right-hand side of its own difference equation. A
multi-step rollout does not — it simulates forward from one initial condition — so output error is the
estimator that removes any regressor-noise bias in `k` structurally rather than by correction. It
happens that §4.3 finds the bias negligible here, but the argument for step 4 does not depend on that.

### 4.2 Whether the two windows may overlap — an objection, retired

`T_m,i` is the mean of zone *i*'s **own sensor** over the plateau. So if the stage-1 window overlaps
the window `T_m` is averaged over, `T_m` contains `T_i(t)` itself, regressor and regressand share that
term, and `k` is biased up. grey-box-technique §4.2 puts stage 1 at 18:00–23:00 and the plateau at
21:00–05:00 — a two-hour overlap. The objection has a target.

**It does not survive the arithmetic.** `T_m` averages ~33 plateau steps, so any single `T_i(t)`
carries about 1/33 of it: the shared-term bias is O(1/N_plateau), not O(1). Measured
([data/stage1_window_variants.csv](data/stage1_window_variants.csv)), `k` moves by **−2.6% (AHU-1) and
−0.9% (AHU-2)** — under 3%, and in the *opposite* direction to the objection. Reported, not asserted.

The disjoint window 18:00–20:59 is still what §6 specifies, on the grounds that it costs nothing and
removes the question. But the reason technique's τ is wrong is not this. It is §5.1.

### 4.3 Sensor noise, retired with a number

Both sides of the stage-1 regression contain `−T_i(t)`, so measurement error in the room sensor is not
classical: it inflates the covariance and the variance together. With `T = θ + ε`,

```
        k̂  =  k + var(ε)·(1 − k)/var(x)
```

— biased **toward 1**, i.e. a zone that looks *faster* than it is, with `τ` biased short. This is the
standard AR(1)-with-measurement-error result and it is a real hazard, since technique §2.3 attributes
AHU-1's poor stage-1 fit to noisy room sensors.

Measured, it is nothing. `σ_ε` from second differences over the flat plateau — an **upper** bound,
since any real curvature inflates it, which is the conservative direction here — against the stage-1
regressor variance:

| | AHU-1 | AHU-2 |
|---|---:|---:|
| `σ_ε` | 0.032 K | 0.006 K |
| `var(x)` = var(`T_m − T`) | 0.329 K² | 0.334 K² |
| **Bias in `k`** | **+0.0028** | **+0.00007** |
| `k` itself | 0.233 | 0.222 |

Three orders of magnitude below `k` on AHU-1 and five on AHU-2. Sensor noise does not measurably bias
this block, and the correction is carried in
[data/stage1_fit.csv](data/stage1_fit.csv) (`k_debiased`) for audit rather than because it changes
anything.

> **A consequence that is not optional.** Since the sensors are *not* noisy in the white-noise sense,
> **grey-box-technique §2.3's explanation of AHU-1's low stage-1 R² — "whose room sensors are noisier"
> — cannot be right.** Under the window this document specifies, AHU-1's stage-1 R² is 0.83 uncentred
> and its σ_ε is 0.032 K. Whatever separates AHU-1 from AHU-2 in §5.2, it is not sensor white noise.
> Candidates, none tested here: quantisation, a slower sample-and-hold in the AHU-1 controllers, or
> genuine unmodelled disturbance. Named, not asserted.

### 4.4 The gains ride a floor-level regressor

[data/gains_identifiability.csv](data/gains_identifiability.csv). Occupancy comes from **four PIR
sensors** — `floor_8_zone_1_iaq_{1..4}__pir` — shared across all 51 fitted zones.

The obvious worry is that `occ` is nearly constant over weekday 08:00–17:00, which would make `δ_i`
and `c_i` inseparable. **It is not:** CV 0.309, p5–p95 spanning 0.17–0.48, and only **17%** of its
variance is the hour-of-day profile. So `δ_i` and `c_i` *are* separately identified.

The real limitation is different, and structural: `occ` is **zone-common**. Every zone sees the same
regressor, so `δ_i` cannot distinguish occupants in zone *i* from any other floor-common driver that
moves on the same schedule — plug load, solar, the AHU's own start ramp. **The cross-zone spread of
`δ̂_i` is a diagnostic, not a measurement of zone gains.** Fitted, its median is −0.007 K/step (AHU-1)
and −0.058 (AHU-2): *negative*, which is not what an occupancy gain should be, and is what a regressor
standing in for the cooling schedule looks like. Reported. `δ_i` is left in the specification because
dropping it moves its content into `c_i` rather than removing it, but no physical reading should be
put on it.

### 4.5 What Block C's error does to `a`

[block-c-derivation/README.md](../block-c-derivation/README.md) §5.2 names Block D as the consumer
that per-room flow error propagates into, and ships a median per-room error of **5.4% (AHU-1) and 9.2%
(AHU-2)**. Closing that loop: multiplicative error on `V̇_i` enters the air regressor and attenuates
`a` classically, by `1/(1 + σ²_v/σ²_x)`. With on-hours flow CV of 0.32 that is
`0.054/0.32 = 0.17` in sd terms, so the attenuation is about **3%**.

Small, and — more importantly — **in the wrong place to matter at fit time.** Stage 2 regresses on
*logged* `V̇_i`, not on Block C's prediction, so Block C's error does not enter the fit at all. It
enters at **rollout**, as model mismatch between the flow Block C hands Block D and the flow the same
damper positions really produced. That is a validation-gate question, not an identification question,
and it belongs to technique §5.3 gate 1.

### 4.6 Non-negativity is physics, and the boundary is evidence

`k_i, a_i, g_i` are conductances divided by a capacitance. A negative value is not a fit, it is a
defect indicator — so the estimator is **non-negative least squares**, and NNLS is a projection onto
the non-negative orthant whose solution satisfies the KKT conditions. The intercept and the occupancy
gain are left free in sign, because a lumped residual gain has no sign to violate.

This changes how the headline `g ≈ 0` should be read. **A parameter pinned at exactly zero by NNLS is
not "no effect measured" — it is the unconstrained fit asking for a negative one**
([data/stage2_fit.csv](data/stage2_fit.csv)):

| | AHU-1 | AHU-2 | Floor |
|---|---:|---:|---:|
| `g` pinned at 0 by NNLS | 17 / 24 | 24 / 27 | **41 / 51** |
| `g` **wanted** negative, unconstrained | 17 / 24 | 24 / 27 | **41 / 51** |
| `a > 0` | 22 / 24 | 19 / 27 | 41 / 51 |

Every zone at the boundary is there because the unconstrained fit wanted `g < 0` — "warmer outdoors
cools the room". That is **stronger** evidence against the outdoor path than `g ≈ 0` would be, and it
agrees with §2.3's estimator-free 0.0057 from a completely different direction.

`a` is pinned at zero on 10 of 51 zones, only 3 of which are on the named parked list. §4.7 is that
subject.

### 4.7 Two of technique's exclusion numbers do not reproduce

Both are minor and both are the kind of thing that becomes load-bearing later, so they are recorded
rather than quietly used.

**The garbage-sensor screen** ([data/sensor_screen.csv](data/sensor_screen.csv)). technique §4.2 drops
`vav_8_1_9` and `vav_8_1_6` on day means of 8.4 and 19.7 °C. Neither reproduces under a screen that
masks the 0.00 bad-read sentinel `config.yml` warns about — they come out at **16.16** and **21.19 °C**
against a floor median of 23.60. The gap is in the direction the sentinel explains: **that screen
appears to have been run without masking it**, and an unmasked AHU-1 mean is exactly what would drift
toward 8 °C. The two named boxes *are* still the floor's two coldest, which is what this script
asserts. But the second clears the third-coldest (`vav_8_2_23`, 21.59 °C) by **0.40 K**, so its
exclusion is a judgement call, not a measurement, and it should be labelled as one.

**Airflow variability.** technique §2.5 reports a median on-hours flow CV of 0.32 with 8 zones below
0.1, and concludes that the eight parked AHU-1 boxes are the ones that cannot identify `a`. On the
rows stage 2 actually fits — weekday 08:00–16:59, fan on, training window — the median CV is **0.115**
and **23 of 51 zones** fall below 0.1 (AHU-1 median 0.042, AHU-2 0.157). The narrower window is the
right one, because it is the variation `a` is identified *from*. **So `a` is unidentifiable on
substantially more zones than the named list of twelve**, and the pooling rule should be a measured CV
threshold rather than a list of box names. Reported; changing the rule is not this document's call.

---

## 5. Two disagreements, stated not hidden

### 5.1 Where 2.5 hours came from

grey-box-technique §2.3 reports the air time constant as `τ = Δt/k`, **median 2.5 h, IQR 2.1–2.9**, and
§4.1 expects 2–3 h. This document gets **0.94 / 1.00 h** from the same record. §2.4 rules out a
masking difference — the two agree on the plateau to 0.01 K. §4.2 rules out the window overlap (<3%).
§4.3 rules out sensor noise (three orders too small). §3.2 accounts for 14% of it, not 140%.

**What accounts for it is which night `T_m` is taken from.** A zone at 19:00 is relaxing toward the
plateau it will reach at 21:00–05:00 *that same night*. technique §4.1 defines `T_m` as "the night that
ended that morning" — correct for a daytime episode, and wrong for the evening transient, where read
literally it makes `T_m` **yesterday's** plateau. Yesterday's plateau is a noisy proxy for tonight's:
the plateau's night-to-night innovation is **0.47 K** — from a plateau sd of 0.54 K at lag-1
autocorrelation 0.50 — against a regressor sd of **0.57 K**, so the proxy error is nearly as large as
the signal. A noisy regressor attenuates the slope, and here it more than halves it.

Fitting it that way reproduces technique's number ([data/stage1_window_variants.csv](data/stage1_window_variants.csv)):

| Reading of `T_m`, stage 1 | `k` | `τ = Δt/k` | `τ` exact | R² uncentred |
|---|---:|---:|---:|---:|
| **Tonight's plateau, disjoint window** — specified in §6 | **0.233 / 0.222** | 1.07 / 1.13 h | **0.94 / 1.00 h** | **0.83 / 0.87** |
| Tonight's plateau, technique's overlapping window | 0.227 / 0.220 | 1.10 / 1.14 h | 0.97 / 1.01 h | — |
| **Yesterday's plateau** — technique read literally | **0.108 / 0.098** | **2.32 / 2.56 h** | 2.19 / 2.43 h | 0.51 / 0.56 |
| technique §2.3 as published | ~0.1 | **2.5 h** (IQR 2.1–2.9) | — | 0.16 |

`Δt/k` under the wrong-night reading lands at **2.32 / 2.56 h** against a published 2.5 h, and the R²
falls in the same direction. **This document asserts that agreement as a guard** — not because 2.5 h is
right, but because reproducing an artifact on demand is the evidence that it *is* the artifact. If it
ever stops reproducing, the explanation here is wrong and must be withdrawn.

The corrected τ of **0.94 / 1.00 h** sits inside technique §4.2's acceptance range of 0.5–8 h and above
surrogate §5.2's "below ~30 min is reporting noise" floor, so nothing downstream breaks. But it is
**below technique §4.1's expected 2–3 h**, and the zone-level minimum is **0.56 h** (`vav_8_1_7`),
close enough to the 0.5 h floor that the acceptance range will reject real zones rather than bad fits.
§11 lists the consequences.

### 5.2 The two-node form loses on AHU-2

Stage 2 fitted three ways on identical rows — NNLS with `a, g ≥ 0`; unconstrained, to read the sign `g`
wanted; and the one-node form with the mass term dropped, which is what surrogate §5.2 currently
specifies ([data/stage2_fit.csv](data/stage2_fit.csv)):

| Median per-zone R² | AHU-1 | AHU-2 | Floor |
|---|---:|---:|---:|
| Two-node, `k` fixed from stage 1 | **0.318** | 0.055 | **0.200** |
| One-node to outdoor — the current specification | 0.066 | **0.066** | 0.066 |
| Ratio | **4.82×** | **0.83×** | **3.01×** |
| Zones | 24 | 27 | 51 |

**The pooled claim holds and reproduces technique's closely** — 3.01× here against its 3.3×, on a
pooled median over the floor, which is the comparison that document actually makes. **On AHU-2 alone it
reverses: the two-node form is slightly worse than the form it replaces.**

Say plainly what this does and does not mean. It does not overturn the structural case, which rests on
§2.3's estimator-free bound and §2.4's plateau, neither of which is a fit statistic and both of which
hold on both AHUs. It does mean **the two-node form's *quantitative* advantage is an AHU-1 result that
survives pooling**, and that stating it as a floor-wide 3× improvement — as both upstream documents do
— hides a counter-example. The guard in the script is deliberately pooled for this reason, and the
per-AHU split is printed on every run: a guard written per-AHU would have failed, and a guard that
hides its own counter-example would be worse than none.

Why AHU-2 behaves differently is **not settled here**. §4.3 rules out sensor noise. Its flow CV is
higher, not lower (0.157 against 0.042), so it is not lack of excitation. Its `a` is pinned at zero on
8 of 27 zones against 2 of 24 on AHU-1, and it is the AHU that surrogate §4.4 records as genuinely
short of capacity — 12% of steps below 0.8× its pressure setpoint and a SAT loop saturated on 29.6% of
large-error steps. A zone whose supply conditions are frequently not what the setpoint says is a zone
whose air term is mis-measured, and that is the candidate worth testing first. Named, not asserted.

---

## 6. The specification, assembled

Per zone *i*, Δt = 0.25 h:

```
        T_i(t+1) − T_i(t) = k_i · (T_m,i − T_i(t))              air ↔ internal mass
                          + a_i · V̇_i(t) · (T_sa(t) − T_i(t))    supply air
                          + g_i · (T_oa(t) − T_i(t))             outdoor, expected 0
                          + δ_i · occ(t) + c_i                   gains

        k_i, a_i, g_i ≥ 0        τ_i = −Δt / ln(1 − k_i − a_i·V̇_i − g_i)
```

`T_m,i` is the mean of zone *i*'s own reading over **21:00–05:00 of the night the step belongs to** —
the night *ahead* for an evening step, the night *behind* for a daytime step (§5.1) — replayed as an
exogenous input, falling back to the floor mean when a zone's night data is missing. For multi-day
rollouts the mass relaxes toward the air at 0.65 K per uncooled day (§2.4).

| Parameter | Provenance | Value |
|---|---|---|
| `k_i` | **fitted**, stage 1, free response, fan off, 18:00–20:59, `T_m` = tonight's plateau | 0.233 / 0.222 median → **τ 0.94 / 1.00 h** |
| `a_i` | **fitted**, stage 2 NNLS, `k` fixed | 2.4e−4 / 1.6e−4 median; pinned at 0 on 10 / 51 |
| `g_i` | **fitted, expected zero** — and 41 / 51 want it negative (§4.6) | 0 on 41 / 51 |
| `δ_i` | **fitted**, but on a zone-common regressor — a diagnostic, not a gain (§4.4) | −0.007 / −0.058 |
| `c_i` | **fitted**, lumped residual, free in sign | −0.24 / −0.28 |
| `T_m,i` | **measured**, replayed, one value per zone per night | 24.76 ± 0.54 °C floor mean |
| `τ_m` | **not identified** — the record gives the rate, not the gradient (§2.4) | 0.65 K/uncooled day |
| airflow unit | **UNRESOLVED**, surrogate §4.6 — absorbed into `a_i` (§3.1) | invariant if fit and rollout agree |

---

## 7. Falsifiers

technique §5.3's gates, restated as the mathematics they are:

1. **Open loop.** An 8-step rollout on the test month must beat persistence on zone-temperature RMSE
   for the floor mean and ≥80% of active zones. Persistence is the **nested `k = a = g = δ = 0` model**,
   so this is a comparison against a special case of the block itself — which is what makes it a real
   test and not a benchmark of convenience. Given §5.2, it must be reported **per AHU**.
2. **Free response.** Switching the fan off in simulation must settle every active zone to its `T_m`
   within 3τ — that is `e^(−3) = 5%` of the initial gap remaining, ≈2.8 h at the fitted τ, which the
   record can check directly against any evening.
3. **Undercooling.** The rollout must reproduce AHU-1's −2.60 K mean offset below setpoint
   (surrogate §5.5). Weaker than gates 1–2 because it involves the controller.
4. **Synthetic recovery.** The two-stage fit must recover a known two-node zone's `k, a, g, δ, c` from
   data it generated itself. Asserted at 1e-6; achieved at **1e-16**
   ([data/synthetic_recovery.csv](data/synthetic_recovery.csv)). An estimator that cannot pass this has
   no business being pointed at a CSV.
5. **The pole identity** (§3.3), asserted at 1e-10.

---

## 8. Guards

Asserted — a reversal fails the run:

| Guard | Rule | Status |
|---|---|---|
| **Pole under averaging** | block averaging preserves `e^(−Δt/τ)` exactly | < 1e-10, 4 τ values |
| **Synthetic recovery** | two-stage fit recovers known `k, a, g, δ, c` | < 1e-15 |
| **No outdoor path** | `g/(k+g)` < 0.05, measured estimator-free | 0.0057 |
| **Two-node beats one-node** | pooled floor median R², the comparison technique makes | 0.200 vs 0.066 |
| **Sensor screen** | the two excluded sensors are still the floor's two coldest | passes, margin 0.40 K |
| **`k > 0`** | on every zone not pooled | 51 / 51 |
| **The 2.5 h artifact reproduces** | wrong-night `T_m` gives `Δt/k` within 0.5 h of 2.5 | 2.32 / 2.56 h |

Reported, never asserted: **τ itself** (§5.1), the **AHU-2 reversal** (§5.2), every per-zone R², the
unconstrained sign of `g` (§4.6), the window contrast (§4.2), the sensor-screen margin and the flow-CV
count (§4.7), and `δ_i` (§4.4). Each is a place this block is weak or a place it disagrees with a
document upstream of it, and a guard that passed by not looking would be worse than no guard.

---

## 9. Tables shipped

| File | One row per | Contents |
|---|---|---|
| [data/mass_node.csv](data/mass_node.csv) | floor | plateau mean, sd, autocorrelation, weekly cycle, drift rates — §2.4 |
| [data/outdoor_share.csv](data/outdoor_share.csv) | AHU + floor | the within-night slope, `g`, and the outdoor τ in days — §2.3 |
| [data/stage1_fit.csv](data/stage1_fit.csv) | zone | `k`, both τ inversions, `σ_ε`, the EIV bias and correction, R² — §4.3, §5.1 |
| [data/stage1_window_variants.csv](data/stage1_window_variants.csv) | AHU | the three readings of `T_m` — §4.2, §5.1 |
| [data/stage2_fit.csv](data/stage2_fit.csv) | zone | `a, g, δ, c`, constrained and not, against the one-node form — §4.6, §5.2 |
| [data/discretisation.csv](data/discretisation.csv) | τ × sampling mode | the pole identity and the Euler error — §3.2, §3.3 |
| [data/synthetic_recovery.csv](data/synthetic_recovery.csv) | parameter | estimator recovery on its own data — §7 |
| [data/capacitance_check.csv](data/capacitance_check.csv) | assumed coefficient | air-only τ and the capacitance ratio implied — §2.2 |
| [data/gains_identifiability.csv](data/gains_identifiability.csv) | floor | occupancy variability, PIR count, flow CV — §4.4, §4.7 |
| [data/sensor_screen.csv](data/sensor_screen.csv) | zone | day means, ranks, margins against technique's quoted values — §4.7 |
| [data/tau_disagreement.csv](data/tau_disagreement.csv) | AHU | §5.1's table, at full precision |

---

## 10. What this block does and does not settle

**Settles:**

1. **The outdoor coupling is absent, and for a physical reason.** 0.6% of the zone equilibrium, an
   implied outdoor time constant of 8 days, and 41 of 51 zones wanting a *negative* coefficient. These
   are interior zones on a floor with conditioned space above and below.
2. **The mass node is an input, not a state.** It moves under 0.3 K per episode and 0.65 K per
   uncooled day, so it is measured once a night and replayed — which is what makes the two-node form
   cost one parameter rather than a filter.
3. **The 15-minute average does not bias this block's time constant**, because block averaging
   preserves the pole exactly under held input — a proof, not an assurance, and the reason the fan-off
   window is the right place to identify `k`.
4. **grey-box-technique §2.3's τ of 2.5 h is an artifact of the wrong night's plateau**, reproduced on
   demand. The value is 0.94 / 1.00 h.
5. **The fast node is not the zone air** — it carries 4–12× the air's capacitance, so "zone air
   temperature" is a lumped node including furnishings and the interior surface layer.

**Does not settle:**

1. **Why AHU-2 does not benefit from the two-node form** (§5.2). The pooled claim holds; the per-AHU
   split reverses; the capacity shortfall in surrogate §4.4 is the candidate and is untested.
2. **Whether any of this survives a multi-step rollout.** Every R² here is one-step, and one-step R² of
   0.06–0.32 is not evidence about a 44-step episode. Gate 1 (§7) is the only number that matters for
   the environment, and it is not built.
3. **`τ_m`**, the mass node's own dynamics, needed for multi-day rollouts (§2.4).
4. **`a_i` as a physical quantity.** The airflow unit is still open (surrogate §4.6) and `C_i` is
   divided out (§3.1), so `a_i` cannot be checked against a design airflow.
5. **`δ_i` as an occupancy gain** (§4.4). Four PIR sensors, zone-common, and the fitted sign is wrong.
6. **Which zones to pool.** §4.7 finds 23 of 51 zones below the CV threshold technique uses to name 8.
7. **The identity of the mass node** — slab, structure, or the conditioned floors above and below.
   Irrelevant within an episode, since it is replayed; it decides whether `τ_m` is a property of this
   floor.

---

## 11. Corrections this forces upstream

Listed, not applied — the same way [block-c-derivation](../block-c-derivation/README.md) §9.1 recorded
its consequences for control-gap §5 rather than editing it.

**[grey-box-surrogate.md](../grey-box-surrogate.md):**

- **§5.1–5.4** — replace with §6 above. The correction banner already says so; this supplies the
  derivation and the fitted values.
- **§5.2** — the implied time constant `τ_i = 1/(α_i·V̇_i + γ_i)` is missing its Δt and inverts the
  wrong discretisation. Use `τ_i = −Δt/ln(1 − k_i − a_i·V̇_i − g_i)` (§3.2).
- **§5.5** — add the night plateau (24.76 °C) and the evening settling (τ ≈ 0.94 h, **not** 2.5 h) to
  what the fit must reproduce.
- **§8.1** — the Block D row becomes `k_i, a_i, g_i, δ_i, c_i`, with `k_i` provenance "fitted, stage 1,
  free response"; add a row for `T_m` as measured-and-replayed; note that `a_i` absorbs the airflow
  unit.
- **§8.2** — the zone DR axis becomes the day-block bootstrap ensemble, per technique §3.3 U2.
- **§10** — add "the AHU-2 two-node reversal" to what the surrogate cannot support.

**[grey-box-technique.md](../grey-box-technique.md):**

- **§2.3** — τ of 2.5 h is the wrong-night artifact of §5.1; the value is 0.94 / 1.00 h. The
  explanation of AHU-1's low R² as noisy sensors is inconsistent with σ_ε = 0.032 K (§4.3). The
  two-node R² advantage reverses on AHU-2 (§5.2).
- **§2.5** — the median on-hours flow CV on the rows stage 2 fits is 0.115, not 0.32, and 23 zones fall
  below 0.1, not 8 (§4.7).
- **§4.1** — expected τ "2–3 h at zero flow" should be ≈1 h. State which night `T_m` comes from
  separately for the evening and daytime windows; the single phrase "the night that ended that morning"
  is what produced the artifact.
- **§4.2** — make the stage-1 and plateau windows disjoint (18:00–20:59 against 21:00–05:59). The bias
  is small (§4.2) but the ambiguity is free to remove. The τ acceptance range 0.5–8 h now sits directly
  on the fitted values and will reject real zones; re-centre it.
- **§4.2 step 1** — the day means quoted for the two excluded sensors do not reproduce under a
  sentinel-masked screen (§4.7); rerun that screen with the mask `config.yml` specifies.
- **§5.3** — gate 1 must be reported per AHU, given §5.2.

**[config.yml](../config.yml)** — the `rl:` block technique §5.1 calls for does not exist yet. When it
is added it should carry the plateau and stage-1 windows as *disjoint* values, the τ acceptance range
re-centred on ~1 h, and a measured flow-CV pooling threshold in place of the named parked list (§4.7).
The guards this document asserts are currently declared at the head of
[reproduce_zone_thermal.py](reproduce_zone_thermal.py) and belong there instead.
