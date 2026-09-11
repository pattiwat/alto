# Block B, derived — from the affinity laws to `P = a + b·(f/50)³`

> **What this is.** The audit trail behind six lines of
> [grey-box-surrogate.md](../grey-box-surrogate.md) §3:
>
> ```
>         P_fan(f) = a + b·(f/50)³        a, b = 0.719, 6.199 (AHU-1) · 0.537, 6.806 (AHU-2)
>         E_fan    = P_fan · Δt
> ```
>
> That document states the result and cites "§1.9 of the data memo". It does not show the chain:
> how the affinity laws produce a cube law, why the fitted form carries an additive offset, which
> two columns of which table the coefficients came from, and what arithmetic turns those rows into
> `a` and `b`. This folder shows the chain and ships the intermediate tables, so the number can be
> recomputed by hand rather than taken on trust.
>
> **Run it:** `python block-b-derivation/reproduce_fan_power.py` — refits from the raw CSV under
> four row masks, asserts the headline matches [report/fan_models.csv](../report/fan_models.csv) to
> 1e-9, and writes every intermediate to [data/](data/).
>
> **Two findings up front.** The fit is reproducible — but **not the one the specification quotes**.
> Three different versions of `a` and `b` are in circulation and the specification's is reproducible
> under no mask in this repo (§8). And the cause is diagnosable: **a mask component that has no
> business in a static fit is deleting a third of the rows** (§7).

---

## 1. The affinity laws, and the cube law they produce

Three proportionalities hold for a fan running along a fixed system curve at speed `N`:

```
        Q  ∝  N              volume flow scales linearly with speed
        Δp ∝  N²             pressure rise scales with the square
        P  =  Q · Δp  ∝  N³  so air power scales with the cube
```

The third is not an independent law — it is the product of the first two. That is the entire
physical content of Block B, and it is why the cube exponent is **not fitted**: it is arithmetic on
the other two.

Reading the same three relations in the other direction gives Block A. There, static pressure is
the input and frequency the output, so `N ∝ √Δp` and `d ln f / d ln SP = ½`
([block-a-derivation §1](../block-a-derivation/README.md)). **Blocks A and B are one derivation read
from opposite ends**, which has a consequence worth stating before either is fitted:

```
        d lnP / d lnSP  =  (d lnP / d lnf) · (d lnf / d lnSP)  =  3 · β
```

Under the textbook affinity value β = ½ this is **1.5**, and that is exactly what
[config.yml:138-139](../config.yml#L138-L139) records:

```yaml
    # ...and the fan-power elasticity that follows from it under a cubic fan law.
    assumed_power_elasticity: 1.5
```

It is the null hypothesis in the `assumed` column of `report/elasticities.csv`, never an applied
constant. Under the *measured* β̂ = 0.382 the same composition predicts `3 × 0.382 = 1.145`, against
**1.130 measured** on AHU-1 — agreement to 1.3%, from two independently estimated quantities. §9
returns to this, because AHU-2 does not behave.

### 1.1 From shaft power to the meter

`ahu_b8_{n}__power` is an electrical measurement, and the cube law above is aerodynamic. The chain
between them is:

```
        P_elec  =  Q · Δp / ( η_fan · η_belt · η_motor · η_vsd )
```

Only `Q · Δp` carries the cube. The efficiencies do not, and they are not constant — motor and drive
efficiency fall off at part load, which is a *load-dependent* loss, not a fixed one. The standard
engineering simplification is to split the result into a speed-independent term and a cube term:

```
        P_elec  ≈  a  +  b · (f/50)³
```

with `f/50` standing in for `N/N_rated` because the VSD commands frequency and the motor is
synchronous to it up to slip. `rated_hz = 50.0` is set at [config.yml:181](../config.yml#L181) and is
the only place the 50 enters.

**This form is a two-term approximation, not a derivation**, and §2 is about what happens when you
ask the plant whether its first term exists.

---

## 2. The offset is not a standby loss — the plant says so

[effects.py:19-20](../src/effects.py#L19-L20) motivates the offset physically:

> *"The offset form is both better-fitting and physically motivated: motor and VSD losses are
> roughly fixed, so a fixed loss plus a cube-law aerodynamic term is the right shape."*

Half of that is right. It **is** better-fitting (§4). But "fixed loss" makes a prediction that this
dataset can check directly, because the fan spends most of the record switched off:

| | AHU-1 | AHU-2 |
|---|---:|---:|
| steps with `f = 0` | **5,909** | **5,733** |
| mean power over those steps | **0.0004 kW** | 0.0008 kW |
| median power over those steps | **0.000 kW** | 0.000 kW |
| fitted `a`, i.e. `P(0)` predicted | **1.097 kW** | 0.496 kW |

The meter records nothing. There is no standby draw to be a fixed loss: when the VSD stops, the
whole subsystem stops. **`a` is a local linearisation coefficient, not a physical constant.** It is
the intercept of a straight line fitted to `P` against `c = (f/50)³` over a band that never comes
near `c = 0`, and extrapolating it to zero speed is reading a regression outside its support.

This is not an argument against the offset cubic — a local linearisation is a perfectly good model
*locally*, and §4 shows it is the best of the three inside the band. It is an argument about what
the parameter means, and it is why the extrapolation guard in
[grey-box-surrogate.md](../grey-box-surrogate.md) §3 is load-bearing rather than decorative. An
agent that drives `f` toward the bottom of the clip is not collecting a real fixed loss; it is
collecting an intercept.

The fan-off rows are the first line of [data/binned_curve_ahu1.csv](data/binned_curve_ahu1.csv).

---

## 3. From the equation to the raw data

### 3.1 Two columns

```python
    frame = pd.DataFrame(
        {"hz": df[f"ahu_b8_{ahu}__frequency"], "kw": df[f"ahu_b8_{ahu}__power"]}
    )[keep].dropna()
```
— [effects.py:72-75](../src/effects.py#L72-L75)

| Symbol | Column | Unit | Position in the raw CSV | Provenance |
|---|---|---|---|---|
| `f` | `ahu_b8_{1,2}__frequency` | Hz | col 17 / 55 — Excel `Q` / `BC` | convention; VSD commands 0–50 |
| `P` | `ahu_b8_{1,2}__power` | kW | col 23 / 61 — Excel `W` / `BI` | [units.yml:89-93](../src/units.yml#L89-L93) |
| `50` | `control_gap.effects.rated_hz` | Hz | — | [config.yml:181](../config.yml#L181) |
| `Δt` | `data.expected_freq` | 0.25 h | — | [config.yml:8](../config.yml#L8) |

**That is the whole input.** No airflow, no pressure, no efficiency curve, no nameplate: the
fit sees a commanded VFD frequency and a metered kW, and nothing else. §5.1 does it in a
spreadsheet to make the point concrete.

Two notes on the mapping, both easy to get wrong:

- **`__frequency`, not `__frequency_read` or `__frequency_write`.** All three exist. The fit uses the
  bare point; the read/write pair are the loop's own view of it and are not interchangeable with it.
- **The Hz unit entry does not apply to this dataset.** [units.yml:77](../src/units.yml#L77)
  registers the pattern `.*__fan_frequency.*`, and the columns are named `..__frequency`. The
  pattern matches **zero columns** here, so the 0–50 Hz range check never fires on the fan fit. The
  band limit that is enforced comes from `config.yml:96` and only inside the loop-authority
  diagnostic. Worth fixing; it is not what causes §8.

`E_fan = P·Δt` uses `Δt = 0.25 h` rather than the logged `ahu_b8_{n}__cumulative_energy`, because
that point is **dead — constant 0 on both AHUs** ([units.yml:95-100](../src/units.yml#L95-L100)).
That is the origin of the bare `* 0.25` at [effects.py:192-193](../src/effects.py#L192-L193).

### 3.2 Which rows

9,305 raw steps become 1,807. Every stage:

| Stage | Rule | AHU-1 | AHU-2 |
|---|---|---:|---:|
| raw grid | 15-min, 18 May → 23 Aug | 9,305 | 9,305 |
| both points present | `notna()` on `hz` and `kw` | 8,646 | 8,650 |
| **fan on** | `hz > 0 ∧ kw > 0` inside `fit_fan_models` | **2,724** | **2,720** |
| ∧ on-hours | `hz > min_fan_hz = 5.0` — [masks.py:43-46](../src/masks.py#L43-L46) | 2,720 | 2,719 |
| ∧ no outage | `≤ 50%` nulls across the row — [masks.py:73-84](../src/masks.py#L73-L84) | 2,720 | 2,719 |
| ∧ no override | override points at 1.0 — [masks.py:49-70](../src/masks.py#L49-L70) | **2,678** | **2,681** |
| ∧ **settling** | 2 steps after any SP or SAT setpoint change — [masks.py:87-109](../src/masks.py#L87-L109) | **1,807** | **1,911** |
| ∧ complete | priced inputs present — [run_control_gap.py:231-247](../scripts/run_control_gap.py#L231-L247) | 1,807 | 1,911 |

Two of those rows carry the whole of §7 and §8:

- **the completeness component is non-binding.** It requires airflow and setpoint to be present and
  positive, and it removes **nothing** the fan fit would have used. `reproduce_fan_power.py` asserts
  this, so if it ever stops being true the document fails rather than misleading.
- **settling removes 871 rows on AHU-1 and 770 on AHU-2** — a third of the sample, on its own.

Per-row flags for all of this are in [data/fit_inputs_ahu1.csv](data/fit_inputs_ahu1.csv) and
[data/fit_inputs_ahu2.csv](data/fit_inputs_ahu2.csv), one column per mask component.

---

## 4. The estimator — three fits, all of them one regression

[effects.py:65-103](../src/effects.py#L65-L103) fits three models and returns them as a band, because
"the spread between them is wider than most of the effects being argued about"
([effects.py:12-13](../src/effects.py#L12-L13)). All three are ordinary least squares. Writing
`c = (f/50)³`:

| Model | Regression | Closed form | Code |
|---|---|---|---|
| empirical power law | `ln P` on `ln f` | slope and intercept in logs | [effects.py:82-87](../src/effects.py#L82-L87) |
| **offset cubic** | `P` on `c`, with intercept | `b = (n·Σcp − Σc·Σp)/(n·Σc² − (Σc)²)`, `a = (Σp − b·Σc)/n` | [effects.py:89-94](../src/effects.py#L89-L94) |
| ideal cubic | `P` on `c`, **through the origin** | `b = Σcp / Σc²` | [effects.py:96-101](../src/effects.py#L96-L101) |

There is no iteration, no regularisation and no weighting. **The headline is five sums and two
divisions**, and those five sums are shipped in
[data/normal_equations.csv](data/normal_equations.csv) precisely so the claim can be checked without
running anything.

Note what is *not* fitted: the exponent 3. It comes from §1. The empirical power law exists only to
show what happens when you try to fit it, which is §6.

`R²` is the plain coefficient of determination, [effects.py:61-62](../src/effects.py#L61-L62) —
computed against the model's own predictions, so for the ideal cubic (which is not a least-squares
minimiser of that quantity, being constrained) it can and does fall well below the offset form's.

---

## 5. The arithmetic

AHU-1, evaluation mask — the row that produced the shipped headline. From
[data/normal_equations.csv](data/normal_equations.csv):

```
n        = 1807
Σc       =   663.3693289490137           c = (f/50)^3
Σc²      =   250.17453145978976
Σp       =  5354.28868                   kW, summed over the masked steps
Σcp      =  1999.3930112489847
```

Then, in full:

```
b = ( n·Σcp − Σc·Σp ) / ( n·Σc² − (Σc)² )
  = ( 1807 × 1999.3930112489847 − 663.3693289490137 × 5354.28868 )
    / ( 1807 × 250.17453145978976 − 663.3693289490137² )
  = ( 3612903.1713269153 − 3551870.8886509000 )
    / (  452065.3783478401 −  440058.8665902648 )
  =     61032.2826760150  /  12006.5117575753
  =         5.0832651404774305                                 <- the published b

a = ( Σp − b·Σc ) / n
  = ( 5354.28868 − 5.0832651404774305 × 663.3693289490137 ) / 1807
  = ( 5354.28868 − 3372.0821851084270 ) / 1807
  =    1982.2064948915727 / 1807
  =         1.0969598754242240                                 <- the published a
```

`reproduce_fan_power.py` performs exactly this recomputation for every AHU and every mask and
asserts it against `np.polyfit`'s answer to 1e-9, so §4's claim that the fit *is* these five sums is
enforced rather than asserted. The ideal cubic is the same table read differently:
`Σcp / Σc² = 1999.393.../250.174... = 7.99199262843566`.

**The band, on this AHU, from these same rows:**

```
    empirical power law   P = 0.1232 · f^0.889                R² 0.509
    offset cubic          P = 1.097 + 5.083 · (f/50)³         R² 0.816     <- headline
    ideal cubic           P =         7.992 · (f/50)³         R² 0.542
```

### 5.1 The same fit in Excel

§5 is five sums and two divisions, which means it is a spreadsheet exercise. Nothing in Block
B requires Python — the script exists to *assert* the answer, not to reach it. This section is
the no-Python route, and it exists because the natural next question after §3.1 is "so I take
the VFD frequency and the metered power and fit a curve?" The answer is yes, and here it is.

**The one transform.** The cube exponent is not fitted — §1 derives it — so Excel only ever
runs a **two-parameter straight line** of `P` against `c`:

```
        c  =  (f/50)³             the 50 is config.yml:181, the only place it enters
        P  =  a  +  b·c           b = SLOPE, a = INTERCEPT
```

#### The short route — the shipped intermediate

[data/fit_inputs_ahu1.csv](data/fit_inputs_ahu1.csv) exists for exactly this. It is 2,724
rows (Excel rows 2–2725) with `hz`, `kw` and `cube` already computed and one boolean column
per mask component, ready to autofilter:

| Excel column | Field |
|---|---|
| `A` | `timestamp` |
| `B` | `hz` — the raw `ahu_b8_1__frequency` |
| `C` | `kw` — the raw `ahu_b8_1__power` |
| `D` | `cube` — `(B/50)^3` |
| `E`–`H` | `mask_on-hours`, `mask_outage`, `mask_override`, `mask_settling` |
| `I`–`L` | `in_evaluation`, `in_analysis`, `in_no_settling`, `in_all_usable` |

Sort on the `in_*` column you want, delete the `FALSE` block, then over what remains:

```
        =SLOPE(C:C, D:D)            -> b
        =INTERCEPT(C:C, D:D)        -> a
        =RSQ(C:C, D:D)              -> R² for the offset cubic
        =LINEST(C:C, D:D)           -> {b, a} in one call
```

`SLOPE` and `LINEST` ignore an autofilter — they read the whole range, hidden rows included —
so the rows must actually be removed, or masked arithmetically. The masked form is the better
one anyway, because it *is* §5 written in cell formulas. With the mask expression
`m = --($L$2:$L$2725&""="TRUE")` (the `&""` makes it survive whether Excel imported the
column as text or as booleans):

```
        n    =SUMPRODUCT(m)                          2724
        Σc   =SUMPRODUCT(m, $D$2:$D$2725)            1033.504922495066
        Σc²  =SUMPRODUCT(m, $D$2:$D$2725, $D$2:$D$2725)    432.21934576178194
        Σp   =SUMPRODUCT(m, $C$2:$C$2725)            8347.318777
        Σcp  =SUMPRODUCT(m, $D$2:$D$2725, $C$2:$C$2725)   3416.1649959450624

        b    =(n*Σcp - Σc*Σp) / (n*Σc² - Σc^2)       6.212755756499197
        a    =(Σp - b*Σc) / n                        0.7071971807630413
```

Those five sums are the `all_usable` row of
[data/normal_equations.csv](data/normal_equations.csv), so the spreadsheet and the script are
checkable against each other cell by cell — to about the twelfth digit, below which the two
disagree only on the order they added 2,724 numbers in. Swap `$L$` for `$I$` / `$K$` to get the other
masks. The other two arms are different formulas rather than different ranges:

```
        ideal cubic       =SUMPRODUCT(m, D, C) / SUMPRODUCT(m, D, D)      (no intercept)
        power-law slope   =SLOPE(LN(kw range), LN(hz range))              (fitted in logs)
```

#### The long route — straight from the raw CSV

`alto_building_floor8_15min_2026-05-18_to_2026-08-23.csv` is 665 columns × 9,305 rows (Excel
rows 2–9306). Keep `Q` and `W` for AHU-1 (`BC` and `BI` for AHU-2), put `=(Q2/50)^3` in a
scratch column, and rebuild the mask from §3.2 with these conditions:

| Component | Excel condition, AHU-1 | AHU-2 | Source |
|---|---|---|---|
| fan on | `Q > 0` **and** `W > 0` | `BC`, `BI` | `fit_fan_models` — [effects.py:75](../src/effects.py#L75) |
| on-hours | `Q > 5` | `BC > 5` | `min_fan_hz: 5.0` |
| no override | `V = 1` **and** `C = 1` | `BH = 1` only | `override_tolerance: 1e-6` |
| no outage | fewer than half the row's cells blank | same | `outage_null_fraction: 0.5` |
| settling | drop a row where `AC` or `AH` changed, and the 2 rows after it | `BO`, `BT` | `settling_steps: 2` |

Two of those are worth knowing before you build them. **On-hours is nearly free** — `>5` Hz
removes only 4 rows beyond `>0` on AHU-1 and 1 on AHU-2, so it changes no coefficient you
care about. **Settling is not** — it removes 871 rows on AHU-1, a third of the sample, and it
is the single choice that decides which `a` and `b` you get. §7 argues it does not belong in
a static fit at all. The override rule is a strict equality by design (tolerance 1e-6):
binaries are logged as fraction-of-slot-on, so `0.333` means a human held the plant for part
of the bucket, and partial is still not automatic.

`auto_manual_control_mode_read` (`C`) exists on AHU-1 only — there is no AHU-2 counterpart, so
that AHU is filtered on `BH` alone.

#### What each filter actually yields

This is the part worth reading before you spend an afternoon in a spreadsheet:

| Excel filter | n | `a` | `b` | R² |
|---|---:|---:|---:|---:|
| `in_evaluation` — **what the code emits** | 1,807 | 1.0970 | 5.0833 | 0.816 |
| `in_no_settling` | 2,678 | 0.8038 | 5.9447 | 0.850 |
| `in_all_usable` | 2,724 | **0.7072** | **6.2128** | 0.906 |
| *[grey-box-surrogate.md](../grey-box-surrogate.md) §3 quotes* | *2,679* | *0.719* | *6.199* | *0.903* |

**No Excel filter reproduces `0.719 + 6.199`.** The nearest is `in_all_usable` — 1.7% out on
`a`, 0.2% on `b`, and its R² of 0.906 is within rounding of the quoted 0.903 — but its row
count is 2,724, and the 2,679 in the specification belongs to the *no-settling* fit, whose
coefficients are `0.804 + 5.945`. AHU-2 behaves the same way: the specification quotes
`0.537 + 6.806` against an `in_all_usable` fit of `0.5226 + 6.8199` on 2,720 rows.

So if you are fitting this in Excel to check the specification, you will not match it, and the
mismatch is not yours. §8 has the diagnosis; the short version is that the quoted tuple was
assembled from more than one fit.

---

## 6. Why the power-law arm finds ~1 instead of 3

The specification explains this as a narrow operating band
([grey-box-surrogate.md](../grey-box-surrogate.md) §3), and that is right, but it is worth
quantifying because the arm is one of the three domain-randomisation branches.

Fitting `ln P = ln a + n·ln f` needs leverage in `ln f`. There is almost none:

| | AHU-1 | AHU-2 |
|---|---:|---:|
| p5–p95 of `f` | 33.3 – 40.9 Hz | 31.8 – 43.9 Hz |
| **span in `ln f`** | **0.204** | 0.322 |
| share of rows in 34–38 Hz | **79%** | 49% |

A 0.20 lever arm is being asked to resolve an exponent of 3, against a signal that also contains a
large speed-independent component. The regression does what it must: it splits the difference and
returns something near 1.

Two consequences the specification does not record:

1. **The shipped AHU-1 exponent is 0.889 — sub-linear.** Not "near 1.5"; not a fan law at any
   exponent. A fan whose power rises more slowly than its speed is not a fan.
2. **The exponent is unstable across masks**, moving 0.889 → 1.003 → 1.107 on AHU-1 as rows are
   added ([data/models_by_mask.csv](data/models_by_mask.csv)). An estimate that moves that much
   under a mask change is reporting the mask.

The synthetic test at [test_effects.py:46-51](../tests/test_effects.py#L46-L51) makes the same
point deliberately — it generates data from a *known* offset cubic and asserts the recovered power
law comes out below 3. The estimator is behaving correctly; the band is the problem.

---

## 7. The settling mask does not belong in a static fit

This is the diagnosis behind §8, and it is a scope error rather than a bug.

[masks.py:87-109](../src/masks.py#L87-L109) `settled` drops every step within 2 steps of a change in
`static_pressure_setpoint_read` **or** `supply_air_temperature_setpoint_read`. Its purpose is sound:
if you are estimating a *response* to a setpoint, steps mid-transient contaminate it.

Block B is not estimating a response. It fits a **static** relation `P(f)` — the fan curve. A step
taken during a setpoint transient still pairs a real frequency with a real power; the fan does not
know why it is at 38 Hz. Worse, transient steps are the ones that *populate the ends of the band*,
which is where §6 showed the fit is starved.

The cost, from [data/models_by_mask.csv](data/models_by_mask.csv):

```
AHU-1   with settling (shipped)   n=1,807   a=1.097  b=5.083   R²=0.816   power law 0.889
        without settling          n=2,678   a=0.804  b=5.945   R²=0.850   power law 1.003
        every usable row          n=2,724   a=0.707  b=6.213   R²=0.906   power law 1.107
```

It is not merely noisier — it is **biased where the money is**. Residuals of each fit against the
binned means ([data/binned_curve_ahu1.csv](data/binned_curve_ahu1.csv)):

| AHU-1 offset cubic | RMSE, all rows | RMSE, 32–45 Hz | mean residual above 45 Hz |
|---|---:|---:|---:|
| shipped (with settling) | 0.281 kW | 0.125 kW | **+0.984 kW** |
| every usable row | 0.242 kW | 0.098 kW | +0.344 kW |

The shipped curve under-predicts by nearly **1 kW — 15%** at the top of the range, on the 85 steps
where the fan draws the most power and where a pressure reset would save the most. Inflating `a`
from 0.707 to 1.097 buys a slightly better fit in the crowded middle by flattening the curve, and it
pays for it exactly where Block B is used.

**Block A got this right and said why.** Its README §3 drops the settling component from the episode
mask, reasoning that "an episode *is* a setpoint change, so a mask that removes steps near setpoint
changes would delete the subject", and re-applies settling inside the estimator relative to the
signal actually being manipulated. Block B inherited the shared mask without that argument. The two
blocks need different row sets for the same reason they need different estimators: one measures a
response, the other measures a curve.

> **Recommendation, not a change.** Fitting Block B on the mask *without* settling — on-hours ∧
> no-outage ∧ no-override — is the defensible choice, and it happens to move the coefficients back
> toward the values the specification quotes. Changing
> [run_control_gap.py:256](../scripts/run_control_gap.py#L256) is outside this folder's scope; the
> evidence is here and the call is yours.

---

## 8. Three sets of coefficients — the honest answer

There are three versions of `a` and `b` in this repository, and they disagree by up to 55% on `a`.

| Source | AHU-1 | R² | n | AHU-2 | R² | n |
|---|---|---:|---:|---|---:|---:|
| `report/fan_models.csv` — **what the code emits** | `1.097 + 5.083` | 0.816 | 1,807 | `0.496 + 6.796` | 0.947 | 1,911 |
| refit, no settling | `0.804 + 5.945` | 0.850 | 2,678 | `0.504 + 6.871` | 0.904 | 2,681 |
| refit, every usable row | `0.707 + 6.213` | 0.906 | 2,724 | `0.523 + 6.820` | 0.920 | 2,720 |
| **`grey-box-surrogate.md` §3 states** | **`0.719 + 6.199`** | **0.903** | **2,679** | **`0.537 + 6.806`** | **0.960** | **2,677** |

Row 1 reproduces from the raw CSV **to the last digit** — `reproduce_fan_power.py` asserts it at
1e-9. The code is internally consistent and the pipeline is sound.

**The specification's numbers reproduce under no mask.** Worse, its own figures are not mutually
consistent:

- its AHU-1 coefficients and R² (`0.719 + 6.199`, 0.903) sit within rounding of the **every-usable-row**
  fit (`0.707 + 6.213`, 0.906) — n = 2,724;
- but its row count, 2,679, matches the **no-settling** fit (2,678) — whose coefficients are
  `0.804 + 5.945` and whose R² is 0.850;
- and its AHU-2 R² of **0.960 exceeds every value I could produce** under any mask (max 0.920).
- The power-law arm reproduces nowhere: the specification says `n = 1.42 / 1.73`, the code gives
  `0.889 / 1.301`, and no mask brings them together.

So the `(a, b, R², n)` tuple in §3 was not produced by a single fit. It is most likely a fit from a
superseded revision, transcribed alongside a row count from a different one.

Two further provenance errors in the specification's own description:

- It calls these "**2,679 / 2,677 coil-on steps**". `coil_on_mask` lives in
  [energy_balance.py](../src/energy_balance.py) and is **never called on the fan path**. The
  2,678 / 2,681 figure is the plain analysis mask without settling. The label is wrong even where
  the number is close.
- It calls Block B "the best-determined block in the surrogate" whose "residual is the *smallest*
  uncertainty the agent faces". On the level fit that is defensible. §9 argues it is not true of the
  quantity savings actually depend on.

The bare fact of the mismatch is already flagged in
[rl-sac-feasibility.md](../rl-sac-feasibility.md) §7.1. What is added here is the diagnosis (§7) and
the demonstration that no mask closes it.

**A third fact matters for how this gets fixed:** `report/fan_models.csv` is **write-only**. It is
produced at [run_control_gap.py:381-384](../scripts/run_control_gap.py#L381-L384) and nothing in the
repo reads it back. Every consumer of Block B today is a human reading a memo, which is exactly how
three versions came to coexist. The surrogate should load its fan curve from the CSV.

> **What the surrogate should carry, if asked:** the no-settling fit (§7), regenerated by the
> pipeline and read from `fan_models.csv` rather than transcribed. Editing
> [grey-box-surrogate.md](../grey-box-surrogate.md) §3 is outside this folder's scope.

### 8.1 Why the test suite does not catch this

[tests/test_effects.py:16-21](../tests/test_effects.py#L16-L21) builds its fixture from a known
offset cubic — and the coefficients it uses are the specification's:

```python
def fan_frame(a: float, b: float, n: int = 400, rated: float = 50.0):
    """Power generated from a known offset cubic, so the fit has a right answer."""
```
called as `fan_frame(a=0.719, b=6.199)` at [test_effects.py:24-25](../tests/test_effects.py#L24-L25).

That is a correct and valuable test: it pins the *estimator* against synthetic truth, exactly as
[test_excitation.py](../tests/test_excitation.py) `test_recovers_a_known_elasticity` does for
Block A. But it can only test that least squares recovers what it was given. Nothing in the suite
compares a fit against the real CSV, so the suite passes while the shipped numbers and the
documented numbers differ. `reproduce_fan_power.py`'s assertion is the missing half.

---

## 9. Level fit versus marginal response — which arm is really "central"

The three models are ordered in [effects.py:15-17](../src/effects.py#L15-L17) as pessimistic /
central / optimistic, with the offset cubic central because it fits best. **A saving does not depend
on how well a curve fits levels. It depends on its slope in logs**, `d lnP / d lnf` — how fast power
falls when the fan slows.

For the offset cubic that slope is strictly below 3, and the larger the offset the further below:

```
        d lnP / d lnf  =  3·b·c / (a + b·c)  ,   c = (f/50)³
```

The setpoint episodes of Block A measured the frequency response and the power response
independently on the same events, so their ratio is a measurement of that slope — and it can be
compared with what each fitted curve implies. From
[data/marginal_vs_level.csv](data/marginal_vs_level.csv), at the median masked frequency:

| | **measured `d lnP/d lnf`** | power law | offset cubic (shipped) | ideal cubic |
|---|---:|---:|---:|---:|
| AHU-1, at 35.79 Hz | **2.962** | 0.889 | 1.889 | **3.000** |
| AHU-2, at 35.48 Hz | **4.734** | 1.301 | 2.491 | 3.000 |

**AHU-1's marginal response is the ideal cube law, to 1.3%.** The offset cubic — the better fit to
levels — implies barely two-thirds of it. On the quantity that determines savings, the arm labelled
"optimistic" is the accurate one and the arm labelled "central" is 36% low.

That is the same finding [report/fan_law_consistency.csv](../report/fan_law_consistency.csv) already
records as a −7.6 point gap, restated as an elasticity so the direction is legible. And it is
consistent with §2: the offset is not physical, so a model that carries a large one will
under-predict how much power a real speed reduction removes.

**AHU-2 is different, and not in a reassuring way.** Its measured 4.734 is *above* 3 — a power
response steeper than any fan law permits. No curve reconciles it, so the discrepancy is not a
choice between arms; something other than the fan is moving with the setpoint on AHU-2. This is the
same AHU whose β episodes are 93% start-up transients at 08h
([grey-box-surrogate.md](../grey-box-surrogate.md) §2.2) and which falls below 0.8× its pressure
setpoint on 12% of steps. The right reading is that AHU-2's episodes measure a start-up ramp, not a
setpoint response, and its marginal numbers should not be used at all.

### 9.1 What the choice is worth

A blunt 20% setpoint cut priced under each arm, both β values, from
[data/savings_by_model.csv](data/savings_by_model.csv):

| AHU-1, measured β = 0.382 | saving | vs baseline 1,326–1,339 kWh |
|---|---:|---:|
| empirical power law | 97 kWh | 7.3% |
| offset cubic *(headline)* | 190 kWh | 14.2% |
| ideal cubic | 299 kWh | 22.5% |

The specification says "reporting the ideal cubic alone would overstate savings by roughly 2×".
Measured here, ideal / offset is **1.57×** on AHU-1 and 1.18× on AHU-2; the ~2× figure is closer to
the span of the *whole* band, power law → ideal cubic (3.07× / 2.21×). The claim is directionally
right and numerically loose.

But §9's result inverts the argument it supports. If AHU-1's true marginal elasticity is 2.96 rather
than 1.89, the ideal cubic is not overstating the saving — the offset cubic is understating it. The
honest statement is that **the fan model contributes a factor of ~1.6 to the fan-side answer, in a
direction the level-fit R² cannot resolve**, which is a reason to keep all three arms in the
domain-randomisation set (§8.2 of the specification) rather than to trust the middle one.

---

## 10. The extrapolation guard, and what Block B supports

The fit is valid over the band it was measured on. Outside it, §2 and §6 both bite: the intercept is
an artefact and the power-law arm is unidentified. From
[data/binned_curve_ahu1.csv](data/binned_curve_ahu1.csv):

```
f_mean    n     P_meas   shipped   all-rows   ideal
  0.00  5909    0.000     1.097     0.707     0.000    <- §2: nothing is drawn at zero
 16.22    35    2.380     1.271     0.919     0.270    <- 15-min averages spanning a fan start
 28.46    40    2.132     2.034     1.853     1.457    <- lower than the 16 Hz bin: not a curve
 33.62   187    2.596     2.643     2.596     2.403
 35.10  1377    2.806     2.856     2.857     2.734    <- 79% of rows live here
 38.77   133    3.596     3.467     3.604     3.685
 48.44    85    6.714     5.718     6.355     7.185    <- the arms bracket; shipped is 1 kW low
```

The sub-30 Hz rows are non-monotonic — the 25–30 Hz bin draws *less* than the 10–20 Hz bin. That is
not a fan curve; it is the signature of a 15-minute mean spanning a start or a stop, which is the
same Nyquist problem §0 of the specification states for the whole record. Those rows should not be
read as evidence about low-speed operation in either direction.

This is why [grey-box-surrogate.md](../grey-box-surrogate.md) §9 clips the agent to the **observed
setpoint support** rather than to engineering limits. A fitted polynomial has a tail; the plant does
not.

**Block B supports:** relative comparison of fan-side policies inside 32–45 Hz, under a stated
three-model band. **It does not support:** any absolute kW figure that depends on the intercept, any
extrapolation below ~30 Hz, or a claim that the fan model is the smallest uncertainty in the
surrogate — on levels it is, on the margin (§9) it contributes a factor of ~1.6.

---

## 11. Reverse-engineering recipe

**Without Python at all:** §5.1 does the whole fit in a spreadsheet, from either the raw CSV
or `data/fit_inputs_ahu{n}.csv`. Everything below is the same arithmetic with assertions
attached.

```
$ python block-b-derivation/reproduce_fan_power.py
```

The script refits under four masks, asserts its offset-cubic coefficients against
`report/fan_models.csv` to 1e-9, asserts the normal equations rebuild `np.polyfit`'s answer to 1e-9,
asserts the completeness component is non-binding (§3.2), and prints the gap to the specification
without asserting it. It fails loudly on all four.

Then, to confirm the headline really is five sums and two divisions:

```
$ python -c "import pandas as pd; d = pd.read_csv('block-b-derivation/data/normal_equations.csv'); \
             r = d[(d.ahu==1) & (d['mask']=='evaluation')].iloc[0]; \
             b = (r.n*r.sum_cp - r.sum_c*r.sum_p) / (r.n*r.sum_c2 - r.sum_c**2); \
             print(b, (r.sum_p - b*r.sum_c) / r.n)"
5.0832651404774305 1.096959875424224
```

And to confirm the estimator is sound rather than merely self-consistent:

```
$ pytest tests/test_effects.py tests/test_excitation.py tests/test_masks.py -q
```

— bearing §8.1 in mind about what that suite can and cannot see.

### Emitted tables

| File | Grain | Key columns |
|---|---|---|
| `data/fit_inputs_ahu{n}.csv` | one **usable step** (2,724 / 2,720) | `timestamp, hz, kw, cube`, then `mask_{on-hours,outage,override,settling}` and `in_{evaluation,analysis,no_settling,all_usable}` — the `in_*` columns are there to be autofiltered in a spreadsheet, §5.1 |
| `data/normal_equations.csv` | one AHU × mask | `n, sum_c, sum_c2, sum_p, sum_cp, sum_lnf, sum_lnf2, sum_lnp, sum_lnflnp, a_from_sums, b_from_sums` — §5 |
| `data/models_by_mask.csv` | one model × AHU × mask (24 rows) | `ahu, mask, kind, name, a, b, rated_hz, r2, n, formula` — §7, §8 |
| `data/binned_curve_ahu{n}.csv` | one frequency bin, plus a fan-off row | `bin, n, f_mean, p_mean, p_median, p_sd`, then `pred_*` and `resid_*` per model — §2, §10 |
| `data/marginal_vs_level.csv` | one model × AHU × mask | `hz_ref, measured_d_ln_hz, measured_d_ln_kw, measured_d_lnP_d_lnf, model_d_lnP_d_lnf, gap` — §9 |
| `data/savings_by_model.csv` | one model × AHU × exponent | `kind, exponent, exponent_source, cut_fraction, n, baseline_kwh, counterfactual_kwh, saving_kwh, saving_fraction` — §9.1 |

**Worked example, `normal_equations.csv` row 1** (AHU-1, evaluation mask — the shipped headline):

```
ahu             1
mask            evaluation
n            1807            masked steps with hz > 0 and kw > 0
sum_c         663.3693289490137      Σ (f/50)³
sum_c2        250.17453145978976     Σ (f/50)⁶
sum_p        5354.28868              Σ kW
sum_cp       1999.3930112489847      Σ (f/50)³·kW
a_from_sums     1.096959875424224  = (Σp − b·Σc)/n            <- the published a
b_from_sums     5.0832651404774305 = (nΣcp − ΣcΣp)/(nΣc² − (Σc)²)   <- the published b
```

### Where each published figure lives

| Figure | Source |
|---|---|
| `a, b`, R², n as shipped | `report/fan_models.csv` |
| `a, b`, R², n under all four masks | `data/models_by_mask.csv` — §7, §8 |
| the five sums behind every one of them | `data/normal_equations.csv` — §5 |
| fan-off power ≈ 0 at `f = 0` | `data/binned_curve_ahu{n}.csv`, first row — §2 |
| `d lnkW/d lnSP` = 1.130 / 1.096, β̂ = 0.382 / 0.232 | `report/elasticities.csv` |
| measured `d lnP/d lnf` = 2.962 / 4.734 | `data/marginal_vs_level.csv` — §9 |
| fan-law gaps −7.6 / −16.1 pts | `report/fan_law_consistency.csv` |
| 190 / 175 kWh, and the same cut under all three arms | `report/naive_cut_savings.csv`, `data/savings_by_model.csv` |
| the affinity derivation these share | [block-a-derivation/README.md](../block-a-derivation/README.md) §1 |
