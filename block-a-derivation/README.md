# Block A, derived — from the affinity law to β̂ = 0.382 / 0.232

> **What this is.** The audit trail behind one line of
> [grey-box-surrogate.md](../grey-box-surrogate.md) §2.2:
>
> ```
>         f(t) = f_ref · ( SP_act(t) / SP_ref )^β        β̂ = 0.382 (AHU-1), 0.232 (AHU-2)
> ```
>
> That document states the result and cites [control-gap-method.md](../control-gap-method.md)
> §4.2–4.3. It does not show the chain: how the affinity law collapses to a one-parameter power
> law, what estimator produced 0.382, which rows of which table were summed to get it, and where
> `SP_ref` and `f_ref` come from. This folder shows the chain and ships the intermediate tables, so
> the number can be recomputed by hand rather than taken on trust.
>
> **Run it:** `python block-a-derivation/reproduce_beta.py` — recomputes β from the raw CSV,
> asserts it matches [report/elasticities.csv](../report/elasticities.csv) to 1e-12, and writes
> every intermediate to [data/](data/).
>
> **One finding up front.** β is fully reproducible. **`SP_ref` and `f_ref` are not defined
> anywhere in this repository** — see §6, which is the honest answer to "how did you get these
> data" rather than a reconstructed provenance.

---

## 1. The affinity law, and why it collapses to one parameter

At a **fixed system curve**, fan pressure rise scales with the square of volume flow and flow
scales linearly with shaft speed ([src/excitation.py:1-16](../src/excitation.py#L1-L16)):

```
        Δp ∝ Q²  ,   Q ∝ N        ⇒        N ∝ √Δp
```

Read as a control law from static-pressure setpoint to fan frequency, that is

```
        Hz_new = Hz_old · √( SP_new / SP_old )          ⇒        d ln f / d ln SP = ½
```

Now generalise the ½ to a free exponent β. Any power law through a reference state `(SP_ref, f_ref)`

```
        f = f_ref · ( SP / SP_ref )^β
```

is, in logs, a straight line:

```
        ln f − ln f_ref  =  β · ( ln SP − ln SP_ref )
```

**This is the step that makes `SP_ref` and `f_ref` vanish from the estimation.** Take any two states
on that curve and difference them: the anchors are common to both sides and cancel identically.

```
        ln f₁ − ln f₀  =  β · ( ln SP₁ − ln SP₀ )        ⇒        β = Δln f / Δln SP
```

So β is identified from *changes alone* and never needs a reference point. The anchors return only
when the law is used to **simulate** — to turn a counterfactual setpoint into a frequency. That
asymmetry is why the estimator below is complete while §6 has to report that the anchors are
undefined: they were never needed to get 0.382, and they are needed to use it.

Note that ½ enters this codebase in exactly one place — [config.yml:137](../config.yml#L137):

```yaml
    # What the textbook affinity law assumes: at a fixed system curve fan pressure
    # goes as Hz^2, so Hz scales as sqrt(SP) -> dlnHz/dlnSP = 0.5. This is the value
    # under test, NOT a value being applied.
    assumed_affinity_exponent: 0.5
```

It is the null hypothesis, never an applied constant. [src/effects.py:106-122](../src/effects.py#L106-L122)
`sp_to_hz` takes the exponent as an argument for the same reason.

---

## 2. Why ½ is the wrong value on this floor

The affinity derivation assumes a **fixed** system curve. This floor's is not fixed, because the VAV
dampers move.

The p10 of the per-step maximum damper position is **100.0 on both AHUs** — at every occupied step
at least one box is already wide open. When duct static falls, those saturated boxes cannot open
further, but the partly-open ones can and do. Opening dampers *flattens* the system curve, so a given
pressure reduction requires less speed reduction than a fixed curve would. The fan slows **less** than
√SP predicts.

Two testable consequences, which is what makes this a prediction rather than a story:

1. **β < ½.** Measured below in §4.
2. **The boxes that cannot open further lose air.** That is Block C of the surrogate
   ([grey-box-surrogate.md](../grey-box-surrogate.md) §4.2), and the reason 64–73% of the modelled
   pressure-reset saving turns out to be ventilation reduction rather than efficiency.

The same physics produces both, which is why Block A's exponent and Block C's flow loss must never be
tuned independently.

---

## 3. The estimator, layer by layer

[src/excitation.py](../src/excitation.py) is a four-layer design. Layer 2 is where the causal work
happens; Layers 1 and 3 are bookkeeping around it.

### Layer 1 — episode detection · [excitation.py:51-117](../src/excitation.py#L51-L117)

The plant lowered its own static-pressure setpoint on tens of occasions, none under operator
override. Those are the natural experiment.

**The baseline is the mode, not the mean** ([excitation.py:51-61](../src/excitation.py#L51-L61)):

```python
def baseline_level(setpoint, eligible):
    values = setpoint[eligible].dropna()
    return float(values.round(3).mode().iloc[0])
```

A commanded setpoint sits at one value most of the time; a *mean* would be dragged off it by the very
episodes being detected. Measured: **0.550 inWG on AHU-1, 0.600 inWG on AHU-2**.

An episode is then a contiguous run held away from that level:

| Rule | Value | Config |
|---|---|---|
| Departure that starts an episode | `\|SP − base\| > 0.02` inWG | [config.yml:118](../config.yml#L118) |
| Steps trimmed from the front (transient) | `settling_steps = 2` (30 min) | [config.yml:81](../config.yml#L81) |
| Minimum run length, before trim | `min_episode_steps + settling = 3 + 2 = 5` | [config.yml:121](../config.yml#L121) |

**Eligibility is the *episode* mask** ([run_control_gap.py:89-92](../scripts/run_control_gap.py#L89-L92)):
on-hours ∧ no-outage ∧ no-override, deliberately **without** the settling component. An episode *is*
a setpoint change, so a mask that removes steps near setpoint changes would delete the subject.
Settling is reapplied inside Layers 1–2 instead, relative to the signal actually being manipulated.
That is the methodologically correct scope for it, and it is easy to get wrong in the other
direction.

**Result: 22 episodes on AHU-1, 30 on AHU-2** → [data/episodes_ahu1.csv](data/episodes_ahu1.csv),
[data/episodes_ahu2.csv](data/episodes_ahu2.csv).

### Layer 2 — pairing each episode against its own control window · [excitation.py:135-202](../src/excitation.py#L135-L202)

Each episode is compared against the eligible steps **immediately before and after it that were at
the normal setpoint** — same day, same weather, same occupancy, same zone loading. This is what does
the causal work, and it is far more convincing than a pooled regression across three months.

| Rule | Value | Config |
|---|---|---|
| Control window width, each side | `control_window_steps = 8` | [config.yml:126](../config.yml#L126) |
| Minimum control steps to keep an episode | `min_control_steps = 4` | [config.yml:127](../config.yml#L127) |
| Post-window starts after | `settling_steps` from episode end | [config.yml:81](../config.yml#L81) |

Both sides are used so that a monotone drift through the day cancels to first order rather than
loading onto the estimate. The pre-window ends *before* the change so it needs no settling trim; the
post-window does, because the loop is still recovering after the setpoint returns.

Then, for the setpoint, the frequency, the fan power and the outdoor-drybulb control:

```python
t, c = np.nanmean(values[treated]), np.nanmean(values[control])
row[f"{col}__dln"] = float(np.log(t / c))
```

Both sides must be finite and strictly positive or the episode is dropped
([excitation.py:190-195](../src/excitation.py#L190-L195)) — imputing either side would invent the
very response being measured.

**The `min_control_steps = 4` requirement is what takes AHU-1 from 22 episodes to 15.** AHU-2 keeps
all 30. → [data/pairs_ahu1.csv](data/pairs_ahu1.csv), [data/pairs_ahu2.csv](data/pairs_ahu2.csv).

> **This table did not previously exist on disk.** `scripts/run_control_gap.py` computes it in
> memory and discards it, writing only the fitted result. It is the table the entire headline
> reduces to, so it is emitted here.

### Layer 3 — the estimator · [excitation.py:287-359](../src/excitation.py#L287-L359)

Two estimators are computed, and the choice between them was stated **before** the answer was seen:

```python
def ratio(a, b): return float(b.sum() / a.sum())              # headline
def slope(a, b): return float((a * b).sum() / (a * a).sum())  # reported alongside
```

So the headline is the **ratio of sums**:

```
                    Σ_e  Δln f_e
        β̂    =    ───────────────
                    Σ_e  Δln SP_e
```

summed over paired episodes *e*. **That is the whole estimator.** Everything in Layers 1–2 exists
only to decide which episodes enter those two sums.

**Why ratio of sums and not least squares** ([excitation.py:295-315](../src/excitation.py#L295-L315)):
these episodes are not a dose-response experiment. The plant cut its setpoint by roughly the same
amount every time, so treatment intensity is nearly constant across episodes (CV **0.33 / 0.36**). In
that regime a through-origin slope is identified almost entirely by the small residual spread in the
treatment — which here is mostly noise — while the ratio of total response to total excitation is
identified by the *level*. The ratio of sums is also the estimator whose bias maps directly onto a
reported kWh total. Both are reported; on this data they agree within 12%, so the choice does not
move the conclusion.

This also explains AHU-1's **negative centred R² (−0.237)**: with almost no spread in the treatment
there is nothing for a slope to explain *across* episodes, and a through-origin fit can be worse than
the mean of the responses while still being the right physical description. The R² is a diagnostic of
the design, not of the effect.

**The interval** is a block bootstrap over whole **episodes** — one episode, one block
([masks.py:210-224](../src/masks.py#L210-L224)); 2,000 resamples, seed 0, percentile 2.5/97.5
([config.yml:130-133](../config.yml#L130-L133)). Resampling *steps* would treat four heavily
autocorrelated 15-minute readings as four independent ones and report an interval roughly half as
wide as it should be.

### Layer 4 — conditioning · [excitation.py:461-481](../src/excitation.py#L461-L481)

A robustness check only, and deliberately limited to **exogenous** controls
([config.yml:152-153](../config.yml#L152-L153) lists exactly one: outdoor drybulb). Airflow, damper
position and cooling rate are all *mediators* of a static-pressure change — conditioning on them
would block the very causal path being measured and drive the estimate toward zero for reasons that
have nothing to do with the plant. On 15 episodes with a free intercept the interval is wide by
construction; it tests whether the headline moves, not what the headline is.

---

## 4. The arithmetic

Because `Σdy/Σdx ≡ mean(dy)/mean(dx)`, and both means are published columns of
[report/elasticities.csv](../report/elasticities.csv), the headline is a division you can do on a
calculator.

| | AHU-1 | AHU-2 *(transient-contaminated)* |
|---|---:|---:|
| Episodes detected → paired | 22 → **15** | 30 → **30** |
| Treated steps | 136 | 118 |
| Baseline setpoint (mode) | 0.550 inWG | 0.600 inWG |
| Median episode length | 13 steps | 5 steps |
| **Σ Δln SP** | −2.614809465184165 | −9.106698231246114 |
| **Σ Δln f** | −0.997769366487989 | −2.108712579151972 |
| mean Δln SP (excitation) | −0.17432063101227768 (−17.4%) | −0.30355660770820380 (−30.4%) |
| mean Δln f (response) | −0.06651795776586593 (−6.7%) | −0.07029041930506573 (−7.0%) |
| **β̂ = Σ Δln f / Σ Δln SP** | **0.3815839661639417** | **0.2315562156124533** |
| 95% CI (episode block bootstrap) | [0.24134, 0.52930] | [0.17783, 0.28272] |
| Slope through origin | 0.34342 | 0.23932 |
| Median episode ratio | 0.43476 | 0.27269 |
| Excitation CV | 0.32812 | 0.35667 |
| Centred R² | −0.23673 | +0.33633 |
| **vs affinity ½** | **not rejected** | **rejected** |

Check the two divisions:

```
        AHU-1:   −0.997769366487989 / −2.614809465184165  =  0.3815839661639417
        AHU-2:   −2.108712579151972 / −9.106698231246114  =  0.2315562156124533
```

Every figure above is a cell of `report/elasticities.csv` rows 2 and 4, or a column sum of
[data/contributions_ahu1.csv](data/contributions_ahu1.csv) / `_ahu2.csv`. Nothing is retyped.

### 4.1 Watching 0.382 assemble

`contributions_ahu1.csv` in full. `running_beta` is the ratio of sums over the first *n* episodes —
its **last row is the published number**, with no further step:

| ep | start | hr | n_t | n_c | SP treat | SP ctrl | Δln SP | Hz treat | Hz ctrl | Δln f | ratio | running β |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-05-18 12:00 | 12 | 3 | 8 | 0.4237 | 0.5500 | −0.2610 | 35.183 | 37.276 | −0.0578 | 0.221 | 0.2214 |
| 2 | 2026-05-19 11:15 | 11 | 25 | 8 | 0.4361 | 0.5485 | −0.2292 | 33.153 | 35.975 | −0.0817 | 0.356 | 0.2845 |
| 6 | 2026-05-25 07:30 | 8 | 13 | 5 | 0.4675 | 0.5486 | −0.1600 | 33.841 | 38.480 | −0.1285 | 0.803 | 0.4121 |
| 7 | 2026-05-25 16:45 | 17 | 3 | 5 | 0.4633 | 0.5452 | −0.1627 | 33.524 | 36.157 | −0.0756 | 0.465 | 0.4226 |
| 8 | 2026-05-26 07:45 | 8 | 26 | 6 | 0.4763 | 0.5443 | −0.1336 | 33.770 | 38.071 | −0.1199 | 0.897 | 0.4896 |
| 9 | 2026-05-27 07:45 | 8 | 7 | 10 | 0.4716 | 0.5494 | −0.1527 | 33.648 | 35.957 | −0.0664 | 0.435 | 0.4820 |
| 10 | 2026-05-28 07:45 | 8 | 5 | 10 | 0.4172 | 0.5500 | −0.2764 | 33.711 | 33.984 | −0.0081 | 0.029 | 0.3910 |
| **11** | 2026-05-29 07:45 | 8 | 6 | 10 | 0.4386 | 0.5500 | −0.2264 | 33.637 | 33.397 | **+0.0072** | **−0.032** | 0.3313 |
| 12 | 2026-06-08 07:45 | 8 | 5 | 9 | 0.4648 | 0.5458 | −0.1606 | 33.772 | 36.271 | −0.0714 | 0.444 | 0.3416 |
| 13 | 2026-06-08 15:30 | 16 | 8 | 8 | 0.4955 | 0.5493 | −0.1030 | 33.127 | 34.526 | −0.0414 | 0.402 | 0.3449 |
| 14 | 2026-06-09 07:45 | 8 | 5 | 5 | 0.4229 | 0.5452 | −0.2539 | 34.119 | 39.575 | −0.1483 | 0.584 | 0.3736 |
| 15 | 2026-06-09 11:15 | 11 | 10 | 4 | 0.4840 | 0.5465 | −0.1214 | 35.583 | 36.437 | −0.0237 | 0.195 | 0.3639 |
| 20 | 2026-06-16 12:00 | 12 | 3 | 15 | 0.4800 | 0.5489 | −0.1342 | 35.294 | 36.323 | −0.0287 | 0.214 | 0.3554 |
| 21 | 2026-06-17 07:45 | 8 | 14 | 6 | 0.4765 | 0.5493 | −0.1423 | 34.606 | 38.592 | −0.1090 | 0.766 | 0.3787 |
| 22 | 2026-06-17 12:00 | 12 | 3 | 9 | 0.4973 | 0.5482 | −0.0974 | 34.800 | 36.385 | −0.0445 | 0.457 | **0.3816** |

Three things this table makes visible that the summary number cannot:

- **Episodes 3, 4, 5, 16–19 are missing.** They were detected but could not be paired: fewer than 4
  eligible at-baseline steps adjacent to them. They are in `episodes_ahu1.csv` and absent from
  `pairs_ahu1.csv`, which is the whole of the 22 → 15 attrition.
- **Episode 11 has a positive response.** The setpoint was cut 20% and the fan ran *faster*, giving a
  per-episode ratio of **−0.032**. This is not an outlier to be removed — it is what a 15-minute
  average of a plant with other things going on looks like, and it is precisely why the median
  episode ratio (0.435) is not used as the headline and why the ratio of sums, which weights by
  excitation, is.
- **The running estimate ranges over 0.22 → 0.49 before settling at 0.38.** Fifteen episodes is a
  small sample, and the interval [0.241, 0.529] is honest about that.

### 4.2 The power elasticity, estimated independently

The same pairs table also yields `d ln kW / d ln SP` (rows 3 and 5 of `elasticities.csv`):

| | AHU-1 | AHU-2 | assumed under a cubic fan law |
|---|---:|---:|---:|
| `d ln kW / d ln SP` | 1.130 [0.770, 1.553] | 1.096 [1.001, 1.196] | 1.500 |

This matters because the frequency response and the power response are measured **independently**,
so the fitted fan curve can be asked to reconcile them —
[effects.py:221-244](../src/effects.py#L221-L244) `check_fan_law_consistency`. It does not close:

```
report/fan_law_consistency.csv
    AHU-1  measured dHz −6.7% implies dkW −12.1%, measured dkW −19.7%   (gap −7.6 pts)
    AHU-2  measured dHz −7.0% implies dkW −17.2%, measured dkW −33.3%   (gap −16.1 pts)
```

The fan is dropping more power than moving along one curve at that speed change can explain. **That
residual is itself the moving system curve** — the same conclusion the sub-affinity exponent points
to, arrived at from a different direction.

---

## 5. Why the number is bounded, not settled

### 5.1 The estimator is pinned against synthetic truth

[tests/test_excitation.py:68-79](../tests/test_excitation.py#L68-L79) builds a square-wave setpoint
with a response of *exactly* `hz = 35·(sp/0.55)^β` and requires the estimator to return β:

```python
@pytest.mark.parametrize("beta", [0.5, 0.183, 1.5, 0.0])
def test_recovers_a_known_elasticity(config, beta):
    ...
    assert result.estimate    == pytest.approx(beta, abs=1e-9)
    assert result.slope_origin == pytest.approx(beta, abs=1e-9)
    assert result.ci_low       == pytest.approx(beta, abs=1e-9)
    assert result.ci_high      == pytest.approx(beta, abs=1e-9)
```

Both CI bounds are pinned too, because a noiseless response has a zero-width interval. A companion
test corrupts the first two steps of every episode with uniform noise and requires the settling trim
to make it change nothing. Without these, a plausible-looking number from real data would only be
plausible.

### 5.2 The specification sweep — the part that decides the verdict

Three judgement calls define an episode, and none is forced by the data
([config.yml:144-147](../config.yml#L144-L147)):

```yaml
    sweep:
      settling_steps: [1, 2, 3]
      min_episode_steps: [2, 3, 4, 6]
      control_window_steps: [4, 8, 12]
```

3 × 4 × 3 = **36 combinations**, all run ([excitation.py:365-415](../src/excitation.py#L365-L415)),
all in [report/specification_sweep.csv](../report/specification_sweep.csv):

| | Usable | β spans | Below 0.50 | Excludes 0.50 at 95% |
|---|---:|---|---:|---:|
| AHU-1 | 36 / 36 | 0.208 → 0.785 | 28 / 36 | **15 / 36** |
| AHU-2 | **24** / 36 | 0.126 → 0.337 | 24 / 24 | 23 / 24 |

**Where AHU-2's "24" comes from** — the number `grey-box-surrogate.md` §2.2 quotes without
explanation: 12 of AHU-2's 36 specifications leave fewer than 3 paired episodes, so `fit_elasticity`
**raises rather than returning a number** ([excitation.py:319-320](../src/excitation.py#L319-L320))
and the row is recorded with a `note` instead of an estimate. The sweep reports 24 usable, not 24
run.

The headline is the grid point `settling = 2, min_episode = 3, control_window = 8` — it is a row of
the sweep, not a separate fit. Verify with:

```
settling_steps,min_episode_steps,control_window_steps,n_episodes,estimate,ci_low,ci_high,excludes_assumed,ahu
2,3,8,15,0.381584,0.241336,0.529300,False,1
2,3,8,30,0.231556,0.177834,0.282716,True,2
```

**This is the finding.** On AHU-1 — the clean unit — the specification moves the answer (0.208 →
0.785) more than the effect being measured does. The data therefore **bounds** the exponent rather
than settling it. Reporting only the specification that rejects ½ would have been indefensible, and
the sweep is what makes that visible.

### 5.3 AHU-2 is precise and untrustworthy

**28 of AHU-2's 30 episodes begin in hour 08** (93%), one at 07, one at 11. Those are the morning
start-up: the plant is warming up, the loops are chasing, and a response measured on them is a
measure of the transient rather than of the setpoint. The runner detects this itself and warns
([run_control_gap.py:166-172](../scripts/run_control_gap.py#L166-L172)).

So the two units fail in opposite directions, and it has to be reported rather than resolved:

|  | AHU-1 | AHU-2 |
|---|---|---|
| Episodes | 15, spread 07h–17h | 30, 93% at 08h |
| Interval | wide, includes ½ | tight, excludes ½ |
| Verdict | trustworthy, imprecise | precise, contaminated |

### 5.4 The confound, stated rather than removed

The episodes were **not randomised**. If the setpoint was lowered *because* load was low, part of the
observed drop belongs to the load. The direction is knowable: that confound biases β̂ **upward**,
i.e. conservatively with respect to the conclusion that the affinity assumption inflates the prize.
Conditioning on outdoor drybulb is a weak control on 15 episodes and the conditioned interval is wide
by construction ([report/elasticities_conditioned.csv](../report/elasticities_conditioned.csv)); it
shows the headline does not collapse, and claims no more.

### 5.5 What the difference is worth

A blunt 20% setpoint cut, priced twice over an identical step set
([report/naive_cut_savings.csv](../report/naive_cut_savings.csv)):

| | measured β | assumed ½ | inflation |
|---|---:|---:|---:|
| AHU-1 | 190 kWh (14.2%) | 240 kWh (17.9%) | **1.26×** |
| AHU-2 | 175 kWh (12.0%) | 346 kWh (23.8%) | **1.98×** |

**The affinity assumption inflates the fan-side prize by roughly 1.3–2× on this floor.** That is what
the 15 rows of §4.1 are worth.

---

## 6. `SP_ref` and `f_ref` — the honest answer

**They are not derived from the data, because they are not defined anywhere in this repository.**

```
$ grep -rn "SP_ref\|f_ref\|sp_ref" .
grey-box-surrogate.md:72     ├─► §2  Block A  : f = f_ref·(SP_act/SP_ref)^β
grey-box-surrogate.md:111            f(t) = f_ref · ( SP_act(t) / SP_ref )^β
grey-box-surrogate.md:260            f_req = f_ref·(SP_sp'/SP_ref)^β  ;   f = min(f_req, 50)
grey-box-surrogate.md:261            SP_act = SP_ref · ( f / f_ref )^{1/β}      when f_req > 50
```

Four hits, one file, all of them the *statement* of the law. The symbols appear in no source file, no
config key, and no `report/` artefact. There is no fitting procedure to reverse-engineer, and the
right response is to say so rather than to reconstruct a provenance after the fact.

**What the code actually does.** [effects.py:106-122](../src/effects.py#L106-L122):

```python
def sp_to_hz(hz, sp_from, sp_to, exponent):
    ratio = np.asarray(sp_to, dtype=float) / np.asarray(sp_from, dtype=float)
    return np.asarray(hz, dtype=float) * np.power(ratio, exponent)
```

and its only caller, [effects.py:177-183](../src/effects.py#L177-L183):

```python
frame = pd.DataFrame({"hz": df[p + "frequency"], "sp": df[p + "static_pressure_setpoint_read"]})[keep]
hz_new = sp_to_hz(frame["hz"], frame["sp"], frame["sp"] * (1 - cut), exponent)
```

`sp_from` and `hz` are **each step's own logged values**. The anchor is per-step and implicit. **The
190 / 175 kWh figures in control-gap §4.4 were produced without any fixed reference point**, which is
legitimate for a one-step replay against logged data and is why the gap went unnoticed.

The only fixed frequency anchor anywhere in the repo is
[run_control_gap.py:270](../scripts/run_control_gap.py#L270), `hz_ref = median frequency over the
evaluation set`, used solely for the fan-law consistency check of §4.2.

### 6.1 Why the surrogate cannot inherit the per-step anchor

Block A in the surrogate runs **closed-loop**. At step 1 there is a logged `(SP, f)` to anchor on; at
step 2 the state is counterfactual and no logged `f` exists for it. The per-step formulation has
nothing to multiply. A fixed `(SP_ref, f_ref)` must therefore be *chosen* and written down.

### 6.2 Three candidate anchors

Computed by `reproduce_beta.py` → [data/candidate_anchors.csv](data/candidate_anchors.csv):

| Anchor | SP_ref (AHU-1 / AHU-2) | f_ref (AHU-1 / AHU-2) | Steps used |
|---|---|---|---|
| **A** per-step (status quo) | — | — | 2,682 / 2,868 |
| **B** modal baseline | 0.550 / 0.600 inWG | **35.919 / 35.625 Hz** | 2,204 / 2,614 |
| **C** episode-mask median | 0.550 / 0.600 inWG | 35.633 / 35.448 Hz | 2,682 / 2,868 |

**A — per-step anchor (what the code does today).** `SP_ref = SP(t)`, `f_ref = f(t)`, both logged;
the law is an incremental multiplier off the current logged state.
*For:* it is exactly what produced every published kWh figure, so adopting it moves no number, and it
introduces no new constant that can go stale.
*Against:* it is not a single `(SP_ref, f_ref)` pair, so Block A as written in §2.2 is not literally
what the code does; and per §6.1 it does not extend to a closed-loop rollout at all.

**B — modal baseline (recommended).** `SP_ref = baseline_level()` — the same mode Layer 1 already
computes, **0.550 / 0.600 inWG**. `f_ref` = mean frequency over eligible steps at that baseline —
**35.919 / 35.625 Hz** — which is the pooled version of the `frequency__control` quantity Layer 2
already computes per episode.
*For:* identification and simulation then share one anchor, so the law is **exact at the operating
point β was measured around** and first-order everywhere else — the strongest property available.
Both numbers fall out of code that already exists.
*Against:* it is a derived constant per AHU that must be recomputed whenever the mask or the episode
definition changes, so it belongs in a `report/` artefact, never hard-coded into the surrogate.

**C — episode-mask median.** `SP_ref` = median setpoint over the episode mask, `f_ref` = median
frequency, following the `hz_ref` precedent at
[run_control_gap.py:270](../scripts/run_control_gap.py#L270).
*For:* precedent already in the repo; a median is robust to the episode definition entirely.
*Against:* the median mixes episode steps into the anchor, so it does not sit at the point β was
identified at. On this data the cost is small but real: `SP_ref` coincides with the mode (0.550 /
0.600) because the plant sits at baseline most of the time, and `f_ref` lands **0.8% below** B
(35.633 vs 35.919 Hz) — the episodes pulling it down. That 0.8% is a pure level shift in every
frequency the surrogate produces, and it propagates cubed through Block B.

**Recommendation: B for the surrogate, A retained for the naive-cut replay** so the published 190 /
175 kWh figures do not move. Whichever is chosen, it must be emitted to `report/` or declared in
`config.yml`. The present situation — a symbol appearing in a specification with no definition
anywhere in the codebase — is the defect this section exists to close.

### 6.3 The extrapolation guard

β was identified from cuts of **~17% (AHU-1) and ~30% (AHU-2)** around baselines of 0.550 / 0.600
inWG. Outside that band the law is extrapolating, which is why
[grey-box-surrogate.md](../grey-box-surrogate.md) §9 clips the agent's actions to the **observed
setpoint support** (0.30–0.64 inWG on AHU-1, 0.40–0.78 on AHU-2) rather than to engineering limits. A
reference point does not extend the range over which β was measured.

---

## 7. Reverse-engineering recipe

```
$ python block-a-derivation/reproduce_beta.py
```

The script asserts its own result against `report/elasticities.csv` to 1e-12 and fails loudly
otherwise. Then, to confirm the headline really is one division of two column sums:

```
$ python -c "import pandas as pd; d = pd.read_csv('block-a-derivation/data/contributions_ahu1.csv'); \
             print(d.dln_f.sum() / d.dln_sp.sum())"
0.3815839661639416
```

(The final digit differs from the in-memory 0.3815839661639**417** because the CSV round-trip sums
the column in a different order. That is float addition being non-associative, not a discrepancy —
the script's own assertion runs before serialisation and holds to 1e-12.)

And to confirm the estimator itself is sound rather than merely self-consistent:

```
$ pytest tests/test_excitation.py -q
```

### Emitted tables

| File | Grain | Key columns |
|---|---|---|
| `data/episodes_ahu{n}.csv` | one **detected** episode | `episode, start_pos, end_pos, treated_start_pos, start, end, n_steps, n_treated, settling_dropped, level, baseline_level, direction, hour` |
| `data/pairs_ahu{n}.csv` | one **paired** episode | `episode, start, hour, n_treated, n_control`, then `{col}__treated`, `{col}__control`, `{col}__dln` for the setpoint, frequency and power |
| `data/contributions_ahu{n}.csv` | one paired episode | `dln_sp, dln_f, episode_ratio, cum_dln_sp, cum_dln_f, running_beta` — §4.1 |
| `data/beta_summary.csv` | one AHU | `episodes_detected, episodes_paired, treated_steps, baseline_sp_inwg, sum_dln_sp, sum_dln_f, beta, ci_low, ci_high, excludes_affinity_half` |
| `data/candidate_anchors.csv` | one anchor × AHU | `anchor, sp_ref, f_ref, n_steps, note` — §6.2 |

**Worked example, `contributions_ahu1.csv` episode 22** (the last row, and the one that lands the
headline):

```
episode          22
start            2026-06-17 12:00:00+07:00
hour             12
n_treated        3            three 15-min steps after the 2-step settling trim
n_control        9            nine adjacent at-baseline steps, pre + post
sp_treated       0.497333     mean setpoint during the episode
sp_control       0.548222     mean setpoint in its control window
dln_sp           -0.097420  = ln(0.497333 / 0.548222)
hz_treated       34.800000
hz_control       36.384622
dln_f            -0.044529  = ln(34.800000 / 36.384622)
episode_ratio     0.457080  = -0.044529 / -0.097420      (this episode alone)
cum_dln_sp       -2.614809    running sum, all 15 episodes
cum_dln_f        -0.997769
running_beta      0.381584  = -0.997769 / -2.614809      <- the published value
```

### Where each published figure lives

| Figure | Source |
|---|---|
| β̂ = 0.382 / 0.232, CIs, slope, CV, R² | `report/elasticities.csv` rows 2, 4 |
| `d ln kW/d ln SP` = 1.130 / 1.096 | `report/elasticities.csv` rows 3, 5 |
| 22 / 30 episodes, baselines, hour distribution | `report/setpoint_episodes.csv` |
| 0.208→0.785, 0.126→0.337, 15/36, 23/24 | `report/specification_sweep.csv` |
| 190 / 175 vs 240 / 346 kWh | `report/naive_cut_savings.csv` |
| fan-law gaps −7.6 / −16.1 pts | `report/fan_law_consistency.csv` |
| every intermediate above | `block-a-derivation/data/` |
