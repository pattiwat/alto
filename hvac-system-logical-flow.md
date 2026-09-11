# Alto Building Floor 8 — HVAC System and Its Logical Flow

> **What this is.** A reconstruction of the plant on Floor 8 and the control logic that runs it,
> derived from the point profile in
> [from-altotech-take-home-pdf-and-alto-bui-jolly-turing.md](from-altotech-take-home-pdf-and-alto-bui-jolly-turing.md).
> That document is a delivery-strategy memo, not a system description — it profiles the data but never
> lays out how the machine actually works. This file fills that gap.
>
> Section references (§1.1, §1.5, …) point back to the source memo, which remains the authority on
> every number quoted here. Where something is inferred rather than stated, it is flagged as such —
> see the closing section.


## 1. The physical chain

```
Central chilled-water plant   ◄── OFF-FLOOR. Not metered, not in the dataset.
        │                          (This is why no chiller COP exists — §1.9)
        │ CHWS ≈ 44.8 °F (7.1 °C)
        ▼
┌──────────────────── AHU-1 / AHU-2 (identical, no staging) ────────────────────┐
│                                                                               │
│  return air ──►(+ outside air)──► COOLING COIL ──► VSD SUPPLY FAN ──► duct    │
│                                        ▲                 ▲                    │
│                              cooling_valve %      fan_frequency Hz            │
│                                        ▲                 ▲                    │
│                              [SAT PID loop]     [static-pressure PID loop]    │
└────────┬──────────────────────────────────────────────────┬───────────────────┘
   CHWR (ΔT 10–11 K)                                        │ SAT 16.8 / 18.3 °C
                                                            ▼
                       duct ──► static pressure sensor ──► (feeds fan PID)
                                                            │
        ┌──────────────┬───────────────┬───────────────┬────┴─────────┐
        ▼              ▼               ▼               ▼              ▼
     VAV box        VAV box         VAV box         VAV box    … 53 boxes total
   damper % / flow                                            (26 on AHU-1, 27 on AHU-2)
        │  ▲
        ▼  └── [zone PI: room_temperature vs setpoint → flow setpoint → damper]
      ZONE ──► room_temperature ──┘

Parallel, not in the AHU chain:
  14 FCUs        — independent perimeter / after-hours cooling
  2 exhaust fans — toilet / pantry extract
  13 IAQ sensors — CO2 / PM / RH / PIR: monitoring only, not in any control loop
  weather station — outdoor drybulb (°F) / wetbulb, observed but not used by the loops
```

Cool water comes from a plant you cannot see. It removes heat from air at the coil. A variable-speed
fan pushes that cold air down a duct. Each zone throttles how much of it it takes with a damper.

That is the whole machine. Everything else is the control layer deciding two numbers: **valve position**
and **fan speed**.

---

## 2. The control layers, innermost first

### Layer 1 — zone loops (53 of them; fastest-acting on comfort)

Each VAV box compares `room_temperature` against `setpoint_read`. Too warm → it raises its airflow
setpoint (bounded below by `minimum_air_flow_rate_setpoint_read` and above by a max-flow setpoint) →
modulates the damper until measured flow matches. These are pressure-independent boxes: the damper
chases *flow*, not position, so it opens further as duct pressure sags. The collective output of these
53 loops is the load the AHU sees.

Zone setpoints are not uniform, and the spread is itself a finding (§1.6):

| Setpoint | Zones |
|---|---|
| **27.0 °C** | `vav_8_1_{1,2,3,4,5,10,11,14}`, `vav_8_2_{19,20,21,22}` — 12 zones, effectively switched off |
| 24.4 °C | `vav_8_1_{6,8}`, `vav_8_2_{6,8}` |
| 24.0 °C | `vav_8_2_26` |
| 23.5 °C | `vav_8_1_{15,17,18}` |
| 23.0 °C | the rest — **these are the only zones actually driving demand** |

### Layer 2 — duct static pressure loop (one per AHU)

A pressure sensor partway down the duct feeds a PID that drives `fan_frequency` on the VSD. The
setpoint sits at 0.55 inWG on AHU-1 and 0.60 on AHU-2 (≈ 137 / 149 Pa) most of the time — but it is
**modal, not fixed**: 122 / 146 distinct values, ranging 0.300–0.636 and 0.400–0.780. Those
departures are the natural experiment that
[control-gap-method.md](control-gap-method.md) uses to measure the setpoint → frequency link.

This is the loop that closes the system: dampers open → pressure falls → fan speeds up → more air,
more fan kW. AHU-1's loop tracks its setpoint with a median error of 0.000 — it is healthy. AHU-2 falls
below 0.8× setpoint on **12% of steps**, i.e. it sometimes cannot make pressure at all (§1.5).

### Layer 3 — supply air temperature loop (one per AHU)

A second PID compares supply air temperature against the SAT setpoint (18.3 °C on AHU-1, 16.8 °C on
AHU-2) and modulates the chilled-water `cooling_valve`. The valve sits around **43% mean** on both
units and is saturated above 95% only 2.4% / 3.9% of the time — there is substantial coil authority
in reserve (§1.5).

### Layer 4 — schedule

A BMS time schedule starts both AHUs on weekdays around 07:00 and stops them around 18:00. On-fraction
is 0.87–0.90 from 08:00–17:00, 0.70 at 07:00, 0.05 at 06:00 (§1.4).

Both AHUs run together on 29.3% of steps, AHU-2 alone on 2.0%, neither on 68.5% — **there is no staging
logic; they run as a pair.** AHU-2 carries a small weekend/overnight baseline (~7% of weekend hours);
AHU-1 essentially none. PIR occupancy sensors exist and broadly agree with the schedule, but they are
observation, not input.

### Layer 5 — the operator

Above all of it, `override_control` and `auto_manual_control_mode_read` show humans taking the plant by
hand on **~8% of steps** (7.85% AHU-1, 8.36% AHU-2; manual mode 7.64% on AHU-1). The values are
fractional — 1.067, 1.2, 0.333 — because on/off points are logged as fraction-of-slot-on, so an
override engaged for part of a 15-minute bucket. Whatever the loops decided in those windows, a person
decided otherwise (§1.8).

### The missing Layer 6 — supervisory reset

There is no *demand-driven* one: nothing resets a setpoint from zone demand, and no logic in the
record responds to how open the dampers are. **That absence is the gap Task B fills.**

> **CORRECTED.** An earlier version of this section said the setpoints "are constants for the entire
> three-month period". They are not. Both static-pressure setpoints move (§2 Layer 2 above), and both
> SAT setpoints move too — 1.27 K of on-hours range on AHU-1, 3.00 K on AHU-2, stepping hour to hour.
> Something is writing them; it is simply not doing so from zone demand. The correction matters
> because those movements are measurable excitation — see
> [control-gap-method.md](control-gap-method.md).

---

## 3. The causal chain that makes it all move

This is the loop to hold in your head, because every energy argument rides on it:

```
zone gets warm → damper opens → duct static pressure drops
   → fan PID raises Hz → fan power rises as ≈ a + b·(f/50)³
   → more airflow across the coil → SAT rises above setpoint
   → SAT PID opens the cooling valve → more chilled water flow
   → more coil load (cooling_rate) → more chiller energy off-floor
```

Two energy paths follow from it, and they are not equally knowable:

| Path | Magnitude | Observability |
|---|---|---|
| **Fan (electrical)** | 4,297 kWh = 19.4% of floor electricity | **Fully measured inside this dataset.** damper % → SP setpoint → Hz → kW is logged end to end. |
| **Coil (thermal)** | 18,732 kWh-thermal — **4.4× the fan** | Measured *thermally* via `cooling_rate`, but stops at the floor boundary. Converting to electricity needs a chiller COP that no point in this dataset carries. |

That asymmetry — the bigger prize sitting behind an unmeasurable conversion — is what drives the
strategy trade-off in the source memo (§1.9, §2.3).

---

## 4. How it is actually behaving, loop by loop

The system works in the sense that it runs. It does not work in the sense that each loop is doing its
job.

**The SAT loop is mis-tuned, not starved.** AHU-1 overcools by **1.13 K on average** against its own
setpoint while its valve sits at 43%. There is authority to spare — the loop simply is not holding.
|error| > 1 K on 43% of steps (39% on AHU-2). Any SAT reset must therefore be presented as *fix the
loop, then reset it* (§1.5).

**The valves are hunting.** Mean absolute valve travel is 6.1% / 6.9% per *15-minute average*, with p95
at 27.9 / 30.8. Averaged data that agitated implies the underlying 1-minute signal is oscillating hard.

**The zones are overcooled downstream of that.** AHU-1 rooms average **2.60 K below** their setpoint and
are more than 1 K below on **50.3%** of on-hours steps (p5 of zone temperature = 15.4 °C). AHU-2 is
−0.62 K. This is comfort being *over*-bought (§1.6).

**The pressure loop can never relax, because at least one box is always wide open.** The p10 of the
per-step maximum damper position is **100.0 on both AHUs** — at literally every occupied timestep, some
zone is calling flat out. A textbook trim-and-respond that trims only when zero zones request would
therefore trim on 0.0% of steps.

Some of those calls are false:

| Box | Symptom | Reading |
|---|---|---|
| `vav_8_1_28` | damper has exactly **1 unique value** (pinned 100%) | Room T median 28.9 °C, max 30.8 — hottest zone on the floor. Genuinely starved, or a failed actuator. |
| `vav_8_2_28` | damper pinned 100%, flow max 50.5 against a 1,000 max-flow setpoint | Disconnected / failed box. Commands 100%, delivers nothing. |
| `vav_8_2_15`, `vav_8_2_23`, `vav_8_1_26`, `vav_8_1_16` | mean damper 81–97% | Near-rogue |

**The water side is fine.** Water ΔT is 10.2–11.3 K against a typical 5.5 K design — *high*, not low.
This floor has the opposite of the over-pumping condition, which is what rules out chilled-water
delta-T management as a strategy (§1.5, §2.3 Option 5).

Put together: **a healthy pressure loop chasing a fixed setpoint it can never be allowed to lower,
feeding zones that are already too cold, fed by a temperature loop that overshoots while sitting at
less than half its valve authority.**

---

## 5. The telemetry layer (and how it constrains a controller)

A Modbus gateway polls 93 devices. The historian buckets everything to a 15-minute grid — analogues
averaged, binaries reported as **fraction of the slot the point was on**, which is why status and
override points carry fractional values.

Four contiguous collector outages account for **7.0% of steps** (§1.1), including a 4-day gap
30 July → 3 August. The gateway's own health block sailed straight through it reporting
`connection 1, devices 4/4, error_count 0` — the health telemetry never detected the outage (§1.7).

One structural point matters for control rather than analysis:

> `ex_b8_{1,2}__status_write` and `ahu_b8_1__status_write` are **constant 0** while the matching
> `status_read` shows the equipment genuinely running (0.116 / 0.355 / 0.31).

**The write points are not commanding.** Any supervisory controller built on this floor cannot actuate
through those channels as they stand — it can only write setpoints (static pressure setpoint, SAT
setpoint), not start/stop commands (§1.7).

---

## 6. Inferred vs. stated

Two things above go beyond what the source document actually asserts:

1. **The mixing section is partly inferred.** ~~Nothing confirms an OA damper~~ — **corrected:**
   `ahu_b8_{1,2}__fresh_air_damper_position_read` is a logged point (909 unique values, 0–100% on
   AHU-1), so the outside-air damper is observed, not inferred. What remains inferred is the
   *economizer logic*: nothing in the data shows the damper being sequenced against outdoor
   conditions, and in Bangkok an economizer would be near-useless anyway, so a fixed minimum-OA
   arrangement with a modulating damper is the likely reality.

2. **The internal structure of the zone loop is inferred.** "PI on room temperature producing a flow
   setpoint, damper chasing flow" is the standard pressure-independent VAV arrangement and is
   consistent with the logged points (`damper`, `flow`, `minimum_air_flow_rate_setpoint_read`,
   max-flow setpoint) — but the document does not state the control algorithm inside the box
   controllers.

Everything else — setpoints, tracking errors, damper statistics, energy totals, outage extents,
override fractions — is taken directly from the profiled data and traces to the section references
given inline.
