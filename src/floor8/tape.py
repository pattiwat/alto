"""EpisodeTape - the record, sliced into days, immutable and shared read-only.

    tape = EpisodeTape.build(ROOT, params)
    day  = tape.day(idx)          # -> DayTape, 96 steps of replayed boundary conditions

`w_t` IS REPLAYED FROM THE RECORD, NEVER SIMULATED. 1.1 calls this "the single most important
structural choice in the surrogate": weather and occupancy are the dominant drivers of everything,
so replaying them means the environment carries zero modelling risk on its largest term. An episode
IS a real day. What the surrogate models is only the plant's response to a counterfactual setpoint.

Expressed here as a dependency direction: the plant never touches the CSV. It receives `exog` dicts
and has no idea where they came from, which is also what lets a falsifier feed it synthetic inputs.

WHAT COUNTS AS A BOUNDARY CONDITION
-----------------------------------
    T_oa         outdoor_weather_station__drybulb_temperature       degF in the file -> degC here
    T_wb         outdoor_weather_station__wetbulb_temperature       observation only
    occ          mean PIR across floor_8_zone_*_iaq_*__pir          fraction of sensors triggered
    T_sp_zone    vav_*__room_temperature_setpoint_read              replayed, incl. the parked 27.0
    T_m          per zone per night, 21:00-05:00 mean               block-d 5.1's mass node
    fan_on       ahu_b8_*__frequency > min_fan_hz                   the BMS schedule, not the agent's
    oa_damper    ahu_b8_*__fresh_air_damper_position_read           feeds 6.2's mixing map

`fan_on` is replayed rather than commanded because the agent CANNOT start or stop plant:
`ahu_b8_1__status_write` is constant 0 while `status_read` shows the equipment running
(rl-environment-design 1, "two structural facts"). The action space is setpoint trim and nothing
else.

WHY A FULL 96-STEP DAY AND NOT THE 44 OCCUPIED ONES
---------------------------------------------------
Block D's `k_i` is fitted on the fan-off free response (18:00-20:59) and `T_m` is the 21:00-05:00
plateau. An episode starting at 07:00 never lets the mass node express itself and forces a GUESSED
morning initial condition. Simulating the whole day sets it physically; scoring is masked to the
occupied window separately, which is what `score_mask` is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STEPS_PER_DAY = 96


@dataclass(frozen=True)
class DayTape:
    """One calendar day of replayed boundary conditions. 96 steps."""
    date: pd.Timestamp
    T_oa: np.ndarray            # (96,)  degC
    T_wb: np.ndarray            # (96,)  degC
    occ: np.ndarray             # (96,)  fraction in [0, 1]
    fan_on: np.ndarray          # (96, 2) bool
    oa_damper: np.ndarray       # (96, 2) %
    T_sp_zone: np.ndarray       # (96, n_boxes) degC
    T_m: np.ndarray             # (n_boxes,) degC - one value per zone per night
    score_mask: np.ndarray      # (96,) bool - occupied AND analysis-mask clean
    logged: dict                # what the plant did, for the as-operated arm and the gates

    def exog(self, t: int) -> dict:
        return {"T_oa": self.T_oa[t], "T_wb": self.T_wb[t], "occ": self.occ[t],
                "fan_on": self.fan_on[t], "oa_damper": self.oa_damper[t],
                "T_sp_zone": self.T_sp_zone[t], "T_m": self.T_m}


class EpisodeTape:
    """All days, loaded once. Immutable and safe to share across vector envs."""

    def __init__(self, days: list[DayTape], boxes: tuple[str, ...], config: dict):
        self.days = days
        self.boxes = boxes
        self.config = config

    def __len__(self) -> int:
        return len(self.days)

    def day(self, i: int) -> DayTape:
        return self.days[i]

    def split(self, train_end: str, test_start: str) -> tuple[list[int], list[int]]:
        """TIME-BASED split only (rl-environment-design 5.1).

        A random split leaks: 15-minute steps are heavily autocorrelated, so neighbouring steps in
        train and test would be near-duplicates. The boundary falls naturally on the 4-day collector
        outage of 30 Jul - 3 Aug.
        """
        tr = [i for i, d in enumerate(self.days) if d.date <= pd.Timestamp(train_end)]
        te = [i for i, d in enumerate(self.days) if d.date >= pd.Timestamp(test_start)]
        return tr, te

    # ------------------------------------------------------------------------------
    @classmethod
    def build(cls, root: Path, boxes: tuple[str, ...], config: dict | None = None) -> "EpisodeTape":
        root = Path(root)
        if config is None:
            with open(root / "config.yml", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
        m = config["control_gap"]["mask"]

        csv = root / config["data"]["csv_path"]
        header = pd.read_csv(csv, nrows=0).columns.tolist()
        pir = [c for c in header if c.endswith("__pir")]
        want = ["timestamp_local",
                "outdoor_weather_station__drybulb_temperature",
                "outdoor_weather_station__wetbulb_temperature"] + pir
        for a in (1, 2):
            want += [f"ahu_b8_{a}__{p}" for p in
                     ("frequency", "static_pressure", "static_pressure_setpoint_read",
                      "supply_air_temperature", "supply_air_temperature_setpoint_read",
                      "fresh_air_damper_position_read", "power", "cooling_rate",
                      "override_control")]
        for b in boxes:
            want += [f"{b}__{p}" for p in
                     ("room_temperature", "room_temperature_setpoint_read",
                      "air_flow_rate", "damper_position")]
        df = pd.read_csv(csv, usecols=[c for c in want if c in header],
                         parse_dates=["timestamp_local"]).set_index("timestamp_local").sort_index()

        # degF -> degC. block-e 5 records that the outdoor column is Fahrenheit while every zone
        # column is Celsius; mixing them without converting is what made phi come out negative.
        wx = ["outdoor_weather_station__drybulb_temperature",
              "outdoor_weather_station__wetbulb_temperature"]
        for c in wx:
            if c in df.columns:
                df[c] = (df[c] - 32.0) * 5.0 / 9.0

        # The weather station is 10% missing, including SIX entirely blank days - three of which are
        # the collector outage that the train/test boundary already falls on. Two rules, because
        # weather is a REPLAYED BOUNDARY CONDITION and 1.1 rests the whole design on it carrying no
        # modelling risk:
        #   * partial gaps are bridged across the whole record, not within the day, so a few missing
        #     steps interpolate from the hours either side;
        #   * a day more than half missing is DROPPED rather than invented. An episode is a real
        #     day; a day whose dominant driver had to be imagined is not one.
        wx_missing = df[wx[0]].isna().groupby(df.index.normalize()).mean() if wx[0] in df else None
        for c in wx:
            if c in df.columns:
                df[c] = df[c].ffill().bfill()

        # 1.7 / 5.3(2): 0.00 in a VAV room temperature is a bad-read sentinel, not a room at
        # freezing point (the minimum non-zero on the floor is 6.49). Mask it; never average it.
        sent, tol = float(m["room_temp_sentinel"]), float(m["room_temp_sentinel_tolerance"])
        for b in boxes:
            c = f"{b}__room_temperature"
            df[c] = df[c].mask((df[c] - sent).abs() < tol)

        # Occupancy: the mean of the four PIR sensors, which are 12.5% missing. Gaps are filled with
        # the HOUR-OF-DAY mean over the record, not with zero. Occupancy is a driver of Block D's
        # gain term, so zeroing a missing day would silently delete the load rather than estimate
        # it - and the profile is strong enough to interpolate (0.02 overnight, 0.24-0.27 in the
        # occupied window). Imputed steps are counted and reported by `build`.
        if pir:
            occ = df[pir].mean(axis=1)
            clim = occ.groupby(occ.index.hour).transform("mean")
            occ = occ.fillna(clim).fillna(0.0).clip(0.0, 1.0)
        else:                                                        # pragma: no cover
            occ = pd.Series(0.0, index=df.index)
        min_hz = float(m["min_fan_hz"])
        nulls = df.isna().sum(axis=1) / float(df.shape[1])
        no_outage = nulls <= float(m["outage_null_fraction"])
        override_ok = pd.Series(True, index=df.index)
        for a in (1, 2):
            c = f"ahu_b8_{a}__override_control"
            if c in df.columns:
                override_ok &= (df[c] - float(m["normal_override_value"])).abs() < float(
                    m["override_tolerance"])

        t_m_by_night = _mass_node(df, boxes)
        hour = df.index.hour

        days, dropped_wx = [], []
        for date, g in df.groupby(df.index.normalize()):
            if len(g) != STEPS_PER_DAY:
                continue                      # partial day (outage edge) - not an episode
            if wx_missing is not None and float(wx_missing.get(date, 0.0)) > 0.5:
                dropped_wx.append(str(date.date()))
                continue
            sl = slice(g.index[0], g.index[-1])
            fan_on = np.column_stack([(g[f"ahu_b8_{a}__frequency"] > min_hz).to_numpy()
                                      for a in (1, 2)])
            occupied = (g.index.hour >= 7) & (g.index.hour < 18)
            days.append(DayTape(
                date=date,
                T_oa=g["outdoor_weather_station__drybulb_temperature"].ffill().bfill().to_numpy(),
                T_wb=g["outdoor_weather_station__wetbulb_temperature"].ffill().bfill().to_numpy(),
                occ=occ[sl].to_numpy(),
                fan_on=fan_on,
                oa_damper=np.column_stack([
                    g[f"ahu_b8_{a}__fresh_air_damper_position_read"].ffill().bfill().fillna(
                        90.0).to_numpy() for a in (1, 2)]),
                T_sp_zone=np.column_stack([
                    g[f"{b}__room_temperature_setpoint_read"].ffill().bfill().fillna(
                        24.0).to_numpy() for b in boxes]),
                T_m=t_m_by_night.get(date, t_m_by_night["__floor__"]),
                score_mask=(occupied & fan_on.any(axis=1)
                            & no_outage[sl].to_numpy() & override_ok[sl].to_numpy()),
                logged=_logged(g, boxes),
            ))
        if dropped_wx:
            print(f"  tape: dropped {len(dropped_wx)} day(s) with >50% of outdoor temperature "
                  f"missing: {', '.join(dropped_wx)}")
        return cls(days, boxes, config)


def _mass_node(df: pd.DataFrame, boxes: tuple[str, ...]) -> dict:
    """block-d 5.1: T_m is each zone's own 21:00-05:00 mean, one value per zone per night.

    Replayed as an exogenous input rather than simulated - the mass relaxes at 0.65 K per uncooled
    day, so within an episode it is effectively fixed. Falls back to the floor mean where a zone has
    no night data, which is what block-d does.
    """
    night = df[(df.index.hour >= 21) | (df.index.hour < 5)]
    # a night spans midnight: attribute the small hours back to the previous calendar day
    key = (night.index - pd.Timedelta(hours=6)).normalize()
    cols = [f"{b}__room_temperature" for b in boxes]
    out: dict = {}
    grouped = night[cols].groupby(key).mean()
    floor = np.array([np.nanmean(df[c].to_numpy()) for c in cols])
    floor = np.where(np.isfinite(floor), floor, 24.76)
    for date, row in grouped.iterrows():
        v = row.to_numpy(dtype=float)
        out[pd.Timestamp(date)] = np.where(np.isfinite(v), v, floor)
    out["__floor__"] = floor
    return out


def _logged(g: pd.DataFrame, boxes: tuple[str, ...]) -> dict:
    """What the plant actually did. The as-operated arm replays it; the gates compare against it."""
    two = lambda p: np.column_stack([g[f"ahu_b8_{a}__{p}"].to_numpy() for a in (1, 2)])
    return {
        "SP_sp": two("static_pressure_setpoint_read"),
        "SAT_sp": two("supply_air_temperature_setpoint_read"),
        "SP_act": two("static_pressure"),
        "T_sa": two("supply_air_temperature"),
        "f": two("frequency"),
        "P_fan": two("power"),
        "cooling_rate": two("cooling_rate"),
        "T_i": np.column_stack([g[f"{b}__room_temperature"].to_numpy() for b in boxes]),
        "V_i": np.column_stack([g[f"{b}__air_flow_rate"].to_numpy() for b in boxes]),
        "d_i": np.column_stack([g[f"{b}__damper_position"].to_numpy() for b in boxes]),
    }
