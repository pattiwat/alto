"""Frozen parameters for the Floor 8 grey-box, assembled from the derivation artifacts.

    from floor8.plant.params import PlantParams
    p = PlantParams.from_derivations(ROOT)

Nothing here fits anything. Every number is read from a `block-*-derivation/data/*.csv` that some
other script produced, and the provenance of each is carried alongside it so a run can print where
its numbers came from. Fitting at environment-construction time would be slow, would silently use
the test window, and would make the parameter vector an agent trained on unreproducible.

WHY THERE IS AN IMPUTATION LAYER AT ALL
---------------------------------------
The blocks were fitted on overlapping but different box sets, because each excluded what would have
corrupted IT:

    Block C   fits `a_i` on CLEAN boxes only - 23 of 26 and 25 of 27. The dead box and the five
              rogues are excluded so no bad flow reading becomes a room weight.
    Block D   fits 51 of 53. Two boxes never produced a usable thermal regression.
    Block F   reports 41 of 53 as identified; the twelve parked at 27.0 degC never call for cooling,
              so their gain is unidentified rather than wrong.

The SIMULATOR cannot inherit those exclusions, because a zone that was excluded from a fit still
occupies the floor. grey-box-surrogate 4.3 is explicit: rogue boxes leave the demand statistic the
agent observes, but they do NOT leave the comfort constraint - "a starved zone is still a zone."

So every box gets parameters, and every box carries a `provenance` string saying whether they were
fitted or imputed. `PlantParams.provenance_table()` prints it. The imputations are:

  * `a_i` for an excluded box: its own mean flow per point of damper opening, from the record, then
    the whole AHU renormalised to sum to 1. For `vav_8_2_28` - which commands 100% damper and
    delivers ~50 against a 1,000 setpoint - this lands near zero, which is the correct simulated
    behaviour rather than a workaround.
  * Block D parameters for an unfitted box: the AHU median, pooled.
  * Block F gain for an unidentified box: the AHU median, pooled - which is what 5.3(4) instructs
    for the parked twelve anyway.

WHAT IS NOT REPRODUCIBLE IN THIS TREE
-------------------------------------
Blocks A and B both open `from src import ...` and there is no `src/` package here, so
`reproduce_beta.py` and `reproduce_fan_power.py` do not run. Their `data/*.csv` were produced
elsewhere. They are loaded and marked `not-reproduced-here`, never presented as freshly fitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

AHUS = (1, 2)
DT_H = 0.25


# --------------------------------------------------------------------------------------
# per-block parameter blocks
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockA:
    """SP -> Hz elasticity.  f = f_ref * (SP_act/SP_ref)**beta."""
    beta: tuple[float, float]              # per AHU
    beta_ci: tuple[tuple[float, float], ...]
    f_ref: tuple[float, float]
    sp_ref: tuple[float, float]
    provenance: str = "block-a-derivation/data/beta_summary.csv (NOT reproduced here)"


@dataclass(frozen=True)
class BlockB:
    """Hz -> kW.  ALL THREE models are carried; `kind` selects which is live.

    Each kind has its OWN (a, b) - they are not interchangeable. `P = a + b*(f/50)**3` with the
    offset-cubic pair is ~5 kW; the same pair read as `a*f**b` is 1e8. Carrying one pair and
    switching only the functional form is a silent catastrophe, so the coefficients travel with the
    kind and `coeffs()` is the only way to get them.
    """
    models: dict                     # kind -> (a per AHU, b per AHU)
    kind: str = "offset_cubic"
    rated_hz: float = 50.0
    mask: str = "analysis"
    provenance: str = "block-b-derivation/data/models_by_mask.csv (NOT reproduced here)"

    def coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.models[self.kind]
        return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


@dataclass(frozen=True)
class BlockC:
    """Share form.  split by a_i*d_i ; total = C * (sum a_j d_j)**q * SP**s."""
    a_i: np.ndarray                        # (n_boxes,) normalised within AHU
    ln_C: tuple[float, float]
    q: tuple[float, float]
    s: tuple[float, float]
    p: float = 1.0
    provenance: str = "block-c-derivation/data/{room_weights,total_fit}.csv"


@dataclass(frozen=True)
class BlockD:
    """Two-node zone thermal, discrete at dt = 0.25 h.  Coefficients ALREADY carry dt."""
    k: np.ndarray                          # air <-> mass
    a: np.ndarray                          # supply air advection
    g: np.ndarray                          # outdoor (0 on 41 of 51)
    delta: np.ndarray                      # occupancy gain
    c: np.ndarray                          # lumped residual
    parked: np.ndarray                     # bool
    provenance: str = "block-d-derivation/data/stage2_fit.csv"


@dataclass(frozen=True)
class BlockE:
    """SAT bias, mixing map, coil.  `b1 = 0`: block-e ships a CONSTANT bias (see module docstring)."""
    b0: tuple[float, float]
    b1: tuple[float, float]
    sigma: tuple[float, float]
    phi0: tuple[float, float]
    phi1: tuple[float, float]
    cop: float = 4.0
    rho_cp_kj_per_m3_k: float = 1.21
    airflow_unit: str = "m3/h"
    provenance: str = "block-e-derivation/data/{sat_bias_fit,mixing_map,airflow_unit}.csv"


@dataclass(frozen=True)
class BlockF:
    """Zone control.  Stage 1 is a LEVEL: V_sp = A_i + G_i*(T_i - Tsp_i), clipped."""
    G: np.ndarray
    A: np.ndarray
    lam: float = 1.0                       # compensation axis, DR
    provenance: str = "block-f-derivation/data/zone_gain_fit.csv"


@dataclass(frozen=True)
class PlantParams:
    boxes: tuple[str, ...]
    ahu_of_box: np.ndarray                 # (n_boxes,) values in {0, 1} -> index into per-AHU tuples
    A: BlockA
    B: BlockB
    C: BlockC
    D: BlockD
    E: BlockE
    F: BlockF
    v_min: np.ndarray                      # logged minimum_air_flow_rate_setpoint_read, per box
    v_max: np.ndarray                      # logged maximum_air_flow_rate_setpoint_read, per box
    rogue: np.ndarray                      # bool - out of the DEMAND statistic, in the comfort one
    box_provenance: tuple[str, ...] = ()
    dt_h: float = DT_H

    # ---- convenience -------------------------------------------------------------
    @property
    def n_boxes(self) -> int:
        return len(self.boxes)

    def mask_ahu(self, ahu_idx: int) -> np.ndarray:
        return self.ahu_of_box == ahu_idx

    def with_(self, **kw) -> "PlantParams":
        """Return a copy with block(s) replaced. Used by the DR sampler; never mutates."""
        return replace(self, **kw)

    # ---- provenance --------------------------------------------------------------
    def provenance_table(self) -> pd.DataFrame:
        rows = [{"block": n, "source": getattr(self, n).provenance,
                 "reproduces_here": "NOT reproduced here" not in getattr(self, n).provenance}
                for n in ("A", "B", "C", "D", "E", "F")]
        return pd.DataFrame(rows)

    def box_provenance_table(self) -> pd.DataFrame:
        return pd.DataFrame({"box": self.boxes,
                             "ahu": [a + 1 for a in self.ahu_of_box],
                             "provenance": self.box_provenance,
                             "rogue": self.rogue,
                             "a_i": self.C.a_i,
                             "parked": self.D.parked})

    # ---- (de)serialisation -------------------------------------------------------
    def to_json(self, path: Path) -> None:
        def enc(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.bool_, np.integer, np.floating)):
                return o.item()
            if hasattr(o, "__dataclass_fields__"):
                return {k: enc(getattr(o, k)) for k in o.__dataclass_fields__}
            if isinstance(o, tuple):
                return [enc(x) for x in o]
            return o
        Path(path).write_text(json.dumps(enc(self), indent=2), encoding="utf-8")

    # ---- the loader --------------------------------------------------------------
    @classmethod
    def from_derivations(cls, root: Path, config: dict | None = None,
                         fan_mask: str = "analysis") -> "PlantParams":
        root = Path(root)
        if config is None:
            with open(root / "config.yml", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)

        csv = root / config["data"]["csv_path"]
        header = pd.read_csv(csv, nrows=0).columns.tolist()
        boxes = sorted({c.split("__")[0] for c in header if c.endswith("__air_flow_rate")})
        ahu_of_box = np.array([0 if b.startswith("vav_8_1_") else 1 for b in boxes])
        n = len(boxes)
        idx = {b: i for i, b in enumerate(boxes)}
        prov = ["fitted"] * n

        stats = _box_stats(csv, boxes)
        rogue_names = set(config["control_gap"]["benchmark"]["rogue_boxes"])
        rogue = np.array([b in rogue_names for b in boxes])

        # ---- Block A -------------------------------------------------------------
        ba = pd.read_csv(root / "block-a-derivation" / "data" / "beta_summary.csv").set_index("ahu")
        block_a = BlockA(
            beta=tuple(float(ba.beta[a]) for a in AHUS),
            beta_ci=tuple((float(ba.ci_low[a]), float(ba.ci_high[a])) for a in AHUS),
            f_ref=tuple(float(stats["f_ref"][a]) for a in AHUS),
            sp_ref=tuple(float(ba.baseline_sp_inwg[a]) for a in AHUS),
        )

        # ---- Block B -------------------------------------------------------------
        bb = pd.read_csv(root / "block-b-derivation" / "data" / "models_by_mask.csv")
        bb = bb[bb["mask"] == fan_mask]
        models = {}
        for kind in ("offset_cubic", "power_law", "ideal_cubic"):
            k = bb[bb.kind == kind].set_index("ahu")
            models[kind] = (tuple(float(k.a[a]) for a in AHUS),
                            tuple(float(k.b[a]) for a in AHUS))
        block_b = BlockB(models=models, rated_hz=float(bb.rated_hz.iloc[0]), mask=fan_mask)

        # ---- Block C -------------------------------------------------------------
        rw = pd.read_csv(root / "block-c-derivation" / "data" / "room_weights.csv")
        tot = pd.read_csv(root / "block-c-derivation" / "data" / "total_fit.csv").set_index("ahu")
        a_i = np.zeros(n)
        fitted_a = {r.box: float(r.a) for r in rw.itertuples()}
        # `a_i` IS air drawn per point of damper opening, so `mean_flow/mean_damper` should be
        # proportional to it with one constant per AHU. Measured on the FITTED boxes that holds
        # almost exactly - corr 0.9986 (AHU-1) and 0.9955 (AHU-2), with the ratio's interquartile
        # range spanning 0.00771-0.00779 on AHU-1. So the scale is read off the boxes Block C did
        # fit and applied to the ones it excluded, rather than inventing a number. The tightness of
        # that ratio is also a small independent check that the weight means what 4.2 says it does.
        for ai, ahu in enumerate(AHUS):
            sel = [i for i in range(n) if ahu_of_box[i] == ai]
            vd = {b: (stats["mean_flow"][b] / stats["mean_damper"][b]
                      if stats["mean_damper"].get(b, 0.0) > 0 else np.nan) for b in boxes}
            ratios = [fitted_a[boxes[i]] / vd[boxes[i]] for i in sel
                      if boxes[i] in fitted_a and np.isfinite(vd[boxes[i]]) and vd[boxes[i]] > 0]
            scale = float(np.median(ratios)) if ratios else 0.0
            for i in sel:
                b = boxes[i]
                if b in fitted_a:
                    a_i[i] = fitted_a[b]
                else:
                    # For vav_8_2_28 - 100% damper, ~50 delivered against a 1,000 setpoint - this
                    # lands near zero, which is the behaviour the box actually exhibits rather than
                    # a workaround for it.
                    a_i[i] = 0.0 if not np.isfinite(vd[b]) else max(scale * vd[b], 0.0)
                    prov[i] = "a_i IMPUTED (excluded from Block C's fit)"
            # renormalise the AHU back to sum 1 - the split is relative by construction
            tot_a = a_i[sel].sum()
            if tot_a > 0:
                a_i[sel] /= tot_a
        block_c = BlockC(a_i=a_i,
                         ln_C=tuple(float(tot.ln_C[a]) for a in AHUS),
                         q=tuple(float(tot.q[a]) for a in AHUS),
                         s=tuple(float(tot.s[a]) for a in AHUS),
                         p=float(config["control_gap"]["zone_flow"]["damper_share_exponent"]))

        # ---- Block D -------------------------------------------------------------
        bd = pd.read_csv(root / "block-d-derivation" / "data" / "stage2_fit.csv")
        k, a_d, g, delta, c_d = (np.zeros(n) for _ in range(5))
        parked = np.zeros(n, dtype=bool)
        have = {r.box: r for r in bd.itertuples()}
        med = {ai: bd[bd.ahu == ahu].median(numeric_only=True) for ai, ahu in enumerate(AHUS)}
        for i, b in enumerate(boxes):
            r = have.get(b)
            m = med[int(ahu_of_box[i])]
            if r is None:
                k[i], a_d[i], g[i] = float(m.k_fixed), float(m.a), float(m.g)
                delta[i], c_d[i] = float(m.delta), float(m.c)
                parked[i] = True
                prov[i] = (prov[i] + "; " if prov[i] != "fitted" else "") + "Block D POOLED (unfitted)"
            else:
                k[i], a_d[i], g[i] = float(r.k_fixed), float(r.a), float(r.g)
                delta[i], c_d[i] = float(r.delta), float(r.c)
                parked[i] = bool(r.parked)
        block_d = BlockD(k=k, a=a_d, g=g, delta=delta, c=c_d, parked=parked)

        # ---- Block E -------------------------------------------------------------
        be = pd.read_csv(root / "block-e-derivation" / "data" / "sat_bias_fit.csv")
        be = be[be.regressor.str.startswith("none")].set_index("ahu")
        mx = pd.read_csv(root / "block-e-derivation" / "data" / "mixing_map.csv").set_index("ahu")
        block_e = BlockE(b0=tuple(float(be.b0[a]) for a in AHUS),
                         b1=(0.0, 0.0),
                         sigma=tuple(float(be.sigma_k[a]) for a in AHUS),
                         phi0=tuple(float(mx.phi0[a]) for a in AHUS),
                         phi1=tuple(float(mx.phi1[a]) for a in AHUS))

        # ---- Block F -------------------------------------------------------------
        bf = pd.read_csv(root / "block-f-derivation" / "data" / "zone_gain_fit.csv")
        G, A_f = np.zeros(n), np.zeros(n)
        ident = bf[bf.identified]
        gmed = {ai: (float(ident[ident.ahu == ahu].G.median()),
                     float(ident[ident.ahu == ahu].A.median())) for ai, ahu in enumerate(AHUS)}
        rows_f = {r.box: r for r in bf.itertuples()}
        for i, b in enumerate(boxes):
            r = rows_f.get(b)
            if r is not None and bool(r.identified) and np.isfinite(r.G):
                G[i], A_f[i] = float(r.G), float(r.A)
            else:
                # 5.3(4): pool rather than report a confident number derived from no excitation.
                G[i], A_f[i] = gmed[int(ahu_of_box[i])]
                prov[i] = (prov[i] + "; " if prov[i] != "fitted" else "") + "Block F POOLED (unidentified)"
        block_f = BlockF(G=G, A=A_f)

        return cls(boxes=tuple(boxes), ahu_of_box=ahu_of_box,
                   A=block_a, B=block_b, C=block_c, D=block_d, E=block_e, F=block_f,
                   v_min=np.array([stats["v_min"].get(b, 0.0) for b in boxes]),
                   v_max=np.array([stats["v_max"].get(b, 0.0) for b in boxes]),
                   rogue=rogue, box_provenance=tuple(prov))


def _box_stats(csv: Path, boxes: list[str]) -> dict:
    """Record-level statistics the imputations and the reference points need."""
    cols = ["timestamp_local"]
    for b in boxes:
        cols += [f"{b}__air_flow_rate", f"{b}__damper_position",
                 f"{b}__minimum_air_flow_rate_setpoint_read",
                 f"{b}__maximum_air_flow_rate_setpoint_read"]
    for a in AHUS:
        cols += [f"ahu_b8_{a}__frequency"]
    head = pd.read_csv(csv, nrows=0).columns.tolist()
    df = pd.read_csv(csv, usecols=[c for c in cols if c in head])
    on = {}
    for a in AHUS:
        f = df[f"ahu_b8_{a}__frequency"]
        on[a] = float(f[f > 5.0].median())
    return {
        "mean_flow": {b: float(df[f"{b}__air_flow_rate"].mean()) for b in boxes},
        "mean_damper": {b: float(df[f"{b}__damper_position"].mean()) for b in boxes},
        "v_min": {b: float(df[f"{b}__minimum_air_flow_rate_setpoint_read"].median()) for b in boxes},
        "v_max": {b: float(df[f"{b}__maximum_air_flow_rate_setpoint_read"].median()) for b in boxes},
        "f_ref": on,
    }
