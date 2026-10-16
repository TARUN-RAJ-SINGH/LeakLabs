"""leaklab.score - turn model output into a manager-facing priority queue.

Two quantities are kept deliberately separate:

* **Observed leak value** - summed realised line value of records whose verified
  status is already Cancelled/Returned.  Real money, no double counting.
* **Expected exposure** (*Estimated*) - ``line value x predicted risk`` for a set
  of records.  A forward-looking scenario, never presented as realised revenue.

Tiers come from the operating threshold chosen during training, and the queue is
ranked by expected exposure so limited review capacity goes to the largest
exposure first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .train import score_frame

DRIVER_DIMS = ["category", "channel", "payment_method", "customer_segment", "region",
               "preferred_channel"]


@dataclass
class QueueSummary:
    population: int
    flagged: int
    observed_adverse: int
    base_rate: float
    mean_risk: float
    total_value: float
    observed_leak_value: float
    expected_exposure: float
    threshold: float
    capacity: float

    @property
    def flagged_share(self) -> float:
        return self.flagged / self.population if self.population else 0.0


def score_all(payload: dict, ft: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Score every record with features and attach exposure, tier and drivers."""
    proba = score_frame(payload, ft)
    out = pd.DataFrame(index=ft.index)
    out["record_id"] = ft["record_id"].astype(str)
    out["date"] = pd.to_datetime(ft["record_date"], errors="coerce")
    out["status"] = ft["status_raw"].astype("string")
    out["target_known"] = ft["target_known"].astype(bool)
    out["observed_adverse"] = (ft["target"] == 1) & out["target_known"]
    out["value"] = pd.to_numeric(ft["leak_value"], errors="coerce").fillna(0.0).clip(lower=0)
    out["risk"] = proba

    thr = float(threshold if threshold is not None else payload.get("threshold", 0.5))
    out["flagged"] = out["risk"] >= thr

    # expected exposure: value x risk (Estimated)
    out["expected_exposure"] = out["value"] * out["risk"]

    # tiers: operating cut plus a softer review band
    base = float(pd.to_numeric(ft.loc[ft["eligible"], "target"], errors="coerce").mean()) if "eligible" in ft else np.nan
    band = thr * 0.85
    tier = np.where(out["flagged"], "Review now",
                    np.where(out["risk"] >= band, "Watch", "Routine"))
    out["tier"] = tier

    for d in DRIVER_DIMS:
        if d in ft.columns:
            out[d] = ft[d].astype("string")
    out["record_key"] = out["record_id"]
    return out.sort_values("expected_exposure", ascending=False).reset_index(drop=True)


def summarise(scored: pd.DataFrame, threshold: float, capacity: float) -> QueueSummary:
    pop = int(len(scored))
    flagged = int(scored["flagged"].sum())
    obs_adv = int(scored["observed_adverse"].sum())
    eligible = scored[scored["target_known"]]
    base = float(eligible["observed_adverse"].mean()) if len(eligible) else float("nan")
    return QueueSummary(
        population=pop,
        flagged=flagged,
        observed_adverse=obs_adv,
        base_rate=base,
        mean_risk=float(scored["risk"].mean()) if pop else float("nan"),
        total_value=float(scored["value"].sum()),
        observed_leak_value=float(scored.loc[scored["observed_adverse"], "value"].sum()),
        expected_exposure=float(scored["expected_exposure"].sum()),
        threshold=float(threshold),
        capacity=float(capacity),
    )


def segment_summary(scored: pd.DataFrame, dim: str, top: int = 12) -> pd.DataFrame:
    """Aggregate observed rate and estimated exposure by a driver dimension."""
    if dim not in scored.columns:
        return pd.DataFrame(columns=[dim, "records", "observed_rate", "observed_value",
                                     "mean_risk", "expected_exposure"])
    d = scored.dropna(subset=[dim]).copy()
    if not len(d):
        return pd.DataFrame(columns=[dim, "records", "observed_rate", "observed_value",
                                     "mean_risk", "expected_exposure"])
    g = d.groupby(dim, observed=True)
    out = pd.DataFrame({
        "records": g.size(),
        "observed_rate": g["observed_adverse"].mean(),
        "observed_value": g.apply(lambda x: x.loc[x["observed_adverse"], "value"].sum(), include_groups=False),
        "mean_risk": g["risk"].mean(),
        "expected_exposure": g["expected_exposure"].sum(),
    }).reset_index()
    return out.sort_values("expected_exposure", ascending=False).head(top).reset_index(drop=True)


def exposure_over_time(scored: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Monthly observed leak value and estimated exposure (the two series)."""
    d = scored.dropna(subset=["date"]).copy()
    if not len(d):
        return pd.DataFrame(columns=["period", "observed_value", "expected_exposure", "records"])
    d["period"] = d["date"].dt.to_period(freq).astype(str)
    g = d.groupby("period")
    return pd.DataFrame({
        "records": g.size(),
        "observed_value": g.apply(lambda x: x.loc[x["observed_adverse"], "value"].sum(), include_groups=False),
        "expected_exposure": g["expected_exposure"].sum(),
    }).reset_index()


def recoverable_scenario(expected_exposure: float, recovery_pct: float) -> dict:
    """Estimated recoverable value at a user-visible recovery assumption."""
    pct = float(min(max(recovery_pct, 0.0), 1.0))
    return {
        "expected_exposure": float(expected_exposure),
        "recovery_pct": pct,
        "recoverable": float(expected_exposure) * pct,
        "basis": "Estimated",
        "note": ("Scenario only: expected exposure x the recovery share you choose. "
                 "Not a forecast and not realised revenue."),
    }


def driver_evidence(scored: pd.DataFrame, record: pd.Series) -> list[dict]:
    """For one record, show how it compares with the population on key drivers."""
    rows = []
    for dim in ["category", "channel", "payment_method", "customer_segment", "region"]:
        if dim not in scored.columns or pd.isna(record.get(dim)):
            continue
        val = record[dim]
        grp = scored[scored[dim] == val]
        if not len(grp):
            continue
        rows.append({
            "driver": dim.replace("_", " ").title(),
            "value": str(val),
            "segment_risk": float(grp["risk"].mean()),
            "population_risk": float(scored["risk"].mean()),
            "segment_observed_rate": float(grp["observed_adverse"].mean()) if grp["target_known"].any() else float("nan"),
            "segment_records": int(len(grp)),
        })
    return rows
