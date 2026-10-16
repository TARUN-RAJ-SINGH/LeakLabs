"""leaklab.actions - map a leak to a concrete intervention, and log it.

The action library is deliberately small and operational: each entry names the
leak type, the recommended intervention, an owner role, the expected direction of
the metric, and the metric that will *validate* the intervention.  The log is a
plain CSV on disk (plus session state) - enough to close the loop in a demo
without a database.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

LOG_COLUMNS = ["logged_at", "scope", "leak_type", "action", "owner",
               "expected_direction", "validation_metric", "baseline_value", "due_date", "status"]


@dataclass
class ActionTemplate:
    leak_type: str
    action: str
    owner: str
    expected_direction: str
    validation_metric: str
    horizon_days: int = 30


#: Transparent, rule-of-thumb interventions.  None of these assume a verified
#: cost/margin field, so none promises a profit figure.
LIBRARY: list[ActionTemplate] = [
    ActionTemplate(
        "Cancellations",
        "Re-check payment authorisation and stock availability for the flagged SKUs before they are promoted again.",
        "Operations / Payments manager",
        "Cancellation rate of flagged SKUs should fall.",
        "Cancellation rate of flagged SKUs over the next 30 days.", 30),
    ActionTemplate(
        "Returns",
        "Audit the top returned products for listing-accuracy or quality issues and pause the worst offenders.",
        "Category / Merchandising manager",
        "Return rate of the audited products should fall.",
        "Fulfilled return rate by product over the next 30 days.", 30),
    ActionTemplate(
        "Low stock",
        "Replenish the highest-demand below-reorder SKUs first; escalate suppliers with recorded delay.",
        "Supply chain / Replenishment manager",
        "Below-reorder product count should fall.",
        "Count of products at or below the reorder point after replenishment.", 21),
    ActionTemplate(
        "Discount erosion",
        "Pause the lowest-yield discount bands and reprice where the discount no longer lifts volume.",
        "Pricing / Marketing manager",
        "Discount rate should fall without a matching drop in volume.",
        "Discount rate and units sold over the next 30 days.", 30),
    ActionTemplate(
        "Revenue anomaly",
        "Drill the low-value window by status, product and region before any pricing change.",
        "Commercial manager",
        "7-day realised revenue should recover toward the prior window.",
        "7-day realised revenue vs the prior window.", 14),
    ActionTemplate(
        "Priority queue (model)",
        "Work the flagged queue top-down: contact the highest-exposure open orders and confirm delivery intent.",
        "Revenue operations manager",
        "Observed adverse rate among worked records should fall below the population base rate.",
        "Adverse rate among worked records in the next cohort.", 30),
]


def templates_for(leak_type: str) -> list[ActionTemplate]:
    hits = [t for t in LIBRARY if t.leak_type.lower() == str(leak_type).lower()]
    return hits or [t for t in LIBRARY if t.leak_type == "Priority queue (model)"]


def all_templates() -> list[ActionTemplate]:
    return list(LIBRARY)


@dataclass
class ActionRecord:
    scope: str
    leak_type: str
    action: str
    owner: str
    expected_direction: str
    validation_metric: str
    baseline_value: str = ""
    due_date: str = ""
    status: str = "Open"
    logged_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    def row(self) -> dict:
        d = asdict(self)
        return {c: d.get(c, "") for c in LOG_COLUMNS}


def default_due_date(horizon_days: int) -> str:
    return (date.today() + pd.Timedelta(days=horizon_days)).strftime("%Y-%m-%d")


def load_log(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(p)
        for c in LOG_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[LOG_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=LOG_COLUMNS)


def append_log(path: str | Path, record: ActionRecord) -> pd.DataFrame:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    try:
        with p.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(record.row())
    except Exception:
        # fall back to a session-only record if the disk is not writable
        pass
    return load_log(p)


def log_summary(df: pd.DataFrame) -> dict:
    if df is None or not len(df):
        return {"total": 0, "open": 0, "owners": 0, "leak_types": 0}
    return {
        "total": int(len(df)),
        "open": int((df["status"].astype(str).str.lower() == "open").sum()),
        "owners": int(df["owner"].astype(str).replace("", pd.NA).nunique()),
        "leak_types": int(df["leak_type"].astype(str).replace("", pd.NA).nunique()),
    }
