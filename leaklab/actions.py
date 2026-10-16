"""leaklab.actions - map a leak to a concrete intervention, and log it.

The action library is deliberately small and operational: each entry names the
leak type, the recommended intervention, an owner role, the expected direction of
the metric, and the metric that will *validate* the intervention.  The log is a
plain CSV on disk (plus session state) - enough to close the loop in a demo
without a database.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

LOG_COLUMNS = ["logged_at", "scope", "leak_type", "action", "owner",
               "expected_direction", "validation_metric", "baseline_value", "due_date", "status",
               "action_id"]
ACTION_ID_COLUMN = "action_id"
ACTION_STATUSES = ("Open", "In progress", "Done")
_STATUS_ALIASES = {
    "open": "Open",
    "in progress": "In progress",
    "in-progress": "In progress",
    "in_progress": "In progress",
    "progress": "In progress",
    "done": "Done",
    "complete": "Done",
    "completed": "Done",
    "closed": "Done",
}


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
    # Keep this field last so existing positional construction remains valid.
    action_id: str = field(default_factory=lambda: new_action_id())

    def __post_init__(self) -> None:
        self.action_id = normalize_action_id(self.action_id) or new_action_id()
        self.status = normalize_status(self.status)

    def row(self) -> dict:
        d = asdict(self)
        return {c: d.get(c, "") for c in LOG_COLUMNS}


def new_action_id() -> str:
    """Return a unique identifier that remains stable once written to a log."""
    return f"act_{uuid4().hex}"


def normalize_action_id(value: Any) -> str:
    """Return a safe, comparable action ID; missing values become an empty string."""
    if value is None or value is pd.NA:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        # IDs should be scalar, but avoid raising if a caller passes an unusual value.
        pass
    return str(value).strip()


def normalize_status(value: Any, default: str = "Open") -> str:
    """Normalize the statuses used by the action UI without raising on bad input."""
    fallback_text = normalize_action_id(default)
    fallback = (_STATUS_ALIASES.get(fallback_text.casefold(), fallback_text)
                if fallback_text else "")
    text = normalize_action_id(value).casefold()
    return _STATUS_ALIASES.get(text, fallback)


def _value_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _legacy_id(row: Mapping[str, Any], occurrence: int) -> str:
    """Build a repeatable ID for a row from a pre-action_id CSV."""
    parts = [f"{column}={_value_text(row.get(column, ''))}"
             for column in LOG_COLUMNS if column != ACTION_ID_COLUMN]
    payload = "\x1f".join(parts) + f"\x1foccurrence={occurrence}"
    return f"act_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _with_action_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with one stable, non-empty ID per row."""
    out = df.copy()
    if ACTION_ID_COLUMN not in out.columns:
        out[ACTION_ID_COLUMN] = ""

    seen: set[str] = set()
    occurrences: dict[str, int] = {}
    id_position = out.columns.get_loc(ACTION_ID_COLUMN)
    for position, (_, row) in enumerate(out.iterrows()):
        current = normalize_action_id(row[ACTION_ID_COLUMN])
        fingerprint = "\x1f".join(
            f"{column}={_value_text(row.get(column, ''))}"
            for column in LOG_COLUMNS if column != ACTION_ID_COLUMN
        )
        occurrence = occurrences.get(fingerprint, 0)
        if not current or current in seen:
            current = _legacy_id(row, occurrence)
            while current in seen:
                occurrence += 1
                current = _legacy_id(row, occurrence)
        occurrences[fingerprint] = occurrence + 1
        seen.add(current)
        # iloc avoids surprises if a caller supplied a DataFrame with duplicate indexes.
        out.iloc[position, id_position] = current
    return out


def _log_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a log frame to the current schema without mutating the input."""
    out = df.copy()
    for column in LOG_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return _with_action_ids(out[LOG_COLUMNS])


def update_row_by_id(df: pd.DataFrame, action_id: Any,
                     updates: Mapping[str, Any]) -> pd.DataFrame:
    """Return a copy with one uniquely identified log row updated.

    Unknown IDs, blank IDs, duplicate IDs, and unsupported columns are safe no-ops.
    The identifier itself is immutable so callers cannot accidentally retarget a row.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(updates, Mapping):
        return df.copy() if isinstance(df, pd.DataFrame) else df

    target = normalize_action_id(action_id)
    if not target:
        return _log_frame(df)
    if ACTION_ID_COLUMN in df.columns:
        raw_matches = sum(normalize_action_id(value) == target
                          for value in df[ACTION_ID_COLUMN])
        if raw_matches != 1:
            return _log_frame(df)

    out = _log_frame(df)
    matches = [position for position, value in enumerate(out[ACTION_ID_COLUMN])
               if normalize_action_id(value) == target]
    if len(matches) != 1:
        return out

    row_position = matches[0]
    for column, value in updates.items():
        if column in LOG_COLUMNS and column != ACTION_ID_COLUMN:
            if column == "status":
                value = normalize_status(value, default="")
                if not value:
                    continue
            out.iloc[row_position, out.columns.get_loc(column)] = value
    return out


def update_action_status(df: pd.DataFrame, action_id: Any, status: Any) -> pd.DataFrame:
    """Return a copy with a valid status applied to exactly one action row."""
    normalized = normalize_status(status, default="")
    if not normalized:
        return _log_frame(df) if isinstance(df, pd.DataFrame) else df
    return update_row_by_id(df, action_id, {"status": normalized})


def default_due_date(horizon_days: int) -> str:
    return (date.today() + pd.Timedelta(days=horizon_days)).strftime("%Y-%m-%d")


def load_log(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        return _log_frame(pd.read_csv(p))
    except Exception:
        return pd.DataFrame(columns=LOG_COLUMNS)


def _write_log(path: Path, df: pd.DataFrame) -> None:
    """Write a complete log atomically so schema migration cannot half-write a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            for row in _log_frame(df).to_dict(orient="records"):
                writer.writerow({column: row.get(column, "") for column in LOG_COLUMNS})
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def append_log(path: str | Path, record: ActionRecord) -> pd.DataFrame:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_log(p) if p.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    combined = pd.concat(
        [existing, pd.DataFrame([record.row()], columns=LOG_COLUMNS)],
        ignore_index=True,
    )
    try:
        # Rewriting the complete file also upgrades old headers before appending,
        # preventing a new value from being shifted under a legacy header.
        _write_log(p, combined)
    except Exception:
        # fall back to a session-only record if the disk is not writable
        pass
    return load_log(p)


def update_log_status(path: str | Path, action_id: Any, status: Any) -> pd.DataFrame:
    """Safely persist a status change and return the resulting log frame.

    Invalid statuses and unknown/ambiguous IDs leave the on-disk log untouched.
    """
    p = Path(path)
    current = load_log(p)
    updated = update_action_status(current, action_id, status)
    if not isinstance(updated, pd.DataFrame) or updated.equals(current):
        return current
    try:
        _write_log(p, updated)
    except Exception:
        # Return what is actually on disk if a local write fails.
        return load_log(p)
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
