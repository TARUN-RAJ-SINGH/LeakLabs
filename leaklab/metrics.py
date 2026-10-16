"""leaklab.metrics - metric functions returning value, definition, availability."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Model, classify_status  # noqa: F401  (classify_status re-exported)


def format_inr(value, decimals: int = 0) -> str:
    """Format a number as Indian Rupees with Indian digit grouping (e.g. ₹1,23,45,678)."""
    if value is None:
        return "Unavailable"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    neg = num < 0
    raw = f"{abs(num):.{decimals}f}"
    int_part, dot, frac = raw.partition(".")
    int_part = int_part.replace(",", "")
    if len(int_part) > 3:
        head, tail = int_part[:-3], int_part[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        int_part = ",".join(groups) + "," + tail
    sign = "-" if neg else ""
    return f"{sign}₹{int_part}" + (dot + frac if dot else "")


@dataclass
class Metric:
    key: str
    label: str
    value: float | int | str | None
    unit: str = ""
    definition: str = ""
    formula: str = ""
    availability: str = "Observed"      # Observed | Estimated | Unavailable
    numerator: str = ""
    denominator: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def display(self) -> str:
        if self.value is None:
            return "Unavailable"
        v = self.value
        if isinstance(v, str):
            return v
        if self.unit == "currency":
            return format_inr(v)
        if self.unit == "pct":
            return f"{v*100:,.1f}%"
        if isinstance(v, float):
            return f"{v:,.2f}"
        return f"{v:,}"


# --------------------------------------------------------------------------- #
# Status role classification now lives in leaklab.data (single source of truth).
# --------------------------------------------------------------------------- #
def _sum_value(df: pd.DataFrame, statuses: set[str]) -> float:
    if "__status" in df.columns and statuses:
        df = df[df["__status"].isin(statuses)]
    return float(df["line_value"].sum())


# --------------------------------------------------------------------------- #
def compute_metrics(model: Model) -> dict[str, Metric]:
    tx = model.tx
    roles = classify_status(tx["__status"]) if "__status" in tx.columns else None
    status_ok = bool(model.flags.get("status_mapped")) and "__status" in tx.columns
    m: dict[str, Metric] = {}

    m["records"] = Metric(
        "records", "Transaction records", int(len(tx)), unit="count",
        definition="Rows in the transaction base grain after any date filter.",
        availability="Observed")

    if model.flags.get("date_mapped"):
        lo, hi = tx["__date"].min(), tx["__date"].max()
        m["date_range"] = Metric("date_range", "Date range", f"{lo:%Y-%m-%d} to {hi:%Y-%m-%d}",
                                 definition="Min/max parsed transaction date.",
                                 availability="Observed")
    else:
        m["date_range"] = Metric("date_range", "Date range", None,
                                 definition="No parseable date column.",
                                 availability="Unavailable")

    # ---- value metrics -----------------------------------------------------
    if model.flags.get("amount_available") and status_ok:
        completed = roles["completed"] if roles else set()
        m["realized_revenue"] = Metric(
            "realized_revenue", "Net realized revenue", _sum_value(tx, completed), unit="currency",
            definition="Realized line value of fulfilled/completed orders only (excludes cancelled, returned and pending).",
            formula=f"sum(quantity x {model.flags.get('amount_field')}) where status in {sorted(completed)}",
            availability="Observed")
        m["cancelled_value"] = Metric(
            "cancelled_value", "Cancelled order value", _sum_value(tx, roles["cancelled"]), unit="currency",
            definition="Realized line value of orders whose status indicates cancellation.",
            formula="sum(line value) where status is cancelled", availability="Observed")
        m["returned_value"] = Metric(
            "returned_value", "Returned/refunded value", _sum_value(tx, roles["returned"]), unit="currency",
            definition="Realized line value of orders whose status indicates a return/refund.",
            formula="sum(line value) where status is returned", availability="Observed")
        m["pending_value"] = Metric(
            "pending_value", "Open/pending value", _sum_value(tx, roles["pending"]), unit="currency",
            definition="Realized line value of orders not yet fulfilled or closed (excluded from realized revenue).",
            formula="sum(line value) where status is pending", availability="Observed")
    else:
        for k, lbl in [("realized_revenue", "Net realized revenue"),
                       ("cancelled_value", "Cancelled order value"),
                       ("returned_value", "Returned/refunded value"),
                       ("pending_value", "Open/pending value")]:
            m[k] = Metric(k, lbl, None, unit="currency",
                          definition="Requires a verified amount/price, quantity and order-status field.",
                          availability="Unavailable")

    # ---- rates -------------------------------------------------------------
    if status_ok and roles:
        total = len(tx)
        n_can = int(tx["__status"].isin(roles["cancelled"]).sum())
        m["cancellation_rate"] = Metric(
            "cancellation_rate", "Cancellation rate", (n_can / total) if total else None, unit="pct",
            definition="Cancelled order lines divided by all placed order lines (base grain).",
            formula=f"{n_can} cancelled / {total} placed",
            availability="Observed" if total else "Unavailable", denominator=float(total),
            numerator=f"{n_can} cancelled lines")

        fulfilled = int(tx["__status"].isin(roles["completed"] | roles["returned"]).sum())
        n_ret = int(tx["__status"].isin(roles["returned"]).sum())
        m["return_rate"] = Metric(
            "return_rate", "Return rate (of fulfilled)", (n_ret / fulfilled) if fulfilled else None, unit="pct",
            definition="Returned/refunded order lines divided by fulfilled/delivered order lines (completed + returned).",
            formula=f"{n_ret} returned / {fulfilled} fulfilled",
            availability="Observed" if fulfilled else "Unavailable", denominator=float(fulfilled),
            numerator=f"{n_ret} returned lines",
            warnings=[] if fulfilled else ["No fulfilled orders - denominator unavailable."])
        m["return_rate_all"] = Metric(
            "return_rate_all", "Return rate (of all lines)", (n_ret / total) if total else None, unit="pct",
            definition="Returned order lines divided by all placed order lines (alternative denominator).",
            formula=f"{n_ret} returned / {total} placed", availability="Observed" if total else "Unavailable",
            numerator=f"{n_ret} returned lines",
            denominator=float(total))
    else:
        for k, lbl in [("cancellation_rate", "Cancellation rate"),
                       ("return_rate", "Return rate (of fulfilled)"),
                       ("return_rate_all", "Return rate (of all lines)")]:
            m[k] = Metric(k, lbl, None, unit="pct",
                          definition="Requires a verified order status field.", availability="Unavailable")

    # ---- discount ----------------------------------------------------------
    if "__discount" in tx.columns and model.flags.get("discount_mapped") and model.flags.get("amount_available"):
        if "__gross_value" in tx.columns and tx["__gross_value"].notna().any():
            gross = float(tx["__gross_value"].sum())
            disc_val = float(tx["__discount_value"].sum())
            m["discount_value"] = Metric("discount_value", "Discount value", disc_val, unit="currency",
                                         definition="Gross line value minus realized line value, using the observed discount rate.",
                                         formula="sum(gross - realized)", availability="Observed")
            m["discount_rate"] = Metric("discount_rate", "Discount rate", (disc_val / gross) if gross else None,
                                        unit="pct",
                                        definition="Discount value divided by gross (pre-discount) line value.",
                                        formula=f"{disc_val:,.0f} / {gross:,.0f}",
                                        availability="Observed" if gross else "Unavailable", denominator=gross)
        else:
            m["discount_rate"] = Metric("discount_rate", "Discount rate", float(tx["__discount"].mean()), unit="pct",
                                        definition="Mean observed discount rate per line.",
                                        formula="mean(discount)", availability="Observed")
    else:
        for k, lbl in [("discount_rate", "Discount rate"), ("discount_value", "Discount value")]:
            m[k] = Metric(k, lbl, None, unit="currency" if k.endswith("value") else "pct",
                          definition="Requires a verified discount field.", availability="Unavailable")

    # ---- inventory ---------------------------------------------------------
    inv = model.latest_inventory
    if inv is not None and "__on_hand" in inv.columns:
        below = None
        if "__reorder" in inv.columns:
            below = inv[inv["__on_hand"] <= inv["__reorder"]]
            m["low_stock_count"] = Metric(
                "low_stock_count", "Products at/below reorder point", int(len(below)), unit="count",
                definition="Latest inventory snapshot per product with on-hand quantity <= the mapped reorder threshold.",
                formula="count(product) where on_hand <= reorder_threshold",
                availability="Observed", denominator=int(len(inv)))
        zero = inv[inv["__on_hand"] == 0]
        m["zero_stock_count"] = Metric("zero_stock_count", "Products with zero on-hand", int(len(zero)), unit="count",
                                       definition="Latest snapshot with zero on-hand quantity.",
                                       formula="count(product) where on_hand == 0", availability="Observed")
    else:
        for k, lbl in [("low_stock_count", "Products at/below reorder point"),
                       ("zero_stock_count", "Products with zero on-hand")]:
            m[k] = Metric(k, lbl, None, unit="count",
                          definition="Requires an inventory file with on-hand quantity.",
                          availability="Unavailable")

    # ---- campaigns (independent) ------------------------------------------
    camp = model.campaigns
    if camp is not None:
        sm = next((c for c in camp.columns if re.search(r"spend", str(c), re.I)), None)
        cv = next((c for c in camp.columns if re.search(r"conversion", str(c), re.I)), None)
        if sm:
            spend = float(pd.to_numeric(camp[sm], errors="coerce").fillna(0).sum())
            m["campaign_spend"] = Metric("campaign_spend", "Campaign spend (independent)", spend, unit="currency",
                                         definition="Total campaign spend. Campaigns have no shared key with transactions, so this is shown independently (no attribution).",
                                         formula=f"sum({sm})", availability="Observed")
            if cv:
                conv = float(pd.to_numeric(camp[cv], errors="coerce").fillna(0).sum())
                m["campaign_cpa"] = Metric("campaign_cpa", "Avg cost / conversion", (spend / conv) if conv else None,
                                           unit="currency",
                                           definition="Total campaign spend divided by total recorded conversions (independent of transactions).",
                                           formula=f"{spend:,.0f} / {conv:,.0f}",
                                           availability="Observed" if conv else "Unavailable")
    return m


# --------------------------------------------------------------------------- #
# helpers used by rules
# --------------------------------------------------------------------------- #
def rate_table(tx: pd.DataFrame, group_col: str, statuses: set[str],
               denominator_statuses: set[str] | None = None) -> pd.DataFrame:
    """Return per-group numerator/denominator/rate for a status-rate rule."""
    if group_col not in tx.columns or "__status" not in tx.columns:
        return pd.DataFrame(columns=[group_col, "num", "den", "rate"])
    if denominator_statuses is not None:
        elig = tx[tx["__status"].isin(denominator_statuses)].copy()
    else:
        elig = tx.copy()
    num = elig.assign(_n=elig["__status"].isin(statuses).astype(int)).groupby(group_col)["_n"].sum()
    den = elig.groupby(group_col).size()
    out = pd.DataFrame({"num": num, "den": den}).dropna()
    out["rate"] = out["num"] / out["den"]
    out = out.reset_index()
    return out
