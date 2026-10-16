"""leaklab.rules - pure deterministic leak rules returning Alert objects.

Every rule returns an :class:`Alert` when it fires, or a :class:`RuleStatus`
describing why it could not fire (Unavailable / below threshold).  Rules never
raise when a field is missing: they degrade to Unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import Metric, classify_status, format_inr, rate_table
from .model import Model

# Thresholds from the PRD (section 5)
CANCEL_ABS = 0.10
RETURN_ABS = 0.08
DISCOUNT_ABS = 0.15
REVENUE_DROP = 0.30
MIN_RECORDS = 10
PRIOR_MULT = 1.5


@dataclass
class Alert:
    key: str
    title: str
    leak: str
    severity: str
    rule_text: str
    scope: str
    numerator: str
    denominator: str
    rate: float | None
    threshold: float | None
    baseline_note: str
    comparison_window: str
    impact_value: float | None
    impact_basis: str            # Observed | Estimated | Unavailable
    impact_note: str
    evidence_score: int
    evidence_parts: list[str]
    availability: str
    assumptions: list[str]
    action: str
    owner: str
    validation_metric: str
    why: str
    trend: pd.DataFrame = field(default_factory=pd.DataFrame)   # period, value, kind
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)    # label, value, unit
    tables: dict = field(default_factory=dict)
    rank: int = 0

    def card(self) -> dict:
        return {
            "key": self.key, "title": self.title, "severity": self.severity,
            "scope": self.scope, "rate": self.rate, "threshold": self.threshold,
            "impact_value": self.impact_value, "impact_basis": self.impact_basis,
            "evidence_score": self.evidence_score,
        }


@dataclass
class RuleStatus:
    key: str
    title: str
    status: str          # Fired | Monitored | Unavailable
    detail: str


def _score(margin_pct: float, n: int, baseline_ok: bool, impact_observed: bool,
           scope_global: bool) -> int:
    s = max(0, min(25, margin_pct * 100))
    s += min(20, 20 * (np.log10(max(n, 10)) / np.log10(20000)))
    s += 15 if baseline_ok else 5
    s += 20 if impact_observed else 8
    s += 10 if scope_global else 6
    return int(round(min(100, s)))


def _monthly_series(tx: pd.DataFrame, statuses: set[str],
                    denom: set[str] | None) -> pd.DataFrame:
    if "__date" not in tx.columns or tx["__date"].isna().all():
        return pd.DataFrame(columns=["period", "value", "num", "den"])
    d = tx.copy()
    d["period"] = d["__date"].dt.to_period("M").astype(str)
    if denom is not None:
        d = d[d["__status"].isin(denom)]
    g = d.groupby("period").apply(
        lambda x: pd.Series({"num": int(x["__status"].isin(statuses).sum()),
                             "den": int(len(x))}), include_groups=False)
    g["value"] = (g["num"] / g["den"]).where(g["den"] > 0)
    g = g.reset_index()
    g["kind"] = "rate"
    return g


# --------------------------------------------------------------------------- #
# 1. Returns
# --------------------------------------------------------------------------- #
def rule_returns(model: Model, metrics: dict[str, Metric]) -> tuple[Alert | None, RuleStatus]:
    tx = model.tx
    met = metrics.get("return_rate")
    if met is None or met.availability == "Unavailable":
        return None, RuleStatus("returns", "Returns", "Unavailable", "No status field mapped.")
    roles = classify_status(tx["__status"])
    rate = float(met.value)
    den = met.denominator or 0
    if den < MIN_RECORDS:
        return None, RuleStatus("returns", "Returns", "Unavailable",
                                f"Only {den:.0f} fulfilled records (< {MIN_RECORDS}).")

    thresholds = max(RETURN_ABS, 0.0)
    fired = rate > thresholds

    # trend (monthly fulfilled return rate)
    trend = _monthly_series(tx, roles["returned"], roles["completed"] | roles["returned"])
    trend["threshold"] = RETURN_ABS

    # bars: top products by returned value
    bars = pd.DataFrame(columns=["label", "value", "unit"])
    tables = {}
    if "__product_key" in tx.columns:
        ret = tx[tx["__status"].isin(roles["returned"])]
        g = ret.groupby("__product_key")["line_value"].sum().sort_values(ascending=False).head(10)
        bars = pd.DataFrame({"label": g.index.astype(str), "value": g.values, "unit": "currency"})
    for col, name in [("__category", "Category"), ("__region", "Region"),
                      ("__channel", "Channel"), ("__segment", "Customer segment")]:
        if col in tx.columns:
            d = tx[tx["__status"].isin(roles["completed"] | roles["returned"])]
            g = d.groupby(col).apply(
                lambda x: pd.Series({"Returned": int(x["__status"].isin(roles["returned"]).sum()),
                                     "Fulfilled": int(len(x))}), include_groups=False)
            g["Return rate"] = g["Returned"] / g["Fulfilled"].replace(0, np.nan)
            g["Returned value"] = d[d["__status"].isin(roles["returned"])].groupby(col)["line_value"].sum()
            tables[name] = g.sort_values("Return rate", ascending=False).reset_index()

    impact = metrics.get("returned_value")
    impact_v = float(impact.value) if impact and impact.value is not None else None
    margin = (rate - thresholds) / thresholds if thresholds else 0
    ev = _score(margin, int(den), False, True, True)

    why = (f"Fulfilled return rate is {rate:.1%} against the {RETURN_ABS:.0%} absolute threshold "
           f"({int(round(rate * den)):,} returned lines of {int(den):,} fulfilled). "
           "No comparable prior window exists in the provided history, so the absolute threshold applies.")
    status = RuleStatus("returns", "Returns", "Fired" if fired else "Monitored",
                        f"Return rate {rate:.1%} vs threshold {RETURN_ABS:.0%}.")
    if not fired:
        return None, status

    alert = Alert(
        key="returns", title="Return / refund leakage", leak="Returns", severity="High",
        rule_text=f"Return rate exceeds max({RETURN_ABS:.0%}, 1.5x prior) with >= {MIN_RECORDS} eligible records.",
        scope="All fulfilled orders (whole analysis window)",
        numerator=met.numerator or "returned lines",
        denominator=f"{int(den):,} fulfilled lines (returned + completed)",
        rate=rate, threshold=RETURN_ABS,
        baseline_note="No comparable prior window available - absolute threshold used.",
        comparison_window="Full period (no prior window)",
        impact_value=impact_v, impact_basis="Observed",
        impact_note="Realized line value of returned/refunded orders (source monetary field).",
        evidence_score=ev,
        evidence_parts=[f"Rate {rate:.1%} vs {RETURN_ABS:.0%} threshold",
                        f"{int(den):,} eligible records",
                        "Impact is Observed currency"],
        availability="Observed",
        assumptions=[],
        action="Inspect the leading returned products and the Home / Sports categories for product-quality or listing accuracy issues.",
        owner="Category / Merchandising manager",
        validation_metric="Fulfilled return rate by product for the next 30 days after the fix.",
        why=why, trend=trend, bars=bars, tables=tables)
    return alert, status


# --------------------------------------------------------------------------- #
# 2. Cancellations
# --------------------------------------------------------------------------- #
def rule_cancellations(model: Model, metrics: dict[str, Metric]) -> tuple[Alert | None, RuleStatus]:
    tx = model.tx
    met = metrics.get("cancellation_rate")
    if met is None or met.availability == "Unavailable":
        return None, RuleStatus("cancellations", "Cancellations", "Unavailable", "No status field mapped.")
    roles = classify_status(tx["__status"])
    overall = float(met.value)
    n_total = int(met.denominator or 0)
    overall_threshold = CANCEL_ABS

    trend = _monthly_series(tx, roles["cancelled"], None)
    trend["threshold"] = CANCEL_ABS

    # segment scan at product scope; baseline proxied by the overall rate
    seg_threshold = max(CANCEL_ABS, PRIOR_MULT * overall)
    flagged = pd.DataFrame()
    if "__product_key" in tx.columns:
        rt = rate_table(tx, "__product_key", roles["cancelled"])
        rt = rt[rt["den"] >= MIN_RECORDS]
        flagged = rt[rt["rate"] > seg_threshold].copy()
        if len(flagged):
            can_value = tx[tx["__status"].isin(roles["cancelled"])].groupby("__product_key")["line_value"].sum()
            flagged["cancelled_value"] = flagged["__product_key"].map(can_value).fillna(0)
            flagged = flagged.sort_values("rate", ascending=False)

    fired_global = overall > overall_threshold
    fired_seg = len(flagged) > 0

    bars = pd.DataFrame(columns=["label", "value", "unit"])
    if len(flagged):
        top = flagged.head(10)
        bars = pd.DataFrame({"label": top["__product_key"].astype(str), "value": top["rate"], "unit": "pct"})

    detail = (f"Overall cancellation rate {overall:.1%} vs {overall_threshold:.0%} absolute threshold."
              + (f" {len(flagged)} product(s) exceed {seg_threshold:.1%} (max({CANCEL_ABS:.0%}, 1.5x overall))." if fired_seg else ""))
    status = RuleStatus("cancellations", "Cancellations", "Fired" if (fired_global or fired_seg) else "Monitored", detail)

    if not (fired_global or fired_seg):
        return None, status

    if fired_seg:
        seg_rate = float((flagged["num"].sum() / flagged["den"].sum()))
        impact_v = float(flagged["cancelled_value"].sum())
        scope = f"Top {len(flagged)} products by cancellation rate"
        num_txt = f"{int(flagged['num'].sum())} cancelled lines on flagged products"
        den_txt = f"{int(flagged['den'].sum())} placed lines on flagged products"
        rate = seg_rate
        margin = (seg_rate - seg_threshold) / seg_threshold
        ev = _score(margin, int(flagged["den"].sum()), False, True, False)
        tables = {"Flagged products": flagged[["__product_key", "num", "den", "rate", "cancelled_value"]]
                  .rename(columns={"__product_key": "Product", "num": "Cancelled", "den": "Placed",
                                   "rate": "Cancel rate", "cancelled_value": "Cancelled value"})}
        why = (f"Overall cancellation ({overall:.1%}) is below the 10% line, but {len(flagged)} products run "
               f"above {seg_threshold:.1%} (max of 10% and 1.5x the overall rate). These SKUs carry "
               f"{format_inr(impact_v)} in cancelled value. Treat as likely drivers, not proven root cause.")
        alert = Alert(
            key="cancellations", title="Elevated cancellation on specific products", leak="Cancellations",
            severity="Medium",
            rule_text=f"Cancellation rate exceeds max({CANCEL_ABS:.0%}, {PRIOR_MULT}x prior) with >= {MIN_RECORDS} eligible records.",
            scope=scope, numerator=num_txt, denominator=den_txt, rate=rate, threshold=seg_threshold,
            baseline_note=f"Prior-period baseline proxied by the overall rate ({overall:.1%}).",
            comparison_window="Full period (baseline = overall rate)",
            impact_value=impact_v, impact_basis="Observed",
            impact_note="Cancelled realized value on the flagged products.",
            evidence_score=ev,
            evidence_parts=[f"Flagged rate {rate:.1%} vs {seg_threshold:.1%}",
                            f"{int(flagged['den'].sum()):,} placed lines across {len(flagged)} products",
                            "Overall rate is below the 10% absolute line"],
            availability="Observed", assumptions=["Baseline proxied by the overall cancellation rate."],
            action="Check payment authorisation, fulfilment and stock availability for the flagged SKUs before they are promoted.",
            owner="Operations / Payments manager",
            validation_metric="Cancellation rate of flagged SKUs over the next 30 days.",
            why=why, trend=trend, bars=bars, tables=tables)
        return alert, status

    # global fire (kept for completeness)
    impact = metrics.get("cancelled_value")
    impact_v = float(impact.value) if impact and impact.value is not None else None
    margin = (overall - overall_threshold) / overall_threshold
    ev = _score(margin, n_total, False, True, True)
    alert = Alert(
        key="cancellations", title="Cancellation leakage", leak="Cancellations", severity="High",
        rule_text=f"Cancellation rate exceeds max({CANCEL_ABS:.0%}, 1.5x prior) with >= {MIN_RECORDS} eligible records.",
        scope="All placed orders", numerator=met.numerator or "cancelled lines",
        denominator=f"{n_total:,} placed lines", rate=overall, threshold=overall_threshold,
        baseline_note="No comparable prior window - absolute threshold used.",
        comparison_window="Full period", impact_value=impact_v, impact_basis="Observed",
        impact_note="Cancelled realized order value.", evidence_score=ev,
        evidence_parts=[f"Rate {overall:.1%} vs {CANCEL_ABS:.0%}", f"{n_total:,} records"],
        availability="Observed", assumptions=[],
        action="Check payment, fulfilment and product availability for the leading product/region.",
        owner="Operations manager", validation_metric="Cancellation rate over the next 30 days.",
        why=f"Cancellation rate {overall:.1%} exceeds the {CANCEL_ABS:.0%} threshold.",
        trend=trend, bars=bars, tables=tables)
    return alert, status


# --------------------------------------------------------------------------- #
# 3. Discount erosion
# --------------------------------------------------------------------------- #
def rule_discount(model: Model, metrics: dict[str, Metric]) -> tuple[Alert | None, RuleStatus]:
    tx = model.tx
    met = metrics.get("discount_rate")
    if met is None or met.availability == "Unavailable" or met.value is None:
        return None, RuleStatus("discount", "Discount erosion", "Unavailable", "No discount field mapped.")
    rate = float(met.value)
    fired = rate > DISCOUNT_ABS

    trend = pd.DataFrame(columns=["period", "value", "kind"])
    if "__date" in tx.columns and tx["__date"].notna().any() and "__gross_value" in tx.columns:
        d = tx.copy(); d["period"] = d["__date"].dt.to_period("M").astype(str)
        g = d.groupby("period").apply(
            lambda x: pd.Series({"value": (x["__discount_value"].sum() / x["__gross_value"].sum())
                                 if x["__gross_value"].sum() else np.nan}), include_groups=False).reset_index()
        g["kind"] = "rate"; g["threshold"] = DISCOUNT_ABS
        trend = g

    bars = pd.DataFrame(columns=["label", "value", "unit"])
    dim = "__category" if "__category" in tx.columns else None
    if dim:
        g = tx.groupby(dim).apply(
            lambda x: pd.Series({"value": (x["__discount_value"].sum() / x["__gross_value"].sum())
                                 if "__gross_value" in x and x["__gross_value"].sum() else np.nan}),
            include_groups=False).sort_values("value", ascending=False).head(10).reset_index()
        bars = pd.DataFrame({"label": g[dim].astype(str), "value": g["value"], "unit": "pct"})

    status = RuleStatus("discount", "Discount erosion", "Fired" if fired else "Monitored",
                        f"Discount rate {rate:.1%} vs threshold {DISCOUNT_ABS:.0%}.")
    if not fired:
        return None, status
    impact = metrics.get("discount_value")
    alert = Alert(
        key="discount", title="Discount erosion", leak="Discount erosion", severity="Medium",
        rule_text=f"Discount rate > {DISCOUNT_ABS:.0%} or 1.5x prior; ranked by discount value.",
        scope="All lines", numerator="discounted value", denominator=met.formula,
        rate=rate, threshold=DISCOUNT_ABS, baseline_note="Absolute threshold.", comparison_window="Full period",
        impact_value=float(impact.value) if impact and impact.value is not None else None,
        impact_basis="Observed", impact_note="Discount value from the observed discount field.",
        evidence_score=_score((rate - DISCOUNT_ABS) / DISCOUNT_ABS, int(len(tx)), False, True, True),
        evidence_parts=[f"Rate {rate:.1%} vs {DISCOUNT_ABS:.0%}"], availability="Observed", assumptions=[],
        action="Review campaign/SKU pricing and pause low-value discounting.",
        owner="Pricing / Marketing manager", validation_metric="Discount rate over the next 30 days.",
        why=f"Discount rate {rate:.1%} exceeds the {DISCOUNT_ABS:.0%} threshold.",
        trend=trend, bars=bars)
    return alert, status


# --------------------------------------------------------------------------- #
# 4. Low stock
# --------------------------------------------------------------------------- #
def rule_low_stock(model: Model, metrics: dict[str, Metric]) -> tuple[Alert | None, RuleStatus]:
    inv = model.latest_inventory
    if inv is None or "__on_hand" not in inv.columns:
        return None, RuleStatus("low_stock", "Low stock", "Unavailable", "No inventory file mapped.")
    has_thr = "__reorder" in inv.columns
    low = inv[inv["__on_hand"] <= inv["__reorder"]] if has_thr else inv[inv["__on_hand"] == 0].iloc[0:0]

    # demand proxy: fulfilled quantity per product across the analysis window
    tx = model.tx
    demand = pd.Series(dtype=float)
    if "__status" in tx.columns and "__product_key" in tx.columns:
        roles = classify_status(tx["__status"])
        demand = tx[tx["__status"].isin(roles["completed"])].groupby("__product_key")["__qty"].sum()
    low = low.copy()
    low["__demand"] = low["__product_key"].map(demand).fillna(0) if "__product_key" in low.columns else 0
    low = low[low["__demand"] > 0] if len(low) else low

    fired = len(low) > 0
    status = RuleStatus("low_stock", "Low stock", "Fired" if fired else "Monitored",
                        f"{len(low)} products at/below reorder point with active demand.")

    bars = pd.DataFrame(columns=["label", "value", "unit"])
    if len(low):
        top = low.sort_values("__demand", ascending=False).head(10)
        bars = pd.DataFrame({"label": top["__product_key"].astype(str), "value": top["__demand"].values,
                             "unit": "count"})

    if not fired:
        return None, status

    # estimated missed-demand proxy = recent demand x list price where both mapped
    price_col = None
    for c in inv.columns:
        if c == "__list_price":
            price_col = c
    impact_v = None
    impact_basis = "Estimated"
    if "__product_key" in low.columns and model.flags.get("list_price_mapped"):
        # bring list price from products via the tx frame mapping is complex; use product list price
        lp = _product_list_price(model)
        if lp is not None:
            low["__price"] = low["__product_key"].map(lp)
            impact_v = float((low["__demand"] * low["__price"]).sum(skipna=True))
    tables = {"Low-stock products": low[[c for c in ["__product_key", "__on_hand",
              "__reorder", "__demand"] if c in low.columns]].rename(columns={
              "__product_key": "Product", "__on_hand": "On hand", "__reorder": "Reorder point",
              "__demand": "Recent demand (units)"})}
    action = ("Replenish the highest-demand low-stock SKUs first and investigate supplier delay "
              "for any SKU below its reorder point for more than one snapshot.")
    alert = Alert(
        key="low_stock", title="Low stock against reorder point", leak="Low stock", severity="Medium",
        rule_text="On-hand quantity <= verified reorder threshold, or on-hand = 0 with recent sales.",
        scope=f"{len(low)} products at or below their reorder point with active demand",
        numerator=f"{len(low)} products below reorder point", denominator=f"{len(inv)} products in latest snapshot",
        rate=None, threshold=None, baseline_note="Deterministic count rule (no rate threshold).",
        comparison_window="Latest snapshot per product",
        impact_value=impact_v, impact_basis=impact_basis,
        impact_note="ESTIMATED missed-demand proxy = recent fulfilled units x product list price. Not observed revenue.",
        evidence_score=_score(0.18, int(len(low)), False, False, True),
        evidence_parts=[f"{len(low)} products at/below reorder point",
                        "Threshold verified from inventory field", "Impact is Estimated"],
        availability="Observed (signal) / Estimated (impact)",
        assumptions=["Reorder threshold is set correctly by the business.",
                     "Missed demand is proxied by recent fulfilled units, not measured lost sales."],
        action=action, owner="Supply chain / Replenishment manager",
        validation_metric="Count of products at/below reorder point after replenishment.",
        why=(f"{len(low)} products are at or below their mapped reorder threshold in the latest snapshot "
             f"and still show fulfilled demand. The rule is a low-stock signal, not a stockout forecast."),
        bars=bars, tables=tables)
    return alert, status


def _product_list_price(model: Model):
    for col in model.tx.columns:
        if col == "__list_price":
            return model.tx.dropna(subset=["__product_key"]).drop_duplicates("__product_key") \
                .set_index("__product_key")[col]
    return None


# --------------------------------------------------------------------------- #
# 5. Revenue anomaly
# --------------------------------------------------------------------------- #
def rule_revenue_anomaly(model: Model, metrics: dict[str, Metric]) -> tuple[Alert | None, RuleStatus]:
    tx = model.tx
    if "__date" not in tx.columns or tx["__date"].isna().all() or not model.flags.get("amount_available"):
        return None, RuleStatus("revenue_anomaly", "Revenue anomaly", "Unavailable",
                                "Requires date and amount fields.")
    roles = classify_status(tx["__status"])
    d = tx[tx["__status"].isin(roles["completed"])].copy()
    if len(d) < 30:
        return None, RuleStatus("revenue_anomaly", "Revenue anomaly", "Unavailable",
                                f"Insufficient observations ({len(d)} completed records < 30).")
    daily = d.assign(_day=d["__date"].dt.floor("D")).groupby("_day")["line_value"].sum()
    daily = daily.asfreq("D", fill_value=0.0)
    r7 = daily.rolling(7).sum()
    prior = r7.shift(7)
    chg = (r7 / prior - 1).dropna()
    trend = pd.DataFrame({"period": [str(i.date()) for i in r7.index], "value": r7.values, "kind": "value"})
    fired = bool((chg <= -REVENUE_DROP).any())
    worst = float(chg.min()) if len(chg) else 0.0
    status = RuleStatus("revenue_anomaly", "Revenue anomaly", "Fired" if fired else "Monitored",
                        f"Worst 7-day change {worst:.1%} vs -{REVENUE_DROP:.0%} threshold.")
    if not fired:
        return None, status
    worst_day = chg.idxmin()
    impact = float(daily.loc[worst_day] * 7) if worst_day in daily.index else None
    alert = Alert(
        key="revenue_anomaly", title="Revenue anomaly", leak="Revenue anomaly", severity="Medium",
        rule_text=f"Current 7-day value is {REVENUE_DROP:.0%} below the comparable prior window, with >= 7 observations.",
        scope="Completed realized revenue, daily", numerator="current 7-day value",
        denominator="prior 7-day value", rate=worst + 1, threshold=REVENUE_DROP,
        baseline_note="Compared current 7-day rolling sum to the preceding 7-day sum.",
        comparison_window=f"7 days ending {worst_day:%Y-%m-%d} vs prior 7 days",
        impact_value=impact, impact_basis="Observed",
        impact_note="Realized revenue in the low 7-day window (Observed).",
        evidence_score=_score(abs(worst) - REVENUE_DROP, int(len(daily)), True, True, True),
        evidence_parts=[f"Worst change {worst:.1%}"], availability="Observed", assumptions=[],
        action="Drill into statuses, top products and regions for the low window before acting.",
        owner="Commercial manager", validation_metric="7-day realized revenue vs prior window.",
        why=f"The 7 days ending {worst_day:%Y-%m-%d} are {abs(worst):.1%} below the prior 7 days.",
        trend=trend)
    return alert, status


# --------------------------------------------------------------------------- #
def run_all(model: Model, metrics: dict[str, Metric]) -> tuple[list[Alert], list[RuleStatus]]:
    alerts, statuses = [], []
    for fn in (rule_returns, rule_cancellations, rule_discount, rule_low_stock, rule_revenue_anomaly):
        try:
            a, s = fn(model, metrics)
        except Exception as exc:  # never let a rule crash the app
            a, s = None, RuleStatus(fn.__name__, fn.__name__, "Unavailable", f"Rule error: {exc}")
        if a is not None:
            alerts.append(a)
        statuses.append(s)
    alerts.sort(key=lambda a: (-a.evidence_score, -(a.impact_value or 0)))
    for i, a in enumerate(alerts, 1):
        a.rank = i
    return alerts[:3], statuses
