"""Robustness tests: missing fields, alternate column names, dup keys, tiny samples.

Every case must run without raising and must degrade to Unavailable/Monitored
rather than crash (PRD section 9).
"""
from __future__ import annotations

import pandas as pd

from leaklab.profile import inspect_file, suggest_mapping
from leaklab.data import build_model, build_feature_table, base_grain_report
from leaklab.metrics import compute_metrics
from leaklab.rules import run_all

D = r"D:\DATATHON"
BASE = {n: pd.read_csv(f"{D}/{n}.csv" if not n.endswith(".csv") else f"{D}/{n}")
        for n in ["customers.csv", "products.csv", "transactions.csv", "inventory.csv", "campaigns.csv"]}


def pipeline(files, label):
    profiles = {n: inspect_file(n, df) for n, df in files.items()}
    mapping = suggest_mapping(files, profiles)
    model = build_model(files, mapping)
    metrics = compute_metrics(model)
    alerts, statuses = run_all(model, metrics)
    st = {s.title: s.status for s in statuses}
    print(f"[{label}] alerts={len(alerts)} statuses={st}")
    assert not any("Rule error" in s.detail for s in statuses), f"rule crashed in {label}"
    # Round 2 path must never crash either; training may legitimately be refused.
    try:
        ft = build_feature_table(model, mapping, files)
        assert len(ft) == len(model.tx), f"feature table lost rows in {label}"
        _ = base_grain_report(model)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"feature table crashed in {label}: {exc}")
    return alerts, metrics


def copy():
    return {k: v.copy() for k, v in BASE.items()}


# 1. missing status column
f = copy(); f["transactions.csv"] = f["transactions.csv"].drop(columns=["status"])
a, m = pipeline(f, "no-status")
assert m["cancellation_rate"].availability == "Unavailable"

# 2. missing price and amount
f = copy(); f["transactions.csv"] = f["transactions.csv"].drop(columns=["unit_price"])
a, m = pipeline(f, "no-price")
assert m["realized_revenue"].availability == "Unavailable"

# 3. missing date
f = copy(); f["transactions.csv"] = f["transactions.csv"].drop(columns=["timestamp"])
a, m = pipeline(f, "no-date")
assert m["date_range"].availability == "Unavailable"

# 4. no inventory file
f = copy(); del f["inventory.csv"]
a, m = pipeline(f, "no-inventory")
assert m["low_stock_count"].availability == "Unavailable"

# 5. alternate column names (mapping must still resolve)
f = copy()
tx = f["transactions.csv"].rename(columns={
    "transaction_id": "order_id", "product_id": "sku", "customer_id": "customer_ref",
    "timestamp": "event_time", "quantity": "qty", "unit_price": "price",
    "discount_pct": "disc", "status": "order_state"})
f["transactions.csv"] = tx
a, m = pipeline(f, "renamed-columns")
assert m["return_rate"].availability == "Observed"

# 6. duplicate product keys -> warning, no crash
f = copy()
f["products.csv"] = pd.concat([f["products.csv"], f["products.csv"].head(3)], ignore_index=True)
a, m = pipeline(f, "duplicate-product-keys")

# 7. tiny sample (< 10 records) -> rate rules Unavailable
f = copy(); f["transactions.csv"] = f["transactions.csv"].head(6)
a, m = pipeline(f, "tiny-sample")

# 8. nulls in key columns
f = copy()
t = f["transactions.csv"].copy()
t.loc[t.index[:50], "customer_id"] = None
f["transactions.csv"] = t
a, m = pipeline(f, "null-keys")

print("\nOK: all edge cases ran without crashing.")
