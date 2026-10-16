"""leaklab.data - the data layer.

Two responsibilities:

1. **Safe joins** that preserve the base (transaction) grain and report join
   health.  Nothing downstream may assume a literal source column name; every
   field is resolved through the schema-first :class:`~leaklab.profile.Mapping`.
2. **Canonical feature table** - a single, leakage-safe modelling frame at the
   base grain.  Only *pre-outcome* information is placed in the feature columns;
   the label is derived exclusively from the verified status field, and
   post-outcome fields (status, refund amounts, return reason, realised revenue)
   are never exposed as features.

Inventory attributes are joined **as-of the order date** (most recent snapshot on
or before each order) so a later snapshot can never inform an earlier order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profile import Mapping, normalize_keys, parse_dates

# --------------------------------------------------------------------------- #
# Status role classification (verified status field only)
# --------------------------------------------------------------------------- #
#: Statuses that carry a *terminal, verified* outcome.  Anything else - e.g.
#: "pending"/"processing" - has an unobserved outcome and is excluded from the
#: labelled population.  This is a leakage guard, not a modelling choice.
TERMINAL_ROLES = ("completed", "cancelled", "returned")
#: Roles counted as a leakage (adverse) outcome.  Used to build the binary label.
ADVERSE_ROLES = ("cancelled", "returned")


def classify_status(values) -> dict[str, set[str]]:
    """Map raw status strings onto semantic roles by keyword.

    Never raises; an unrecognised status falls through to ``pending`` so it is
    excluded from the labelled set rather than silently treated as harmless.
    """
    roles: dict[str, set[str]] = {"completed": set(), "cancelled": set(),
                                  "returned": set(), "pending": set()}
    for v in pd.Series(values).dropna().astype(str).str.strip().unique():
        low = v.lower()
        if re.search(r"cancel|void|reject", low):
            roles["cancelled"].add(v)
        elif re.search(r"return|refund|rma", low):
            roles["returned"].add(v)
        elif re.search(r"complete|deliver|fulfil|fulfill|paid|success|closed|shipped", low):
            roles["completed"].add(v)
        else:
            # pending / processing / open / unknown -> outcome not yet verified
            roles["pending"].add(v)
    return roles


# --------------------------------------------------------------------------- #
# Join health
# --------------------------------------------------------------------------- #
@dataclass
class JoinHealth:
    left: str
    right: str
    key: str
    left_rows: int
    right_rows: int
    matched_rows: int
    unmatched_rows: int
    unmatched_pct: float
    row_count_preserved: bool

    def as_row(self) -> dict:
        return {
            "left": self.left, "right": self.right, "key": self.key,
            "left_rows": self.left_rows, "right_rows": self.right_rows,
            "matched": self.matched_rows, "unmatched": self.unmatched_rows,
            "unmatched_%": round(self.unmatched_pct * 100, 2),
            "grain_preserved": self.row_count_preserved,
        }


@dataclass
class Model:
    tx: pd.DataFrame
    flags: dict
    join_health: list[JoinHealth] = field(default_factory=list)
    countries: dict = field(default_factory=dict)
    latest_inventory: pd.DataFrame | None = None
    campaigns: pd.DataFrame | None = None

    @property
    def grain(self) -> str:
        return self.flags.get("grain", "row")

    def has(self, col: str) -> bool:
        return col in self.tx.columns

    def amount(self, statuses: list[str] | None = None) -> float:
        d = self.tx
        if statuses is not None and "status" in d.columns:
            d = d[d["status"].isin(statuses)]
        return float(d["line_value"].sum())


# --------------------------------------------------------------------------- #
# Safe joins -> base grain frame
# --------------------------------------------------------------------------- #
def build_model(files: dict[str, pd.DataFrame], mapping: Mapping,
                date_range: tuple | None = None) -> Model:
    flags: dict = {}
    joins: list[JoinHealth] = []

    tx_role = _file_with(mapping, "order_key") or _role_by_name(files, "transactions")
    if tx_role is None or tx_role not in files:
        raise ValueError("No transactions-like file could be identified.")

    tx = files[tx_role].copy()
    tm = mapping.by_file.get(tx_role, {})
    order_key = tm.get("order_key")
    if order_key is None:
        flags["grain"] = "row"
        flags["grain_note"] = "No order/transaction key - aggregated at observed row grain."
    elif tx[order_key].astype("string").str.strip().nunique() == len(tx):
        flags["grain"] = "order"
    else:
        flags["grain"] = "row"
        flags["grain_note"] = "Order key is not unique - treated as order-line grain."

    key_cols = [c for c in [tm.get("product_key"), tm.get("customer_key")] if c]
    tx = normalize_keys(tx, key_cols)

    date_col = tm.get("date")
    tx["__date"] = parse_dates(tx[date_col]) if date_col else pd.NaT
    flags["date_mapped"] = bool(date_col) and tx["__date"].notna().any()
    if date_range and flags["date_mapped"]:
        lo, hi = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        tx = tx[(tx["__date"] >= lo) & (tx["__date"] <= hi + pd.Timedelta(days=1))]

    flags["status_mapped"] = bool(tm.get("status"))
    tx["__status"] = tx[tm["status"]].astype("string").str.strip() if flags["status_mapped"] else pd.NA

    if tm.get("product_key"):
        tx["__product_key"] = tx[tm["product_key"]]
    if tm.get("customer_key"):
        tx["__customer_key"] = tx[tm["customer_key"]]

    qty = pd.to_numeric(tx[tm["quantity"]], errors="coerce").fillna(0) if tm.get("quantity") else pd.Series(1.0, index=tx.index)
    if tm.get("amount"):
        line_value = pd.to_numeric(tx[tm["amount"]], errors="coerce").fillna(0)
        flags["amount_field"] = tm["amount"]
        flags["amount_basis"] = "amount column"
    elif tm.get("unit_price"):
        line_value = qty * pd.to_numeric(tx[tm["unit_price"]], errors="coerce").fillna(0)
        flags["amount_field"] = tm["unit_price"]
        flags["amount_basis"] = f"{tm.get('quantity') or '1'} x {tm['unit_price']}"
    else:
        line_value = pd.Series(np.nan, index=tx.index)
        flags["amount_field"] = None
        flags["amount_basis"] = None
    tx["line_value"] = line_value
    tx["__qty"] = qty
    flags["amount_available"] = bool(tm.get("amount") or tm.get("unit_price"))

    if tm.get("discount"):
        tx["__discount"] = pd.to_numeric(tx[tm["discount"]], errors="coerce").fillna(0)
        flags["discount_mapped"] = True
    else:
        tx["__discount"] = np.nan
        flags["discount_mapped"] = False

    # ---- product join ------------------------------------------------------
    prod_role = _file_with(mapping, "product_key") if _file_with(mapping, "on_hand") is None else None
    prod_role = prod_role or _role_by_name(files, "products")
    if prod_role and tm.get("product_key") and mapping.by_file.get(prod_role, {}).get("product_key"):
        prod = files[prod_role].copy()
        pm = mapping.by_file[prod_role]
        pkey = pm["product_key"]
        prod = normalize_keys(prod, [pkey])
        right = prod.drop_duplicates(subset=[pkey])
        merged = tx.merge(right, left_on=tm["product_key"], right_on=pkey,
                          how="left", suffixes=("", "_prod"))
        joins.append(_health("transactions", prod_role, tm["product_key"], tx, merged, pkey))
        tx = merged
        flags["product_join"] = True
        for concept in ["category", "cost", "list_price", "lead_days"]:
            col = pm.get(concept)
            if col and col in tx.columns:
                tx[f"__{concept}"] = tx[col]
        if pm.get("list_price") and tm.get("unit_price"):
            tx["__gross_value"] = qty * pd.to_numeric(tx[pm["list_price"]], errors="coerce")
            tx["__discount_value"] = tx["__gross_value"] - tx["line_value"]
        elif tm.get("discount"):
            disc = tx["__discount"].clip(0, 0.99)
            tx["__gross_value"] = tx["line_value"] / (1 - disc)
            tx["__discount_value"] = tx["__gross_value"] - tx["line_value"]
        flags["list_price_mapped"] = bool(pm.get("list_price"))
    else:
        flags["product_join"] = False
        flags["list_price_mapped"] = False

    # ---- customer join -----------------------------------------------------
    cust_role = _role_by_name(files, "customers")
    if cust_role and tm.get("customer_key") and mapping.by_file.get(cust_role, {}).get("customer_key"):
        cust = files[cust_role].copy()
        cm = mapping.by_file[cust_role]
        ckey = cm["customer_key"]
        cust = normalize_keys(cust, [ckey])
        right = cust.drop_duplicates(subset=[ckey])
        merged = tx.merge(right, left_on=tm["customer_key"], right_on=ckey,
                          how="left", suffixes=("", "_cust"))
        joins.append(_health("transactions", cust_role, tm["customer_key"], tx, merged, ckey))
        tx = merged
        flags["customer_join"] = True
        for concept in ["region", "segment", "preferred_channel", "signup_date"]:
            col = cm.get(concept)
            if col and col in tx.columns:
                tx[f"__{concept}"] = tx[col]
    else:
        flags["customer_join"] = False

    # ---- inventory (latest snapshot on/before analysis date) ---------------
    latest_inv = None
    inv_role = _role_by_name(files, "inventory")
    if inv_role:
        im = mapping.by_file.get(inv_role, {})
        if im.get("product_key") and im.get("on_hand"):
            inv = files[inv_role].copy()
            inv = normalize_keys(inv, [im["product_key"]])
            inv["__idate"] = parse_dates(inv[im["date"]]) if im.get("date") else pd.NaT
            if date_range and im.get("date"):
                hi = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
                inv = inv[inv["__idate"] <= hi]
            sort_cols = ["__idate"] if inv["__idate"].notna().any() else []
            inv = inv.sort_values(sort_cols) if sort_cols else inv
            latest_inv = inv.groupby(im["product_key"], as_index=False).tail(1)
            latest_inv = latest_inv.rename(columns={im["product_key"]: "__product_key",
                                                    im["on_hand"]: "__on_hand"})
            if im.get("reorder_threshold"):
                latest_inv["__reorder"] = latest_inv[im["reorder_threshold"]]
            if im.get("stockout_flag"):
                latest_inv["__stockout_flag"] = latest_inv[im["stockout_flag"]]
            if im.get("supplier_delay"):
                latest_inv["__supplier_delay"] = latest_inv[im["supplier_delay"]]
            flags["inventory_available"] = True
        else:
            flags["inventory_available"] = False
    else:
        flags["inventory_available"] = False

    # ---- campaigns (independent, no key join) ------------------------------
    campaigns = None
    camp_role = _role_by_name(files, "campaigns")
    if camp_role:
        cm = mapping.by_file.get(camp_role, {})
        camp = files[camp_role].copy()
        camp["__cdate"] = parse_dates(camp[cm["date"]]) if cm.get("date") else pd.NaT
        campaigns = camp
        flags["campaigns_available"] = True
        flags["campaign_key_shared"] = False
    else:
        flags["campaigns_available"] = False

    tx = tx.reset_index(drop=True)
    return Model(tx=tx, flags=flags, join_health=joins, latest_inventory=latest_inv,
                 campaigns=campaigns, countries={})


# --------------------------------------------------------------------------- #
# Canonical feature table
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSpec:
    name: str
    kind: str          # "numeric" | "categorical"
    label: str
    source: str = ""   # human-readable provenance
    pre_outcome: bool = True
    note: str = ""


#: The modelling contract.  Every column here is available *before* the outcome
#: is known, so it can be used as a predictor without leakage.  The label is not
#: in this list: it is produced separately from the verified status field.
FEATURE_SPEC: list[FeatureSpec] = [
    FeatureSpec("quantity", "numeric", "Quantity", "transactions.quantity"),
    FeatureSpec("unit_price", "numeric", "Unit price", "transactions.unit_price"),
    FeatureSpec("discount_pct", "numeric", "Discount rate", "transactions.discount_pct"),
    FeatureSpec("line_value", "numeric", "Order line value", "quantity x unit_price"),
    FeatureSpec("price_ratio", "numeric", "Price / list ratio", "unit_price / list_price",
                note="Captures how deeply a line is discounted versus catalogue."),
    FeatureSpec("list_price", "numeric", "List price", "products.list_price"),
    FeatureSpec("unit_cost", "numeric", "Unit cost", "products.unit_cost",
                note="Association only - 76 products have cost above list price."),
    FeatureSpec("cost_exceeds_list", "numeric", "Cost above list (0/1)", "unit_cost > list_price"),
    FeatureSpec("supplier_lead_days", "numeric", "Supplier lead time", "products.supplier_lead_days"),
    FeatureSpec("stock_on_hand", "numeric", "Stock on hand (as-of)", "inventory.stock_on_hand"),
    FeatureSpec("reorder_threshold", "numeric", "Reorder threshold", "inventory.reorder_threshold"),
    FeatureSpec("stockout_flag", "numeric", "Stockout flag (as-of)", "inventory.stockout_flag"),
    FeatureSpec("supplier_delay_days", "numeric", "Supplier delay (as-of)", "inventory.supplier_delay_days"),
    FeatureSpec("below_reorder", "numeric", "Below reorder (0/1)", "stock_on_hand <= reorder_threshold"),
    FeatureSpec("order_month", "numeric", "Order month", "timestamp"),
    FeatureSpec("order_dow", "numeric", "Order weekday", "timestamp"),
    FeatureSpec("order_hour", "numeric", "Order hour", "timestamp"),
    FeatureSpec("customer_tenure_days", "numeric", "Customer tenure (days)", "order date - signup date"),
    FeatureSpec("category", "categorical", "Product category", "products.category"),
    FeatureSpec("channel", "categorical", "Sales channel", "transactions.channel"),
    FeatureSpec("payment_method", "categorical", "Payment method", "transactions.payment_method"),
    FeatureSpec("customer_segment", "categorical", "Customer segment", "customers.customer_segment"),
    FeatureSpec("region", "categorical", "Customer region", "customers.region"),
    FeatureSpec("preferred_channel", "categorical", "Preferred channel", "customers.preferred_channel"),
]

#: Post-outcome / identification fields that must never become features.
EXCLUDED_FIELDS = [
    "status (the label source)",
    "refund / return amount (not present, and would leak the outcome)",
    "return reason (not present)",
    "realised revenue after outcome",
    "transaction / order id (identifier, not signal)",
]


def _num(series, default=np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def build_feature_table(model: Model, mapping: Mapping,
                        files: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the canonical modelling frame at the base grain.

    Columns:
      ``record_id``  stable identifier (order key or row index)
      ``record_date`` parsed order date (may be NaT)
      ``status_raw``  verified status string
      ``target``      1 for cancelled/returned, 0 for completed, NaN otherwise
      ``target_known`` whether the outcome is verified (terminal status)
      ``eligible``    label is known -> usable for supervised training
      ``leak_value``  realised line value (Observed currency)
      plus every column in :data:`FEATURE_SPEC`.

    The function never raises: missing sources simply drop their columns, and the
    caller sees a shorter feature list.  Inventory attributes are joined as-of the
    order date (backward direction) so no future snapshot informs a past order.
    """
    tx = model.tx.copy()
    n = len(tx)
    ft = pd.DataFrame(index=tx.index)

    # resolve the transaction file mapping (the one owning the order key / status)
    tx_role = _file_with(mapping, "order_key") or _role_by_name(files, "transactions")
    tm = mapping.by_file.get(tx_role, {}) if tx_role else {}

    order_key = tm.get("order_key")
    if order_key and order_key in tx.columns:
        ft["record_id"] = tx[order_key].astype("string").fillna("row")
    else:
        ft["record_id"] = pd.Series([f"row{i}" for i in range(n)], index=tx.index)

    ft["record_date"] = tx["__date"] if "__date" in tx.columns else pd.NaT
    ft["status_raw"] = tx["__status"] if "__status" in tx.columns else pd.NA
    ft["leak_value"] = tx["line_value"] if "line_value" in tx.columns else np.nan

    # ---- label (verified status only) --------------------------------------
    if "__status" in tx.columns and model.flags.get("status_mapped"):
        roles = classify_status(tx["__status"])
        terminal = set().union(*[roles[r] for r in TERMINAL_ROLES])
        adverse = set().union(*[roles[r] for r in ADVERSE_ROLES])
        known = tx["__status"].isin(terminal)
        ft["target_known"] = known.fillna(False).astype(bool)
        ft["target"] = np.where(known, tx["__status"].isin(adverse).astype(int), np.nan)
    else:
        ft["target_known"] = False
        ft["target"] = np.nan
    ft["eligible"] = ft["target_known"]

    # ---- numeric features ---------------------------------------------------
    q = _num(tx.get("__qty"))
    ft["quantity"] = q
    up = _num(tx.get(tm.get("unit_price")) if tm.get("unit_price") else None)
    ft["unit_price"] = up
    ft["discount_pct"] = _num(tx.get("__discount"))
    lv = _num(tx.get("line_value"))
    ft["line_value"] = lv
    lp = _num(tx.get("__list_price")) if "__list_price" in tx.columns else pd.Series(np.nan, index=tx.index)
    ft["list_price"] = lp
    ft["unit_cost"] = _num(tx.get("__cost")) if "__cost" in tx.columns else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ft["price_ratio"] = (up / lp.replace(0, np.nan))
    ft["cost_exceeds_list"] = (ft["unit_cost"] > lp).astype("float").where(lp.notna(), np.nan)
    ft["supplier_lead_days"] = _num(tx.get("__lead_days")) if "__lead_days" in tx.columns else np.nan

    # time features
    d = ft["record_date"]
    ft["order_month"] = d.dt.month
    ft["order_dow"] = d.dt.dayofweek
    ft["order_hour"] = d.dt.hour

    # customer tenure
    ft["customer_tenure_days"] = np.nan
    if "__signup_date" in tx.columns and ft["record_date"].notna().any():
        signup = parse_dates(tx["__signup_date"])
        ft["customer_tenure_days"] = (ft["record_date"] - signup).dt.days

    # ---- categorical features ----------------------------------------------
    if "__category" in tx.columns:
        ft["category"] = tx["__category"].astype("string")
    if tm.get("channel") and tm["channel"] in tx.columns:
        ft["channel"] = tx[tm["channel"]].astype("string")
    if tm.get("payment") and tm["payment"] in tx.columns:
        ft["payment_method"] = tx[tm["payment"]].astype("string")
    if "__segment" in tx.columns:
        ft["customer_segment"] = tx["__segment"].astype("string")
    if "__region" in tx.columns:
        ft["region"] = tx["__region"].astype("string")
    if "__preferred_channel" in tx.columns:
        ft["preferred_channel"] = tx["__preferred_channel"].astype("string")

    # ---- inventory as-of the order date ------------------------------------
    inv_role = _role_by_name(files, "inventory")
    if inv_role and tm.get("product_key") and ft["record_date"].notna().any():
        im = mapping.by_file.get(inv_role, {})
        if im.get("product_key") and im.get("on_hand"):
            inv = files[inv_role].copy()
            inv = normalize_keys(inv, [im["product_key"]])
            inv["__idate"] = parse_dates(inv[im["date"]]) if im.get("date") else pd.NaT
            inv = inv[inv["__idate"].notna()]
            keep = ["__idate", im["product_key"], im["on_hand"]]
            ren = {"__idate": "__idate", im["product_key"]: "__product_key", im["on_hand"]: "stock_on_hand"}
            for concept, out in [("reorder_threshold", "reorder_threshold"),
                                 ("stockout_flag", "stockout_flag"),
                                 ("supplier_delay", "supplier_delay_days")]:
                if im.get(concept):
                    keep.append(im[concept]); ren[im[concept]] = out
            inv = inv[keep].rename(columns=ren).sort_values("__idate")
            left = pd.DataFrame({
                "__row": ft.index,
                "__date": ft["record_date"],
                "__product_key": tx["__product_key"].astype("string") if "__product_key" in tx.columns else pd.NA,
            })
            left = left[left["__date"].notna() & left["__product_key"].notna()]
            if len(left):
                left = left.sort_values("__date").reset_index(drop=True)
                asof = pd.merge_asof(left, inv.reset_index(drop=True),
                                     left_on="__date", right_on="__idate",
                                     by="__product_key", direction="backward")
                for col in ["stock_on_hand", "reorder_threshold", "stockout_flag", "supplier_delay_days"]:
                    if col in asof.columns:
                        ft.loc[asof["__row"].values, col] = asof[col].values
    ft["stock_on_hand"] = ft.get("stock_on_hand", pd.Series(np.nan, index=tx.index))
    ft["reorder_threshold"] = ft.get("reorder_threshold", pd.Series(np.nan, index=tx.index))
    ft["stockout_flag"] = ft.get("stockout_flag", pd.Series(np.nan, index=tx.index))
    ft["supplier_delay_days"] = ft.get("supplier_delay_days", pd.Series(np.nan, index=tx.index))
    with np.errstate(invalid="ignore"):
        ft["below_reorder"] = (ft["stock_on_hand"] <= ft["reorder_threshold"]).astype("float")
        ft.loc[ft["stock_on_hand"].isna() | ft["reorder_threshold"].isna(), "below_reorder"] = np.nan

    # order the feature columns deterministically
    ordered = [s.name for s in FEATURE_SPEC if s.name in ft.columns]
    meta = ["record_id", "record_date", "status_raw", "target", "target_known",
            "eligible", "leak_value"]
    return ft[meta + [c for c in ordered if c not in meta]].reset_index(drop=True)


def available_features(ft: pd.DataFrame) -> list[FeatureSpec]:
    """Feature specs present in the frame and carrying at least some information."""
    out = []
    for spec in FEATURE_SPEC:
        if spec.name not in ft.columns:
            continue
        col = ft[spec.name]
        if col.notna().sum() == 0:
            continue
        if spec.kind == "numeric" and col.nunique(dropna=True) <= 1:
            continue
        if spec.kind == "categorical" and col.nunique(dropna=True) <= 1:
            continue
        out.append(spec)
    return out


def base_grain_report(model: Model) -> dict:
    """Describe the base grain and whether joins preserved it."""
    return {
        "grain": model.grain,
        "rows": int(len(model.tx)),
        "grain_note": model.flags.get("grain_note", ""),
        "joins_preserved": all(j.row_count_preserved for j in model.join_health) if model.join_health else True,
        "max_unmatched_pct": max((j.unmatched_pct for j in model.join_health), default=0.0),
    }


# --------------------------------------------------------------------------- #
def _file_with(mapping: Mapping, concept: str) -> str | None:
    return mapping.file_of(concept)


def _role_by_name(files: dict[str, pd.DataFrame], role: str) -> str | None:
    for f in files:
        if role.rstrip("s") in f.lower():
            return f
    return None


def _health(left: str, right: str, lkey: str, before: pd.DataFrame,
            after: pd.DataFrame, rkey: str) -> JoinHealth:
    matched = int(after[rkey].notna().sum()) if rkey in after.columns else 0
    return JoinHealth(
        left=left, right=right, key=str(lkey),
        left_rows=len(before), right_rows=matched, matched_rows=matched,
        unmatched_rows=len(before) - matched,
        unmatched_pct=(len(before) - matched) / len(before) if len(before) else 0.0,
        row_count_preserved=len(after) == len(before),
    )
