"""leaklab.profile - schema-first CSV inspection, type parsing and field mapping.

Nothing in the app may reference a literal source column name.  Every rule and
metric resolves its inputs through the :class:`Mapping` produced here, so an
unused / renamed / missing column degrades to ``Unavailable`` instead of a
KeyError.

Design notes
------------
* Columns are *scored* against concept hints, then *validated* (keys must be
  near-unique, dates must parse, amounts must be numeric).  A column only wins a
  concept if both the name hint and the validator agree.
* File roles (transactions / products / customers / inventory / campaigns) are
  inferred from the mapped concepts and the file name, never hard coded.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Concept hints: ordered regexes (first hit scores highest).
# --------------------------------------------------------------------------- #
CONCEPT_HINTS: dict[str, list[str]] = {
    "order_key":        [r"transaction.?id", r"order.?id", r"txn.?id", r"invoice.?id", r"^id$"],
    "product_key":      [r"product.?id", r"sku", r"item.?id", r"^product", r"^sku"],
    "customer_key":     [r"customer.?id", r"client.?id", r"account.?id", r"^customer", r"^client"],
    "campaign_key":     [r"campaign.?id", r"^campaign"],
    "date":             [r"^timestamp", r"^datetime", r"^date$", r"order.?date", r"_date$", r"^date", r"time"],
    "status":           [r"^status", r"order.?status", r"state"],
    "quantity":         [r"^quantity$", r"^qty$", r"units", r"^count$"],
    "unit_price":       [r"unit.?price", r"sale.?price", r"^price$", r"selling.?price"],
    "list_price":       [r"list.?price", r"regular.?price", r"msrp", r"catalog.?price", r"^price$"],
    "amount":           [r"^amount$", r"line.?total", r"^total$", r"revenue", r"sales.?amount", r"^value$"],
    "discount":         [r"discount.?pct", r"discount.?rate", r"^discount", r"^disc$", r"markdown"],
    "cost":             [r"unit.?cost", r"^cost$", r"cogs", r"cost.?price"],
    "on_hand":          [r"stock.?on.?hand", r"on.?hand", r"^stock$", r"available", r"inventory.?qty", r"quantity.?on.?hand"],
    "reorder_threshold":[r"reorder.?threshold", r"reorder.?point", r"^threshold$", r"reorder.?level"],
    "stockout_flag":    [r"stockout", r"out.?of.?stock", r"is.?stockout"],
    "supplier_delay":   [r"supplier.?delay", r"delay.?days"],
    "lead_days":        [r"lead.?time", r"lead.?days", r"supplier.?lead"],
    "category":         [r"^category$", r"product.?category", r"department", r"^dept$"],
    "region":           [r"^region$", r"^area$", r"territory", r"^geo", r"^state$"],
    "segment":          [r"^segment$", r"customer.?segment", r"^tier$"],
    "channel":          [r"^channel$", r"^source$", r"^medium$", r"sales.?channel"],
    "payment":          [r"^payment", r"tender"],
    "signup_date":      [r"signup", r"sign.?up", r"registration"],
    "preferred_channel":[r"preferred.?channel"],
    "spend":            [r"^spend$", r"^cost$", r"budget", r"ad.?spend"],
    "impressions":      [r"^impressions$", r"^impressions", r"^views$"],
    "clicks":           [r"^clicks$", r"^click"],
    "conversions":      [r"^conversions$", r"^conversion", r"^orders$", r"^signups$"],
}


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
def parse_dates(series: pd.Series) -> pd.Series:
    """Best-effort date parsing; never raises."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kwargs in ({"format": "mixed"}, {}):
            try:
                return pd.to_datetime(series, errors="coerce", **kwargs)
            except (ValueError, TypeError):
                continue
    return pd.Series(pd.NaT, index=series.index)


def _date_ratio(series: pd.Series) -> float:
    parsed = parse_dates(series)
    return float(parsed.notna().mean()) if len(series) else 0.0


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _uniqueness(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 0.0
    norm = series.astype("string").str.strip()
    return float(norm.nunique(dropna=True) / n)


def _low_cardinality(series: pd.Series, k: int = 25) -> bool:
    return int(series.astype("string").str.strip().nunique(dropna=True)) <= k


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class ColumnInfo:
    name: str
    dtype: str
    null_rate: float
    n_unique: int
    parse_ok: float | None = None
    minimum: Any = None
    maximum: Any = None
    top_values: dict = field(default_factory=dict)


@dataclass
class FileProfile:
    name: str
    role: str
    rows: int
    columns: list[ColumnInfo]
    sample: pd.DataFrame

    @property
    def col_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class Mapping:
    """Resolved role -> column map, plus the file that supplied each concept."""

    by_file: dict[str, dict[str, str | None]] = field(default_factory=dict)
    source: dict[str, tuple[str, str]] = field(default_factory=dict)  # concept -> (file, column)
    warnings: list[str] = field(default_factory=list)

    def get(self, concept: str, file_role: str | None = None) -> str | None:
        if file_role is not None:
            return self.by_file.get(file_role, {}).get(concept)
        src = self.source.get(concept)
        return src[1] if src else None

    def has(self, concept: str) -> bool:
        return concept in self.source

    def file_of(self, concept: str) -> str | None:
        src = self.source.get(concept)
        return src[0] if src else None


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def inspect_file(name: str, df: pd.DataFrame) -> FileProfile:
    cols: list[ColumnInfo] = []
    for c in df.columns:
        s = df[c]
        ci = ColumnInfo(
            name=str(c),
            dtype=str(s.dtype),
            null_rate=round(float(s.isna().mean()), 4),
            n_unique=int(s.astype("string").str.strip().nunique(dropna=True)),
        )
        if _is_numeric(s):
            ci.minimum, ci.maximum = float(s.min()), float(s.max())
        else:
            dr = _date_ratio(s)
            if dr >= 0.8:
                ci.parse_ok = round(dr, 4)
                p = parse_dates(s)
                ci.minimum, ci.maximum = str(p.min()), str(p.max())
            elif s.nunique(dropna=True) <= 12:
                ci.top_values = {str(k): int(v) for k, v in s.value_counts(dropna=False).head(12).items()}
        cols.append(ci)
    return FileProfile(name=name, role="unknown", rows=len(df), columns=cols, sample=df.head(5).copy())


def _score(concept: str, col: str) -> int:
    pats = CONCEPT_HINTS.get(concept, [])
    n = str(col).lower()
    for i, p in enumerate(pats):
        if re.search(p, n):
            return 100 - i * 5
    return 0


def _validate(concept: str, series: pd.Series) -> bool:
    if concept.endswith("_key"):
        return _uniqueness(series) >= 0.5 or _low_cardinality(series, 5000)
    if concept in {"date", "signup_date"}:
        return _date_ratio(series) >= 0.8
    if concept == "status":
        return _low_cardinality(series, 12)
    if concept == "discount":
        if not _is_numeric(series):
            return False
        mx = float(series.max())
        return mx <= 1.0  # pct or rate
    if concept in {"quantity", "on_hand", "reorder_threshold", "stockout_flag",
                   "lead_days", "supplier_delay", "impressions", "clicks", "conversions"}:
        return _is_numeric(series)
    if concept in {"unit_price", "list_price", "amount", "cost", "spend"}:
        return _is_numeric(series)
    if concept in {"category", "region", "segment", "channel", "payment",
                   "preferred_channel"}:
        return _low_cardinality(series, 40)
    return True


def suggest_mapping(files: dict[str, pd.DataFrame], profiles: dict[str, FileProfile]) -> Mapping:
    m = Mapping()
    role_of_file: dict[str, str] = {}

    # --- 1. detect file roles -------------------------------------------------
    for fname, df in files.items():
        cols = list(df.columns)
        look = {c: str(c).lower() for c in cols}
        has_camp = any(re.search(r"campaign|impression|click|conversion", v) for v in look.values())
        has_stock = any(re.search(r"stock|on.?hand|reorder", v) for v in look.values())
        has_status = any(re.search(r"status|state", v) for v in look.values())
        has_cust = any(re.search(r"customer|client", v) for v in look.values())
        has_prod = any(re.search(r"product|sku|item", v) for v in look.values())
        has_ord = any(re.search(r"transaction|order|invoice|txn", v) for v in look.values())
        has_metrics = any(re.search(r"spend|impressions|clicks|conversions", v) for v in look.values())

        role = "unknown"
        if has_camp or has_metrics or "campaign" in fname.lower():
            role = "campaigns"
        elif has_stock and has_prod:
            role = "inventory"
        elif has_ord and (has_status or has_metrics is False and _date_ratio(df[cols[0]]) < 0.5):
            role = "transactions"
        elif has_ord:
            role = "transactions"
        elif has_prod and not has_ord:
            role = "products"
        elif has_cust:
            role = "customers"
        elif "product" in fname.lower():
            role = "products"
        elif "inventory" in fname.lower():
            role = "inventory"
        elif "customer" in fname.lower():
            role = "customers"
        role_of_file[fname] = role
        profiles[fname].role = role

    # disambiguate transactions: must contain something resembling an amount/qty/status
    for fname, df in files.items():
        if role_of_file[fname] == "transactions":
            look = " ".join(str(c).lower() for c in df.columns)
            if not re.search(r"quantity|qty|price|amount|status|discount", look):
                role_of_file[fname] = "unknown"

    # --- 2. map concepts per file --------------------------------------------
    # concepts that make sense for a given file role
    ROLE_CONCEPTS: dict[str, list[str]] = {
        "transactions": ["order_key", "product_key", "customer_key", "date", "status",
                         "quantity", "unit_price", "amount", "discount", "channel", "payment"],
        "products":     ["product_key", "category", "cost", "list_price", "lead_days"],
        "customers":    ["customer_key", "region", "segment", "signup_date", "preferred_channel"],
        "inventory":    ["product_key", "date", "on_hand", "reorder_threshold",
                         "stockout_flag", "supplier_delay"],
        "campaigns":    ["campaign_key", "date", "channel", "category", "spend",
                         "impressions", "clicks", "conversions"],
        "unknown":      list(CONCEPT_HINTS.keys()),
    }

    for fname, df in files.items():
        role = role_of_file[fname]
        m.by_file[fname] = {}
        concepts = ROLE_CONCEPTS.get(role, list(CONCEPT_HINTS.keys()))
        for concept in concepts:
            best, best_score = None, 0
            for c in df.columns:
                sc = _score(concept, c)
                if sc > best_score and _validate(concept, df[c]):
                    best, best_score = c, sc
            m.by_file[fname][concept] = best
            if best is not None and concept not in m.source:
                m.source[concept] = (fname, best)

    # --- 3. sanity warnings ---------------------------------------------------
    if "product_key" in m.source:
        f = m.file_of("product_key")
        kc = m.get("product_key", f)
        if not _unique_keys(files[f][kc]):
            m.warnings.append(
                f"Product key '{kc}' in {f} is not unique - product-level joins disabled."
            )
    if "customer_key" in m.source:
        f = m.file_of("customer_key")
        kc = m.get("customer_key", f)
        if not _unique_keys(files[f][kc]):
            m.warnings.append(
                f"Customer key '{kc}' in {f} is not unique - customer-level joins disabled."
            )
    return m


def _unique_keys(series: pd.Series) -> bool:
    return _uniqueness(series) >= 0.999


def load_csv_bytes(name: str, data: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(data))


def normalize_keys(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Trim + stringify join keys so joins do not silently drop on whitespace."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    return df
