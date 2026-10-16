"""LeakLens (The Money Leak - Round 2) — a manager decision system.

Run with:  python -m streamlit run app.py

The app profiles the CSVs, builds a leakage-safe feature table, trains a baseline
+ final model *in this project* on the provided data, audits whether the model
learned anything at all, then turns the result into a ranked priority queue, an
explainable investigation, an action, and a validation loop.  All money is INR.
No external API or LLM is used as a predictor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from leaklab import views
from leaklab.actions import (ActionRecord, all_templates, append_log, default_due_date,
                             load_log, templates_for)
from leaklab.data import (build_feature_table, build_model, base_grain_report)
from leaklab.metrics import compute_metrics, format_inr
from leaklab.profile import inspect_file, suggest_mapping
from leaklab.rules import run_all
from leaklab.score import (driver_evidence, exposure_over_time, score_all,
                           segment_summary, summarise)
from leaklab.train import load_artifact, result_from_dict, train_model

DATA_DIR = Path(__file__).resolve().parent
ASSETS = DATA_DIR / "assets"
ARTIFACT_DIR = DATA_DIR / "artifacts"
ARTIFACT = ARTIFACT_DIR / "leaklens_model.joblib"
ACTION_LOG = DATA_DIR / "action_log.csv"
BUNDLED = ["customers.csv", "products.csv", "transactions.csv", "inventory.csv", "campaigns.csv"]

_favicon = ASSETS / "favicon.png"
st.set_page_config(
    page_title="LeakLens",
    page_icon=str(_favicon) if _favicon.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

views.inject_css()
views.sidebar_hotkey()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_bundled(dir_str: str, names: tuple[str, ...], stamps: tuple[float, ...]) -> dict:
    return {n: pd.read_csv(Path(dir_str) / n) for n in names}


def load_uploads(files) -> dict:
    out = {}
    for uf in files:
        out[uf.name] = pd.read_csv(pd.io.common.BytesIO(uf.getvalue()))
    return out


def make_sample() -> dict:
    rng = np.random.default_rng(7)
    n = 400
    products = pd.DataFrame({
        "product_id": [f"P{i:03d}" for i in range(1, 21)],
        "category": rng.choice(["A", "B", "C"], 20),
        "unit_cost": rng.uniform(5, 60, 20).round(2),
        "list_price": rng.uniform(10, 90, 20).round(2),
    })
    cust = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(1, 101)],
                         "region": rng.choice(["North", "South", "East", "West"], 100)})
    status = rng.choice(["Completed", "Cancelled", "Returned", "Pending"], n, p=[.75, .1, .1, .05])
    tx = pd.DataFrame({
        "transaction_id": [f"T{i:05d}" for i in range(n)],
        "customer_id": rng.choice(cust["customer_id"], n),
        "product_id": rng.choice(products["product_id"], n),
        "timestamp": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 360, n), "D"),
        "quantity": rng.integers(1, 5, n),
        "unit_price": rng.uniform(10, 90, n).round(2),
        "discount_pct": rng.choice([0, .1, .2], n),
        "status": status,
    })
    inv = pd.DataFrame({"inventory_record_id": [f"I{i}" for i in range(40)],
                        "product_id": rng.choice(products["product_id"], 40),
                        "date": pd.to_datetime("2025-12-01"),
                        "stock_on_hand": rng.integers(0, 60, 40),
                        "reorder_threshold": rng.integers(10, 40, 40)})
    return {"sample_transactions.csv": tx, "sample_products.csv": products,
            "sample_customers.csv": cust, "sample_inventory.csv": inv}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
views.brand()

st.sidebar.markdown("##### Data source")
mode = st.sidebar.radio("Data source", ["Bundled dataset", "Upload CSVs"],
                        index=0, label_visibility="collapsed")

files: dict = {}
mode_label = ""
if mode == "Upload CSVs":
    ups = st.sidebar.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    if ups:
        files = load_uploads(ups)
        mode_label = f"Uploaded · {len(files)} files"
    else:
        st.sidebar.caption("Upload one or more CSVs. A transactions file with a status column is required.")
else:
    present = [n for n in BUNDLED if (DATA_DIR / n).exists()]
    if present:
        stamps = tuple((DATA_DIR / n).stat().st_mtime for n in present)
        files = load_bundled(str(DATA_DIR), tuple(present), stamps)
        mode_label = f"Bundled · {len(present)} files"
    else:
        st.sidebar.warning("No bundled CSVs found next to app.py.")
        st.sidebar.button("Load sample data", on_click=lambda: st.session_state.update(use_sample=True))
        if st.session_state.get("use_sample"):
            files = make_sample()
            mode_label = "Sample data (synthetic)"

if not files:
    st.title("LeakLens")
    st.warning("No data loaded. Select *Bundled dataset* or upload CSVs to begin.")
    st.stop()

# --------------------------------------------------------------------------- #
# Profile -> mapping -> model -> features
# --------------------------------------------------------------------------- #
profiles = {n: inspect_file(n, df) for n, df in files.items()}
mapping = suggest_mapping(files, profiles)

tx_name = next((n for n, p in profiles.items() if p.role == "transactions"), None)
if tx_name is None:
    st.error("Could not identify a transactions-like file (needs an order/transaction ID plus "
             "quantity/price/status).")
    st.stop()

date_col = mapping.by_file[tx_name].get("date")
date_range = None
if date_col:
    parsed = pd.to_datetime(files[tx_name][date_col], errors="coerce")
    if parsed.notna().any():
        lo, hi = parsed.min().date(), parsed.max().date()
        st.sidebar.markdown("##### Analysis window")
        picked = st.sidebar.date_input("Analysis window", value=(lo, hi),
                                       min_value=lo, max_value=hi,
                                       label_visibility="collapsed")
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            date_range = picked

model = build_model(files, mapping, date_range=date_range)
metrics = compute_metrics(model)
alerts, statuses = run_all(model, metrics)
ft = build_feature_table(model, mapping, files)
grain = base_grain_report(model)

# --------------------------------------------------------------------------- #
# Model controls
# --------------------------------------------------------------------------- #
st.sidebar.divider()
st.sidebar.markdown("##### Model")
capacity = st.sidebar.slider("Review capacity (% of orders flagged)", 2, 40, 10, 1,
                             help="The manager can act on this share of orders; the threshold is "
                                  "the matching quantile of predicted risk.") / 100.0
final_kind = st.sidebar.selectbox(
    "Final model", ["rf", "hgb"], index=0,
    format_func=lambda k: {"rf": "Random Forest (final)",
                           "hgb": "Histogram Gradient Boosting"}.get(k, k))

# data + settings signature: changing any of these invalidates a loaded model
sig = (round(len(ft), -1), round(float(ft["line_value"].sum()), -2),
       tuple(sorted(files.keys())), str(date_range), round(capacity, 3), final_kind)


def _train_now() -> None:
    try:
        with st.spinner("Training baseline + final model on the provided data…"):
            res = train_model(ft, capacity=capacity, final_kind=final_kind,
                              artifact_path=str(ARTIFACT))
        st.session_state["res"] = res
        st.session_state["res_sig"] = sig
        st.session_state["payload"] = load_artifact(ARTIFACT)
        st.session_state.pop("train_error", None)
    except Exception as exc:  # noqa: BLE001 - surface, never crash the app
        st.session_state["train_error"] = str(exc)
        st.session_state.pop("res", None)
        st.session_state.pop("payload", None)


# load a persisted model for this exact data window, else require training
if st.session_state.get("res_sig") != sig:
    st.session_state.pop("res", None)
    st.session_state.pop("payload", None)
if "res" not in st.session_state and ARTIFACT.exists():
    payload0 = load_artifact(ARTIFACT)
    if payload0 and payload0.get("result"):
        st.session_state["res"] = result_from_dict(payload0["result"])
        st.session_state["res_sig"] = sig
        st.session_state["payload"] = payload0

train_clicked = st.sidebar.button("Train model", type="primary", width="stretch")
st.sidebar.caption("Runs locally in-project. No external API or LLM is used as a predictor.")

res = st.session_state.get("res")
payload = st.session_state.get("payload")

if train_clicked:
    _train_now()
    res = st.session_state.get("res")
    payload = st.session_state.get("payload")
if st.session_state.get("train_error"):
    st.error(f"Model could not be trained: {st.session_state['train_error']}")

# --------------------------------------------------------------------------- #
# Sidebar summary
# --------------------------------------------------------------------------- #
st.sidebar.divider()
unmatched = sum(j.unmatched_rows for j in model.join_health)
st.sidebar.markdown(
    f'<div class="ll-kpi-l">Dataset</div>'
    f'<div class="ll-meta-line">{len(files)} files · {len(ft):,} rows · grain “{grain["grain"]}”</div>'
    f'<div class="ll-meta-line">{int(ft["eligible"].sum()):,} labelled · {len(alerts)} rule leak(s) · '
    f'{unmatched:,} unmatched join rows</div>'
    f'<div class="ll-meta-line">Model: {"trained" if res else "not trained"}'
    f'{"" if res else " — click Train model"}</div>'
    f'<div class="ll-hint">Press <kbd>S</kbd> to collapse or expand this panel.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
views.header("LeakLens",
             "A leakage decision system: profile the data, train a model in-project, audit whether "
             "it actually predicts, then prioritise a queue, explain it, act, and measure. ₹ (INR).",
             mode_label)
if "Sample" in mode_label:
    st.warning("Running on synthetic sample data — results do not reflect the provided dataset.")

# --------------------------------------------------------------------------- #
# Six screens
# --------------------------------------------------------------------------- #
tabs = st.tabs(["1 · Data trust", "2 · Train & evaluate", "3 · Priority queue",
                "4 · Why this risk", "5 · Action lab", "6 · Measure"])

# --- 1 Data trust -----------------------------------------------------------
with tabs[0]:
    views.render_data_trust(files, profiles, mapping, model, ft, res.specs if res else [], grain,
                            mapping.warnings)

# --- 2 Train & evaluate -----------------------------------------------------
with tabs[1]:
    clicked = views.render_train(res, capacity=capacity, kind=final_kind)
    if clicked and not train_clicked:
        _train_now()
        st.rerun()

# --- 3 Priority queue -------------------------------------------------------
active_idx = None
with tabs[2]:
    if res is None or payload is None:
        views.banner("warn", "Model required",
                     "Train the model in <b>2 · Train & evaluate</b> (or the sidebar) to build the "
                     "priority queue.")
    else:
        scored = score_all(payload, ft, threshold=res.threshold)
        st.session_state["scored"] = scored
        summary = summarise(scored, res.threshold, res.threshold_info.get("capacity", capacity))
        st.session_state["summary"] = summary
        dims = [d for d in ["category", "channel", "payment_method", "customer_segment", "region",
                            "preferred_channel"] if d in scored.columns]
        dim = st.selectbox("Rank segments by driver", dims, index=0, key="queue_dim")
        seg = segment_summary(scored, dim)
        sel = views.render_queue(summary, scored, seg, dim, alerts, statuses)
        if sel is not None:
            st.session_state["active_idx"] = sel
        active_idx = st.session_state.get("active_idx", 0)

# --- 4 Why this risk --------------------------------------------------------
with tabs[3]:
    scored = st.session_state.get("scored")
    if res is None or payload is None or scored is None:
        views.note("Train the model and open the priority queue first.")
    else:
        dim = st.session_state.get("queue_dim", "category")
        idx = int(st.session_state.get("active_idx", 0))
        idx = min(max(idx, 0), len(scored) - 1)
        record = scored.iloc[idx]
        over_time = exposure_over_time(scored)
        seg = segment_summary(scored, dim)
        evidence = driver_evidence(scored, record)
        tpls = templates_for("Priority queue (model)")
        if record.get("observed_adverse"):
            tpls = templates_for("Returns") + tpls
        views.render_why(res, scored, record, over_time, seg, dim, evidence, tpls)

# --- 5 Action lab -----------------------------------------------------------
with tabs[4]:
    summary = st.session_state.get("summary")
    exposure = float(summary.expected_exposure) if summary else 0.0
    baseline_value = format_inr(exposure)
    if res is not None:
        baseline_value = f"Estimated exposure {format_inr(exposure)} at base rate {res.base_rate*100:.1f}%"
    scope_default = "Priority queue — top exposure"
    rec = views.render_action_lab(all_templates(), load_log(ACTION_LOG), scope_default,
                                  exposure, baseline_value)
    if rec:
        append_log(ACTION_LOG, ActionRecord(**rec))
        st.session_state["action_logged"] = True
        st.rerun()
    if st.session_state.pop("action_logged", False):
        st.success("Action logged. See *6 · Measure* for the validation plan.")

# --- 6 Measure --------------------------------------------------------------
with tabs[5]:
    summary = st.session_state.get("summary")
    if summary is None and res is not None and payload is not None:
        scored = score_all(payload, ft, threshold=res.threshold)
        summary = summarise(scored, res.threshold, res.threshold_info.get("capacity", capacity))
    policy = ("The provided data cannot support predictive prioritisation: the permutation test "
              "found no signal. The highest-value next investment is <b>data collection</b> — capture "
              "return reasons, payment authorisation outcomes, fulfilment timestamps, and promotion "
              "exposure per order. Re-run this app on that richer data; the same pipeline will then "
              "report whether the new fields actually predict leakage.")
    if summary is None:
        views.banner("warn", "Train the model first", "The validation plan appears once a model exists.")
    else:
        views.render_measure(load_log(ACTION_LOG), summary, metrics, model, policy)
