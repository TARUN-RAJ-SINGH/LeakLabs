"""LeakLabs (The Money Leak - Round 2) — a simple leak-to-action dashboard.

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
from leaklab.actions import ActionRecord, all_templates, append_log, load_log, templates_for
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
    page_title="LeakLabs",
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
    st.title("LeakLabs")
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
# Technical settings are rendered only inside the Technical details (ML) tab.
# --------------------------------------------------------------------------- #
if "ml_capacity" not in st.session_state:
    st.session_state["ml_capacity"] = 10
if "ml_final_kind" not in st.session_state:
    st.session_state["ml_final_kind"] = "rf"
capacity = float(st.session_state["ml_capacity"]) / 100.0
final_kind = st.session_state["ml_final_kind"]

# data + settings signature: changing any of these invalidates a loaded model
sig = (round(len(ft), -1), round(float(ft["line_value"].sum()), -2),
       tuple(sorted(files.keys())), str(date_range), round(capacity, 3), final_kind)


def _train_now() -> None:
    try:
        with st.spinner("Checking the data and building the order list…"):
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

res = st.session_state.get("res")
payload = st.session_state.get("payload")

if st.session_state.get("train_error"):
    st.error(f"Detailed analysis could not run: {st.session_state['train_error']}")

# --------------------------------------------------------------------------- #
# Sidebar summary
# --------------------------------------------------------------------------- #
st.sidebar.divider()
unmatched = sum(j.unmatched_rows for j in model.join_health)
st.sidebar.markdown(
    f'<div class="ll-kpi-l">Dataset</div>'
    f'<div class="ll-meta-line">{len(files)} files · {len(ft):,} rows · grain “{grain["grain"]}”</div>'
    f'<div class="ll-meta-line">{int(ft["eligible"].sum()):,} records with known outcomes · {len(alerts)} warning(s) · '
    f'{unmatched:,} unmatched join rows</div>'
    f'<div class="ll-meta-line">Review list: {"ready" if res else "not ready"}'
    f'{"" if res else " — open Technical details (ML)"}</div>'
    f'<div class="ll-hint">Press <kbd>S</kbd> to collapse or expand this panel.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
views.header("LeakLabs",
             "Find where money is leaking, choose what to fix, and check whether the fix worked. ₹ (INR).",
             mode_label)
if "Sample" in mode_label:
    st.warning("Running on synthetic sample data — results do not reflect the provided dataset.")

# Build the queue once before rendering the screens so the first viewport can
# show the same decision state as the tabs below. This makes the app feel like
# a product: a manager sees the current recommendation immediately after
# training instead of having to discover it by opening a tab.
scored = None
summary = None
if res is not None and payload is not None:
    scored = score_all(payload, ft, threshold=res.threshold)
    summary = summarise(scored, res.threshold, res.threshold_info.get("capacity", capacity))
    st.session_state["scored"] = scored
    st.session_state["summary"] = summary

views.render_command_center(
    files=files,
    mode_label=mode_label,
    ft=ft,
    res=res,
    summary=summary,
    alerts=alerts,
    log_df=load_log(ACTION_LOG),
)

# --------------------------------------------------------------------------- #
# Six screens
# --------------------------------------------------------------------------- #
tabs = st.tabs(["1 · Dashboard", "2 · Where is the leak?", "3 · Check an order",
                "4 · Action plan", "5 · Measure results", "6 · Technical details (ML)"])

# --- 1 Dashboard ------------------------------------------------------------
with tabs[0]:
    views.render_simple_dashboard(metrics, alerts, statuses, summary, res, load_log(ACTION_LOG))

# --- 2 Where is the leak? ---------------------------------------------------
active_idx = None
with tabs[1]:
    if res is None or payload is None:
        views.banner("warn", "Order review list not ready",
                     "Open <b>Technical details (ML)</b> and choose <b>Run analysis</b> to build the "
                     "list of orders to check first.")
    else:
        dims = [d for d in ["category", "channel", "payment_method", "customer_segment", "region",
                            "preferred_channel"] if d in scored.columns]
        dim = st.selectbox("Group orders by", dims, index=0, key="queue_dim",
                           format_func=lambda value: value.replace("_", " ").title())
        seg = segment_summary(scored, dim)
        sel = views.render_queue(summary, scored, seg, dim, alerts, statuses)
        if sel is not None:
            st.session_state["active_idx"] = sel
        active_idx = st.session_state.get("active_idx", 0)

# --- 3 Check an order -------------------------------------------------------
with tabs[2]:
    scored = st.session_state.get("scored")
    if res is None or payload is None or scored is None:
        views.note("Run the checks in **Technical details (ML)**, then choose an order from **Where is the leak?**.")
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

# --- 4 Action plan ----------------------------------------------------------
with tabs[3]:
    summary = st.session_state.get("summary") or summary
    exposure = float(summary.expected_exposure) if summary else 0.0
    baseline_value = format_inr(exposure)
    if res is not None:
        baseline_value = f"Estimated amount at risk {format_inr(exposure)} at starting problem rate {res.base_rate*100:.1f}%"
    scope_default = "Top orders to check first"
    preferred_leak = "Priority queue (model)"
    if scored is not None and len(scored):
        idx = int(st.session_state.get("active_idx", 0))
        idx = min(max(idx, 0), len(scored) - 1)
        focus = scored.iloc[idx]
        focus_dim = st.session_state.get("queue_dim", "category")
        scope_default = (f"{focus.get('record_id', 'Selected record')} · "
                         f"{focus_dim.replace('_', ' ').title()}: {focus.get(focus_dim, '—')}")
        if str(focus.get("status", "")).lower().startswith("return"):
            preferred_leak = "Returns"
        elif str(focus.get("status", "")).lower().startswith("cancel"):
            preferred_leak = "Cancellations"
    rec = views.render_action_lab(
        all_templates(), load_log(ACTION_LOG), scope_default, exposure, baseline_value,
        action_log_path=ACTION_LOG, preferred_leak_type=preferred_leak,
    )
    if rec:
        append_log(ACTION_LOG, ActionRecord(**rec))
        st.session_state["action_logged"] = True
        st.rerun()
    if st.session_state.pop("action_logged", False):
        st.success("Action saved. See *Measure results* to track the change.")

# --- 5 Measure results ------------------------------------------------------
with tabs[4]:
    summary = st.session_state.get("summary")
    if summary is None and res is not None and payload is not None:
        scored = score_all(payload, ft, threshold=res.threshold)
        summary = summarise(scored, res.threshold, res.threshold_info.get("capacity", capacity))
    if res is not None and res.audit.get("usable"):
        policy = ("The detailed checks found a useful pattern, but they cannot prove why a problem happened. "
                  "Run the same checks after each new file and compare the problem rate for worked orders "
                  "with the overall starting rate.")
    else:
        policy = ("The current files cannot reliably tell us which orders will have a problem. The next "
                  "best step is to collect return reasons, payment results, delivery times, and promotion "
                  "details. Run the same checks again when those details are available.")
    if summary is None:
        views.banner("warn", "Review list not ready", "The results page appears after the order list is created.")
    else:
        views.render_measure(load_log(ACTION_LOG), summary, metrics, model, policy)

# --- 6 Technical details (ML) ----------------------------------------------
with tabs[5]:
    tech_tabs = st.tabs(["ML check", "Data check"])
    with tech_tabs[0]:
        settings_left, settings_right = st.columns(2, gap="large")
        with settings_left:
            st.slider("How many orders can we check?", 2, 40,
                      key="ml_capacity", help="Leave the default for the demo.")
        with settings_right:
            st.selectbox("Analysis method", ["rf", "hgb"], key="ml_final_kind",
                         format_func=lambda k: {"rf": "Standard", "hgb": "Alternative"}.get(k, k))
        clicked = views.render_train(res, capacity=capacity, kind=final_kind)
        if clicked:
            _train_now()
            st.rerun()
    with tech_tabs[1]:
        views.render_data_trust(files, profiles, mapping, model, ft, res.specs if res else [], grain,
                                mapping.warnings)
