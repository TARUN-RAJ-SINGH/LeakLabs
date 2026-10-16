"""leaklab.train - train a leakage-aware model from scratch on the provided data.

The pipeline is deliberately small and reproducible:

* **Target** - derived only from the *verified* status field.  ``Cancelled`` and
  ``Returned`` are the adverse outcome (1); ``Completed`` is 0.  Rows whose
  status has no verified outcome (``Pending``/unknown) are excluded from the
  labelled population entirely.
* **Features** - only pre-outcome fields (see :data:`leaklab.data.FEATURE_SPEC`).
  Post-outcome columns are never exposed to the model.
* **Split** - temporal hold-out (train on the past, test on the future) so the
  evaluation mimics how the model would actually be used.  A stratified split is
  used only when no usable date exists.  A middle validation slice chooses the
  decision threshold; the final artifact is refit on everything *before* the test
  period, so the test period is never seen during training.
* **Baseline and final** - Logistic Regression (interpretable baseline) and
  Random Forest (final).  Scikit-learn pipelines fit imputers/encoders on the
  training fold only.
* **Signal audit** - a label-permutation test asks whether the model learned
  anything at all, rather than assuming it did.  If the observed AUC is inside
  the null distribution, the honest verdict is "no detectable signal".

Nothing here calls an external API or an LLM.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import FeatureSpec, available_features

try:  # joblib ships with scikit-learn
    import joblib
except Exception:  # pragma: no cover
    joblib = None

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DEFAULT_CAPACITY = 0.10          # fraction of the population a manager can review
TEST_FRAC = 0.20
VAL_FRAC = 0.10                  # slice carved from the front of the training data
N_PERMUTATIONS = 12              # label-permutation repetitions for the signal audit


def _frozen(estimator):
    """Wrap a fitted estimator so a calibrator can be trained on held-out data."""
    try:
        from sklearn.frozen import FrozenEstimator
        return FrozenEstimator(estimator)
    except Exception:  # older scikit-learn
        return estimator


def _calibrate_on(estimator, X_val: pd.DataFrame, y_val: np.ndarray, prefit: bool = True):
    """Isotonic calibration so predicted probabilities are interpretable.

    With ``prefit=True`` the base estimator must already be fitted and the
    calibrator is trained on the separate validation slice; otherwise an internal
    3-fold CV is used (for the deployment refit, which has no spare slice).
    """
    if len(X_val) < 20 or len(np.unique(y_val)) < 2:
        return estimator
    try:
        if prefit:
            return CalibratedClassifierCV(_frozen(estimator), method="isotonic").fit(X_val, y_val)
        return CalibratedClassifierCV(estimator, cv=3, method="isotonic").fit(X_val, y_val)
    except Exception:
        return estimator


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    kind: str                    # "temporal" | "stratified"
    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray
    note: str = ""
    train_period: tuple = ()
    test_period: tuple = ()


def make_split(ft: pd.DataFrame, eligible: pd.Series) -> Split:
    """Temporal hold-out where possible, stratified otherwise."""
    idx = np.where(eligible.to_numpy())[0]
    d = pd.to_datetime(ft["record_date"], errors="coerce")
    usable = idx[d.iloc[idx].notna().to_numpy()] if len(idx) else idx

    if len(usable) >= 0.8 * len(idx) and len(idx) >= 60:
        order = usable[np.argsort(d.iloc[usable].to_numpy(), kind="stable")]
        n = len(order)
        n_test = max(1, int(round(n * TEST_FRAC)))
        n_val = max(1, int(round(n * VAL_FRAC)))
        test = order[n - n_test:]
        val = order[n - n_test - n_val: n - n_test]
        fit = order[: n - n_test - n_val]
        if len(fit) < 20:  # too small to train - fall back
            return _stratified_split(ft, eligible, "Temporal slice too small - stratified split used.")
        return Split(
            kind="temporal", fit=fit, val=val, test=test,
            note="Rows ordered by date: fit (past) / validation / test (future) - the test window is never seen in training.",
            train_period=(d.iloc[fit].min(), d.iloc[val].max()),
            test_period=(d.iloc[test].min(), d.iloc[test].max()),
        )
    return _stratified_split(ft, eligible, "No usable date column - stratified random split used.")


def _stratified_split(ft: pd.DataFrame, eligible: pd.Series, note: str) -> Split:
    from sklearn.model_selection import train_test_split

    idx = np.where(eligible.to_numpy())[0]
    y = ft.loc[idx, "target"].to_numpy()
    strat = y if len(np.unique(y)) > 1 else None
    train, test = train_test_split(idx, test_size=TEST_FRAC, random_state=SEED, stratify=strat)
    strat2 = ft.loc[train, "target"].to_numpy()
    strat2 = strat2 if len(np.unique(strat2)) > 1 else None
    fit, val = train_test_split(train, test_size=VAL_FRAC / (1 - TEST_FRAC),
                                random_state=SEED, stratify=strat2)
    return Split(kind="stratified", fit=np.sort(fit), val=np.sort(val), test=np.sort(test), note=note)


# --------------------------------------------------------------------------- #
# Design matrix + estimators
# --------------------------------------------------------------------------- #
def _design(ft: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for s in specs:
        if s.name in ft.columns:
            col = ft[s.name]
        else:
            col = pd.Series(np.nan, index=ft.index)
        if s.kind == "numeric":
            cols[s.name] = pd.to_numeric(col, errors="coerce")
        else:
            cols[s.name] = col.astype("string").fillna("(missing)")
    return pd.DataFrame(cols, index=ft.index)


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])
    return ColumnTransformer([("num", num_pipe, numeric),
                              ("cat", cat_pipe, categorical)],
                             remainder="drop", sparse_threshold=0.3)


def make_pipeline(kind: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = _preprocessor(numeric, categorical)
    if kind == "logreg":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=160, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                             random_state=SEED)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("pre", pre), ("clf", clf)])


MODEL_LABELS = {
    "logreg": "Logistic Regression (baseline)",
    "rf": "Random Forest (final)",
    "hgb": "Histogram Gradient Boosting (optional)",
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metric_block(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    base = float(np.mean(y_true)) if len(y_true) else float("nan")
    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "base_rate": base,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flagged": int(pred.sum()),
        "flagged_rate": float(pred.mean()) if len(pred) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "brier": float(brier_score_loss(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "lift": (float(precision_score(y_true, pred, zero_division=0) / base)
                 if base and base > 0 else float("nan")),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
        out["pr_auc"] = float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def auc_ci(y_true: np.ndarray, proba: np.ndarray, n_boot: int = 300, seed: int = SEED) -> tuple[float, float]:
    """Bootstrap 95% CI for ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], proba[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #
def choose_threshold(y_val: np.ndarray, proba_val: np.ndarray, capacity: float) -> dict:
    """Pick the probability cut that flags roughly ``capacity`` of the population.

    Business rationale: a manager has finite review capacity, so the operating
    point is defined by *how many cases they can action*, not by an arbitrary
    0.5 cut.  The same capacity is applied to the hold-out set for reporting.
    """
    capacity = float(min(max(capacity, 0.005), 0.9))
    if len(proba_val) == 0:
        return {"threshold": 0.5, "capacity": capacity, "reason": "No validation rows; default 0.5."}
    thr = float(np.quantile(proba_val, 1 - capacity))
    flagged = float((proba_val >= thr).mean())
    return {
        "threshold": thr,
        "capacity": capacity,
        "flagged_rate_val": flagged,
        "reason": (f"Review capacity {capacity:.0%}: flag the top {capacity:.0%} of scored orders by "
                   "predicted risk. Precision/recall at this cut are reported on the hold-out period."),
    }


# --------------------------------------------------------------------------- #
# Signal audit (label permutation)
# --------------------------------------------------------------------------- #
def signal_audit(ft: pd.DataFrame, split: Split, specs: list[FeatureSpec],
                 observed_auc: float, fast_kind: str = "logreg",
                 n_perm: int = N_PERMUTATIONS) -> dict:
    """Permutation test: could this AUC arise from pure noise?

    We refit the *baseline* pipeline on shuffled training labels and score the
    same hold-out set.  The fraction of permuted AUCs at least as large as the
    observed AUC is an empirical p-value for "the features carry no signal".
    """
    numeric = [s.name for s in specs if s.kind == "numeric"]
    categorical = [s.name for s in specs if s.kind == "categorical"]
    X = _design(ft, specs)
    y = ft["target"].to_numpy()

    X_fit, y_fit = X.iloc[split.fit], y[split.fit]
    X_test, y_test = X.iloc[split.test], y[split.test]
    if len(np.unique(y_test)) < 2 or len(X_fit) == 0:
        return {"available": False, "observed_auc": observed_auc, "p_value": float("nan"),
                "permuted_aucs": [], "n_perm": 0,
                "verdict": "Not enough held-out data with both classes to audit."}

    rng = np.random.default_rng(SEED)
    perm_aucs: list[float] = []
    for _ in range(int(n_perm)):
        pipe = make_pipeline(fast_kind, numeric, categorical)
        y_perm = rng.permutation(y_fit)
        if len(np.unique(y_perm)) < 2:
            continue
        pipe.fit(X_fit, y_perm)
        try:
            perm_aucs.append(float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])))
        except ValueError:
            continue

    if not perm_aucs or not np.isfinite(observed_auc):
        return {"available": False, "observed_auc": observed_auc, "p_value": float("nan"),
                "permuted_aucs": perm_aucs, "n_perm": len(perm_aucs),
                "verdict": "Permutation test could not be completed."}

    arr = np.array(perm_aucs)
    p = float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1))
    null_hi = float(np.percentile(arr, 95))
    if p > 0.05:
        verdict = ("No detectable signal: the hold-out AUC is inside the label-permutation "
                   "null distribution. The model must not be trusted as a predictor on this data.")
        usable = False
    else:
        verdict = ("Signal detected: the hold-out AUC exceeds the permutation null, so the model "
                   "discriminates better than chance on this data.")
        usable = True
    return {"available": True, "observed_auc": float(observed_auc), "p_value": p,
            "permuted_aucs": perm_aucs, "n_perm": len(perm_aucs),
            "null_mean": float(arr.mean()), "null_95": null_hi,
            "verdict": verdict, "usable": usable}


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class TrainingResult:
    target_label: str
    population: int
    positives: int
    base_rate: float
    specs: list[FeatureSpec]
    numeric: list[str]
    categorical: list[str]
    split: Split | None
    threshold_info: dict
    baseline: dict
    final: dict
    baseline_label: str
    final_label: str
    pr_curve: pd.DataFrame
    roc_curve: pd.DataFrame
    calibration: pd.DataFrame
    importance: pd.DataFrame
    audit: dict
    auc_ci: tuple
    seed: int
    trained_at: str
    seconds: float
    pipeline: Any = None
    artifact_path: str | None = None
    notes: list[str] = field(default_factory=list)
    final_kind: str = "rf"
    # flat split fields (survive serialization, unlike the numpy Split object)
    split_kind: str = "temporal"
    split_note: str = ""
    fit_n: int = 0
    val_n: int = 0
    test_n: int = 0
    train_span: str = ""
    test_span: str = ""

    @property
    def threshold(self) -> float:
        return float(self.threshold_info.get("threshold", 0.5))

    def model_card(self) -> dict:
        f, b = self.final, self.baseline
        return {
            "target": self.target_label,
            "population": self.population,
            "positives": self.positives,
            "base_rate": self.base_rate,
            "features": [s.name for s in self.specs],
            "split": self.split_kind,
            "split_note": self.split_note,
            "train_span": self.train_span,
            "test_span": self.test_span,
            "baseline": self.baseline_label,
            "final": self.final_label,
            "baseline_auc": b.get("roc_auc"),
            "final_auc": f.get("roc_auc"),
            "pr_auc": f.get("pr_auc"),
            "threshold": self.threshold,
            "capacity": self.threshold_info.get("capacity"),
            "seed": self.seed,
            "trained_at": self.trained_at,
            "signal_verdict": self.audit.get("verdict"),
            "signal_p": self.audit.get("p_value"),
        }


def _jsonify(obj):
    """Recursively convert numpy/pandas scalars to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj)):
        return None
    return obj


def result_to_dict(res: TrainingResult) -> dict:
    """Serialise everything the UI needs (no fitted pipeline)."""
    return {
        "target_label": res.target_label,
        "population": res.population,
        "positives": res.positives,
        "base_rate": res.base_rate,
        "specs": [[s.name, s.kind, s.label, s.source, bool(s.pre_outcome), s.note] for s in res.specs],
        "numeric": res.numeric,
        "categorical": res.categorical,
        "threshold_info": _jsonify(res.threshold_info),
        "baseline": _jsonify(res.baseline),
        "final": _jsonify(res.final),
        "baseline_label": res.baseline_label,
        "final_label": res.final_label,
        "pr_curve": res.pr_curve.to_dict("records"),
        "roc_curve": res.roc_curve.to_dict("records"),
        "calibration": res.calibration.to_dict("records"),
        "importance": res.importance.to_dict("records"),
        "audit": _jsonify(res.audit),
        "auc_ci": _jsonify(list(res.auc_ci)),
        "seed": res.seed,
        "trained_at": res.trained_at,
        "seconds": res.seconds,
        "notes": res.notes,
        "final_kind": res.final_kind,
        "artifact_path": res.artifact_path,
        "split_kind": res.split_kind,
        "split_note": res.split_note,
        "fit_n": res.fit_n,
        "val_n": res.val_n,
        "test_n": res.test_n,
        "train_span": res.train_span,
        "test_span": res.test_span,
    }


def result_from_dict(d: dict) -> TrainingResult:
    specs = [FeatureSpec(n, k, l, s, bool(po), no) for (n, k, l, s, po, no) in d.get("specs", [])]
    return TrainingResult(
        target_label=d.get("target_label", ""),
        population=int(d.get("population", 0)),
        positives=int(d.get("positives", 0)),
        base_rate=float(d.get("base_rate", float("nan"))),
        specs=specs,
        numeric=list(d.get("numeric", [])),
        categorical=list(d.get("categorical", [])),
        split=None,
        threshold_info=d.get("threshold_info", {}),
        baseline=d.get("baseline", {}),
        final=d.get("final", {}),
        baseline_label=d.get("baseline_label", ""),
        final_label=d.get("final_label", ""),
        pr_curve=pd.DataFrame(d.get("pr_curve", [])),
        roc_curve=pd.DataFrame(d.get("roc_curve", [])),
        calibration=pd.DataFrame(d.get("calibration", [])),
        importance=pd.DataFrame(d.get("importance", [])),
        audit=d.get("audit", {}),
        auc_ci=tuple(d.get("auc_ci", (float("nan"), float("nan")))),
        seed=int(d.get("seed", SEED)),
        trained_at=d.get("trained_at", ""),
        seconds=float(d.get("seconds", 0.0)),
        pipeline=None,
        artifact_path=d.get("artifact_path"),
        notes=list(d.get("notes", [])),
        final_kind=d.get("final_kind", "rf"),
        split_kind=d.get("split_kind", "temporal"),
        split_note=d.get("split_note", ""),
        fit_n=int(d.get("fit_n", 0)),
        val_n=int(d.get("val_n", 0)),
        test_n=int(d.get("test_n", 0)),
        train_span=d.get("train_span", ""),
        test_span=d.get("test_span", ""),
    )


def _span(period) -> str:
    try:
        lo, hi = period
        if pd.isna(lo) or pd.isna(hi):
            return "n/a"
        return f"{pd.Timestamp(lo):%Y-%m-%d} to {pd.Timestamp(hi):%Y-%m-%d}"
    except Exception:
        return "n/a"


def _curve_tables(y: np.ndarray, proba: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pr = pd.DataFrame()
    roc = pd.DataFrame()
    if len(np.unique(y)) > 1:
        p, r, _ = precision_recall_curve(y, proba)
        pr = pd.DataFrame({"recall": r, "precision": p})
        fpr, tpr, _ = roc_curve(y, proba)
        roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    return pr, roc


def _calibration_table(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    if len(y) == 0:
        return pd.DataFrame(columns=["bin", "predicted", "observed", "n"])
    edges = np.quantile(proba, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame(columns=["bin", "predicted", "observed", "n"])
    cuts = pd.cut(proba, bins=edges, include_lowest=True, duplicates="drop")
    g = pd.DataFrame({"p": proba, "y": y, "bin": cuts}).groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
    g["bin"] = g["bin"].astype(str)
    return g


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def train_model(ft: pd.DataFrame, capacity: float = DEFAULT_CAPACITY,
                final_kind: str = "rf", seed: int = SEED,
                artifact_path: str | Path | None = None) -> TrainingResult:
    """Train baseline + final models and return a fully-populated result."""
    t0 = time.time()
    eligible = ft["eligible"].astype(bool)
    pop = int(eligible.sum())
    if pop < 30:
        raise ValueError(
            f"Only {pop} labelled rows with a verified outcome - not enough to train. "
            "A verified status field with at least ~30 completed/cancelled/returned rows is required.")

    specs = available_features(ft)
    if len(specs) < 3:
        raise ValueError("Fewer than 3 usable pre-outcome features were found - cannot train.")
    numeric = [s.name for s in specs if s.kind == "numeric"]
    categorical = [s.name for s in specs if s.kind == "categorical"]

    split = make_split(ft, eligible)
    X = _design(ft, specs)
    y = ft["target"].astype(float).to_numpy()
    pos = int(np.nansum(y[split.fit]))

    X_fit, y_fit = X.iloc[split.fit], y[split.fit]
    X_val, y_val = X.iloc[split.val], y[split.val]
    X_test, y_test = X.iloc[split.test], y[split.test]

    notes: list[str] = []

    # --- baseline ------------------------------------------------------------
    baseline = make_pipeline("logreg", numeric, categorical)
    baseline.fit(X_fit, y_fit)
    baseline = _calibrate_on(baseline, X_val, y_val, prefit=True)
    base_val = baseline.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    base_test = baseline.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])

    # --- final ---------------------------------------------------------------
    final = make_pipeline(final_kind, numeric, categorical)
    final.fit(X_fit, y_fit)
    final = _calibrate_on(final, X_val, y_val, prefit=True)
    fin_val = final.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])

    # operating point chosen on the final model's validation scores
    thr_info = choose_threshold(y_val, fin_val if len(fin_val) else base_val, capacity)
    thr = thr_info["threshold"]
    fin_test = final.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])
    baseline_metrics = _metric_block(y_test, base_test, thr)
    final_metrics = _metric_block(y_test, fin_test, thr)

    # --- curves / calibration -----------------------------------------------
    pr_curve, roc_curve_df = _curve_tables(y_test, fin_test)
    calibration = _calibration_table(y_test, fin_test)

    # --- feature importance (permutation on the raw feature columns) ---------
    importance = pd.DataFrame(columns=["feature", "importance", "kind", "label"])
    if len(X_test) and len(np.unique(y_test)) > 1:
        from sklearn.inspection import permutation_importance
        scoring = "average_precision" if len(np.unique(y_test)) > 1 else "roc_auc"
        try:
            r = permutation_importance(final, X_test, y_test, scoring=scoring,
                                       n_repeats=5, random_state=seed, n_jobs=1)
            imp = pd.DataFrame({
                "feature": X_test.columns,
                "importance": r.importances_mean,
                "std": r.importances_std,
            })
            kind_map = {s.name: s.kind for s in specs}
            label_map = {s.name: s.label for s in specs}
            imp["kind"] = imp["feature"].map(kind_map)
            imp["label"] = imp["feature"].map(label_map)
            imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
            importance = imp[["feature", "label", "kind", "importance", "std"]]
        except Exception as exc:  # never let importance break training
            notes.append(f"Permutation importance unavailable: {exc}")

    # --- signal audit --------------------------------------------------------
    audit = signal_audit(ft, split, specs, final_metrics.get("roc_auc", float("nan")))

    # --- confidence interval for the reported AUC ----------------------------
    ci = auc_ci(y_test, fin_test)

    # --- artifact: refit on everything BEFORE the test period ----------------
    deploy_idx = np.concatenate([split.fit, split.val])
    deploy_base = make_pipeline(final_kind, numeric, categorical)
    deploy = _calibrate_on(deploy_base, X.iloc[deploy_idx], y[deploy_idx], prefit=False)
    artifact = None
    if artifact_path is not None and joblib is not None:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": deploy,
            "numeric": numeric,
            "categorical": categorical,
            "specs": [(s.name, s.kind, s.label) for s in specs],
            "target_label": _target_label(ft),
            "threshold": thr,
            "capacity": capacity,
            "seed": seed,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "split_kind": split.kind,
            "test_metrics": final_metrics,
            "signal_report": audit,
            "final_kind": final_kind,
        }
        try:
            joblib.dump(payload, path)
            artifact = str(path)
        except Exception as exc:
            notes.append(f"Could not persist artifact: {exc}")

    notes.append("The final artifact is refit on all rows before the test window, so the "
                 "reported hold-out metrics come from a model that never saw the test period.")

    result = TrainingResult(
        target_label=_target_label(ft),
        population=pop,
        positives=pos,
        base_rate=float(np.nanmean(y[eligible.to_numpy()])),
        specs=specs, numeric=numeric, categorical=categorical,
        split=split, threshold_info=thr_info,
        baseline=baseline_metrics, final=final_metrics,
        baseline_label=MODEL_LABELS["logreg"], final_label=MODEL_LABELS.get(final_kind, final_kind),
        pr_curve=pr_curve, roc_curve=roc_curve_df, calibration=calibration,
        importance=importance, audit=audit, auc_ci=ci, seed=seed,
        trained_at=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        seconds=float(time.time() - t0),
        pipeline=deploy, artifact_path=artifact, notes=notes, final_kind=final_kind,
        split_kind=split.kind, split_note=split.note,
        fit_n=len(split.fit), val_n=len(split.val), test_n=len(split.test),
        train_span=_span(split.train_period), test_span=_span(split.test_period),
    )

    # persist the renderable result alongside the pipeline so the UI can show the
    # evaluation without retraining in a later session
    if artifact is not None and joblib is not None:
        try:
            payload["result"] = result_to_dict(result)
            joblib.dump(payload, Path(artifact))
        except Exception as exc:
            notes.append(f"Could not persist result metadata: {exc}")
    return result


def _target_label(ft: pd.DataFrame) -> str:
    roles = ft.loc[ft["eligible"], "status_raw"].dropna().astype(str).unique().tolist()
    roles = sorted(roles)
    return "Leakage outcome = Cancelled or Returned (vs Completed); eligible statuses: " + ", ".join(roles)


# --------------------------------------------------------------------------- #
# Loading / scoring helpers
# --------------------------------------------------------------------------- #
def load_artifact(path: str | Path) -> dict | None:
    if joblib is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def score_frame(payload: dict, ft: pd.DataFrame) -> np.ndarray:
    """Apply a persisted payload to a feature frame, returning P(adverse)."""
    numeric = payload.get("numeric", [])
    categorical = payload.get("categorical", [])
    specs = [FeatureSpec(n, k, l) for (n, k, l) in payload.get("specs", [])]
    if not specs:
        specs = [FeatureSpec(n, "numeric", n) for n in numeric] + \
                [FeatureSpec(c, "categorical", c) for c in categorical]
    X = _design(ft, specs)
    pipe = payload["pipeline"]
    return pipe.predict_proba(X)[:, 1]
