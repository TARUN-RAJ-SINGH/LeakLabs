"""Command-line smoke test for the leaklab pipeline (no Streamlit).

Verifies the app starts producing a real alert from the provided CSVs and prints
the discovered mapping, join health, KPIs and ranked alerts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Windows consoles often default to cp1252; make ₹-containing output safe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from leaklab.profile import inspect_file, suggest_mapping
from leaklab.data import build_model, build_feature_table, base_grain_report
from leaklab.metrics import compute_metrics
from leaklab.rules import run_all

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\DATATHON")
FILES = ["customers.csv", "products.csv", "transactions.csv", "inventory.csv", "campaigns.csv"]


def main() -> int:
    files, profiles = {}, {}
    for name in FILES:
        p = DATA / name
        if not p.exists():
            print("MISSING", p)
            continue
        df = pd.read_csv(p)
        files[name] = df
        profiles[name] = inspect_file(name, df)

    mapping = suggest_mapping(files, profiles)
    print("=" * 70)
    print("FILE ROLES:", {p.name: p.role for p in profiles.values()})
    for f, m in mapping.by_file.items():
        print(f"  {f} -> {profiles[f].role}: {m}")
    if mapping.warnings:
        print("WARNINGS:", mapping.warnings)

    model = build_model(files, mapping)
    print("=" * 70)
    print("FLAGS:", {k: v for k, v in model.flags.items()})
    print("JOIN HEALTH:")
    for j in model.join_health:
        print("  ", j.as_row())
    print("grain:", model.grain, "rows:", len(model.tx))

    metrics = compute_metrics(model)
    print("=" * 70)
    print("METRICS:")
    for k, met in metrics.items():
        print(f"  {met.label:38s} {met.display:>16s}  [{met.availability}]")

    alerts, statuses = run_all(model, metrics)
    print("=" * 70)
    print("RULE STATUS:")
    for s in statuses:
        print(f"  {s.title:20s} {s.status:12s} {s.detail}")
    print("=" * 70)
    print(f"ALERTS (top {len(alerts)}):")
    for a in alerts:
        print(f"  #{a.rank} {a.title}  [score={a.evidence_score}, sev={a.severity}]")
        print(f"      scope: {a.scope}")
        print(f"      rule : {a.rule_text}")
        print(f"      rate : {a.rate} vs threshold {a.threshold}")
        print(f"      num  : {a.numerator}")
        print(f"      den  : {a.denominator}")
        print(f"      impact: {a.impact_value} ({a.impact_basis})")
        print(f"      action: {a.action}")
        if a.bars is not None and len(a.bars):
            print(f"      bars : {list(a.bars['label'])[:5]}")
        if a.trend is not None and len(a.trend):
            print(f"      trend: {list(a.trend['period'])[:4]} ... {list(a.trend['period'])[-2:]}")

    assert len(alerts) >= 1, "Acceptance: at least one supported leak must surface."
    print("\nOK: pipeline produced", len(alerts), "alert(s).")

    # ---- Round 2: feature table + model -------------------------------------
    from leaklab.train import train_model
    from leaklab.score import score_all, summarise

    print("=" * 70)
    ft = build_feature_table(model, mapping, files)
    print("FEATURE TABLE:", ft.shape, "eligible:", int(ft["eligible"].sum()))
    print("GRAIN:", base_grain_report(model))
    res = train_model(ft, artifact_path=str(DATA / "artifacts" / "selftest.joblib"))
    print("MODEL:", res.final_label, "hold-out ROC-AUC %.3f" % res.final["roc_auc"],
          "PR-AUC %.3f" % res.final["pr_auc"], "| baseline AUC %.3f" % res.baseline["roc_auc"])
    print("AUDIT p=%.3f ->" % res.audit["p_value"], res.audit["verdict"][:72])
    print("IMPORTANCE top:", list(res.importance["feature"][:4]) if len(res.importance) else [])
    sc = score_all({"pipeline": res.pipeline, "threshold": res.threshold,
                    "numeric": res.numeric, "categorical": res.categorical,
                    "specs": [(s.name, s.kind, s.label) for s in res.specs]}, ft, res.threshold)
    s = summarise(sc, res.threshold, res.threshold_info["capacity"])
    print(f"QUEUE: {s.population:,} scored, {s.flagged:,} flagged, observed leak "
          f"₹{s.observed_leak_value:,.0f}, estimated exposure ₹{s.expected_exposure:,.0f}")
    assert res.population >= 30, "Model must train on a labelled population."
    print("\nOK: model trained in-project; queue and audit produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
