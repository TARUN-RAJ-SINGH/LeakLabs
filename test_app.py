"""Headless smoke test of the LeakLabs app via streamlit.testing.v1.AppTest.

Trains the model through the UI, then checks every screen renders, no error
elements are produced, INR formatting is used, and the honest signal-audit
verdict is present.
"""
import sys

from streamlit.testing.v1 import AppTest

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def run():
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception, f"App raised: {at.exception}"

    # Run the detailed analysis through the dedicated technical tab.
    run_buttons = [b for b in at.button if b.label == "Run analysis"]
    assert run_buttons, "No Run analysis button"
    run_buttons[0].click().run()
    assert not at.exception, f"App raised after training: {at.exception}"

    print("titles:", [t.value for t in at.title])
    print("dataframes:", len(at.dataframe))
    print("markdown blocks:", len(at.markdown))
    print("ERROR count:", len(at.error))
    for e in at.error:
        print("  ERROR:", e.value)
    assert len(at.error) == 0, "App produced error elements"

    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    for needle in [
        "Quick look",                           # simple decision brief
        "No clear pattern",                    # honest model state
        "Baseline vs final",                 # technical tab
        "Signal audit",                      # technical tab
        "Calibration",                       # technical tab
        "Orders to check first",             # screen 2
        "Why this order was marked",         # screen 3
        "What you can do",                   # screen 3
        "Make a simple action plan",         # screen 4
        "Before and after",                  # screen 5
        "How money is measured",             # retained round-1 method
    ]:
        assert needle in text, f"Missing section: {needle!r}"

    assert "₹" in text, "No INR-formatted value rendered"
    assert "no detectable signal" in text.lower(), "Signal-audit verdict missing"
    assert any(t.label == "6 · Technical details (ML)" for t in at.tabs), "Technical tab is missing"
    assert any(t.label == "ML check" for t in at.tabs), "ML check sub-tab is missing"
    assert any(w.label == "Search record or segment" for w in at.text_input), "Queue search is missing"
    assert len(at.download_button) >= 1, "Queue CSV export is missing"

    print("OK: app runs, trains, all six screens render, INR present, audit verdict present.")


if __name__ == "__main__":
    run()
