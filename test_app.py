"""Headless smoke test of the LeakLens app via streamlit.testing.v1.AppTest.

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

    # Train through the sidebar button (the model must be trained in-app).
    assert len(at.sidebar.button) > 0, "No sidebar train button"
    at.sidebar.button[0].click().run()
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
        "Base grain",                        # screen 1
        "Modelling eligibility",             # screen 1
        "Baseline vs final",                 # screen 2
        "Signal audit",                      # screen 2
        "Calibration",                       # screen 2
        "Priority queue",                    # screen 3
        "Why this is flagged",               # screen 4
        "Recommended actions",               # screen 4
        "Turn a leak into a measurable next step",   # screen 5
        "Before / after plan",               # screen 6
        "How money is measured",             # retained round-1 method
    ]:
        assert needle in text, f"Missing section: {needle!r}"

    assert "₹" in text, "No INR-formatted value rendered"
    assert "no detectable signal" in text.lower(), "Signal-audit verdict missing"

    print("OK: app runs, trains, all six screens render, INR present, audit verdict present.")


if __name__ == "__main__":
    run()
