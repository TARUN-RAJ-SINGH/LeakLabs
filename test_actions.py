"""Smoke tests for the local action workflow used by the Action lab."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from leaklab.actions import ActionRecord, append_log, load_log, update_log_status


def run() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "action_log.csv"
        created = append_log(
            path,
            ActionRecord(
                scope="Priority queue — top exposure",
                leak_type="Returns",
                action="Audit top returned products",
                owner="Merchandising",
                expected_direction="Return rate should fall",
                validation_metric="Fulfilled return rate",
            ),
        )
        assert len(created) == 1
        action_id = created.iloc[0]["action_id"]
        assert str(action_id).startswith("act_")

        updated = update_log_status(path, action_id, "In progress")
        assert updated.iloc[0]["status"] == "In progress"
        assert load_log(path).iloc[0]["status"] == "In progress"

        unchanged = update_log_status(path, "missing", "Done")
        assert unchanged.iloc[0]["status"] == "In progress"

    print("OK: action IDs persist and statuses update safely.")


if __name__ == "__main__":
    run()
