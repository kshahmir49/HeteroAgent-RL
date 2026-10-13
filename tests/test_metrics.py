from heteroagent_rl.metrics import summarize_verifier_calibration


def test_verifier_calibration_counts_both_error_directions():
    rows = [
        {
            "verifier_verdict": "PASS",
            "base_status": "pass",
            "plus_status": "pass",
        },
        {
            "verifier_verdict": "PASS",
            "base_status": "pass",
            "plus_status": "fail",
        },
        {
            "verifier_verdict": "REVISE",
            "base_status": "fail",
            "plus_status": "fail",
        },
        {
            "verifier_verdict": "REVISE",
            "base_status": "pass",
            "plus_status": "pass",
        },
    ]

    metrics = summarize_verifier_calibration(rows)

    assert metrics["verifier_true_passes"] == 1
    assert metrics["verifier_false_passes"] == 1
    assert metrics["verifier_true_revises"] == 1
    assert metrics["verifier_false_revises"] == 1
    assert metrics["verifier_accuracy"] == 0.5
    assert metrics["verifier_precision"] == 0.5
    assert metrics["verifier_recall"] == 0.5
    assert metrics["verifier_false_pass_rate"] == 0.5
    assert metrics["verifier_false_revise_rate"] == 0.5


def test_verifier_calibration_handles_unknown_and_zero_denominators():
    rows = [
        {
            "verifier_verdict": "UNKNOWN",
            "base_status": "pass",
            "plus_status": "pass",
        }
    ]

    metrics = summarize_verifier_calibration(rows)

    assert metrics["verifier_unknowns"] == 1
    assert metrics["verifier_unknown_rate"] == 1.0
    assert metrics["verifier_accuracy"] == 0.0
    assert metrics["verifier_precision"] == 0.0
    assert metrics["verifier_recall"] == 0.0
