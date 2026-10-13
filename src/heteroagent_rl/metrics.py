from __future__ import annotations


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_verifier_calibration(trajectories: list[dict]) -> dict[str, float | int]:
    """Compare advisory verifier verdicts with strict objective correctness.

    PASS predicts that both base and plus tests pass.
    REVISE predicts that the solution is not strictly correct.
    UNKNOWN verdicts are tracked separately and excluded from accuracy.
    """
    tp = fp = tn = fn = unknown = 0

    for row in trajectories:
        verdict = row.get("verifier_verdict", "UNKNOWN")
        objective_pass = (
            row.get("base_status") == "pass"
            and row.get("plus_status") == "pass"
        )

        if verdict == "PASS":
            if objective_pass:
                tp += 1
            else:
                fp += 1
        elif verdict == "REVISE":
            if objective_pass:
                fn += 1
            else:
                tn += 1
        else:
            unknown += 1

    known = tp + fp + tn + fn
    total = known + unknown
    predicted_pass = tp + fp
    predicted_revise = tn + fn
    actual_pass = tp + fn

    return {
        "verifier_true_passes": tp,
        "verifier_false_passes": fp,
        "verifier_true_revises": tn,
        "verifier_false_revises": fn,
        "verifier_unknowns": unknown,
        "verifier_accuracy": _safe_div(tp + tn, known),
        "verifier_precision": _safe_div(tp, predicted_pass),
        "verifier_recall": _safe_div(tp, actual_pass),
        "verifier_false_pass_rate": _safe_div(fp, predicted_pass),
        "verifier_false_revise_rate": _safe_div(fn, predicted_revise),
        "verifier_unknown_rate": _safe_div(unknown, total),
    }
