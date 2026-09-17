import json
from pathlib import Path


def load_goldens(path):
    """Load a golden dataset from a JSON file path or project-relative string."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[1] / candidate
    with candidate.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def summarize_by_metric(result):
    """Return a lightweight summary of a deepeval result object."""
    try:
        test_results = getattr(result, "test_results", [])
    except Exception:
        test_results = []

    summary = {
        "case_count": len(test_results),
        "results": [],
    }
    for item in test_results:
        metrics = getattr(item, "metrics", [])
        summary["results"].append({
            "name": getattr(item, "name", "unnamed"),
            "metrics": [
                {
                    "name": getattr(metric, "name", type(metric).__name__),
                    "score": getattr(metric, "score", None),
                    "reason": getattr(metric, "reason", None),
                }
                for metric in metrics
            ],
        })
    return summary


def print_summary(label, payload):
    """Pretty-print a compact summary for CLI runs."""
    cases = payload.get("case_count", 0)
    print(f"{label} eval summary: {cases} cases")
    for result in payload.get("results", []):
        print(f"- {result['name']}")
        for metric in result.get("metrics", []):
            score = metric.get("score")
            reason = metric.get("reason")
            print(f"  * {metric['name']}: {score}")
            if reason:
                print(f"    reason: {reason}")
