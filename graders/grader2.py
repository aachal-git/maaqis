class Grader2:
    SCORE_MAP = {
        ("traffic",  "traffic"):  0.93,
        ("industry", "industry"): 0.93,
        ("dust",     "dust"):     0.93,
        ("traffic",  "industry"): 0.25,
        ("traffic",  "dust"):     0.20,
        ("industry", "traffic"):  0.25,
        ("industry", "dust"):     0.30,
        ("dust",     "traffic"):  0.20,
        ("dust",     "industry"): 0.30,
    }

    def __init__(self):
        self.name = "source_classification"
        self.difficulty = "medium"

    def _safe(self, v):
        return round(max(0.03, min(0.97, float(v))), 4)

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])
        scores = []
        for step in steps:
            action      = step.get("action", {})
            true_source = step.get("true_source", None)
            if action.get("action_type") == "classify" and true_source is not None:
                predicted = str(action.get("value", "")).strip().lower()
                true_src  = str(true_source).strip().lower()
                raw       = self.SCORE_MAP.get((true_src, predicted), 0.05)
                scores.append(self._safe(raw))
        if not scores:
            return 0.05
        return self._safe(sum(scores) / len(scores))

if __name__ == "__main__":
    g = Grader2()
    tests = [
        {"steps": []},
        {"steps": [{"action": {"action_type": "classify", "value": "traffic"},  "true_source": "traffic"}]},
        {"steps": [{"action": {"action_type": "classify", "value": "industry"}, "true_source": "industry"}]},
        {"steps": [{"action": {"action_type": "classify", "value": "dust"},     "true_source": "dust"}]},
        {"steps": [{"action": {"action_type": "classify", "value": "dust"},     "true_source": "traffic"}]},
    ]
    for i, ep in enumerate(tests):
        s = g.grade(ep)
        assert 0 < s < 1, f"FAIL test {i}: {s}"
        print(f"Test {i}: {s} OK")
    print("Grader2 ALL PASSED")