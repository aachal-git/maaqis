class Grader1:
    def __init__(self):
        self.name = "aqi_prediction"
        self.difficulty = "easy"

    def _safe(self, v):
        return round(max(0.03, min(0.97, float(v))), 4)

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])
        scores = []
        for step in steps:
            action   = step.get("action", {})
            true_val = step.get("true_prediction", None)
            if action.get("action_type") == "predict" and true_val is not None:
                try:
                    error = abs(float(action.get("value", 0)) - float(true_val))
                    raw   = 0.93 - (error / 500.0) * 0.90
                    scores.append(self._safe(raw))
                except (ValueError, TypeError):
                    scores.append(0.05)
        if not scores:
            return 0.05
        return self._safe(sum(scores) / len(scores))

if __name__ == "__main__":
    g = Grader1()
    tests = [
        ({"steps": []}, None),
        ({"steps": [{"action": {"action_type": "predict", "value": 300},  "true_prediction": 300}]}, None),
        ({"steps": [{"action": {"action_type": "predict", "value": 0},    "true_prediction": 500}]}, None),
        ({"steps": [{"action": {"action_type": "predict", "value": 0},    "true_prediction": 0}]},   None),
        ({"steps": [{"action": {"action_type": "predict", "value": 500},  "true_prediction": 500}]}, None),
    ]
    for i, (ep, _) in enumerate(tests):
        s = g.grade(ep)
        assert 0 < s < 1, f"FAIL test {i}: {s}"
        print(f"Test {i}: {s} OK")
    print("Grader1 ALL PASSED")