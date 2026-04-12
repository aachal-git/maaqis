class Grader3:
    SCORING_TABLE = {
        "alert":   {"alert": 0.93, "monitor": 0.35, "safe": 0.05},
        "monitor": {"monitor": 0.93, "alert": 0.45, "safe": 0.15},
        "safe":    {"safe": 0.93, "monitor": 0.50, "alert": 0.10},
    }

    def __init__(self):
        self.name = "health_recommendation"
        self.difficulty = "hard"

    def _safe(self, v):
        return round(max(0.03, min(0.97, float(v))), 4)

    def _correct(self, aqi):
        if aqi > 300:   return "alert"
        elif aqi >= 200: return "monitor"
        else:            return "safe"

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])
        scores = []
        for step in steps:
            action = step.get("action", {})
            aqi    = step.get("current_aqi", None)
            if action.get("action_type") == "recommend" and aqi is not None:
                predicted = str(action.get("value", "")).strip().lower()
                correct   = self._correct(float(aqi))
                raw       = self.SCORING_TABLE.get(correct, {}).get(predicted, 0.05)
                scores.append(self._safe(raw))
        if not scores:
            return 0.05
        return self._safe(sum(scores) / len(scores))

if __name__ == "__main__":
    g = Grader3()
    tests = [
        {"steps": []},
        {"steps": [{"action": {"action_type": "recommend", "value": "alert"},   "current_aqi": 350}]},
        {"steps": [{"action": {"action_type": "recommend", "value": "monitor"}, "current_aqi": 250}]},
        {"steps": [{"action": {"action_type": "recommend", "value": "safe"},    "current_aqi": 100}]},
        {"steps": [{"action": {"action_type": "recommend", "value": "safe"},    "current_aqi": 350}]},
        {"steps": [{"action": {"action_type": "recommend", "value": "alert"},   "current_aqi": 100}]},
    ]
    for i, ep in enumerate(tests):
        s = g.grade(ep)
        assert 0 < s < 1, f"FAIL test {i}: {s}"
        print(f"Test {i}: {s} OK")
    print("Grader3 ALL PASSED")