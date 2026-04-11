# Public Health Recommendation Grader

class Grader3:
    """
    Hard Task: Public Health Recommendation
    Score strictly in (0, 1) — never 0.0 or 1.0.
    """

    SCORING_TABLE = {
        "alert":   {"alert": 0.95, "monitor": 0.35, "safe": 0.05},
        "monitor": {"monitor": 0.95, "alert": 0.45, "safe": 0.15},
        "safe":    {"safe": 0.95, "monitor": 0.50, "alert": 0.10},
    }

    def __init__(self):
        self.name = "health_recommendation"
        self.difficulty = "hard"

    def _safe_score(self, value: float) -> float:
        clamped = max(0.02, min(0.98, float(value)))
        return round(clamped, 4)

    def _get_correct_action(self, aqi: float) -> str:
        if aqi > 300:
            return "alert"
        elif aqi >= 200:
            return "monitor"
        else:
            return "safe"

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])
        scores = []

        for step in steps:
            action = step.get("action", {})
            aqi    = step.get("current_aqi", None)

            if action.get("action_type") == "recommend" and aqi is not None:
                predicted      = str(action.get("value", "")).strip().lower()
                correct_action = self._get_correct_action(float(aqi))
                row            = self.SCORING_TABLE.get(correct_action, {})
                raw            = row.get(predicted, 0.05)
                scores.append(self._safe_score(raw))

        if not scores:
            return 0.05

        avg = sum(scores) / len(scores)
        return self._safe_score(avg)


if __name__ == "__main__":
    grader = Grader3()
    tests = [
        {"steps": [{"action": {"action_type": "recommend", "value": "alert"},   "current_aqi": 350}]},
        {"steps": [{"action": {"action_type": "recommend", "value": "monitor"}, "current_aqi": 250}]},
        {"steps": [{"action": {"action_type": "recommend", "value": "safe"},    "current_aqi": 100}]},
        {"steps": [{"action": {"action_type": "recommend", "value": "safe"},    "current_aqi": 350}]},
        {"steps": [{"action": {"action_type": "predict",   "value": 200},       "current_aqi": 350}]},
    ]
    for i, episode in enumerate(tests):
        score = grader.grade(episode)
        assert 0 < score < 1, f"Test {i}: Score {score} NOT strictly between 0 and 1!"
        print(f"Test {i}: score={score} OK")
    print("Grader3 ALL PASSED")