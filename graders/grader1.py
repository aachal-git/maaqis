# AQI Prediction Grader

class Grader1:
    """
    Easy Task: AQI Prediction
    Score strictly in (0, 1) — never 0.0 or 1.0.
    """

    def __init__(self):
        self.name = "aqi_prediction"
        self.difficulty = "easy"

    def _safe_score(self, value: float) -> float:
        """Guarantee score is strictly between 0 and 1."""
        clamped = max(0.02, min(0.98, float(value)))
        return round(clamped, 4)

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])
        prediction_scores = []

        for step in steps:
            action = step.get("action", {})
            true_val = step.get("true_prediction", None)

            if action.get("action_type") == "predict" and true_val is not None:
                try:
                    predicted = float(action.get("value", 0))
                    true_val  = float(true_val)
                    error     = abs(predicted - true_val)

                    # Map error to score: error=0 → 0.97, error=500 → 0.03
                    raw = 0.97 - (error / 500.0) * 0.94
                    prediction_scores.append(self._safe_score(raw))
                except (ValueError, TypeError):
                    prediction_scores.append(0.05)

        if not prediction_scores:
            return 0.05

        avg = sum(prediction_scores) / len(prediction_scores)
        return self._safe_score(avg)


if __name__ == "__main__":
    grader = Grader1()
    tests = [
        {"steps": [{"action": {"action_type": "predict", "value": 300}, "true_prediction": 300}]},
        {"steps": [{"action": {"action_type": "predict", "value": 0},   "true_prediction": 500}]},
        {"steps": [{"action": {"action_type": "recommend", "value": "alert"}, "true_prediction": 300}]},
        {"steps": [
            {"action": {"action_type": "predict", "value": 280}, "true_prediction": 300},
            {"action": {"action_type": "predict", "value": 350}, "true_prediction": 300},
        ]},
    ]
    for i, episode in enumerate(tests):
        score = grader.grade(episode)
        assert 0 < score < 1, f"Test {i}: Score {score} NOT strictly between 0 and 1!"
        print(f"Test {i}: score={score} OK")
    print("Grader1 ALL PASSED")