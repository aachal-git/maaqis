#AQI Prediction Grader
class Grader1:
    """
    Easy Task: AQI Prediction
    Scores how close the agent's predicted AQI is to the true AQI.
    Reward = 1.0 - (abs_error / 500), clamped to [0.0, 1.0]
    """

    def __init__(self):
        self.name = "aqi_prediction"
        self.difficulty = "easy"

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])

        prediction_scores = []

        for step in steps:
            action = step.get("action", {})
            true_val = step.get("true_prediction", None)

            if action.get("action_type") == "predict" and true_val is not None:
                try:
                    predicted = float(action.get("value", 0))
                    error = abs(predicted - float(true_val))
                    score = max(0.0, 1.0 - (error / 500.0))
                    prediction_scores.append(round(score, 2))
                except (ValueError, TypeError):
                    prediction_scores.append(0.0)

        if not prediction_scores:
            return 0.0

        return round(sum(prediction_scores) / len(prediction_scores), 3)


# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    grader = Grader1()

    episode = {
        "steps": [
            {"action": {"action_type": "predict", "value": 320}, "true_prediction": 300},
            {"action": {"action_type": "predict", "value": 180}, "true_prediction": 200},
            {"action": {"action_type": "recommend", "value": "alert"}, "true_prediction": 300},
        ]
    }

    score = grader.grade(episode)
    print(f"Grader1 score: {score}")
    assert 0.0 <= score <= 1.0, "Score out of range!"
    print("✅ Grader1 passed")