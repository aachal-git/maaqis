# Public Health Recommendation Grader

class Grader3:
    """
    Hard Task: Public Health Recommendation
    Scores the appropriateness of health recommendations based on AQI level.
    Uses continuous scoring — never exactly 0 or 1.
    """

    # action -> aqi_range -> score
    # Correct answer gets 0.95, close gets partial, wrong gets 0.1
    SCORING_TABLE = {
        # AQI > 300 → correct is "alert"
        "alert":   {"alert": 0.95, "monitor": 0.35, "safe": 0.01},
        # AQI 200-300 → correct is "monitor"
        "monitor": {"monitor": 0.95, "alert": 0.45, "safe": 0.01},
        # AQI < 200 → correct is "safe"
        "safe":    {"safe": 0.95, "monitor": 0.5, "alert": 0.01},
    }

    def __init__(self):
        self.name = "health_recommendation"
        self.difficulty = "hard"

    def _get_correct_action(self, aqi: float) -> str:
        if aqi > 300:
            return "alert"
        elif aqi >= 200:
            return "monitor"
        else:
            return "safe"

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])

        recommendation_scores = []

        for step in steps:
            action = step.get("action", {})
            aqi    = step.get("current_aqi", None)

            if action.get("action_type") == "recommend" and aqi is not None:
                predicted      = str(action.get("value", "")).strip().lower()
                correct_action = self._get_correct_action(float(aqi))

                row   = self.SCORING_TABLE.get(correct_action, {})
                score = row.get(predicted, 0.01)  # unknown action → 0.01

                recommendation_scores.append(round(score, 3))

        if not recommendation_scores:
            return 0.01

        return round(sum(recommendation_scores) / len(recommendation_scores), 3)


# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    grader = Grader3()

    episode = {
        "steps": [
            {"action": {"action_type": "recommend", "value": "alert"},   "current_aqi": 350},
            {"action": {"action_type": "recommend", "value": "monitor"}, "current_aqi": 250},
            {"action": {"action_type": "recommend", "value": "safe"},    "current_aqi": 100},
            {"action": {"action_type": "predict",   "value": 200},       "current_aqi": 350},
        ]
    }

    score = grader.grade(episode)
    print(f"Grader3 score: {score}")
    assert 0.01 <= score <= 0.99, "Score out of range!"
    print("Grader3 passed")