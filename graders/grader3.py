#Policy Recommendation Grader
class Grader3:
    """
    Hard Task: Policy Recommendation
    Scores whether the agent recommends the correct public health action.
    Requires integrating AQI level + trend + source signals.

    Scoring:
      Correct action           → 1.0
      monitor instead of safe  → 0.4  (over-cautious)
      monitor instead of alert → 0.2  (under-responded)
      safe instead of monitor  → 0.3  (slightly under-cautious)
      anything else wrong      → 0.1
    """

    def __init__(self):
        self.name = "policy_recommendation"
        self.difficulty = "hard"

    def _correct_action(self, aqi: float, trend: str) -> str:
        if aqi > 300:
            return "alert"
        elif aqi > 200:
            # If worsening trend, escalate to alert
            if trend == "increasing":
                return "alert"
            return "monitor"
        else:
            # If improving, confirm safe
            if trend == "increasing":
                return "monitor"
            return "safe"

    def _score_recommendation(self, predicted: str, correct: str) -> float:
        if predicted == correct:
            return 1.0

        if predicted == "monitor" and correct == "safe":
            return 0.4
        elif predicted == "monitor" and correct == "alert":
            return 0.2
        elif predicted == "safe" and correct == "monitor":
            return 0.3
        elif predicted == "alert" and correct == "monitor":
            return 0.5  # erring on side of caution is less wrong
        else:
            return 0.1

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])

        recommendation_scores = []

        for step in steps:
            action = step.get("action", {})
            aqi    = step.get("current_aqi", 0)
            trend  = step.get("trend", "stable")

            if action.get("action_type") == "recommend":
                predicted = str(action.get("value", "")).strip().lower()
                correct   = self._correct_action(aqi, trend)
                score     = self._score_recommendation(predicted, correct)
                recommendation_scores.append(round(score, 2))

        if not recommendation_scores:
            return 0.0

        return round(sum(recommendation_scores) / len(recommendation_scores), 3)


# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    grader = Grader3()

    episode = {
        "steps": [
            {"action": {"action_type": "recommend", "value": "alert"},   "current_aqi": 350, "trend": "increasing"},
            {"action": {"action_type": "recommend", "value": "monitor"}, "current_aqi": 250, "trend": "stable"},
            {"action": {"action_type": "recommend", "value": "safe"},    "current_aqi": 120, "trend": "decreasing"},
            {"action": {"action_type": "recommend", "value": "monitor"}, "current_aqi": 120, "trend": "increasing"},
            {"action": {"action_type": "recommend", "value": "alert"},   "current_aqi": 250, "trend": "increasing"},
        ]
    }

    score = grader.grade(episode)
    print(f"Grader3 score: {score}")
    assert 0.0 <= score <= 1.0, "Score out of range!"
    print("✅ Grader3 passed")