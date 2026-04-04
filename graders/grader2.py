#Source Classification Grader
class Grader2:
    """
    Medium Task: Pollution Source Classification
    Scores whether the agent correctly identifies the pollution source.
    Correct → 1.0
    Wrong but adjacent range → 0.3
    Completely wrong → 0.1
    """

    def __init__(self):
        self.name = "source_classification"
        self.difficulty = "medium"

        # AQI range hints per source for partial credit logic
        self.source_ranges = {
            "traffic":  (0,   150),
            "industry": (151, 300),
            "dust":     (301, 500),
        }

    def _partial_credit(self, predicted_source: str, true_source: str, aqi: float) -> float:
        if predicted_source == true_source:
            return 1.0

        # Check if predicted source range overlaps with AQI value
        low, high = self.source_ranges.get(predicted_source, (0, 0))
        if low <= aqi <= high:
            return 0.3  # wrong label but plausible for this AQI

        return 0.1  # completely wrong

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])

        classification_scores = []

        for step in steps:
            action      = step.get("action", {})
            true_source = step.get("true_source", None)
            aqi         = step.get("current_aqi", 0)

            if action.get("action_type") == "classify" and true_source is not None:
                predicted_source = str(action.get("value", "")).strip().lower()
                score = self._partial_credit(predicted_source, true_source, aqi)
                classification_scores.append(round(score, 2))

        if not classification_scores:
            return 0.0

        return round(sum(classification_scores) / len(classification_scores), 3)


# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    grader = Grader2()

    episode = {
        "steps": [
            {"action": {"action_type": "classify", "value": "traffic"},  "true_source": "traffic",  "current_aqi": 120},
            {"action": {"action_type": "classify", "value": "industry"}, "true_source": "dust",     "current_aqi": 350},
            {"action": {"action_type": "classify", "value": "dust"},     "true_source": "industry", "current_aqi": 200},
        ]
    }

    score = grader.grade(episode)
    print(f"Grader2 score: {score}")
    assert 0.0 <= score <= 1.0, "Score out of range!"
    print("✅ Grader2 passed")