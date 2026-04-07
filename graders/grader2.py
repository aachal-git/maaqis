# Pollution Source Classification Grader

class Grader2:
    """
    Medium Task: Pollution Source Classification
    Scores how accurately the agent classifies the pollution source.
    Uses continuous scoring — never exactly 0 or 1.
    """

    VALID_SOURCES = ["traffic", "industry", "dust"]

    # Partial credit matrix: how related wrong answers are
    PARTIAL_CREDIT = {
        ("traffic", "industry"): 0.2,
        ("traffic", "dust"):     0.15,
        ("industry", "traffic"): 0.2,
        ("industry", "dust"):    0.25,
        ("dust", "traffic"):     0.15,
        ("dust", "industry"):    0.25,
    }

    def __init__(self):
        self.name = "source_classification"
        self.difficulty = "medium"

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])

        classification_scores = []

        for step in steps:
            action = step.get("action", {})
            true_source = step.get("true_source", None)

            if action.get("action_type") == "classify" and true_source is not None:
                predicted = str(action.get("value", "")).strip().lower()
                true_src  = str(true_source).strip().lower()

                if predicted == true_src:
                    score = 0.95  # correct but not exactly 1.0
                elif (predicted, true_src) in self.PARTIAL_CREDIT:
                    score = self.PARTIAL_CREDIT[(predicted, true_src)]
                elif predicted not in self.VALID_SOURCES:
                    score = 0.05  # invalid answer, not exactly 0
                else:
                    score = 0.1   # wrong but valid answer

                classification_scores.append(round(score, 3))

        if not classification_scores:
            return 0.1

        return round(sum(classification_scores) / len(classification_scores), 3)


# -----------------------------
# Standalone test
# -----------------------------
if __name__ == "__main__":
    grader = Grader2()

    episode = {
        "steps": [
            {"action": {"action_type": "classify", "value": "traffic"}, "true_source": "traffic"},
            {"action": {"action_type": "classify", "value": "dust"},    "true_source": "industry"},
            {"action": {"action_type": "predict",  "value": 200},       "true_source": "traffic"},
        ]
    }

    score = grader.grade(episode)
    print(f"Grader2 score: {score}")
    assert 0.0 <= score <= 1.0, "Score out of range!"
    print("Grader2 passed")