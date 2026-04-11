# Pollution Source Classification Grader

class Grader2:
    """
    Medium Task: Pollution Source Classification
    Score strictly in (0, 1) — never 0.0 or 1.0.
    """

    VALID_SOURCES = ["traffic", "industry", "dust"]

    SCORE_MAP = {
        ("traffic",  "traffic"):  0.95,
        ("industry", "industry"): 0.95,
        ("dust",     "dust"):     0.95,
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

    def _safe_score(self, value: float) -> float:
        clamped = max(0.02, min(0.98, float(value)))
        return round(clamped, 4)

    def grade(self, episode: dict) -> float:
        steps = episode.get("steps", [])
        scores = []

        for step in steps:
            action = step.get("action", {})
            true_source = step.get("true_source", None)

            if action.get("action_type") == "classify" and true_source is not None:
                predicted = str(action.get("value", "")).strip().lower()
                true_src  = str(true_source).strip().lower()
                key       = (true_src, predicted)
                raw       = self.SCORE_MAP.get(key, 0.05)
                scores.append(self._safe_score(raw))

        if not scores:
            return 0.05

        avg = sum(scores) / len(scores)
        return self._safe_score(avg)


if __name__ == "__main__":
    grader = Grader2()
    tests = [
        {"steps": [{"action": {"action_type": "classify", "value": "traffic"}, "true_source": "traffic"}]},
        {"steps": [{"action": {"action_type": "classify", "value": "dust"},    "true_source": "industry"}]},
        {"steps": [{"action": {"action_type": "classify", "value": "industry"},"true_source": "industry"}]},
        {"steps": [{"action": {"action_type": "recommend","value": "alert"},   "true_source": "traffic"}]},
    ]
    for i, episode in enumerate(tests):
        score = grader.grade(episode)
        assert 0 < score < 1, f"Test {i}: Score {score} NOT strictly between 0 and 1!"
        print(f"Test {i}: score={score} OK")
    print("Grader2 ALL PASSED")