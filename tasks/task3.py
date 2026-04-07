import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graders.grader3 import Grader3

class Task3:
    def __init__(self):
        self.id = "policy_recommendation"
        self.name = "Policy Recommendation"
        self.difficulty = "hard"
        self.grader = Grader3()

    def evaluate(self, episode: dict) -> float:
        return self.grader.grade(episode)


if __name__ == "__main__":
    task = Task3()
    episode = {
        "steps": [
            {"action": {"action_type": "recommend", "value": "alert"},   "current_aqi": 350},
            {"action": {"action_type": "recommend", "value": "monitor"}, "current_aqi": 250},
            {"action": {"action_type": "recommend", "value": "safe"},    "current_aqi": 100},
        ]
    }
    score = task.evaluate(episode)
    print(f"Task3 score: {score}")
    assert 0.01 <= score <= 0.99, f"Score {score} out of range!"
    print("Task3 PASSED")