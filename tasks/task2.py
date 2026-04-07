import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graders.grader2 import Grader2

class Task2:
    def __init__(self):
        self.id = "source_classification"
        self.name = "Pollution Source Classification"
        self.difficulty = "medium"
        self.grader = Grader2()

    def evaluate(self, episode: dict) -> float:
        return self.grader.grade(episode)


if __name__ == "__main__":
    task = Task2()
    episode = {
        "steps": [
            {"action": {"action_type": "classify", "value": "traffic"}, "true_source": "traffic"},
            {"action": {"action_type": "classify", "value": "dust"},    "true_source": "industry"},
        ]
    }
    score = task.evaluate(episode)
    print(f"Task2 score: {score}")
    assert 0.01 <= score <= 0.99, f"Score {score} out of range!"
    print("Task2 PASSED")