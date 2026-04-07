import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graders.grader1 import Grader1

class Task1:
    def __init__(self):
        self.id = "aqi_prediction"
        self.name = "AQI Prediction"
        self.difficulty = "easy"
        self.grader = Grader1()

    def evaluate(self, episode: dict) -> float:
        return self.grader.grade(episode)


if __name__ == "__main__":
    task = Task1()
    episode = {
        "steps": [
            {"action": {"action_type": "predict", "value": 280}, "true_prediction": 300},
            {"action": {"action_type": "predict", "value": 350}, "true_prediction": 300},
        ]
    }
    score = task.evaluate(episode)
    print(f"Task1 score: {score}")
    assert 0.01 <= score <= 0.99, f"Score {score} out of range!"
    print("Task1 PASSED")