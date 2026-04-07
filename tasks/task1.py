from graders.grader1 import Grader1

class Task1:
    def __init__(self):
        self.id = "aqi_prediction"
        self.name = "AQI Prediction"
        self.difficulty = "easy"
        self.grader = Grader1()

    def evaluate(self, episode: dict) -> float:
        return self.grader.grade(episode)