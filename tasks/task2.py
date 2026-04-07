from graders.grader2 import Grader2

class Task2:
    def __init__(self):
        self.id = "source_classification"
        self.name = "Pollution Source Classification"
        self.difficulty = "medium"
        self.grader = Grader2()

    def evaluate(self, episode: dict) -> float:
        return self.grader.grade(episode)