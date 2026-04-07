from graders.grader3 import Grader3

class Task3:
    def __init__(self):
        self.id = "policy_recommendation"
        self.name = "Policy Recommendation"
        self.difficulty = "hard"
        self.grader = Grader3()

    def evaluate(self, episode: dict) -> float:
        return self.grader.grade(episode)