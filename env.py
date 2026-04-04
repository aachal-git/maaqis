from typing import Optional, Dict, Any
from pydantic import BaseModel
import random


class Observation(BaseModel):
    city: str
    current_aqi: float
    predicted_aqi: Optional[float] = None
    source: Optional[str] = None
    suggestion: Optional[str] = None


class Action(BaseModel):
    action_type: str
    value: Optional[Any] = None


class Reward(BaseModel):
    score: float


class MAAQISEnv:

    def __init__(self):
        self.state_data = {}
        self.step_count = 0
        self.max_steps = 5

    def reset(self) -> Observation:
        self.state_data = {
            "city": "Delhi",
            "current_aqi": random.randint(150, 400),
            "true_source": random.choice(["traffic", "industry", "dust"]),
            "true_prediction": random.randint(150, 400),
        }
        self.step_count = 0

        return Observation(
            city=self.state_data["city"],
            current_aqi=self.state_data["current_aqi"]
        )

    def step(self, action: Action) -> Dict[str, Any]:
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}

        obs = Observation(
            city=self.state_data["city"],
            current_aqi=self.state_data["current_aqi"]
        )

        if action.action_type == "predict":
            predicted = action.value
            true_val = self.state_data["true_prediction"]

            error = abs(predicted - true_val)
            reward = max(0.0, 1.0 - (error / 500.0))

            obs.predicted_aqi = predicted

        elif action.action_type == "classify":
            if action.value == self.state_data["true_source"]:
                reward = 1.0
            else:
                reward = 0.1

            obs.source = action.value

        elif action.action_type == "recommend":
            aqi = self.state_data["current_aqi"]

            if aqi > 300:
                correct_action = "alert"
            elif aqi > 200:
                correct_action = "monitor"
            else:
                correct_action = "safe"

            if action.value == correct_action:
                reward = 1.0
            elif action.value == "monitor" and correct_action == "safe":
                reward = 0.4
            elif action.value == "monitor" and correct_action == "alert":
                reward = 0.2
            else:
                reward = 0.1

            obs.suggestion = action.value

        if self.step_count >= self.max_steps:
            done = True

        return {
            "observation": obs,
            "reward": round(reward, 2),
            "done": done,
            "info": info
        }

    def state(self) -> Dict[str, Any]:
        return self.state_data