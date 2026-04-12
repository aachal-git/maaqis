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


def safe_score(value: float) -> float:
    """Strictly between 0 and 1 — floor 0.03, ceiling 0.97."""
    return round(max(0.03, min(0.97, float(value))), 4)


class MAAQISEnv:

    def __init__(self, aqi_range=(150, 400)):
        self.state_data = {}
        self.step_count = 0
        self.max_steps  = 3
        self.aqi_range  = aqi_range

    def reset(self) -> Observation:
        self.state_data = {
            "city": "Delhi",
            "current_aqi": random.randint(self.aqi_range[0], self.aqi_range[1]),
            "true_source": random.choice(["traffic", "industry", "dust"]),
            "true_prediction": random.randint(self.aqi_range[0], self.aqi_range[1]),
        }
        self.step_count = 0
        return Observation(
            city=self.state_data["city"],
            current_aqi=self.state_data["current_aqi"]
        )

    def step(self, action: Action) -> Dict[str, Any]:
        self.step_count += 1
        done = False
        info = {}

        obs = Observation(
            city=self.state_data["city"],
            current_aqi=self.state_data["current_aqi"]
        )

        if action.action_type == "predict":
            predicted = float(action.value)
            true_val  = float(self.state_data["true_prediction"])
            error     = abs(predicted - true_val)
            # error=0 → 0.93, error=500 → 0.03  (never hits 0 or 1)
            raw    = 0.93 - (error / 500.0) * 0.90
            reward = safe_score(raw)
            obs.predicted_aqi = predicted

        elif action.action_type == "classify":
            true_source = self.state_data["true_source"]
            partial = {
                ("traffic",  "traffic"):  0.93,
                ("industry", "industry"): 0.93,
                ("dust",     "dust"):     0.93,
                ("traffic",  "industry"): 0.25,
                ("traffic",  "dust"):     0.20,
                ("industry", "traffic"):  0.25,
                ("industry", "dust"):     0.30,
                ("dust",     "traffic"):  0.20,
                ("dust",     "industry"): 0.30,
            }
            key    = (true_source, str(action.value).strip().lower())
            raw    = partial.get(key, 0.05)
            reward = safe_score(raw)
            obs.source = action.value

        elif action.action_type == "recommend":
            aqi = self.state_data["current_aqi"]
            if aqi > 300:
                correct = "alert"
            elif aqi > 200:
                correct = "monitor"
            else:
                correct = "safe"

            scoring = {
                "alert":   {"alert": 0.93, "monitor": 0.35, "safe": 0.05},
                "monitor": {"monitor": 0.93, "alert": 0.45, "safe": 0.15},
                "safe":    {"safe": 0.93, "monitor": 0.50, "alert": 0.10},
            }
            predicted_action = str(action.value).strip().lower()
            raw    = scoring.get(correct, {}).get(predicted_action, 0.05)
            reward = safe_score(raw)
            obs.suggestion = action.value

        else:
            reward = safe_score(0.05)

        if self.step_count >= self.max_steps:
            done = True

        return {
            "observation": obs,
            "reward":      reward,
            "done":        done,
            "info":        info
        }

    def state(self) -> Dict[str, Any]:
        return self.state_data