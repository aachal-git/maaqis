from typing import Optional, Dict, Any
from pydantic import BaseModel
import random


# -----------------------------
# 🔹 Observation Model
# -----------------------------
class Observation(BaseModel):
    city: str
    current_aqi: float
    predicted_aqi: Optional[float] = None
    source: Optional[str] = None
    suggestion: Optional[str] = None


# -----------------------------
# 🔹 Action Model
# -----------------------------
class Action(BaseModel):
    action_type: str  # "predict", "classify", "recommend"
    value: Optional[Any] = None


# -----------------------------
# 🔹 Reward Model
# -----------------------------
class Reward(BaseModel):
    score: float


# -----------------------------
# 🔹 Environment
# -----------------------------
class MAAQISEnv:

    def __init__(self):
        self.state_data = {}
        self.step_count = 0
        self.max_steps = 5

    # -----------------------------
    # 🔁 RESET
    # -----------------------------
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

    # -----------------------------
    # ⚙️ STEP
    # -----------------------------
    def step(self, action: Action) -> Dict[str, Any]:
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}

        obs = Observation(
            city=self.state_data["city"],
            current_aqi=self.state_data["current_aqi"]
        )

        # 🔹 Task: AQI Prediction
        if action.action_type == "predict":
            predicted = action.value
            true_val = self.state_data["true_prediction"]

            error = abs(predicted - true_val)
            reward = max(0.0, 1 - (error / 500))

            obs.predicted_aqi = predicted

        # 🔹 Task: Source Classification
        elif action.action_type == "classify":
            if action.value == self.state_data["true_source"]:
                reward = 1.0
            else:
                reward = 0.0

            obs.source = action.value

        # 🔹 Task: Policy Recommendation
        elif action.action_type == "recommend":
            aqi = self.state_data["current_aqi"]

            correct_action = "alert" if aqi > 300 else "monitor"

            if action.value == correct_action:
                reward = 1.0
            else:
                reward = 0.5

            obs.suggestion = action.value

        # 🔹 Done condition
        if self.step_count >= self.max_steps:
            done = True

        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }

    # -----------------------------
    # 📊 STATE
    # -----------------------------
    def state(self) -> Dict[str, Any]:
        return self.state_data