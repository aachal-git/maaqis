from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import MAAQISEnv, Action
from graders.grader1 import Grader1
from graders.grader2 import Grader2
from graders.grader3 import Grader3

app = FastAPI(
    title="MAAQIS - Multi-Agent Air Quality Intelligence System",
    description="OpenEnv-compatible environment for air quality monitoring agents.",
    version="1.0.0"
)

env = MAAQISEnv()
grader1 = Grader1()
grader2 = Grader2()
grader3 = Grader3()


class ActionRequest(BaseModel):
    action_type: str
    value: Optional[Any] = None


class EpisodeStep(BaseModel):
    action: dict
    true_prediction: Optional[float] = None
    true_source: Optional[str] = None
    current_aqi: Optional[float] = None
    trend: Optional[str] = "stable"


class Episode(BaseModel):
    steps: List[EpisodeStep]


@app.get("/")
def root():
    return {
        "name": "MAAQIS",
        "description": "Multi-Agent Air Quality Intelligence System",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/grade/1", "/grade/2", "/grade/3", "/tasks"]
    }


@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.dict(),
        "done": False,
        "info": {}
    }


@app.post("/step")
def step(action: ActionRequest):
    action_obj = Action(
        action_type=action.action_type,
        value=action.value
    )
    result = env.step(action_obj)
    return {
        "observation": result["observation"].dict(),
        "reward": result["reward"],
        "done": result["done"],
        "info": result["info"]
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# 🏆 TASKS ENDPOINT
# -----------------------------
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": "aqi_prediction",
                "name": "AQI Prediction",
                "difficulty": "easy",
                "grader": "/grade/1"
            },
            {
                "id": "source_classification",
                "name": "Pollution Source Classification",
                "difficulty": "medium",
                "grader": "/grade/2"
            },
            {
                "id": "policy_recommendation",
                "name": "Policy Recommendation",
                "difficulty": "hard",
                "grader": "/grade/3"
            }
        ]
    }


# -----------------------------
# 🏆 GRADER ENDPOINTS
# -----------------------------
@app.post("/grade/1")
def grade1(episode: Episode):
    episode_dict = {"steps": [s.dict() for s in episode.steps]}
    score = grader1.grade(episode_dict)
    return {
        "task": "aqi_prediction",
        "difficulty": "easy",
        "score": score
    }


@app.post("/grade/2")
def grade2(episode: Episode):
    episode_dict = {"steps": [s.dict() for s in episode.steps]}
    score = grader2.grade(episode_dict)
    return {
        "task": "source_classification",
        "difficulty": "medium",
        "score": score
    }


@app.post("/grade/3")
def grade3(episode: Episode):
    episode_dict = {"steps": [s.dict() for s in episode.steps]}
    score = grader3.grade(episode_dict)
    return {
        "task": "policy_recommendation",
        "difficulty": "hard",
        "score": score
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()