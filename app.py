from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any
from env import MAAQISEnv, Action, Observation

app = FastAPI(
    title="MAAQIS - Multi-Agent Air Quality Intelligence System",
    description="OpenEnv-compatible environment for air quality monitoring agents.",
    version="1.0.0"
)

# -----------------------------
# 🔹 Global Environment Instance
# -----------------------------
env = MAAQISEnv()


# -----------------------------
# 🔹 Request Models
# -----------------------------
class ActionRequest(BaseModel):
    action_type: str
    value: Optional[Any] = None


# -----------------------------
# 🔹 Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "name": "MAAQIS",
        "description": "Multi-Agent Air Quality Intelligence System",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state"]
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


# -----------------------------
# 🔹 Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}