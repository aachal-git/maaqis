print("SCRIPT STARTED", flush=True)
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

from openai import OpenAI
from env import MAAQISEnv, Action

from agents.satellite import SatelliteAgent
from agents.ground import GroundAgent
from agents.prediction import PredictionAgent
from agents.policy import PolicyAgent

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN not found. Check your .env file.")

MAX_STEPS = 3

# 3 tasks: easy / medium / hard
TASKS = [
    {"name": "aqi_prediction",       "difficulty": "easy",   "aqi_range": (80,  150)},
    {"name": "source_classification","difficulty": "medium",  "aqi_range": (151, 300)},
    {"name": "policy_recommendation","difficulty": "hard",    "aqi_range": (301, 400)},
]

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_task(client, task_cfg):
    task_name = task_cfg["name"]
    aqi_range = task_cfg["aqi_range"]

    env = MAAQISEnv(aqi_range=aqi_range)

    satellite_agent  = SatelliteAgent()
    ground_agent     = GroundAgent()
    prediction_agent = PredictionAgent()
    policy_agent     = PolicyAgent()

    rewards = []
    steps   = 0
    score   = 0.01
    success = False

    log_start(task=task_name, env="maaqis_env", model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            trend = satellite_agent.analyze({"city": obs.city, "current_aqi": obs.current_aqi})
            severity = ground_agent.analyze({"city": obs.city, "current_aqi": obs.current_aqi})
            pred_out = prediction_agent.analyze({"city": obs.city, "current_aqi": obs.current_aqi, "trend": trend})
            predicted_aqi = pred_out["predicted_aqi"]
            policy_out = policy_agent.analyze({
                "current_aqi": obs.current_aqi,
                "predicted_aqi": predicted_aqi,
                "risk_level": severity["risk_level"],
                "source": severity["source"],
                "trend": trend,
            })
            policy = policy_out["action"]

            if step == 1:
                action_obj = Action(action_type="predict", value=float(predicted_aqi))
            elif step == 2:
                action_obj = Action(action_type="classify", value=severity["source"])
            else:
                action_obj = Action(action_type="recommend", value=policy)

            result = env.step(action_obj)
            obs    = result["observation"]
            reward = result["reward"]
            done   = result["done"]

            rewards.append(reward)
            steps = step

            action_str = f"{action_obj.action_type}:{action_obj.value}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        score   = round(sum(rewards) / len(rewards), 3) if rewards else 0.0
        score   = min(max(score, 0.01), 0.99)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)
        score   = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_cfg in TASKS:
        run_task(client, task_cfg)


if __name__ == "__main__":
    main()