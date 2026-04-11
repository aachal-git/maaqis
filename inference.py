print("SCRIPT STARTED", flush=True)
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

from openai import OpenAI
from env import MAAQISEnv, Action
from agents.satellite import SatelliteAgent
from agents.ground import GroundAgent
from agents.prediction import PredictionAgent
from agents.policy import PolicyAgent

# ✅ Per hackathon rules: HF_TOKEN required, others have defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

MAX_STEPS = 3

TASKS = [
    {"name": "aqi_prediction",        "difficulty": "easy",   "aqi_range": (80,  150)},
    {"name": "source_classification", "difficulty": "medium", "aqi_range": (151, 300)},
    {"name": "policy_recommendation", "difficulty": "hard",   "aqi_range": (301, 400)},
]


# -----------------------------
# LOG FUNCTIONS
# -----------------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# -----------------------------
# LLM CALL
# -----------------------------
def get_action_from_model(client, observation):
    prompt = f"""You are an AI agent in an air quality monitoring system.
Current AQI: {observation.current_aqi}

Respond with exactly one line, no explanation:
predict:<integer>   OR   classify:<traffic/industry/dust>   OR   recommend:<alert/monitor/safe>

Rules:
- predict: forecast next AQI as integer
- classify: identify pollution source (traffic/industry/dust)
- recommend: AQI>300 → alert, AQI 200-300 → monitor, AQI<200 → safe
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a decision-making agent. Respond with exactly one line."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=50,
        )
        text = response.choices[0].message.content.strip()
        if ":" in text:
            action_type, value = text.split(":", 1)
            action_type = action_type.strip().lower()
            value = value.strip()
            if action_type == "predict":
                value = float(value)
            return Action(action_type=action_type, value=value)
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
    return Action(action_type="recommend", value="monitor")


# -----------------------------
# RUN ONE TASK
# -----------------------------
def run_task(client, task_cfg):
    task_name = task_cfg["name"]
    aqi_range = task_cfg["aqi_range"]

    env              = MAAQISEnv(aqi_range=aqi_range)
    satellite_agent  = SatelliteAgent()
    ground_agent     = GroundAgent()
    prediction_agent = PredictionAgent()
    policy_agent     = PolicyAgent()

    rewards = []
    steps   = 0
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
                "current_aqi": obs.current_aqi, "predicted_aqi": predicted_aqi,
                "risk_level": severity["risk_level"], "source": severity["source"], "trend": trend,
            })
            policy = policy_out["action"]

            model_action = get_action_from_model(client, obs)

            if step == 1:
                action_obj = model_action if model_action.action_type == "predict" else Action(action_type="predict", value=float(predicted_aqi))
            elif step == 2:
                action_obj = model_action if model_action.action_type == "classify" else Action(action_type="classify", value=severity["source"])
            else:
                action_obj = model_action if model_action.action_type == "recommend" else Action(action_type="recommend", value=policy)

            result = env.step(action_obj)
            obs    = result["observation"]
            reward = result["reward"]
            done   = result["done"]

            rewards.append(reward)
            steps = step

            log_step(step=step, action=f"{action_obj.action_type}:{action_obj.value}", reward=reward, done=done, error=None)

            if done:
                break

        score   = round(sum(rewards) / len(rewards), 3) if rewards else 0.05
        score   = max(0.02, min(0.98, score))
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps, rewards=rewards)


# -----------------------------
# MAIN
# -----------------------------
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_cfg in TASKS:
        run_task(client, task_cfg)

if __name__ == "__main__":
    main()