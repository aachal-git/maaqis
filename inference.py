print("🔥 SCRIPT STARTED", flush=True)
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

from openai import OpenAI
from env import MAAQISEnv, Action
print("ENV PATH:", ENV_PATH)
print("HF_TOKEN:", os.getenv("HF_TOKEN"))


# -----------------------------
# 🔑 ENV VARIABLES
# -----------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

if not API_KEY:
    raise ValueError("HF_TOKEN not found. Check your .env file.")

os.environ["OPENAI_API_KEY"] = API_KEY

MAX_STEPS = 5


# -----------------------------
# 🔹 LOG FUNCTIONS (STRICT FORMAT)
# -----------------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# -----------------------------
# 🤖 MODEL CALL
# -----------------------------
def get_action_from_model(client, observation):
    prompt = f"""
    You are an AI agent in an air quality system.

    Current AQI: {observation.current_aqi}

    IMPORTANT:
    - If AQI > 300 → recommend alert
    - If AQI < 200 → recommend monitor
    - Also try prediction and classification in different steps

    Choose different actions across steps.

    Respond strictly:
    predict:<number>
    classify:<traffic/industry/dust>
    recommend:<alert/monitor>
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a decision-making agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=50,
        )

        text = response.choices[0].message.content.strip()

        # Parse response
        if ":" in text:
            action_type, value = text.split(":", 1)
            action_type = action_type.strip()
            value = value.strip()

            # Convert number if needed
            if action_type == "predict":
                value = float(value)

            return Action(action_type=action_type, value=value)

    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)

    # fallback
    return Action(action_type="recommend", value="monitor")


# -----------------------------
# 🚀 MAIN
# -----------------------------
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = MAAQISEnv()

    rewards = []
    steps = 0

    log_start(task="maaqis", env="maaqis_env", model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action_obj = get_action_from_model(client, obs)

            result = env.step(action_obj)

            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]

            rewards.append(reward)
            steps = step

            action_str = f"{action_obj.action_type}:{action_obj.value}"

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        success = sum(rewards) > 1.5  # simple threshold

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps, rewards=rewards)


if __name__ == "__main__":
    main()