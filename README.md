---
title: MAAQIS
emoji: 🌫️
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
---

# MAAQIS — Multi-Agent Air Quality Intelligence System

> **Team:** VOIDERR &nbsp;|&nbsp; **Project:** MAAQIS — Multi-Agent Air Quality Intelligence System

A real-world OpenEnv-compatible environment where AI agents monitor urban air quality by predicting AQI values, classifying pollution sources, and recommending public health policy actions.

## Motivation

Air quality monitoring is a critical real-world problem affecting billions of people. MAAQIS simulates the decision-making pipeline of an intelligent air quality monitoring system — making it an ideal environment for training and evaluating AI agents on multi-step, multi-task reasoning with real environmental consequences.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Environment info |
| `POST` | `/reset` | Reset environment, get initial observation |
| `POST` | `/step` | Take an action, get reward and next observation |
| `GET` | `/state` | Get current environment state |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks with graders |
| `POST` | `/grade/1` | Grade AQI prediction episode |
| `POST` | `/grade/2` | Grade source classification episode |
| `POST` | `/grade/3` | Grade policy recommendation episode |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `city` | str | Name of monitored city |
| `current_aqi` | float | Current AQI value (0–500) |
| `predicted_aqi` | float | Agent-predicted next AQI value |
| `source` | str | Classified pollution source |
| `suggestion` | str | Policy recommendation |

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | str | One of: `predict`, `classify`, `recommend` |
| `value` | any | Depends on action type |

### Action Values
- `predict` → integer AQI forecast (e.g. `285`)
- `classify` → `traffic` / `industry` / `dust`
- `recommend` → `alert` / `monitor` / `safe`

---

## Tasks

### Task 1: AQI Prediction (Easy)
- **Goal:** Predict the next AQI value as close to the true value as possible
- **Reward:** `1.0 - (abs_error / 500)`, clamped to `[0.1, 0.99]`
- **Grader:** `graders/grader1.py`
- **AQI Range:** 80–150

### Task 2: Pollution Source Classification (Medium)
- **Goal:** Identify the dominant pollution source from AQI signal
- **Reward:** `0.99` correct, `0.3` plausible, `0.1` wrong
- **Grader:** `graders/grader2.py`
- **AQI Range:** 151–300

### Task 3: Policy Recommendation (Hard)
- **Goal:** Recommend correct public health action integrating AQI, trend and source
- **Reward:** `0.99` correct, graduated partial credit for near-misses, min `0.1`
- **Grader:** `graders/grader3.py`
- **AQI Range:** 301–400

---

## Reward Function

All rewards strictly between `[0.1, 0.99]` — no binary sparse signals:

| Action | Correct | Partial | Wrong |
|--------|---------|---------|-------|
| predict | 0.99 (no error) | proportional to accuracy | 0.1 (500 off) |
| classify | 0.99 | 0.3 (plausible range) | 0.1 |
| recommend | 0.99 | 0.2–0.5 (graduated) | 0.1 |

---

## Multi-Agent Pipeline

| Agent | Role |
|-------|------|
| `SatelliteAgent` | Detects AQI trend (increasing/stable/decreasing) |
| `GroundAgent` | Classifies pollution source and risk level |
| `PredictionAgent` | Forecasts next AQI value |
| `PolicyAgent` | Recommends public health action |

Each step calls the **LLM via proxy** first, then falls back to agent logic if response is invalid.

---

## Setup & Usage

### Local Setup
```
git clone https://github.com/aachal-git/maaqis
cd maaqis
pip install -r requirements.txt
```

Create `.env` file:
```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
API_KEY=your_api_key_here
HF_TOKEN=your_hf_token_here
```

Run the server:
```
uvicorn app:app --host 0.0.0.0 --port 7860
```

Run inference:
```
python inference.py
```

### Docker
```
docker build -t maaqis .
docker run --env-file .env -p 7860:7860 maaqis
```

---

## Baseline Scores

| Task | Difficulty | Score |
|------|------------|-------|
| AQI Prediction | Easy | ~0.91 |
| Source Classification | Medium | ~0.10 |
| Policy Recommendation | Hard | ~0.40 |
| **Overall** | — | **~0.44** |

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | OpenAI-compatible API endpoint | Yes |
| `MODEL_NAME` | Model identifier | Yes |
| `API_KEY` | LiteLLM proxy API key | Yes |
| `HF_TOKEN` | HuggingFace API token (fallback) | Yes |

---

## Project Structure

```
maaqis/
|-- app.py                  # FastAPI server (root)
|-- env.py                  # OpenEnv environment
|-- inference.py            # Baseline inference script
|-- openenv.yaml            # OpenEnv metadata
|-- Dockerfile              # Container config
|-- requirements.txt        # Dependencies
|-- pyproject.toml          # Project metadata
|-- uv.lock                 # Dependency lock file
|-- server/
|   |-- app.py              # FastAPI server (openenv entry)
|-- agents/
|   |-- __init__.py
|   |-- satellite.py        # AQI trend detection
|   |-- ground.py           # Source classification
|   |-- prediction.py       # AQI forecasting
|   |-- policy.py           # Policy recommendation
|-- graders/
|   |-- __init__.py
|   |-- grader1.py          # Easy task grader
|   |-- grader2.py          # Medium task grader
|   |-- grader3.py          # Hard task grader
|-- tasks/
    |-- __init__.py
    |-- task1.py            # AQI prediction task
    |-- task2.py            # Source classification task
    |-- task3.py            # Policy recommendation task
```

---

## Contributing

Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests.

---

## License

MIT