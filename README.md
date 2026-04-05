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

# 🌫️ MAAQIS — Multi-Agent Air Quality Intelligence System

A real-world OpenEnv-compatible environment where AI agents monitor urban air quality by predicting AQI values, classifying pollution sources, and recommending public health policy actions.

## 🌍 Motivation

Air quality monitoring is a critical real-world problem affecting billions of people. MAAQIS simulates the decision-making pipeline of an intelligent air quality monitoring system — making it an ideal environment for training and evaluating AI agents on multi-step, multi-task reasoning with real environmental consequences.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Environment info |
| `POST` | `/reset` | Reset environment, get initial observation |
| `POST` | `/step` | Take an action, get reward and next observation |
| `GET` | `/state` | Get current environment state |
| `GET` | `/health` | Health check |

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `city` | str | Name of monitored city |
| `current_aqi` | float | Current AQI value (0–500) |
| `predicted_aqi` | float | Agent-predicted next AQI value |
| `source` | str | Classified pollution source |
| `suggestion` | str | Policy recommendation |

---

## ⚡ Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | str | One of: `predict`, `classify`, `recommend` |
| `value` | any | Depends on action type |

### Action Values
- `predict` → integer AQI forecast (e.g. `285`)
- `classify` → `traffic` / `industry` / `dust`
- `recommend` → `alert` / `monitor` / `safe`

---

## 🏆 Tasks

### Task 1: AQI Prediction (Easy)
- **Goal:** Predict the next AQI value as close to the true value as possible
- **Reward:** `1.0 - (abs_error / 500)`, clamped to `[0.0, 1.0]`
- **Grader:** `graders/grader1.py`

### Task 2: Pollution Source Classification (Medium)
- **Goal:** Identify the dominant pollution source from AQI signal
- **Reward:** `1.0` correct, `0.3` plausible, `0.1` wrong
- **Grader:** `graders/grader2.py`

### Task 3: Policy Recommendation (Hard)
- **Goal:** Recommend correct public health action integrating AQI, trend and source
- **Reward:** `1.0` correct, graduated partial credit for near-misses
- **Grader:** `graders/grader3.py`

---

## 🎁 Reward Function

Rewards are meaningful and non-sparse across the full trajectory:

| Action | Correct | Partial | Wrong |
|--------|---------|---------|-------|
| predict | 1.0 (no error) | proportional to accuracy | 0.0 (500 off) |
| classify | 1.0 | 0.3 (plausible range) | 0.1 |
| recommend | 1.0 | 0.2–0.5 (graduated) | 0.1 |

---

## 🤖 Multi-Agent Pipeline

| Agent | Role |
|-------|------|
| `SatelliteAgent` | Detects AQI trend (increasing/stable/decreasing) |
| `GroundAgent` | Classifies pollution source and risk level |
| `PredictionAgent` | Forecasts next AQI value |
| `PolicyAgent` | Recommends public health action |

---

## 🚀 Setup & Usage

### Local Setup
```bash
git clone https://huggingface.co/spaces/NamanTiwari2603/maaqis
cd maaqis
pip install -r requirements.txt
```

Create `.env` file:
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_hf_token_here

Run the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Run inference:
```bash
python inference.py
```

### Docker
```bash
docker build -t maaqis .
docker run --env-file .env -p 7860:7860 maaqis
```

---

## 📊 Baseline Scores

| Task | Score |
|------|-------|
| AQI Prediction | ~0.91 |
| Source Classification | ~0.10 |
| Policy Recommendation | ~0.40 |
| **Overall** | **~0.44** |

---

## 🌐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | OpenAI-compatible API endpoint | Yes |
| `MODEL_NAME` | Model identifier | Yes |
| `HF_TOKEN` | HuggingFace API token | Yes |

---

## 📁 Project Structure
maaqis/
├── app.py              # FastAPI server
├── env.py              # OpenEnv environment
├── inference.py        # Baseline inference script
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile          # Container config
├── requirements.txt    # Dependencies
├── agents/
│   ├── satellite.py    # AQI trend detection
│   ├── ground.py       # Source classification
│   ├── prediction.py   # AQI forecasting
│   └── policy.py       # Policy recommendation
└── graders/
├── grader1.py      # Easy task grader
├── grader2.py      # Medium task grader
└── grader3.py      # Hard task grader


## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests.

---

## 📝 License

This project is licensed under the MIT License.