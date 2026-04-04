# agents/policy.py

class PolicyAgent:
    def __init__(self):
        self.action_log = []

    def analyze(self, observation):
        current_aqi   = observation.get("current_aqi", 0)
        predicted_aqi = observation.get("predicted_aqi", current_aqi)
        risk_level    = observation.get("risk_level", "good")
        source        = observation.get("source", "unknown")
        trend         = observation.get("trend", "stable")

        action     = self._decide_action(current_aqi, predicted_aqi, trend)
        advice     = self._generate_advice(source, risk_level)
        reward     = self._compute_reward(current_aqi, predicted_aqi, action)

        self.action_log.append(action)

        return {
            "action":        action,
            "advice":        advice,
            "reward":        reward,
            "risk_level":    risk_level,
            "source":        source,
        }

    def _decide_action(self, current_aqi, predicted_aqi, trend):
        if current_aqi > 300 or predicted_aqi > 300:
            return "alert"
        elif current_aqi > 200 or (trend == "increasing" and predicted_aqi > 200):
            return "monitor"
        elif trend == "decreasing" and current_aqi < 100:
            return "safe"
        else:
            return "monitor"

    def _generate_advice(self, source, risk_level):
        advice_map = {
            ("traffic",  "hazardous"): "Enforce emergency traffic restrictions immediately.",
            ("traffic",  "unhealthy"): "Limit vehicle usage and activate low-emission zones.",
            ("traffic",  "moderate"):  "Monitor traffic density and encourage public transport.",
            ("traffic",  "good"):      "Normal traffic conditions. Continue monitoring.",
            ("industry", "hazardous"): "Shut down non-essential industrial operations.",
            ("industry", "unhealthy"): "Reduce industrial output and inspect emission controls.",
            ("industry", "moderate"):  "Review industrial activity and emissions logs.",
            ("industry", "good"):      "Industrial emissions within acceptable range.",
            ("dust",     "hazardous"): "Issue public health emergency. Restrict outdoor activity.",
            ("dust",     "unhealthy"): "Advise vulnerable groups to stay indoors.",
            ("dust",     "moderate"):  "Monitor dust levels. Avoid prolonged outdoor exposure.",
            ("dust",     "good"):      "Dust levels normal. No action required.",
        }
        return advice_map.get(
            (source, risk_level),
            "Insufficient data. Continue monitoring all sensors."
        )

    def _compute_reward(self, current_aqi, predicted_aqi, action):
        if current_aqi > 300 and action == "alert":
            reward = 1.0
        elif 200 < current_aqi <= 300 and action == "monitor":
            reward = 0.8
        elif current_aqi <= 200 and action == "safe":
            reward = 0.9
        elif current_aqi <= 200 and action == "monitor":
            reward = 0.6
        elif current_aqi > 300 and action != "alert":
            reward = 0.1
        else:
            reward = 0.4

        gap = abs(predicted_aqi - current_aqi)
        if gap < 20:
            reward = min(1.0, reward + 0.05)
        elif gap > 100:
            reward = max(0.0, reward - 0.1)

        return round(reward, 2)