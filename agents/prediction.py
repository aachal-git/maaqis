# agents/prediction.py

class PredictionAgent:
    def __init__(self):
        self.history = []
        self.window = 5

    def analyze(self, observation):
        current_aqi = observation.get("current_aqi", 0)
        trend = observation.get("trend", "stable")

        self.history.append(current_aqi)

        if len(self.history) >= self.window:
            window_vals = self.history[-self.window:]
            weighted = (
                window_vals[-1] * 0.4 +
                window_vals[-2] * 0.3 +
                window_vals[-3] * 0.2 +
                window_vals[-4] * 0.05 +
                window_vals[-5] * 0.05
            )
            predicted_aqi = round(weighted)
        else:
            predicted_aqi = current_aqi

        if trend == "increasing":
            predicted_aqi = round(predicted_aqi * 1.08)
        elif trend == "decreasing":
            predicted_aqi = round(predicted_aqi * 0.93)

        predicted_aqi = max(0, min(500, predicted_aqi))

        if predicted_aqi > 300:
            confidence = "low"
        elif predicted_aqi > 150:
            confidence = "medium"
        else:
            confidence = "high"

        return {
            "predicted_aqi": predicted_aqi,
            "confidence":    confidence,
            "trend_applied": trend,
        }