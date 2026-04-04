# agents/ground.py

class GroundAgent:
    def __init__(self):
        self.source_thresholds = {
            "traffic":   (0,   150),
            "industry":  (151, 300),
            "dust":      (301, float("inf")),
        }

    def analyze(self, observation):
        city = observation.get("city", "unknown")
        current_aqi = observation.get("current_aqi", 0)

        source = "unknown"
        for src, (low, high) in self.source_thresholds.items():
            if low <= current_aqi <= high:
                source = src
                break

        if current_aqi > 300:
            risk_level = "hazardous"
        elif current_aqi > 200:
            risk_level = "unhealthy"
        elif current_aqi > 100:
            risk_level = "moderate"
        else:
            risk_level = "good"

        return {
            "city":       city,
            "aqi":        current_aqi,
            "source":     source,
            "risk_level": risk_level,
        }