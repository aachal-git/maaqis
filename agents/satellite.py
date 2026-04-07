class SatelliteAgent:
    def __init__(self):
        self.history = []

    def analyze(self, observation):
        city = observation.get("city", "unknown")
        current_aqi = observation.get("current_aqi", 0)

        self.history.append(current_aqi)

        if len(self.history) < 2:
            return "stable"

        recent = self.history[-3:] if len(self.history) >= 3 else self.history

        avg_prev = sum(recent[:-1]) / len(recent[:-1])
        latest = recent[-1]

        delta = latest - avg_prev

        if delta > 10:
            return "increasing"
        elif delta < -10:
            return "decreasing"
        else:
            return "stable"