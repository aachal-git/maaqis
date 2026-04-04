#AQI Prediction Grader
def grade(predicted, actual):
    error = abs(predicted - actual)
    score = max(0.0, 1 - (error / 500))
    return round(score, 2)