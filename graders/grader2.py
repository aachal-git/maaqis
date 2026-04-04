#Source Classification Grader
def grade(predicted, actual):
    if predicted == actual:
        return 1.0
    return 0.0