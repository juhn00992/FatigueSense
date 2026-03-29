import math

def calculate_fatigue(yawn_count):

    # power growth (non-linear)
    fatigue_score = yawn_count ** 1.5

    return fatigue_score


def normalized_fatigue(yawn_count, max_yawns=10):

    raw_score = yawn_count ** 1.5
    max_score = max_yawns ** 1.5

    fatigue_percent = (raw_score / max_score) * 100

    return min(fatigue_percent, 100)