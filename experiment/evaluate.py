import numpy as np


def mrr_cal(
        scores,
        feedback,
):
    """
    :param scores: scores for items (higher ranks higher)
    :param feedback: the item chosen
    :return: reciprocal ranking score
    """
    rank = np.sum(scores > scores[feedback]).astype(np.float32)
    return 1.0 / (1.0 + rank)
