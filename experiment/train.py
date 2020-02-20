"""
train recommender system
"""


def train_seq_w_static(
        n_items,
        data_file,
        rec_model,
):
    """
    train sequence recommender model (e.g., ../recommender.sequence)
        with static implicit rating data
    :param n_items:
    :param data_file:
    :param rec_model:
    :return:
    """
