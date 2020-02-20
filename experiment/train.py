"""
train recommender system
"""
import numpy as np

from experiment import DataLoader, DataLoaderItem, DataLoaderItemValid, mrr_cal


def train_static_with_static(
        n_users,
        n_items,
        data_file,
        rec_model,
):
    """
    train static recommender model (e.g., ..recommender.factor)
        with static implicit rating data
    :param n_users:
    :param n_items:
    :param data_file:
    :param rec_model:
    :return:
    """
    data_loader = DataLoader(
        data_file=data_file,
        n_items=n_items,
        n_users=n_users
    )
    result = rec_model.train(
        data_generator=data_loader
    )
    return result


def train_seq_w_static(
        n_users,
        n_items,
        data_file,
        rec_model,
        data_spec,
        data_file_valid=None,
        data_spec_valid=None,
):
    """
    train sequence recommender model (e.g., ..recommender.sequence)
        with static implicit rating data
    :param n_users:
    :param n_items:
    :param data_file:
    :param rec_model:
    :param data_spec:
    :return:
    """
    data_loader = DataLoaderItem(
        data_file=data_file,
        n_items=n_items,
        n_users=n_users,
        model_spec=data_spec,
    )
    # move epoch iterations here #
    max_epoch = rec_model.model_spec["max_epoch"]

    if data_file_valid is not None:
        if data_spec_valid is None:
            data_spec_valid = data_spec
        data_loader_valid = DataLoaderItemValid(
            data_file=data_file_valid,
            n_items=n_items,
            n_users=n_users,
            model_spec=data_spec_valid,
        )
    else:
        data_loader_valid = None

    for epoch in range(max_epoch):
        rec_model.model_spec["max_epoch"] = epoch + 1         # single epoch training
        result_train = rec_model.train(
            data_generator=data_loader,
            epoch_start=epoch,
        )
        if data_loader_valid is not None:
            valid_result = valid_seq_w_static(
                data_loader=data_loader_valid,
                rec_model=rec_model,
            )
            print("%03d epoch valid result %s" % (epoch, valid_result))
    # restore max_epoch parameter to rec_model
    rec_model.model_spec["max_epoch"] = max_epoch


def valid_seq_w_static(
        data_loader,
        rec_model,
):
    result = {
        "mrr": []
    }
    while data_loader.i_sequence_effect < data_loader.n_sequences_effect:
        scores = rec_model.predict(
            data_generator=data_loader
        )
        feedback = data_loader.sequence_data[data_loader.i_sequence_effect][2]
        result["mrr"].append(mrr_cal(scores, feedback))
    data_loader.i_sequence_effect = -1         # set to start
    for measure in result:
        result[measure] = np.mean(np.array(result[measure]))
    return result
