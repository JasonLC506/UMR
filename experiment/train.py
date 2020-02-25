"""
train recommender system
"""
import numpy as np
from tqdm import tqdm
import time

from experiment import DataLoader, DataLoaderItem, DataLoaderItemValid, mrr_cal
from recommender import Models


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
        epoch_start=0
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

    for epoch in range(epoch_start, max_epoch):
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
            msg = "%03d epoch valid result %s" % (epoch, valid_result)
            print(msg)
            with open("../log/" + rec_model.model_name, "a") as logf:
                logf.write(msg + "\n")
    # restore max_epoch parameter to rec_model
    rec_model.model_spec["max_epoch"] = max_epoch


def valid_seq_w_static(
        data_loader,
        rec_model,
):
    result = {
        "mrr": []
    }
    time_eval = time.time()
    time_predict_cul = 0.0
    time_eval_cul = 0.0
    for i_seq in tqdm(range(data_loader.n_sequences_effect)):
        scores, _ = rec_model.predict(
            data_generator=data_loader
        )
        if "n_neg_sample" in data_loader.model_spec:
            feedback = 0   # see DataLoaderItemValid.candidate_items
        else:
            feedback = data_loader.sequence_data[data_loader.i_sequence_effect][2]
        time_predict = time.time()
        time_predict_cul += (time_predict - time_eval)
        result["mrr"].append(mrr_cal(scores, feedback))
        time_eval = time.time()
        time_eval_cul += (time_eval - time_predict)
    print("time_predict: %f" % time_predict_cul)
    print("time_eval: %f" % time_eval_cul)
    data_loader.i_sequence_effect = -1         # set to start
    for measure in result:
        result[measure] = np.mean(np.array(result[measure]))
    return result


if __name__ == "__main__":
    n_items = 10000
    n_users_train = 10000
    n_users_valid = 1000
    emb_dim = 25
    rec_model = Models["sequence"](
        n_items=n_items,
        model_spec={
            "learning_rate": 0.01,
            "batch_size": 512,
            "max_epoch": 50,
            "emb_dim": emb_dim,
            "max_hist_length": 50,
        },
        model_name="avg_item_emb"
    )
    # rec_model.initialization(
    #     item_emb_data=np.random.normal(
    #         size=[n_items, emb_dim],
    #         scale=0.04
    #     )
    # )
    # rec_model.restore(
    #     save_path="../ckpt/" + rec_model.model_name + "/epoch_009"
    # )
    rec_model.initialization(
        item_emb_file="../data/emb/item_10k_25"
    )
    valid_result = valid_seq_w_static(
        data_loader=DataLoaderItemValid(
            data_file="../data/initial_data_random_u1000_v10000_e25_k20_n50_s2022",
            n_items=n_items,
            n_users=n_users_valid,
            model_spec={
                "min_length_sample": 10,
                "max_length_sample": 49,
                "max_length_input": 50,
                "n_seq_per_user": 1,
                "n_neg_sample": 99,
            },
        ),
        rec_model=rec_model
    )
    with open("../log/KNN", "a") as logf:
        logf.write(str(valid_result) + "\n")
    # train_seq_w_static(
    #     n_users=n_users_train,
    #     n_items=n_items,
    #     data_file="../data/initial_data_random_u10000_v10000_e25_k20_n50_s2021",
    #     rec_model=rec_model,
    #     data_spec={
    #         "min_length_sample": 10,
    #         "max_length_sample": 49,
    #         "max_length_input": 50,
    #         "n_seq_per_user": 1
    #     },
    #     data_file_valid="../data/initial_data_random_u1000_v10000_e25_k20_n50_s2022",
    #     data_spec_valid={
    #         "min_length_sample": 10,
    #         "max_length_sample": 49,
    #         "max_length_input": 50,
    #         "n_seq_per_user": 1,
    #         "n_neg_sample": 99,
    #     },
    #     epoch_start=9,
    # )
