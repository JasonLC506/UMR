"""
train recommender model,
interact with user model
"""
import numpy as np
import pickle
import copy
import time

from data_synthesizer import emb_synthesize
from recommender import RecommenderSeq
from user import User
from experiment import (
    Interactor,
    train_seq_w_static
)


if __name__ == "__main__":
    n_users = 1000
    n_items = 10000
    emb_dim = 25
    random_seed = 2022
    k = 20
    n_steps = 50
    data_file = "../data/initial_data_random_u%d_v%d_e%d_k%d_n%d_s%d" % (
        n_users,
        n_items,
        emb_dim,
        k,
        n_steps,
        random_seed
    )
    item_emb_path = "../data/emb/item_10k_25"
    user_emb_path = "../data/emb/user_temp10k_25"
    np.random.seed(random_seed)

    # --------------------- initial data generation ---------------------- #
    user_embs = emb_synthesize(
        n=n_users,
        model_spec={
            "dim": emb_dim,
            "common_init_alpha": 1.0,
            "rescale_common": 10.0
        },
        save_path=user_emb_path
    )
    # item_embs = emb_synthesize(
    #     n=n_items,
    #     model_spec={
    #         "dim": emb_dim,
    #         "common_init_alpha": 100.0,
    #         "rescale_common": 0.1,
    #     },
    #     save_path=item_emb_path
    # )
    # print(item_embs.shape)
    print("done synthesize embedding")

    recommender = RecommenderSeq(
        model="sequence",
        n_users=n_users,
        n_items=n_items,
        model_spec={
            "learning_rate": 0.01,
            "batch_size": 512,
            "max_epoch": 3,
            "emb_dim": 1,
            "max_hist_length": n_steps,
        },
        model_name="sequence_rec"
    )

    print("done initialize recommender")

    user = User(
        model="base",
        n_users=n_users,
        n_items=n_items,
        model_spec={
            "learning_rate": 0.01,
            "batch_size": 512,
            "max_epoch": 3,
            "emb_dim": emb_dim,
        },
        model_name="base_user"
    )
    user.model.initialization(
        item_emb_file=item_emb_path,
        user_emb_file=user_emb_path,
    )
    print("done initialize user")

    interactor = Interactor(
        user=user,
        recommender=recommender,
        model_spec={
            "data_file": data_file
        }
    )

    # --------------------- interact --------------------------- #

    time_start = time.time()
    for i_step in range(n_steps):
        interactor.iterate(
            steps=1,
            k=k,
            random_rec=True,
            max_length_input=n_steps
        )
        time_step = time.time()
        print("%d steps takes %f s" % (i_step + 1, time_step - time_start))


