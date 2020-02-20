"""
train recommender model,
interact with user model
"""
import numpy as np
import pickle

from data_synthesizer import uv_emb_synthesize
from recommender import RecommenderSeq
from user import User
from experiment import (
    Interactor,
    train_seq_w_static
)


if __name__ == "__main__":
    n_users = 100
    n_items = 1000
    emb_dim = 20
    random_seed = 2020
    data_file = "../data/synthetic_knn"
    np.random.seed(random_seed)

    # --------------------- initial data generation ---------------------- #
    _, _ = uv_emb_synthesize(
        n_users=n_users,
        n_items=n_items,
        model_spec={
            "dim": emb_dim,
            "user": {
                "common_init_alpha": 1.0,
                "rescale_common": 10.0
            },
            "item": {
                "common_init_alpha": 100,
                "rescale_common": 0.1
            }
        },
        save_path="../data/emb/",
    )
    print("done synthesize embedding")

    recommender = RecommenderSeq(
        model="sequence",
        n_users=n_users,
        n_items=n_items,
        model_spec={
            "learning_rate": 0.01,
            "batch_size": 512,
            "max_epoch": 3,
            "emb_dim": 20,
            "max_hist_length": 100,
        },
        model_name="factor_rec"
    )
    recommender.model.initialization(
        item_emb_data=np.random.random(size=[n_items, emb_dim]),
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
            "emb_dim": 20,
        },
        model_name="base_user"
    )
    user.model.initialization(
        item_emb_file="../data/emb/item",
        user_emb_file="../data/emb/user"
    )
    print("done initialize user")

    interactor = Interactor(
        user=user,
        recommender=recommender,
        model_spec={
            "data_file": data_file
        }
    )

    _, candidates_steps_init, feedback_steps_init, scores_steps_init = interactor.iterate(
        steps=1,
        k=10,
        random_rec=True,
        max_length_input=100
    )
    # --------------------- interact --------------------------- #

    # item-KNN setting (item embedding using ground truth)#
    interactor.recommender.model.initialization(
        item_emb_file="../data/emb/item",
    )
    _, candidates_steps, feedback_steps, scores_steps = interactor.iterate(
        steps=100,
        k=10,
        random_rec=False,
        max_length_input=100
    )
    with open(data_file + "_detail", "wb") as f:
        pickle.dump(
            [
                candidates_steps_init + candidates_steps,
                feedback_steps_init + feedback_steps,
                scores_steps_init + scores_steps,
            ],
            f
        )
