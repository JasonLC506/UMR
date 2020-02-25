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


def get_optim_score_step(
        optim_scores,
        recommender,
        k
):
    optim_scores_step = []
    for uid in range(len(optim_scores)):
        inds = recommender.exclude_sort(
            scores=copy.deepcopy(optim_scores[uid]),
            excluded_indices=recommender.user_feedbacks[uid],
            k=k
        )
        optim_scores_step.append(optim_scores[uid][inds])
    return optim_scores_step


if __name__ == "__main__":
    n_users = 100
    n_items = 10000
    emb_dim = 25
    random_seed = 2020
    k = 10
    data_file = "../data/synthetic_KNN_random_10Kitem_25_1000steps"
    item_emb_path = "../data/emb/item_10k_25"
    user_emb_path = "../data/emb/user_new_25"
    np.random.seed(random_seed)

    # --------------------- initial data generation ---------------------- #
    # user_embs = emb_synthesize(
    #     n=n_users,
    #     model_spec={
    #         "dim": emb_dim,
    #         "common_init_alpha": 1.0,
    #         "rescale_common": 10.0
    #     },
    #     save_path=user_emb_path
    # )
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
            "emb_dim": emb_dim,
            "max_hist_length": 100,
        },
        model_name="sequence_rec"
    )
    recommender.model.initialization(
        item_emb_data=np.random.random(size=[n_items, emb_dim]),
    )
    # recommender.model.restore(
    #     save_path="../ckpt/avg_item_emb/epoch_015"
    # )
    recommender.model.initialization(
        item_emb_file=item_emb_path
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

    optim_scores = user.rate(
        [
            np.arange(n_items) for _ in range(n_users)
        ]
    )
    print("done calculating full optim scores")

    interactor = Interactor(
        user=user,
        recommender=recommender,
        model_spec={
            "data_file": data_file
        }
    )

    optim_scores_steps = [
        get_optim_score_step(
            optim_scores=optim_scores,
            recommender=interactor.recommender,
            k=k,
        )
    ]
    _, candidates_steps, feedback_steps, scores_steps, rec_u_embs_steps = interactor.iterate(
        steps=1,
        k=k,
        random_rec=True,
        random_react=True,
        max_length_input=100
    )
    # --------------------- interact --------------------------- #

    time_start = time.time()
    for i_step in range(999):
        optim_scores_steps.append(
            get_optim_score_step(
                optim_scores=optim_scores,
                recommender=interactor.recommender,
                k=k,
            )
        )
        _, candidates_step, feedback_step, scores_step, rec_u_embs_step = interactor.iterate(
            steps=1,
            k=k,
            random_rec=False,
            random_react=True,
            max_length_input=100
        )
        candidates_steps += candidates_step
        feedback_steps += feedback_step
        scores_steps += scores_step
        rec_u_embs_steps += rec_u_embs_step

        time_step = time.time()
        print("%d steps takes %f s" % (i_step + 1, time_step - time_start))

    # --------------------- save results ------------------------- #

    with open(data_file + "_detail", "wb") as f:
        pickle.dump(
            [
                candidates_steps,
                feedback_steps,
                scores_steps,
                optim_scores_steps,
                rec_u_embs_steps
            ],
            f
        )
