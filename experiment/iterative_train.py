"""
iterative training
@article{Chaney2018,
author = {Chaney, Allison J. B. and Stewart, Brandon M. and Engelhardt, Barbara E.},
pages = {224--232},
title = {{How algorithmic confounding in recommendation systems increases homogeneity and decreases utility}},
year = {2018}
}
"""
import numpy as np
from data_synthesizer import generate_emb, save_emb
from recommender import Recommender
from user import User
from experiment import Interactor, DataLoader


# --------------------- initial data generation ---------------------- #
# synthetic
def uv_emb_synthesize(
        n_users,
        n_items,
        model_spec,
        save_path="../data/emb/",
):
    """
    generate user/item embeddings
    """
    user_embs = generate_emb(
        n=n_users,
        dim=model_spec["dim"],
        model_spec=model_spec["user"]
    )
    item_embs = generate_emb(
        n=n_items,
        dim=model_spec["dim"],
        model_spec=model_spec["item"]
    )
    if save_path is not None:
        save_emb(user_embs, save_path + "user")
        save_emb(item_embs, save_path + "item")
    return user_embs, item_embs


# --------------------- interact --------------------------- #
# Interactor().iterate()


# --------------------- train ------------------------- #
def train(
        n_users,
        n_items,
        data_file,
        rec_model,
):
    data_loader = DataLoader(
        data_file=data_file,
        n_items=n_items,
        n_users=n_users
    )
    result = rec_model.train(
        data_generator=data_loader
    )
    return result


if __name__ == "__main__":
    n_users = 100
    n_items = 1000
    emb_dim = 20
    random_seed = 2020
    data_file = "../data/test_full_ideal"
    np.random.seed(random_seed)

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

    recommender = Recommender(
        model="factor",
        n_users=n_users,
        n_items=n_items,
        model_spec={
            "learning_rate": 0.01,
            "batch_size": 512,
            "max_epoch": 3,
            "emb_dim": 20,
        },
        model_name="factor_rec"
    )
    recommender.model.initialization(
        item_emb_data=np.random.random(size=[n_items, emb_dim]),
        user_emb_data=np.random.random(size=[n_users, emb_dim])
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

    interactor.iterate(
        steps=10,
        k=10,
        random_rec=True
    )
    print("done start-up iterations")

    interactor.recommender.model.initialization(
        item_emb_file="../data/emb/item",
        user_emb_file="../data/emb/user"
    )
    for i in range(990):
        # interactor.recommender.model.initialization(
        #     item_emb_data=np.random.random(size=[n_items, emb_dim]),
        #     user_emb_data=np.random.random(size=[n_users, emb_dim])
        # )
        # train(
        #     n_users=n_users,
        #     n_items=n_items,
        #     data_file=data_file,
        #     rec_model=interactor.recommender.model
        # )
        interactor.iterate(
            steps=1,
            k=10,
            random_rec=False
        )
        print("done iteration %d" % i)
