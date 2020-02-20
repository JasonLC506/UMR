"""
generate user and/or item embeddings
@article{Chaney2018,
author = {Chaney, Allison J. B. and Stewart, Brandon M. and Engelhardt, Barbara E.},
pages = {224--232},
title = {{How algorithmic confounding in recommendation systems increases homogeneity and decreases utility}},
year = {2018}
}

"""
import numpy as np
from scipy.stats import dirichlet
import pickle


def generate_emb(
        n,
        dim,
        model_spec,
):
    common_init = dirichlet.rvs(
        np.ones(dim, dtype=np.float32) * model_spec["common_init_alpha"]
    ).squeeze()
    common = model_spec["rescale_common"] * common_init
    embs = dirichlet.rvs(
        common,
        size=n
    )
    return embs


def save_emb(
        embs,
        fn
):
    with open(fn, 'wb') as f:
        pickle.dump(embs, f)


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
