"""
warp recommender
"""
import numpy as np

from recommender import Models
from experiment import DataLoaderRecUsers, DataLoaderItemPredict


class Recommender(object):
    def __init__(
            self,
            model,
            n_users,
            n_items,
            model_spec,
            model_name=None,
    ):
        self.model = Models[model](
            n_users=n_users,
            n_items=n_items,
            model_spec=model_spec,
            model_name=model_name
        )
        self.n_users = n_users
        self.n_items = n_items
        self.user_feedbacks = {}

    def rec(
            self,
            users,
            k=3,
            random_rec=False
    ):
        """
        :param users: np.array([uid])
        :param k: number of returned items
        :param random_rec: circumvent self.model, gives random rec
        :return:
        """
        user_data_loader = DataLoaderRecUsers(
            n_items=self.n_items,
            users=users
        )
        if random_rec:
            cands = [
                np.random.choice(
                    self.n_items, size=k, replace=False
                ) for _ in range(user_data_loader.n_users_effect)
            ]
        else:
            cands = []
            for i in range(user_data_loader.n_users_effect):
                scores = self.model.predict(
                    data_generator=user_data_loader
                )
                cand = cand_ind = self.exclude_sort(
                    scores=scores,
                    excluded_indices=np.array(self.user_feedbacks[user_data_loader.i_user_effect]).astype(np.int64),
                    k=k,
                )
                cands.append(cand)
        return cands

    def update(
            self,
            feedback
    ):
        for uid in range(len(feedback)):
            self.user_feedbacks.setdefault(uid, []).append(feedback[uid])

    @staticmethod
    def exclude_sort(
            scores,
            excluded_indices,
            k
    ):
        min_score = np.min(scores)
        scores[excluded_indices] = min_score - 1.0   # set to minimum
        inds = np.argsort(scores)[-1: -(k+1): -1]
        return inds


class RecommenderSeq(Recommender):
    def __init__(
            self,
            model,
            n_users,
            n_items,
            model_spec,
            model_name=None,
    ):
        super(RecommenderSeq, self).__init__(
            model=model,
            n_users=n_users,
            n_items=n_items,
            model_spec=model_spec,
            model_name=model_name
        )
        for uid in range(self.n_users):
            self.user_feedbacks[uid] = []

    def rec(
            self,
            max_length_input=100,
            k=3,
            random_rec=False,
            **kwargs
    ):
        """
        :param max_length_input:
        :param k: number of returned items
        :param random_rec: circumvent self.model, gives random rec
        :return:
        """
        user_data_loader = DataLoaderItemPredict(
            n_items=self.n_items,
            hist_vs=list(map(
                lambda x: np.array(x[1], dtype=np.int64),
                sorted(
                    self.user_feedbacks.items(),
                    key=lambda y: y[0]
                )
            )),
            max_length_input=max_length_input
        )
        if random_rec:
            cands = [
                np.random.choice(
                    self.n_items, size=k, replace=False
                ) for _ in range(user_data_loader.n_sequences_effect)
            ]
        else:
            cands = []
            for i in range(user_data_loader.n_sequences_effect):
                scores = self.model.predict(
                    data_generator=user_data_loader
                )
                cand = cand_ind = self.exclude_sort(
                    scores=scores,
                    excluded_indices=np.array(self.user_feedbacks[user_data_loader.i_sequence_effect]).astype(np.int64),
                    k=k,
                )
                cands.append(cand)
        return cands
