"""
warp user
"""
import numpy as np

from user import Models
from experiment import DataLoaderRecUsersCand


class User(object):
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

    def users(self):
        """
        sampling users
        :return: np.array([uid])
        """
        return np.arange(self.n_users).astype(dtype=np.int64)

    def react(
            self,
            candidates
    ):
        """
        basic implicit reaction
        :param candidates: [[candidate of length k] for u in self.users().tolist()]
        :return: [item_clicked]
        """
        scores_list = self.rate(candidates=candidates)
        feedbacks = []
        for i in range(len(scores_list)):
            scores = scores_list[i]
            feedback_ind = np.argsort(scores)[-1]           # select highest score item
            feedback = candidates[i][feedback_ind]
            feedbacks.append(feedback)
        return feedbacks, scores_list

    def rate(
            self,
            candidates,
    ):
        cand_data_loader = DataLoaderRecUsersCand(
            candidates=candidates,
            users=self.users()
        )
        scores_list = []
        for i in range(cand_data_loader.n_users_effect):
            scores = self.model.predict(
                data_generator=cand_data_loader
            )
            scores_list.append(scores)
        return scores_list
