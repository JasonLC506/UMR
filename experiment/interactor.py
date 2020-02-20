"""
interaction between user and recommender
"""


class Interactor(object):
    def __init__(
            self,
            user,
            recommender,
            model_spec,
    ):
        self.user = user
        self.recommender = recommender
        self.model_spec = model_spec

    def iterate(
            self,
            steps,
            k,
            random_rec=False,
    ):
        data = []
        for i in range(steps):
            users = self.user.users()
            candidates = self.recommender.rec(
                users=users,
                k=k,
                random_rec=random_rec,
            )
            feedback = self.user.react(candidates)
            self.recommender.update(feedback)
            data.append(feedback)
        self.save_data(
            data=data,
            fn=self.model_spec["data_file"]
        )

    @staticmethod
    def save_data(data, fn):
        # naive write
        with open(fn, 'a') as f:
            for d in data:
                f.write("\t".join(list(map(str, d))) + "\n")
