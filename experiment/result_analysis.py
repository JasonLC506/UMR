import numpy as np
import pickle


def load_data(data_file):
    user_dict = {}
    with open(data_file, 'r') as f:
        for line in f:
            uv = list(map(int, line.rstrip().split("\t")))
            for uid in range(len(uv)):
                user_dict.setdefault(uid, []).append(uv[uid])
    return user_dict


def user_reaction_compare(user_dict):
    users = np.array(list(user_dict.keys()))
    jaccards = []
    for i in range(100):
        uid_pair = np.random.choice(users, size=2, replace=False)
        reaction_pair = [
            user_dict[uid_pair[i]] for i in range(2)
        ]
        jaccard_score = [
            jaccard(
                reaction_pair[0][:j],
                reaction_pair[1][:j],
            ) for j in range(1, min(list(map(len, reaction_pair))))
        ]
        jaccards.append(jaccard_score)
    jaccards = np.array(jaccards)
    jaccard_mean = np.mean(jaccards, axis=0)
    return jaccard_mean


def jaccard(
        la,
        lb
):
    set_a = set(la)
    set_b = set(lb)
    length_a = len(set_a)
    length_b = len(set_b)
    set_union = set_a.union(set_b)
    length_common = length_a + length_b - len(set_union)
    return float(length_common) / float(length_a + length_b - length_common)


if __name__ == "__main__":
    user_dict = load_data("../data/test_full_ideal")
    for uid in user_dict:
        user_dict[uid] = np.random.choice(
            1000, # n_items
            size=1000,   # iterations
            replace=False
        )
    jaccard_mean = user_reaction_compare(user_dict)
    print(jaccard_mean)
    with open("../result/jaccard_test_full_random", "wb") as f:
        pickle.dump(jaccard_mean, f)
