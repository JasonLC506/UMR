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


def user_feedback_trend(
        candidates_steps,
        feedback_steps,
        scores_steps,
        scores_optim_steps,
):
    """

    :param candidates_steps:
    :param feedback_steps:
    :param scores_steps:
    :param scores_optim_steps:
    :return:
    """
    n_steps = len(scores_steps)
    n_users = len(scores_steps[0])
    total_regrets = np.zeros([n_steps, n_users])
    selected_regrets = np.zeros([n_steps, n_users])
    optim_scores = np.zeros([n_steps, n_users])
    optim_selected_scores = np.zeros([n_steps, n_users])
    for i_step in range(n_steps):
        for i_user in range(len(scores_steps[i_step])):
            optim_scores[i_step, i_user] = np.mean(scores_optim_steps[i_step][i_user])
            optim_selected_scores[i_step, i_user] = np.max(scores_optim_steps[i_step][i_user])
            total_regrets[i_step, i_user] = optim_scores[i_step, i_user] - np.mean(scores_steps[i_step][i_user])
            selected_ind = candidates_steps[i_step][i_user].tolist().index(feedback_steps[i_step][i_user])
            selected_regrets[i_step, i_user] = optim_selected_scores[i_step, i_user] - \
                                               scores_steps[i_step][i_user][selected_ind]
    return total_regrets, selected_regrets, optim_scores, optim_selected_scores


def emb_trend_compare(
        emb_base,
        embs,
):
    """

    :param emb_base:
    :param embs: [[np.array(emb)]]
    :return:
    """
    emb_base = np.array(emb_base)
    emb_base_norm = np.linalg.norm(emb_base, axis=-1)
    n_steps = len(embs)
    n_users = len(embs[0])
    emb_comparison = np.zeros([n_steps, n_users, 2])
    for i_step in range(n_steps):
        for i_user in range(len(embs[i_step])):
            emb = embs[i_step][i_user]
            if emb is None:
                emb_comparison[i_step, i_user, :] = 0.0
                continue
            print(emb.shape)   ### test
            emb_cos = np.inner(
                emb,
                emb_base[i_user]
            ) / emb_base_norm[i_user]
            emb_sin = np.sqrt(np.linalg.norm(emb) ** 2 - emb_cos ** 2)
            emb_comparison[i_step, i_user, 0] = emb_cos
            emb_comparison[i_step, i_user, 1] = emb_sin
    return emb_comparison


if __name__ == "__main__":
    # user_dict = load_data("../data/synthetic_knn")
    # # for uid in user_dict:
    # #     user_dict[uid] = np.random.choice(
    # #         1000, # n_items
    # #         size=100,   # iterations
    # #         replace=False
    # #     )
    # jaccard_mean = user_reaction_compare(user_dict)
    # print(jaccard_mean)
    # # with open("../result/jaccard_synthetic_knn", "wb") as f:
    # #     pickle.dump(jaccard_mean, f)


    [
        candidates_steps,
        feedback_steps,
        scores_steps,
        optim_scores_steps,
        rec_u_embs_steps
    ] = pickle.load(open("../data/synthetic_KNN_random_10Kitem_25_1000steps_detail", 'rb'))
    u_embs = pickle.load(open("../data/emb/user_new_25", 'rb'))
    total_regrets, selected_regrets, optim_scores, optim_selected_scores = user_feedback_trend(
        candidates_steps,
        feedback_steps,
        scores_steps,
        optim_scores_steps,
    )
    emb_comparison = emb_trend_compare(
        emb_base=u_embs,
        embs=rec_u_embs_steps
    )
    from matplotlib import pyplot as plt
    for uid in range(total_regrets.shape[1]):
        x = np.arange(total_regrets.shape[0])
        _, axs = plt.subplots(2)
        axs[0].plot(x, total_regrets[:, uid], label="total_regret")
        axs[0].plot(x, selected_regrets[:, uid], label="selected_regret")
        axs[0].plot(x, optim_scores[:, uid], label="optim_total")
        axs[0].plot(x, optim_selected_scores[:, uid], label="optim_selected")
        # axs[1].plot(x, emb_comparison[:, uid, 0], label="emb_cos")
        # axs[1].plot(x, emb_comparison[:, uid, 1], label="emb_sin")
        axs[0].legend()
        # axs[1].legend()
        plt.show()
    x = np.arange(total_regrets.shape[0])
    _, axs = plt.subplots(2)
    axs[0].plot(x, np.mean(total_regrets, axis=1), label="total_regret")
    axs[0].plot(x, np.mean(selected_regrets, axis=1), label="selected_regret")
    axs[0].plot(x, np.mean(optim_scores, axis=1), label="optim_total")
    axs[0].plot(x, np.mean(optim_selected_scores, axis=1), label="optim_selected")
    # axs[1].plot(x, np.mean(emb_comparison[:, :, 0], axis=1), label="emb_cos")
    # axs[1].plot(x, np.mean(emb_comparison[:, :, 1], axis=1), label="emb_sin")
    axs[0].legend()
    # axs[1].legend()
    plt.show()
    with open("../result/trend_synthetic_knn_10Kitem", 'wb') as f:
        pickle.dump(
            [
                total_regrets,
                selected_regrets,
                optim_scores,
                optim_selected_scores,
                emb_comparison
            ],
            f
        )
