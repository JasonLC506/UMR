from matplotlib import pyplot as plt
import numpy as np
import pickle


def lineplot_relative(
        lines,
        baseline,
        names,
):
    x = np.arange(baseline.shape[0])
    for i in range(len(lines)):
        relative = lines[i] - baseline
        plt.plot(x, relative, label=names[i])
    plt.plot(x, np.zeros_like(x), label="baseline")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lineplot_relative(
        lines=[
            pickle.load(open("../result/jaccard_test_full", 'rb')),
            pickle.load(open("../result/jaccard_test_full_ideal", "rb")),
            pickle.load(open("../result/jaccard_test_full_random", "rb")),
        ],
        # baseline=pickle.load(open("../result/jaccard_test_full_random", "rb")),
        baseline=np.zeros(999),
        names=[
            "factor",
            "ideal",
            "random"
        ]
    )
