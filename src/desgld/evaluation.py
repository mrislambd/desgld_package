import numpy as np


class AccuracyCalculator:
    """Accuracy: This class is used to calculate the accuracy"""

    """
    :param x_all: the input data
    :param y_all: the output data
    :param history_all: Combined output from all the nodes
    :param T: the number of iterations
    """

    def __init__(self, x_all, y_all, history_all, T):
        self.x_all = x_all
        self.y_all = y_all
        self.history_all = history_all
        self.T = T

    def calculate(self):
        mis_class = []
        for m in range(self.T + 1):
            mis_class.append([])
        for t in range(self.T + 1):
            for n in range(len(self.history_all[t][0][0])):
                temp0 = 0
                for i in range(len(self.x_all)):
                    z = 1 / (
                        1
                        + np.exp(
                            -np.dot(
                                np.transpose(self.history_all[t][1])[n],
                                self.x_all[i],
                            )
                        )
                    )
                    if z >= 0.5:
                        z = 1
                    else:
                        z = 0
                    if self.y_all[i] != z:
                        temp0 += 1
                mis_class[t].append(1 - temp0 / len(self.x_all))
        result_acc = np.mean(mis_class, axis=1)
        result_std = np.std(mis_class, axis=1)

        return result_acc, result_std
