import math

import numpy as np
from scipy.linalg import sqrtm


class ClassificationAccuracy:
    """Calculate the classification accuracy in Bayesian Logistic Regression"""

    def __init__(self, x_all, y_all, history_all, T):
        """
        :param x_all: the input data
        :param y_all: the output data
        :param: history_all: contains the approximation from all the nodes
        :param T: the number of iterations
        """
        self.x_all = x_all
        self.y_all = y_all
        self.history_all = history_all
        self.T = T

    def compute_accuracy(self):
        """Accuracy computation function"""

        """
        This function is used to compute the classification accuracy
        :return: misclassification
        """

        mis_class = np.empty((self.T + 1, len(self.history_all[0, 0, 0])))

        for t in range(self.T + 1):
            for n in range(len(self.history_all[t, 0, 0])):
                temp0 = 0
                for i in range(len(self.x_all)):
                    z = 1 / (
                        1
                        + np.exp(
                            -np.dot(
                                np.transpose(self.history_all[t, 1])[n],
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
                mis_class[t, n] = 1 - temp0 / len(self.x_all)

        result_acc = np.mean(mis_class, axis=1)
        result_std = np.std(mis_class, axis=1)
        return result_acc, result_std


class Wasserstein2Distance:
    """Calculate the Wasserstein 2 distance in Bayesian Linear Regression"""

    def __init__(
        self, size_w, T, avg_post, cov_post, history_all, beta_mean_all
    ):
        """
        :param size_w: scaler, int: the size of the network
        :param T: the number of iterations
        :param avg_post: the mean of the posterior distribution
        :param cov_post: the covariance of the posterior distribution
        :param: history_all: contains the approximation from all the nodes
        :param: beta_mean_all: contains the mean of the approximation
        from all the nodes
        """
        self.size_w = size_w
        self.T = T
        self.avg_post = avg_post
        self.cov_post = cov_post
        self.history_all = history_all
        self.beta_mean_all = beta_mean_all

    def W2_dist(self):
        w2dis = []
        for i in range(self.size_w):
            temp = []
            w2dis.append(temp)
        temp = []
        w2dis.append(temp)
        """
        W2 distance of each agent
        """
        for i in range(self.size_w):
            for t in range(self.T + 1):
                d = 0
                avg_temp = []
                avg_temp.append(np.mean(self.history_all[t][i][0]))
                avg_temp.append(np.mean(self.history_all[t][i][1]))
                avg_temp = np.array(avg_temp)
                cov_temp = np.cov(self.history_all[t][i])
                d = np.linalg.norm(self.avg_post - avg_temp) * np.linalg.norm(
                    self.avg_post - avg_temp
                )
                d = d + np.trace(
                    self.cov_post
                    + cov_temp
                    - 2
                    * sqrtm(
                        np.dot(
                            np.dot(sqrtm(cov_temp), self.cov_post),
                            sqrtm(cov_temp),
                        )
                    )
                )
                w2dis[i].append(np.array(math.sqrt(abs(d))))
        """
        W2 distance of the mean of agents
        """
        for t in range(self.T + 1):
            d = 0
            avg_temp = []
            avg_temp.append(np.mean(self.beta_mean_all[t][0]))
            avg_temp.append(np.mean(self.beta_mean_all[t][1]))
            avg_temp = np.array(avg_temp)
            cov_temp = np.cov(self.beta_mean_all[t])
            d = np.linalg.norm(self.avg_post - avg_temp) * np.linalg.norm(
                self.avg_post - avg_temp
            )
            d = d + np.trace(
                self.cov_post
                + cov_temp
                - 2
                * sqrtm(
                    np.dot(
                        np.dot(sqrtm(cov_temp), self.cov_post), sqrtm(cov_temp)
                    )
                )
            )
            w2dis[self.size_w].append(np.array(math.sqrt(abs(d))))

        for i in range(len(w2dis)):
            w2dis[i] = np.array(w2dis[i])

        return w2dis
