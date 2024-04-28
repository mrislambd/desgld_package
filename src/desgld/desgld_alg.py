import math

import numpy as np


class DeSGLD:
    def __init__(self, size_w, N, sigma, eta, T, dim, b, lam, x, y, w, hv):
        self.size_w = size_w
        self.N = N
        self.sigma = sigma
        self.eta = eta
        self.T = T
        self.dim = dim
        self.b = b
        self.lam = lam
        self.x = x
        self.y = y
        self.w = w
        self.hv = hv

    def gradient_logreg(self, beta, x, y, dim, lam, b):
        f = np.zeros(dim)
        for i in range(b):
            h = 1 / (1 + np.exp(-np.dot(beta, x[i])))
            f = f - np.dot((y[i] - h), x[i])
        f = f + np.dot(2 / lam, beta)
        return f

    def vanila_desgld_logreg(self):
        # Initialization
        beta = np.random.normal(0, self.sigma, (self.N, self.size_w, self.dim))
        history_all = []
        beta_mean_all = []

        for _ in range(1):
            history = np.empty((self.size_w, self.dim, self.N))
            beta_mean = np.empty((self.dim, self.N))

            for i in range(self.size_w):
                for d in range(self.dim):
                    history[i, d] = beta[:, i, d]
            for d in range(self.dim):
                beta_mean[d] = np.mean(history[:, d], axis=0)
            history_all.append(history)
            beta_mean_all.append(beta_mean)

        # Main loop (update)
        step = self.eta
        for t in range(self.T):
            for n in range(self.N):
                for i in range(self.size_w):
                    g = self.gradient_logreg(
                        beta[n, i],
                        self.x[i],
                        self.y[i],
                        self.dim,
                        self.lam,
                        self.b,
                    )
                    temp = 0
                    for j in range(len(beta[n])):
                        temp = temp + self.w[i, j] * beta[n, j]
                    noise = np.random.normal(0, self.sigma, self.dim)
                    beta[n, i] = temp - step * g + math.sqrt(2 * step) * noise

            history = np.empty((self.size_w, self.dim, self.N))
            beta_mean = np.empty((self.dim, self.N))
            for i in range(self.size_w):
                for d in range(self.dim):
                    history[i, d] = beta[:, i, d]
            for d in range(self.dim):
                beta_mean[d] = np.mean(history[:, d], axis=0)
            history_all.append(history)
            beta_mean_all.append(beta_mean)
