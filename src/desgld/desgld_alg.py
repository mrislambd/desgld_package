import numpy as np


class DeSGLD:
    """Decentralized Stochastic Gradient Langevin Diffuision Algorithm"""

    """
    This class implements the Decentralized Stochastic Grad
    Langevin Diffuision Algorithm (DeSGLD) for both vanila and
    extra convergence analysis
    """

    def __init__(self, size_w, N, sigma, eta, T, dim, b, lam, x, y, w, hv):
        """Construct of the DeSGLD class"""

        """
        :param size_w: the size of the network
        :param N: size of the array in each iteration
        :param sigma: variance of the noise
        :param eta: the step size
        :param T: the number of iterations
        :param dim: the dimension of the input data
        :param b: batch size
        :param lam: the regularization parameter
        :param x: the input data
        :param y: the output data
        :param w: the initial weights/network weights
        :param hv: h values for the extra convergence analysis
        """
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
        """Gradient of the loss function for logistic regression"""

        """
        :param beta: parameters of interest
        :return: the gradient of the loss function
        """
        f = np.zeros(dim)
        for i in range(b):
            h = 1 / (1 + np.exp(-np.dot(beta, x[i])))
            f = f - np.dot((y[i] - h), x[i])
        f = f + np.dot(2 / lam, beta)
        return f

    def gradient_linreg(self, beta, x, y, dim, lam, b):
        """Gradient of the loss function for linear regression"""

        """
        :param beta: parameters of interest
        :return: the gradient of the loss function
        """
        f = []
        for i in range(dim):
            f.append(0)
        f = np.array(f)
        randomList = np.random.randint(0, len(y) - 1, size=int(b))
        for item in randomList:
            f = f - np.dot((y[item] - np.dot(beta, x[item])), x[item])
        f = f + np.dot(2 / lam, beta)
        return f

    def vanila_desgld_logreg(self):
        """Vanilla DeSGLD algorithm for logistic regression"""

        """
        :return: parameters of interest from the vanila DeSGLD algorithm
        :param: history_all: Combined output from all the nodes
        :param: beta_mean_all: Mean approximation from all the nodes
        """

        # Initialization
        beta = np.random.normal(
            0, self.sigma, size=(self.N, self.size_w, self.dim)
        )
        history_all = np.zeros((self.T + 1, self.size_w, self.dim, self.N))
        beta_mean_all = np.zeros((self.T + 1, self.dim, self.N))

        for i in range(self.size_w):
            for d in range(self.dim):
                history_all[0, i, d] = beta[:, i, d].T
                beta_mean_all[0, d] += history_all[0, i, d] / self.size_w

        # Update the iterations
        step = self.eta
        for t in range(1, self.T + 1):
            noise = np.random.normal(
                0, self.sigma, size=(self.N, self.size_w, self.dim)
            )
            for n in range(self.N):
                for i in range(self.size_w):
                    for d in range(self.dim):
                        g = self.gradient_logreg(
                            beta[n, i],
                            self.x[i],
                            self.y[i],
                            self.dim,
                            self.lam,
                            self.b,
                        )
                        temp = np.sum(self.w[i] * beta[n], axis=0)
                        beta[n, i] = (
                            temp - step * g + np.sqrt(2 * step) * noise[n, i]
                        )

            history_all[t] = beta.transpose(1, 2, 0)
            beta_mean_all[t] = np.mean(history_all[t], axis=0)

        return history_all, beta_mean_all

    def vanila_desgld_linreg(self):
        """Vanilla DeSGLD algorithm  for linear regression"""

        """
        :return: parameters of interest from the vanila DeSGLD algorithm
        :param: history_all: Combined output from all the nodes
        :param: beta_mean_all: Mean approximation from all the nodes
        """

        # Initialization
        beta = np.random.normal(
            0, self.sigma, size=(self.N, self.size_w, self.dim)
        )
        history_all = np.zeros((self.T + 1, self.size_w, self.dim, self.N))
        beta_mean_all = np.zeros((self.T + 1, self.dim, self.N))

        for i in range(self.size_w):
            for d in range(self.dim):
                history_all[0, i, d] = beta[:, i, d].T
                beta_mean_all[0, d] += history_all[0, i, d] / self.size_w

        # Update the iterations
        step = self.eta
        for t in range(1, self.T + 1):
            noise = np.random.normal(
                0, self.sigma, size=(self.N, self.size_w, self.dim)
            )
            for n in range(self.N):
                for i in range(self.size_w):
                    for d in range(self.dim):
                        g = self.gradient_linreg(
                            beta[n, i],
                            self.x[i],
                            self.y[i],
                            self.dim,
                            self.lam,
                            self.b,
                        )
                        temp = np.sum(self.w[i] * beta[n], axis=0)
                        beta[n, i] = (
                            temp - step * g + np.sqrt(2 * step) * noise[n, i]
                        )

            history_all[t] = beta.transpose(1, 2, 0)
            beta_mean_all[t] = np.mean(history_all[t], axis=0)

        return history_all, beta_mean_all
