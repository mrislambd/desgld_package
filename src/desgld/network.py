class NetworkArchitecture:
    """Decentralized network architecture"""

    """
    This class is used to generate different network architectures. Here we add
    four different network architectures: fully connected, fully disconnected,
    circular network and star network.
    """

    def __init__(self, size_w):
        """
        :param size_w: the size of the network
        """
        self.size_w = size_w

    def fully_connected(self):
        """
        :return: a fully connected network
        """
        return [
            [1 / self.size_w for _ in range(self.size_w)]
            for _ in range(self.size_w)
        ]

    def circular_network(self):
        """
        :return: a circular network matrix
        """
        x = [[0] * self.size_w for _ in range(self.size_w)]
        for i in range(self.size_w):
            for j in range(self.size_w):
                if i == j:
                    x[i][j] = 1 / 3
                elif i == (j + 1) % self.size_w or j == (i + 1) % self.size_w:
                    x[i][j] = 1 / 3
        return x

    def fully_disconnected(self):
        """
        :return: a fully disconnected network matrix
        """
        x = [[0] * self.size_w for _ in range(self.size_w)]
        for i in range(self.size_w):
            for j in range(self.size_w):
                if i == j:
                    x[i][j] = 1
        return x

    def star_network(self):
        """
        :return: a star network matrix
        """
        s = [
            [
                0.0 if j != self.size_w - 1 else 1 / self.size_w
                for j in range(self.size_w)
            ]
            for _ in range(self.size_w)
        ]
        for i in range(self.size_w):
            s[i][i] = 1 / self.size_w
        return s
