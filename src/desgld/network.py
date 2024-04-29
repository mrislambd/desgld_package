import numpy as np


class NetworkArchitecture:
    """Decentralized network architecture

    Description:
        This class is used to generate different network
        architectures. Here we add four different network architectures:
        fully connected, fully disconnected, circular network and star network.
    Args:
        size_w (int): the size of the network
    """

    def __init__(self, size_w):
        self.size_w = size_w

    def fully_connected(self):
        """Fully Connected Network

        Description:
            A fully connected network is a kind of network in which all nodes
            are connected to all other nodes.

        Returns:
            w (float): a fully connected network matrix

        Examples:
            >>> net = NetworkArchitecture(size_w=6)
            >>> w=net.fully_connected()
            >>> print(w)
            w=[[1/6,1/6,1/6,1/6,1/6,1/6],
                [1/6,1/6,1/6,1/6,1/6,1/6],
                [1/6,1/6,1/6,1/6,1/6,1/6],
                [1/6,1/6,1/6,1/6,1/6,1/6],
                [1/6,1/6,1/6,1/6,1/6,1/6],
                [1/6,1/6,1/6,1/6,1/6,1/6]]

        """
        return np.array(
            [
                [1 / self.size_w for _ in range(self.size_w)]
                for _ in range(self.size_w)
            ]
        )

    def circular_network(self):
        """Circular Network

        Description:
            A circular network is a kind of network in which a particular
            node is connected to its left and right nodes only.

        Returns:
            w (float): a circular network matrix

        Examples:
            >>> net = NetworkArchitecture(size_w=6)
            >>> w=net.circular()
            >>> print(w)
            w=[[1/3,1/3,0,0,0,1/3],
                [1/3,1/3,1/3,0,0,0],
                [0,1/3,1/3,1/3,0,0],
                [0,0,1/3,1/3,1/3,0],
                [0,0,0,1/3,1/3,1/3],
                [1/3,0,0,0,1/3,1/3]]

        """
        x = [[0] * self.size_w for _ in range(self.size_w)]
        for i in range(self.size_w):
            for j in range(self.size_w):
                if i == j:
                    x[i][j] = 1 / 3
                elif i == (j + 1) % self.size_w or j == (i + 1) % self.size_w:
                    x[i][j] = 1 / 3
        return np.array(x)

    def fully_disconnected(self):
        """Completely Disconnected Network

        Description:
            A completely disconnected network is a kind of network in which
            all the nodes are disconnected from each other.

        Returns:
            w (float): a disconnected network matrix

        Examples:
            >>> net = NetworkArchitecture(size_w=6)
            >>> w=net.circular()
            >>> print(w)
            w=[[1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,1,0,0,0],
                [0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]]

        """
        x = [[0] * self.size_w for _ in range(self.size_w)]
        for i in range(self.size_w):
            for j in range(self.size_w):
                if i == j:
                    x[i][j] = 1
        return np.array(x)

    def star_network(self):
        """Star-like Connected Network

        Description:
            A star-like network is a kind of network in which
            there is a central node and all other nodes are
            connected to the central node. However, the individual
            nodes are not connected to each other.

        Returns:
            w (float): a star network matrix

        Examples:
            >>> net = NetworkArchitecture(size_w=6)
            >>> w=net.star_network()
            >>> print(w)
            w=[[1,0,0,0,0,1],
                [0,1,0,0,0,1],
                [0,0,1,0,0,1],
                [0,0,0,1,0,1],
                [0,0,0,0,1,1],
                [0,0,0,0,0,1]]

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
        return np.array(s)
