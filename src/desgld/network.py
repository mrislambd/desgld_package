import random

import numpy as np


class NetworkArchitecture:
    """Decentralized network architecture

    Description:
        This class is used to generate different network
        architectures. Here we add four different network architectures:
        fully connected, fully disconnected, circular network and star network.
    Args:
        size_w (int): the size of the network
    Returns:
        Different network architectures
    """

    def __init__(self, size_w):
        self.size_w = size_w

    def fully_connected(self):
        """Fully Connected Network

        Description:
            A fully connected network is a kind of network in which all nodes
            are connected to all other nodes.

        Args:
            deg_mat(np.ndarray): degree matrix of the graph Laplacian
            adj_mat(np.ndarray): adjacency matrix of the graph Laplacian
            laplacian_matrix(np.ndarray): graph Laplacian matrix
            eigenvalues (np.ndarray): eigenvalues of the graph Laplacian
            max_eigenvalue (float): maximum eigenvalue of the graph Laplacian
            reg_delta (float): regular delta
            min_delta (float): minimum delta for positive w

        Code:
        def fully_connected(self):
            adj_mat=np.ones(self.size_w,self.size_w)-np.eye(self.size_w)
            deg_mat=(self.size_w-1)*np.eye(self.size_w)
            laplacian_matrix=deg_mat-adj_mat
            eigenvalues = np.linalg.eigvals(laplacian_matrix)
            max_eigenvalue = np.max(np.real(eigenvalues))
            reg_delta = random.uniform(0, (2 / max_eigenvalue))
            min_delta = np.min(laplacian_matrix) / np.max(deg_mat)
            delta = max(reg_delta, min_delta)
            w = np.eye(self.size_w) - delta * laplacian_matrix

            return w

        Examples:
            >>> w=NetworkArchitecture(size_w=5)
            >>> w=w.fully_connected()
            >>> print(w)
                [[0.67811004 0.08047249 0.08047249 0.08047249 0.08047249]
                [0.08047249 0.67811004 0.08047249 0.08047249 0.08047249]
                [0.08047249 0.08047249 0.67811004 0.08047249 0.08047249]
                [0.08047249 0.08047249 0.08047249 0.67811004 0.08047249]
                [0.08047249 0.08047249 0.08047249 0.08047249 0.67811004]]
            >>> print("Row sums:", np.sum(w, axis=0))
                Row sums: [1. 1. 1. 1. 1.]
            >>> print("Column sums:", np.sum(w, axis=1))
                Column sums: [1. 1. 1. 1. 1.]
        Returns:
            w (float): a fully connected network matrix
        """
        adj_mat = np.ones((self.size_w, self.size_w)) - np.eye(self.size_w)
        deg_mat = (self.size_w - 1) * np.eye(self.size_w)
        laplacian_matrix = deg_mat - adj_mat
        eigenvalues = np.linalg.eigvals(laplacian_matrix)
        max_eigenvalue = np.max(np.real(eigenvalues))
        reg_delta = random.uniform(0, (2 / max_eigenvalue))
        min_delta = np.min(laplacian_matrix) / np.max(deg_mat)
        delta = max(reg_delta, min_delta)
        w = np.eye(self.size_w) - delta * laplacian_matrix

        return w

    def circular_network(self):
        """Circular Network

        Description:
            A circular network is a kind of network in which a particular
            node is connected to its left and right nodes only.

        Args:
            deg_mat(np.ndarray): degree matrix of the graph Laplacian
            adj_mat(np.ndarray): adjacency matrix of the graph Laplacian
            laplacian_matrix(np.ndarray): graph Laplacian matrix
            eigenvalues (np.ndarray): eigenvalues of the graph Laplacian
            max_eigenvalue (float): maximum eigenvalue of the graph Laplacian
            reg_delta (float): regular delta
            min_delta (float): minimum delta for positive w


        Returns:
            w (float): a circular network matrix

        def circular_network(self):
            adj_mat=np.zeros((self.size_w,self.size_w))
            for i in range(self.size_w):
                for j in range(self.size_w):
                    if (i+1)==j:
                        adj_mat[i,j]=1
                    else:
                        adj_mat[i,j]=adj_mat[i,j]
            adj_mat[0,(self.size_w-1)]=1
            adj_mat=adj_mat+np.transpose(adj_mat)
            deg_mat=2*np.eye(self.size_w)
            laplacian_matrix=deg_mat-adj_mat
            eigenvalues=np.linalg.eigvals(laplacian_matrix)
            max_eigenvalue = np.max(np.real(eigenvalues))
            reg_delta = random.uniform(0, (2 / max_eigenvalue))
            min_delta = np.min(laplacian_matrix) / np.max(deg_mat)
            delta = max(reg_delta, min_delta)
            w=np.eye(self.size_w)-delta*laplacian_matrix
            return w

        Examples:
            >>> net = NetworkArchitecture(size_w=5)
            >>> w=net.circular_network()
            >>> print(w)
                [[0.72162653 0.13918674 0.         0.         0.13918674]
                [0.13918674 0.72162653 0.13918674 0.         0.        ]
                [0.         0.13918674 0.72162653 0.13918674 0.        ]
                [0.         0.         0.13918674 0.72162653 0.13918674]
                [0.13918674 0.         0.         0.13918674 0.72162653]]
            >>> print("Row sums:", np.sum(w, axis=0))
                Row sums: [1. 1. 1. 1. 1.]
            >>> print("Column sums:", np.sum(w, axis=1))
                Column sums: [1. 1. 1. 1. 1.]
        """
        adj_mat = np.zeros((self.size_w, self.size_w))
        for i in range(self.size_w):
            for j in range(self.size_w):
                if (i + 1) == j:
                    adj_mat[i, j] = 1
                else:
                    adj_mat[i, j] = adj_mat[i, j]
        adj_mat[0, (self.size_w - 1)] = 1
        adj_mat = adj_mat + np.transpose(adj_mat)
        deg_mat = 2 * np.eye(self.size_w)
        laplacian_matrix = deg_mat - adj_mat
        eigenvalues = np.linalg.eigvals(laplacian_matrix)
        max_eigenvalue = np.max(np.real(eigenvalues))
        reg_delta = random.uniform(0, (2 / max_eigenvalue))
        min_delta = np.min(laplacian_matrix) / np.max(deg_mat)
        delta = max(reg_delta, min_delta)
        w = np.eye(self.size_w) - delta * laplacian_matrix
        return w

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

        Args:
            deg_mat(np.ndarray): degree matrix of the graph Laplacian
            adj_mat(np.ndarray): adjacency matrix of the graph Laplacian
            laplacian_matrix(np.ndarray): graph Laplacian matrix
            eigenvalues (np.ndarray): eigenvalues of the graph Laplacian
            max_eigenvalue (float): maximum eigenvalue of the graph Laplacian
            reg_delta (float): regular delta
            min_delta (float): minimum delta for positive w

        Returns:
            w (float): a star network matrix

        Code:
            def star_network(self):
                adj_mat=np.zeros((self.size_w,self.size_w))
                for i in range(self.size_w):
                    for j in range(self.size_w):
                        if i==0 or j==0:
                            adj_mat[i,j]=1
                            adj_mat[i,i]=0
                        else:
                            adj_mat[i,j]=adj_mat[i,j]
                deg_mat=np.eye(self.size_w)
                deg_mat[0,0]=(self.size_w-1)
                laplacian_matrix=deg_mat-adj_mat
                eigenvalues=np.linalg.eigvals(laplacian_matrix)
                max_eigenvalue = np.max(np.real(eigenvalues))
                reg_delta = random.uniform(0, (2 / max_eigenvalue))
                min_delta = np.min(laplacian_matrix) / np.max(deg_mat)
                delta = max(reg_delta, min_delta)
                w=np.eye(self.size_w)-delta*laplacian_matrix
            return w

        Examples:
            >>> net = NetworkArchitecture(size_w=5)
            >>> w=net.star_network()
            >>> print(w)
                [[0.71180893 0.07204777 0.07204777 0.07204777 0.07204777]
                [0.07204777 0.92795223 0.         0.         0.        ]
                [0.07204777 0.         0.92795223 0.         0.        ]
                [0.07204777 0.         0.         0.92795223 0.        ]
                [0.07204777 0.         0.         0.         0.92795223]]
            >>> print("Row sums:", np.sum(w, axis=0))
                Row sums: [1. 1. 1. 1. 1.]
            >>> print("Column sums:", np.sum(w, axis=1))
                Column sums: [1. 1. 1. 1. 1.]
        """
        adj_mat = np.zeros((self.size_w, self.size_w))
        for i in range(self.size_w):
            for j in range(self.size_w):
                if i == 0 or j == 0:
                    adj_mat[i, j] = 1
                    adj_mat[i, i] = 0
                else:
                    adj_mat[i, j] = adj_mat[i, j]
        deg_mat = np.eye(self.size_w)
        deg_mat[0, 0] = self.size_w - 1
        laplacian_matrix = deg_mat - adj_mat
        eigenvalues = np.linalg.eigvals(laplacian_matrix)
        max_eigenvalue = np.max(np.real(eigenvalues))
        reg_delta = random.uniform(0, (2 / max_eigenvalue))
        min_delta = np.min(laplacian_matrix) / np.max(deg_mat)
        delta = max(reg_delta, min_delta)
        w = np.eye(self.size_w) - delta * laplacian_matrix
        return w
