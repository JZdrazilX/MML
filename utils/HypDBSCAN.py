# hyperbolic_dbscan.py

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from typing import Optional, List
from .PBCalculations import PoincareBallMapper

class HyperbolicDBSCAN(BaseEstimator, ClusterMixin):
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    adapted for hyperbolic space using the Poincaré ball model.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point.

    dimension : int, default=2
        The dimension of the hyperbolic space (Poincaré ball).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset. Noisy samples are given
        the label -1.

    Examples
    --------
    >>> import numpy as np
    >>> from hyperbolic_dbscan import HyperbolicDBSCAN
    >>> X = np.random.randn(100, 2) * 0.1  # Sample data
    >>> dbscan = HyperbolicDBSCAN(eps=0.2, min_samples=3)
    >>> labels = dbscan.fit_predict(X)
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, dimension: int = 2):
        self.eps = eps
        self.min_samples = min_samples
        self.dimension = dimension
        self.manifold = PoincareBallMapper(self.dimension)
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'HyperbolicDBSCAN':
        """
        Perform DBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.shape[0] == 0:
            raise ValueError("The input data is empty.")

        n_samples = X.shape[0]
        self.X_ = X

        # Compute the distance matrix
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = self.manifold.distance(X[i], X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Symmetric

        # Find the neighbors for each sample
        neighbors = [np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)]

        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -1:
                continue

            if len(neighbors[i]) >= self.min_samples:
                labels[i] = cluster_id
                self._expand_cluster(i, neighbors, labels, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        return self

    def _expand_cluster(self, idx: int, neighbors: List[np.ndarray], labels: np.ndarray, cluster_id: int):
        """
        Expand the cluster to include all density-reachable points.

        Parameters
        ----------
        idx : int
            Index of the starting point.

        neighbors : list of ndarray
            List of neighbor indices for each sample.

        labels : ndarray of shape (n_samples,)
            Cluster labels for each point.

        cluster_id : int
            The ID of the current cluster.
        """
        queue = [idx]
        while queue:
            point_idx = queue.pop(0)
            if labels[point_idx] == -1:
                labels[point_idx] = cluster_id
            elif labels[point_idx] != cluster_id:
                continue

            if len(neighbors[point_idx]) >= self.min_samples:
                for neighbor_idx in neighbors[point_idx]:
                    if labels[neighbor_idx] == -1:
                        labels[neighbor_idx] = cluster_id
                        queue.append(neighbor_idx)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform clustering on X and return cluster labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point in the dataset. Noisy samples are given the label -1.
        """
        self.fit(X, y)
        return self.labels_
