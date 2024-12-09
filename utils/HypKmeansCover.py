import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.kmeans import RiemannianKMeans
from typing import Optional

class GeomstatsKMeansCover(BaseEstimator, TransformerMixin):
    """
    A cover generator based on Riemannian K-Means clustering in hyperbolic space.

    This class performs clustering on data in the Poincaré ball model of hyperbolic space
    using Riemannian K-Means. It then constructs a cover for Mapper algorithms based on
    the clustering results and a specified overlap threshold.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form and the number of centroids to generate.

    overlap_threshold : float, default=0.5
        The overlap threshold determining the extent to which neighboring clusters overlap.
        A higher value increases the overlap between clusters.

    precomputed_centers : array-like of shape (n_clusters, n_features), default=None
        Precomputed cluster centers. If provided, Riemannian K-Means will use these centers
        for initialization.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    cluster_distances_ : ndarray of shape (n_samples, n_clusters)
        Distances from each sample to each cluster center.

    centroid_distances_ : ndarray of shape (n_clusters, n_clusters)
        Pairwise distances between cluster centers.

    manifold : geomstats.geometry.poincare_ball.PoincareBall
        The manifold representing the Poincaré ball model.

    metric : geomstats.geometry.riemannian_metric.RiemannianMetric
        The metric associated with the Poincaré ball model.

    Examples
    --------
    >>> import numpy as np
    >>> from geomstats.geometry.poincare_ball import PoincareBall
    >>> from geomstats_kmeans_cover import GeomstatsKMeansCover
    >>> X = PoincareBall(dim=2).random_uniform(100)  # Sample data on the Poincaré disk
    >>> cover = GeomstatsKMeansCover(n_clusters=5)
    >>> cover_matrix = cover.fit_transform(X)
    """

    def __init__(
        self,
        n_clusters: int,
        overlap_threshold: float = 0.5,
        precomputed_centers: Optional[np.ndarray] = None,
    ):
        self.n_clusters = n_clusters
        self.overlap_threshold = overlap_threshold
        self.precomputed_centers = precomputed_centers

        # Initialize the manifold and metric
        self.manifold = PoincareBall(dim=2)
        self.metric = self.manifold.metric

        # Initialize Riemannian K-Means
        self.init = self.precomputed_centers if self.precomputed_centers is not None else 'kmeans++'
        self.kmeans = RiemannianKMeans(
            n_clusters=self.n_clusters,
            space=self.manifold,
            init=self.init,
            max_iter=300
        )

        # Attributes to be set during fitting
        self.cluster_centers_ = None
        self.cluster_distances_ = None
        self.centroid_distances_ = None

    def fit(self, X, y=None):
        """
        Compute Riemannian K-Means clustering on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Fit Riemannian K-Means
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # Compute distances from each sample to each cluster center
        self.cluster_distances_ = self.metric.dist(
            X[:, np.newaxis, :], self.cluster_centers_[np.newaxis, :, :]
        )

        # Compute pairwise distances between cluster centers
        self._calculate_centroid_distances()
        return self

    def _calculate_centroid_distances(self):
        """
        Calculate pairwise distances between cluster centers.
        """
        self.centroid_distances_ = self.metric.dist(
            self.cluster_centers_[:, np.newaxis, :], self.cluster_centers_[np.newaxis, :, :]
        )

    def transform(self, X):
        """
        Construct the cover matrix based on distances to cluster centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        cover_matrix : ndarray of shape (n_samples, n_clusters)
            Boolean matrix indicating the clusters each sample belongs to.
        """
        n_samples = X.shape[0]
        cover_matrix = np.zeros((n_samples, self.n_clusters), dtype=bool)

        # Compute distances from each sample to each cluster center
        distances_to_centroids = self.metric.dist(
            X[:, np.newaxis, :], self.cluster_centers_[np.newaxis, :, :]
        )

        # Determine cluster memberships based on the overlap threshold
        for i in range(n_samples):
            sample_distances = distances_to_centroids[i]
            nearest_cluster = np.argmin(sample_distances)
            cover_matrix[i, nearest_cluster] = True

            # Include overlapping clusters based on relative distance
            for j in range(self.n_clusters):
                if j != nearest_cluster:
                    relative_distance = sample_distances[j] / sample_distances[nearest_cluster]
                    if relative_distance <= 1 + self.overlap_threshold:
                        cover_matrix[i, j] = True
        return cover_matrix

    def fit_transform(self, X, y=None):
        """
        Fit the model to X and compute the cover matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        cover_matrix : ndarray of shape (n_samples, n_clusters)
            Boolean matrix indicating the clusters each sample belongs to.
        """
        return self.fit(X).transform(X)
