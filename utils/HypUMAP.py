import numpy as np
import umap
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional

class HyperbolicUMAP(BaseEstimator, TransformerMixin):
    """
    A transformer that embeds data into hyperbolic space using UMAP and projects
    the embeddings onto the Poincaré disk.

    This class wraps UMAP with the 'hyperboloid' output metric and transforms
    the embeddings to the Poincaré disk model of hyperbolic space.

    Parameters
    ----------
    n_neighbors : int, default=89
        The size of local neighborhood (in terms of number of neighboring sample points)
        used for manifold approximation.

    min_dist : float, default=0.45
        The effective minimum distance between embedded points.

    spread : float, default=0.51
        The effective scale of embedded points.

    n_components : int, default=2
        The dimension of the space to embed into.

    metric : str or callable, default='cosine'
        The metric to use to compute distances in high-dimensional space.

    random_state : int or None, default=48
        Random seed used for initialization.

    n_epochs : int or None, default=500
        The number of training epochs to use when optimizing the low-dimensional embedding.

    Attributes
    ----------
    umap_model_ : umap.UMAP
        The fitted UMAP model.

    Examples
    --------
    >>> import numpy as np
    >>> from hyperbolic_umap import HyperbolicUMAP
    >>> X = np.random.rand(100, 10)
    >>> hyper_umap = HyperbolicUMAP()
    >>> X_transformed = hyper_umap.fit_transform(X)
    """

    def __init__(
        self,
        n_neighbors: int = 89,
        min_dist: float = 0.45,
        spread: float = 0.51,
        n_components: int = 2,
        metric: str = 'cosine',
        random_state: Optional[int] = None,
        n_epochs: Optional[int] = 500,
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.umap_model_ = None

    def fit(self, X, y=None):
        """
        Fit the UMAP model to the data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.umap_model_ = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            spread=self.spread,
            n_components=self.n_components,
            n_epochs=self.n_epochs,
            metric=self.metric,
            output_metric='hyperboloid',
            random_state=self.random_state,
        )
        self.umap_model_.fit(X)
        return self

    def transform(self, X):
        """
        Transform X into the Poincaré disk model using the fitted UMAP model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data in the Poincaré disk model.
        """
        if self.umap_model_ is None:
            raise RuntimeError("You must fit the model before transforming data.")

        embedding = self.umap_model_.transform(X)
        # Convert from hyperboloid model to Poincaré disk model
        z = np.sqrt(1 + np.sum(embedding**2, axis=1))
        disk_coords = embedding / (1 + z[:, np.newaxis])
        return disk_coords

    def fit_transform(self, X, y=None):
        """
        Fit the model to X and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data in the Poincaré disk model.
        """
        self.fit(X, y)
        return self.transform(X)
