import numpy as np
from numba import njit
from typing import Optional

@njit
def custom_clip_scalar(value: float, min_val: float, max_val: float) -> float:
    """
    Clip a scalar value between a minimum and maximum value.

    Parameters
    ----------
    value : float
        The value to clip.
    min_val : float
        The minimum allowable value.
    max_val : float
        The maximum allowable value.

    Returns
    -------
    float
        The clipped value.
    """
    return max(min_val, min(value, max_val))

class PoincareBallMapper:
    """
    A class representing the Poincaré ball model of hyperbolic space.

    This class provides methods for checking point membership in the ball,
    projecting points onto the ball, performing dilations, and computing
    distances between points in hyperbolic space.

    Parameters
    ----------
    dimension : int
        The dimension of the Poincaré ball.

    Attributes
    ----------
    dimension : int
        The dimension of the Poincaré ball.

    Examples
    --------
    >>> import numpy as np
    >>> from poincare_ball_mapper import PoincareBallMapper
    >>> mapper = PoincareBallMapper(dimension=2)
    >>> point = np.array([0.5, 0.5])
    >>> print(mapper.belongs(point))
    True
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def belongs(self, point: np.ndarray, atol: float = 1e-10) -> bool:
        """
        Check if a point belongs to the Poincaré ball.

        Parameters
        ----------
        point : ndarray of shape (..., dimension)
            The point(s) to check.
        atol : float, optional
            Absolute tolerance for numerical stability (default is 1e-10).

        Returns
        -------
        bool
            True if the point(s) belong to the ball, False otherwise.
        """
        squared_norm = np.sum(point**2, axis=-1)
        return np.all(squared_norm < (1 - atol))

    def projection(self, point: np.ndarray, atol: float = 1e-10) -> np.ndarray:
        """
        Project a point onto the Poincaré ball if it lies outside.

        Parameters
        ----------
        point : ndarray of shape (..., dimension)
            The point(s) to project.
        atol : float, optional
            Absolute tolerance for numerical stability (default is 1e-10).

        Returns
        -------
        ndarray of shape (..., dimension)
            The projected point(s) on the ball.
        """
        l2_norm = np.linalg.norm(point, axis=-1, keepdims=True)
        scaling_factor = np.ones_like(l2_norm)

        mask = l2_norm >= (1 - atol)
        scaling_factor[mask] = (1 - atol) / l2_norm[mask]

        projected_point = point * scaling_factor
        return projected_point

    def dilation(self, points: np.ndarray, factor: float, atol: float = 1e-10) -> np.ndarray:
        """
        Perform a dilation (scaling) of points in the Poincaré ball.

        Parameters
        ----------
        points : ndarray of shape (n_samples, dimension)
            The point(s) to dilate.
        factor : float
            The scaling factor.
        atol : float, optional
            Absolute tolerance for numerical stability (default is 1e-10).

        Returns
        -------
        ndarray of shape (n_samples, dimension)
            The dilated point(s), projected back into the ball if necessary.
        """
        points = self.projection(points, atol=atol)
        norms = np.linalg.norm(points, axis=-1, keepdims=True)
        new_norms = norms * factor
        new_norms = np.minimum(new_norms, 1 - atol)
        scaled_points = points * (new_norms / norms)
        return scaled_points

    @staticmethod
    @njit
    def distance(point_a: np.ndarray, point_b: np.ndarray, epsilon: float = 1e-6) -> float:
        """
        Compute the hyperbolic distance between two points in the Poincaré ball.

        Parameters
        ----------
        point_a : ndarray of shape (dimension,)
            The first point.
        point_b : ndarray of shape (dimension,)
            The second point.
        epsilon : float, optional
            A small value to prevent numerical issues (default is 1e-6).

        Returns
        -------
        float
            The hyperbolic distance between point_a and point_b.
        """
        # Compute squared norms
        point_a_norm_sq = np.sum(point_a ** 2)
        point_b_norm_sq = np.sum(point_b ** 2)

        # Clip norms to avoid numerical issues
        point_a_norm_sq = custom_clip_scalar(point_a_norm_sq, 0.0, 1.0 - epsilon)
        point_b_norm_sq = custom_clip_scalar(point_b_norm_sq, 0.0, 1.0 - epsilon)

        # Compute norm of the difference
        diff_norm_sq = np.sum((point_a - point_b) ** 2)

        # Compute the hyperbolic distance
        denominator = (1.0 - point_a_norm_sq) * (1.0 - point_b_norm_sq)
        norm_function = 1.0 + 2.0 * diff_norm_sq / denominator
        sqrt_term = np.sqrt(norm_function ** 2 - 1.0)
        distance = np.log(norm_function + sqrt_term)
        return distance
