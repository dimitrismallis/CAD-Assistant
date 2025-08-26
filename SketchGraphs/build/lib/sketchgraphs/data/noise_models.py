import logging

import numpy as np
import numpy.random as npr
from numpy.linalg import cholesky
from scipy.stats import truncnorm

from sketchgraphs.data.data_utils import parameterize_entity, entity_from_params
from sketchgraphs.data import Sketch
from sketchgraphs.data._entity import Arc, Circle, Line, Point, Entity


def get_distances(
    x: np.ndarray, length_scale: float, squared: bool = False
) -> np.ndarray:
    x2d = np.atleast_2d(x) / length_scale
    squared_distances = (x2d - x2d.T) ** 2
    distances = np.sqrt(squared_distances)
    return distances if squared is False else (distances, squared_distances)


def jitter(arr: np.ndarray, nugget: float = 1e-6) -> np.ndarray:
    m, _ = arr.shape
    return arr + (np.eye(m) * nugget)


def matern(
    x: np.ndarray, length_scale: float, amplitude: float, **kwargs
) -> np.ndarray:
    nu = kwargs.get("nu", 3)
    if nu == 3:
        distances = get_distances(x, length_scale)
        K = (1 + np.sqrt(nu) * distances) * np.exp(-np.sqrt(nu) * distances)
    elif nu == 5:
        distances, squared_distances = get_distances(x, length_scale, squared=True)
        K = (1 + np.sqrt(nu) * distances + nu * squared_distances / 3) * np.exp(
            -np.sqrt(nu) * distances
        )
    else:
        raise NotImplementedError
    return amplitude**2 * jitter(K)


def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180


def radians_to_degrees(radians: float) -> float:
    return radians * 180 / np.pi


class RenderNoise:
    """
    Hand drawn/rendered noise model. Applies noise to plot coordinates which
    are then rendered. The primary use case of this model is to render images
    to train the ImageToPrimative and Critic models.

    Author: rpa
    """

    def __init__(self, sketch: Sketch, resolution: int = 500):
        self.sketch = sketch
        self.entities = sketch.entities
        self.resolution = resolution

        self._get_ranges()
        self._get_chol()

    @property
    def entities(self):
        return self.sketch.entities

    @entities.setter
    def entities(self, value):
        self.sketch.entities = value

    def _get_ranges(self):
        self.min_x = np.inf
        self.max_x = -np.inf
        self.min_y = np.inf
        self.max_y = -np.inf

        def update_x(x: np.ndarray):
            if x > self.max_x:
                self.max_x = x
            if x < self.min_x:
                self.min_x = x

        def update_y(y: np.ndarray):
            if y > self.max_y:
                self.max_y = y
            if y < self.min_y:
                self.min_y = y

        for ent in self.entities.values():
            if isinstance(ent, (Arc, Circle)):
                update_x(-ent.radius)
                update_x(ent.radius)
                update_y(-ent.radius)
                update_y(ent.radius)
                update_x(ent.xCenter)
                update_y(ent.yCenter)
            elif isinstance(ent, Line):
                update_x(ent.start_point[0])
                update_x(ent.end_point[0])
                update_y(ent.start_point[1])
                update_y(ent.end_point[1])
            elif isinstance(ent, Point):
                update_x(ent.x)
                update_y(ent.y)

    def _get_chol(self, ls: float = 0.05, amp: float = 0.002):
        self.scale = 10 * np.sqrt(
            (self.max_x - self.min_x) ** 2 + (self.max_y - self.min_y) ** 2
        )
        self.x = np.linspace(0, 1, self.resolution)
        K = matern(self.x, ls, amp)
        self.cK = cholesky(K)

    def get_line(self, start_x, start_y, end_x, end_y):
        length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        max_idx = int(np.floor((length / self.scale) * self.resolution))

        y = self.scale * self.cK[:max_idx, :max_idx] @ npr.randn(max_idx)
        x = self.x[:max_idx] * self.scale

        theta = np.arctan2(end_y - start_y, end_x - start_x)
        newx = start_x + x * np.cos(theta) - y * np.sin(theta)
        newy = start_y + y * np.cos(theta) + x * np.sin(theta)

        return newx, newy

    def get_arc(
        self,
        center_x: np.ndarray,
        center_y: np.ndarray,
        radius: float,
        start: float,
        end: float,
    ):
        start = np.pi * start / 180
        end = np.pi * end / 180

        if end < start:
            end += 2 * np.pi

        length = np.abs(radius * (end - start))
        max_idx = np.maximum(int(np.floor((length / self.scale) * self.resolution)), 1)

        y = self.scale * self.cK[:max_idx, :max_idx] @ npr.randn(max_idx)

        thetas = np.linspace(start, end, max_idx)
        newx = center_x + (radius + y) * np.cos(thetas)
        newy = center_y + (radius + y) * np.sin(thetas)

        return newx, newy

    def get_circle(self, center_x, center_y, radius):
        gap = npr.rand() * 360
        return self.get_arc(center_x, center_y, radius, gap, gap + 359)

    def get_point(self, center_x, center_y):
        noise_std = 1e-2
        new_x = noise_std * npr.randn() + center_x
        new_y = noise_std * npr.randn() + center_y
        return new_x, new_y


def _trunc_normal_entity_noise(entity: Entity, std: float = 0.2, max_diff: float = 0.1):
    """
    Applies translation noise drawn from a trunctated normal distribution
    to the parameters of the provided entity.

    Parameters
    ----------
    std : float, optional
        The standard deviation of the truncated normal noise distribution (default 0.2)
    max_diff : float, optional
        Maximum allowed magnitude of change in a parameter value (default 0.1)

    Returns
    ---------
    new_ent : EntityObject
    """
    assert std > 0.0 and max_diff > 0.0

    # Parameterize entity
    params = parameterize_entity(entity)  # TODO: check param val legality
    trial = 0
    while True:
        # Draw noise vector
        noise = truncnorm.rvs(
            a=-max_diff, b=max_diff, scale=std / (2**trial), size=params.size
        )
        new_params = params + noise
        new_ent = entity_from_params(new_params, entity_id=entity.entityId)

        if isinstance(entity, Arc):  # prevent large arc alterations
            trial += 1
            if new_ent is None:  # try again when arc fails
                continue
            actual_new_params = parameterize_entity(new_ent)
            if (np.abs(actual_new_params - params) <= max_diff).all():
                if (
                    np.abs(entity.center_point - new_ent.center_point) <= max_diff
                ).all():
                    break
        else:
            break
    return new_ent


def noisify_sketch_ents(sketch: Sketch, std: float = 0.2, max_diff: float = 0.1):
    """Inject noise into the entity parameters of the given sketch.

    The given sketch is assumed to already be normalized. The sketch is modified
    in place.

    Parameters
    ----------
    sketch : Sketch
        The sketch whose entities are to be noisified
    std : float (see `_trunc_normal_entity_noise`)
    max_diff : float (see `_trunc_normal_entity_noise`)

    Returns
    -------
    None
    """
    for ent_key, ent in sketch.entities.items():
        new_ent = _trunc_normal_entity_noise(ent, std, max_diff)
        new_ent.isConstruction = ent.isConstruction
        sketch.entities[ent_key] = new_ent
    return sketch


class RenderPoints:
    def __init__(self, sketch: Sketch, resolution: int = 500):
        self.sketch = sketch
        self.entities = sketch.entities
        self.resolution = resolution

        self._get_ranges()
        self._get_chol()

    @property
    def entities(self):
        return self.sketch.entities

    @entities.setter
    def entities(self, value):
        self.sketch.entities = value

    def _get_ranges(self):
        self.min_x = np.inf
        self.max_x = -np.inf
        self.min_y = np.inf
        self.max_y = -np.inf

        def update_x(x: np.ndarray):
            if x > self.max_x:
                self.max_x = x
            if x < self.min_x:
                self.min_x = x

        def update_y(y: np.ndarray):
            if y > self.max_y:
                self.max_y = y
            if y < self.min_y:
                self.min_y = y

        for ent in self.entities.values():
            if isinstance(ent, (Arc, Circle)):
                update_x(-ent.radius)
                update_x(ent.radius)
                update_y(-ent.radius)
                update_y(ent.radius)
                update_x(ent.xCenter)
                update_y(ent.yCenter)
            elif isinstance(ent, Line):
                update_x(ent.start_point[0])
                update_x(ent.end_point[0])
                update_y(ent.start_point[1])
                update_y(ent.end_point[1])
            elif isinstance(ent, Point):
                update_x(ent.x)
                update_y(ent.y)

    def _get_chol(self, ls: float = 0.05, amp: float = 0.002):
        self.scale = 10 * np.sqrt(
            (self.max_x - self.min_x) ** 2 + (self.max_y - self.min_y) ** 2
        )
        self.x = np.linspace(0, 1, self.resolution)
        K = matern(self.x, ls, amp)
        self.cK = cholesky(K)

    def get_line(self, start_x, start_y, end_x, end_y):
        # length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        # max_idx = int(np.floor((length / self.scale) * self.resolution))
        max_idx = self.resolution

        newx = np.linspace(start_x, end_x, max_idx)
        newy = np.linspace(start_y, end_y, max_idx)

        noise_x = np.random.uniform(-0.01, 0.01, max_idx - 2)
        noise_y = np.random.uniform(-0.01, 0.01, max_idx - 2)

        # Add noise to the inner points (excluding start and end points)
        newx[1:-1] += noise_x
        newy[1:-1] += noise_y

        return newx, newy

    def get_arc(
        self,
        center_x: np.ndarray,
        center_y: np.ndarray,
        radius: float,
        start: float,
        end: float,
    ):
        start = np.pi * start / 180
        end = np.pi * end / 180

        if end < start:
            end += 2 * np.pi

        # length = np.abs(radius * (end - start))
        # max_idx = np.maximum(int(np.floor((length / self.scale) * self.resolution)), 1)
        max_idx = self.resolution

        thetas = np.linspace(start, end, max_idx)
        newx = center_x + (radius) * np.cos(thetas)
        newy = center_y + (radius) * np.sin(thetas)

        noise_x = np.random.uniform(-0.01, 0.01, max_idx - 2)
        noise_y = np.random.uniform(-0.01, 0.01, max_idx - 2)

        newx[1:-1] += noise_x
        newy[1:-1] += noise_y

        return newx, newy

    def get_circle(self, center_x, center_y, radius):
        return self.get_arc(center_x, center_y, radius, 0, 360)

    def get_point(self, center_x, center_y):
        new_x = center_x
        new_y = center_y
        return new_x, new_y
