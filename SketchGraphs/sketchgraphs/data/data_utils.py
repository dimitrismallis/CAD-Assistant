"""This module contains some utilities for processing the sketch data for
img2cad.
"""

import numpy as np
import matplotlib.pyplot as plt

import sketchgraphs.data as datalib
from sketchgraphs.data import EntityType
from sketchgraphs.data import Arc, Circle, Line, Point




# Positional parameters
POS_PARAMS = {
    EntityType.Arc: ("xCenter", "yCenter"),
    EntityType.Circle: ("xCenter", "yCenter"),
    EntityType.Line: ("pntX", "pntY"),
    EntityType.Point: ("x", "y"),
}


# Scale parameters
SCALE_PARAMS = {
    EntityType.Arc: ("radius",),
    EntityType.Circle: ("radius",),
    EntityType.Line: ("startParam", "endParam"),
    EntityType.Point: (),
}


def _get_entity_bbox(ent):
    """Compute the bounding box for the given entity.

    Parameters
    ----------
    ent : Entity
        The entity of interest for bbox computation

    Returns
    -------
    np.array
        Bounding box of the form `[[x0, y0], [x1, y1]]`
    """

    def get_max_circle_bbox(ent):
        """Get maximum bounding box for an arc or circle."""
        if type(ent) not in [Arc, Circle]:
            raise ValueError("Only applicable to Arc and Circle")
        return np.array(
            [
                [ent.xCenter - ent.radius, ent.yCenter - ent.radius],
                [ent.xCenter + ent.radius, ent.yCenter + ent.radius],
            ]
        )

    def get_relative_quadrant(point, center):
        """Determine relative quadrant for the given point (array-like [x, y])."""
        x = point[0] - center[0]
        y = point[1] - center[1]
        if x >= 0:
            if y >= 0:
                return 1
            else:
                return 4
        else:
            if y >= 0:
                return 2
            else:
                return 3

    if isinstance(ent, Arc):
        x_start, y_start = ent.start_point
        x_end, y_end = ent.end_point
        (x0, y0), (x1, y1) = get_max_circle_bbox(ent)
        # Get start and end relative quadrants
        start_quadrant = get_relative_quadrant(ent.start_point, ent.center_point)
        end_quadrant = get_relative_quadrant(ent.end_point, ent.center_point)
        if ent.clockwise:
            start_quadrant, end_quadrant = end_quadrant, start_quadrant
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        # Get included quadrants
        if start_quadrant < end_quadrant:
            quadrants = list(range(start_quadrant, end_quadrant + 1))
        elif start_quadrant > end_quadrant:
            quadrants = list(range(start_quadrant, 5)) + list(
                range(1, end_quadrant + 1)
            )
        else:
            # Handle case when start & end in same quadrant
            start_ahead = False
            if start_quadrant in [1, 2]:
                if x_start <= x_end:
                    start_ahead = True
            if start_quadrant in [3, 4]:
                if x_start >= x_end:
                    start_ahead = True
            if start_ahead:
                quadrants = list(range(start_quadrant, 5)) + list(
                    range(1, end_quadrant + 1)
                )
            else:
                quadrants = [start_quadrant]

        def has_ordered_quadrants(q1, q2):
            if not (q1 in quadrants and q2 in quadrants):
                return False
            idx1 = quadrants.index(q1)
            idx2 = quadrants.index(q2)
            if idx2 > idx1:
                return True
            return False

        # Replace each coordinate of max bbox as needed
        if not has_ordered_quadrants(4, 1):
            x1 = max(x_start, x_end)
        if not has_ordered_quadrants(1, 2):
            y1 = max(y_start, y_end)
        if not has_ordered_quadrants(2, 3):
            x0 = min(x_start, x_end)
        if not has_ordered_quadrants(3, 4):
            y0 = min(y_start, y_end)
        return np.array([[x0, y0], [x1, y1]])

    if isinstance(ent, Circle):
        return get_max_circle_bbox(ent)
    if isinstance(ent, Line):
        start_x, start_y = ent.start_point
        end_x, end_y = ent.end_point
        return np.array(
            [
                [min(start_x, end_x), min(start_y, end_y)],
                [max(start_x, end_x), max(start_y, end_y)],
            ]
        )
    if isinstance(ent, Point):
        return np.array([[ent.x, ent.y], [ent.x, ent.y]])
    # Return None for unsupported entities
    return None


def get_sketch_bbox(sketch):
    """Compute the bounding box for the sketch.

    The bounding box is only computed based on entities currently supported
    in `_get_entity_bbox` (`Arc`, `Circle`, `Line`, `Point`).
    TODO: raise error if sketch has unsupported entities

    Parameters
    ----------
    sketch : Sketch
        The sketch of interest for bbox computation

    Returns
    -------
    np.array
        Bounding box of the form `[[x0, y0], [x1, y1]]`
    """
    # Get entity bboxes
    bboxes = [_get_entity_bbox(ent) for ent in sketch.entities.values()]
    # Remove any None bboxes
    bboxes = np.array([bbox for bbox in bboxes if bbox is not None])
    if bboxes.size == 0:
        # Return origin as bounding box for empty sketch
        return np.array([[0.0, 0.0], [0.0, 0.0]])
    # Compute overall bbox
    x0, y0 = np.min(bboxes[:, 0, :], axis=0)
    x1, y1 = np.max(bboxes[:, 1, :], axis=0)
    return np.array([[x0, y0], [x1, y1]])


def center_sketch(sketch):
    """Center the sketch's bounding box over the origin.

    The entity parameters of the given sketch are modified in place.

    Parameters
    ----------
    sketch : Sketch
        The sketch to be centered

    Returns
    -------
    None
    """
    # Get bounding box
    (x0, y0), (x1, y1) = get_sketch_bbox(sketch)
    # Compute current offset from origin
    x_offset = np.mean([x0, x1])
    y_offset = np.mean([y0, y1])
    # Modify the sketch's entities
    for ent in sketch.entities.values():
        pos_params = POS_PARAMS.get(ent.type)
        if pos_params is None:
            # Skip unsupported entities
            continue
        for param_id in pos_params:
            this_offset = x_offset if "x" in param_id.lower() else y_offset
            curr_val = getattr(ent, param_id)
            setattr(ent, param_id, curr_val - this_offset)


def rescale_sketch(sketch):
    """Rescale the sketch such that the long axis of the bounding box is one.

    The entity parameters of the given sketch are modified in place. This
    function should only be called on sketches that are centered; an error is
    raised it the given sketch is not.

    Parameters
    ----------
    sketch : Sketch
        The sketch to be rescaled

    Returns
    -------
    float
        The normalizing scale factor. If the sketch is zero-dim,
        -1 is returned instead.
    """

    # Get bounding box
    (x0, y0), (x1, y1) = get_sketch_bbox(sketch)
    # Ensure sketch is already centered
    if not np.isclose(x0, -x1) or not np.isclose(y0, -y1):
        raise ValueError("sketch must be centered before rescaling")
    # Calculate scale factor
    w = x1 - x0
    h = y1 - y0
    factor = max(w, h)
    if factor == 0:
        return -1
    # Modify the sketch's entities
    for ent in sketch.entities.values():
        pos_params = POS_PARAMS.get(ent.type)
        scale_params = SCALE_PARAMS.get(ent.type)
        if pos_params is None:
            # Skip unsupported entities
            continue
        params = pos_params + scale_params  # rescale both types of params
        for param_id in params:
            curr_val = getattr(ent, param_id)
            setattr(ent, param_id, curr_val / factor)
    return factor


def normalize_sketch(sketch):
    """Helper function that both centers and rescales the given sketch in place."""
    center_sketch(sketch)
    scale_factor = rescale_sketch(sketch)
    return scale_factor


def _parameterize_arc(arc: datalib.Arc, sort_parameters=True) -> np.ndarray:
    """Extract parameterization for the given Arc instance."""
    start_point, end_point = arc.start_point, arc.end_point
    is_flipped = False
    if arc.clockwise:
        start_point, end_point = end_point, start_point
        is_flipped = True

    return np.concatenate([start_point, arc.mid_point, end_point]), is_flipped


def _parameterize_circle(circle: datalib.Circle, sort_parameters=True) -> np.ndarray:
    """Extract parameterization for the given Circle instance."""
    return np.append(circle.center_point, circle.radius), False


def _parameterize_line(line: datalib.Line, sort_parameters=True) -> np.ndarray:
    """Extract parameterization for the given Line instance."""
    line_pts = np.array([line.start_point, line.end_point])
    is_flipped = False
    if sort_parameters == True:
        ind_line = np.lexsort((line_pts[:, 0], line_pts[:, 1]))
        line_pts = np.concatenate(line_pts[ind_line])
        if ind_line[0] == 1: is_flipped = True   
    else:
        line_pts = np.concatenate(line_pts)
    return line_pts, is_flipped

    # return np.concatenate([line.start_point, line.end_point])


def _parameterize_point(point: datalib.Point, sort_parameters=True) -> np.ndarray:
    """Extract parameterization for the given Point instance."""
    return np.array([point.x, point.y]), False


NUM_PARAMS = {Arc: 6, Circle: 3, Line: 4, Point: 2}


def parameterize_entity(ent, sort_parameters=True) -> np.array:
    """Extract parameterization for the given entity.

    Only continuous parameters of the entity are included.

    Parameters
    ----------
    ent : Entity
        The entity of interest for parameterization

    Returns
    -------
    np.array
        The entity's parameterization
    """
    param_by_type = {
        Arc: _parameterize_arc,
        Circle: _parameterize_circle,
        Line: _parameterize_line,
        Point: _parameterize_point,
    }
    param_fn = param_by_type.get(type(ent))
    if param_fn is None:
        return None
    return param_fn(ent, sort_parameters)


def _arc_from_params(params, entity_id=None):
    """Instantiate an arc from the given parameterization.
    Implementation from:
    https://stackoverflow.com/questions/52990094/calculate-circle-given-
    3-points-code-explanation
    """
    b, c, d = params[:2], params[2:4], params[4:]

    temp = c[0] ** 2 + c[1] ** 2
    bc = (b[0] ** 2 + b[1] ** 2 - temp) / 2
    cd = (temp - d[0] ** 2 - d[1] ** 2) / 2
    det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

    if abs(det) < 1.0e-10:
        return None

    # Center of circle
    cx = (bc * (c[1] - d[1]) - cd * (b[1] - c[1])) / det
    cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

    radius = ((cx - b[0]) ** 2 + (cy - b[1]) ** 2) ** 0.5

    info = {
        "id": entity_id,
        "center": [cx, cy],
        "radius": radius,
        "startPoint": params[:2],
        "endPoint": params[-2:],
    }
    return Arc.from_info(info)


def _circle_from_params(params, entity_id=None):
    """Instantiate a circle from the given parameterization."""
    return Circle(entity_id, xCenter=params[0], yCenter=params[1], radius=params[2])


def _line_from_params(params, entity_id=None):
    """Instantiate a line from the given parameterization."""
    info = {"id": entity_id, "startPoint": params[:2], "endPoint": params[2:]}
    return Line.from_info(info)


def _point_from_params(params, entity_id=None):
    """Instantiate a point from the given parameterization"""
    return Point(entity_id, x=params[0], y=params[1])


def entity_from_params(params, entity_id: str = None):
    """Instantiate an entity from the given parameterization.

    The length of params uniquely determines the target entity type.

    Parameters
    ----------
    params : np.array
        The entity's parameterization
    entity_id : str, optional
        Optional string specifying the id of the entity to create.

    Returns
    -------
    Entity
        The entity instance corresponding to the input parameterization
    """
    entity_by_num_params = {
        NUM_PARAMS[Arc]: _arc_from_params,
        NUM_PARAMS[Circle]: _circle_from_params,
        NUM_PARAMS[Line]: _line_from_params,
        NUM_PARAMS[Point]: _point_from_params,
    }

    # print(params, len(params))
    ent_build_fn = entity_by_num_params.get(len(params))
    if ent_build_fn is None:
        raise ValueError("Unsupported number of parameters")
    return ent_build_fn(params, entity_id)


# Minimum and maximum parameter values following normalization
MIN_VAL = -0.5
MAX_VAL = 0.5


def quantize_params(params: np.ndarray, n_bins):
    """Convert params in [MIN_VAL, MAX_VAL] to discrete values in [0, n_bins-1].

    Parameters
    ----------
    params : np.array
        The parameters to be quantized
    n_bins : int
        The number of bins

    Returns
    -------
    np.array
        The quantized parameter bins
    """
    min_val, max_val = MIN_VAL, MAX_VAL
    params = np.around(params, decimals=10)
    if (params < min_val).any() or (params > max_val).any():
        raise ValueError(
            "Parameters must be in [%f, %f]. Got [%f, %f]."
            % (min_val, max_val, np.min(params), np.max(params))
        )
    params_quantized = (params - min_val) / (max_val - min_val) * n_bins
    params_quantized = params_quantized.astype("int32")
    # Handle max_val edge case
    params_quantized[params_quantized == n_bins] -= 1
    return params_quantized


def dequantize_params(params, n_bins):
    """Convert quantized parameters to floats in range [MIN_VAL, MAX_VAL].

    Parameters
    ----------
    params : array-like
        The parameters to be dequantized
    n_bins : int
        The number of bins

    Returns
    -------
    np.array
        The dequantized parameter values
    """
    if isinstance(params, list):
        params = np.array(params)
    min_val, max_val = MIN_VAL, MAX_VAL
    if (
        (params < 0).any()
        or (params > (n_bins - 1)).any()
        or not np.issubdtype(params.dtype, np.integer)
    ):
        raise ValueError("Invalid quantized params")
    params = params.astype("float32") + 0.5  # center of each bin
    params = params / n_bins * (max_val - min_val) + min_val
    return params
