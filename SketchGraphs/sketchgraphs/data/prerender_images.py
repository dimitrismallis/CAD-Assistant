import io
import PIL.Image


import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data import data_utils
import os
from sketchgraphs.data import Arc, Circle, Line, Point
from sketchgraphs.data import plotting
import matplotlib.pyplot as plt


def render_sketch(
    sketch: datalib.Sketch,
    ax=None,
    pad: float = 0.1,
    size_pixels: int = 128,
    sketch_extent=None,
    hand_drawn: bool = False,
    return_fig: bool = False,
    return_img=False,
    is_colored=False,
    plot_constraints=False,
    constraint_per_primitive={},
    show_subnodes=False,
    linewidth=1,
    show_borders=False,
    render_points=False,
    render_points_resolution=500,
):
    """Renders the given sketch to a byte buffer.

    Parameters
    ----------
    sketch : datalib.Sketch
        The sketch to render.
    pad : float
        The amount to pad the sketch on each side by, as a fraction of the longest
        side of the unpadded sketch.
    size_pixels : int
        The size of the sketch to render, in pixels.
    sketch_extent : float, optional
        If not None, the size of the bounding box to use for the sketch. Otherwise, this
        is automatically deduced from the sketch.

    Returns
    -------
    bytes
        A byte array representing the PNG encoded image, if rendering was succesful,
        or `None` otherwise.
    """
    # Verify sketch is normalized
    if sketch_extent is None:
        (x0, y0), (x1, y1) = data_utils.get_sketch_bbox(sketch)
        w = x1 - x0
        h = y1 - y0
        sketch_extent = max(w, h)

    # Render sketch
    try:
        fig = plotting.render_sketch(
            sketch,
            ax=ax,
            hand_drawn=hand_drawn,
            is_colored=is_colored,
            plot_constraints=plot_constraints,
            constraint_per_primitive=constraint_per_primitive,
            show_subnodes=show_subnodes,
            linewidth=linewidth,
            show_borders=show_borders,
            render_points=render_points,
            render_points_resolution=render_points_resolution,
        )
    except Exception as exc:
        import pickle
        import random
        import logging

        filename = os.path.abspath(f"error_sketch_{random.randint(0, 10000)}.pkl")
        logging.getLogger(__name__).error(
            f"Error processing sketch! Sketch dumped to {filename}"
        )

        with open(filename, "wb") as f:
            pickle.dump(sketch, f, protocol=4)
        raise

    # Adjust lims according to pad
    curr_lim = sketch_extent / 2
    new_lim = curr_lim + pad

    if ax is None:
        fig.axes[0].set_xlim(-new_lim, new_lim)
        fig.axes[0].set_ylim(-new_lim, new_lim)
    else:
        ax.set_xlim(-new_lim, new_lim)
        ax.set_ylim(-new_lim, new_lim)

    # Verify correct radii
    for ent in sketch.entities.values():
        if hasattr(ent, "radius"):
            if getattr(ent, "radius") == 0:
                return None

    if ax is None:
        fig.set_size_inches(1, 1)
        fig.tight_layout()

    if return_fig is True:
        return fig

    # Save image
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=size_pixels)
    plt.close(fig)

    buffer.seek(0)

    img = PIL.Image.open(buffer, formats=("PNG",))
    if is_colored == False:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    if return_img == True:
        return img

    output_buffer = io.BytesIO()
    img.save(output_buffer, format="PNG", optimize=True)
    return output_buffer.getvalue()


def _get_entity_length(ent):
    if isinstance(ent, Line):
        return np.linalg.norm(ent.end_point - ent.start_point)
    if isinstance(ent, Circle):
        return ent.radius
    if isinstance(ent, Arc):
        return np.linalg.norm(ent.end_point - ent.mid_point) + np.linalg.norm(
            ent.mid_point - ent.start_point
        )
    if isinstance(ent, Point):
        return 0
