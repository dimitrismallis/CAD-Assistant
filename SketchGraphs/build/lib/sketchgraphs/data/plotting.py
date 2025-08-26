"""Functions for drawing sketches using matplotlib
This module implements local plotting functionality in order to render Onshape sketches using matplotlib.
"""

import math
from contextlib import nullcontext
import warnings
import matplotlib as mpl
import matplotlib.cbook
import matplotlib.patches
import matplotlib.pyplot as plt
from sketchgraphs.data import noise_models
from sketchgraphs.data._entity import Arc, Circle, Line, Point
import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
plt.rcParams.update({"figure.max_open_warning": 2000})

# Extract colors from multiple tab colormaps
tab20_colors = [plt.cm.tab20(i) for i in range(20)]
tab20b_colors = [plt.cm.tab20b(i) for i in range(20)]
tab20c_colors = [plt.cm.tab20c(i) for i in range(20)]  # We only need 8 more colors

# Combine them to get 48 colors
tab_colors = tab20_colors + tab20b_colors + tab20c_colors + tab20_colors

def get_image_by_constraint(constraint):
    img = Image.open(f"resources/constraint_icons_v2/{constraint.lower()}.png")
    return img


def plot_img_at_point(ax, img, point):
    # Create an OffsetImage
    imagebox = OffsetImage(img, zoom=0.07)  # Adjust 'zoom' to scale your image

    # Create an AnnotationBbox to place the image at the specific point
    ab = AnnotationBbox(imagebox, (point[0], point[1]), frameon=False)

    # Add the AnnotationBbox to the plot
    ax.add_artist(ab)
    return ax


def _get_linestyle(entity):
    return "--" if entity.isConstruction else "-"


def sketch_point(
    ax,
    point: Point,
    color: str = "black",
    show_subnodes: bool = False,
    noisy: noise_models.RenderNoise = None,
    is_colored=False,
    plot_constraints=False,
    constraints=[],
    ent_pos=0.0,
    linewidth=1,
):
    if is_colored == True:
        color = tab_colors[ent_pos]
    if noisy:
        x, y = noisy.get_point(point.x, point.y)
        ax.plot(x, y, c=color, marker="o", markersize=linewidth + 1)
    else:
        ax.plot(point.x, point.y, c=color, marker="o", markersize=linewidth + 1)


def sketch_line(
    ax,
    line: Line,
    color="black",
    show_subnodes=False,
    noisy: noise_models.RenderNoise = None,
    is_colored=False,
    plot_constraints=False,
    constraints=[],
    ent_pos=0.0,
    linewidth=1,
):
    if is_colored == True:
        color = tab_colors[ent_pos]
    start_x, start_y = line.start_point
    end_x, end_y = line.end_point
    if show_subnodes:
        marker = "."
    else:
        marker = None
    if noisy:
        x, y = noisy.get_line(start_x, start_y, end_x, end_y)
        ax.plot(
            x,
            y,
            c=color,
            linestyle=_get_linestyle(line),
            linewidth=linewidth,
            marker=marker,
        )
    else:
        ax.plot(
            (start_x, end_x),
            (start_y, end_y),
            c=color,
            linestyle=_get_linestyle(line),
            linewidth=linewidth,
            marker=marker,
            markersize=2,
        )


def sketch_circle(
    ax,
    circle: Circle,
    color="black",
    show_subnodes=False,
    noisy: noise_models.RenderNoise = None,
    is_colored=False,
    plot_constraints=False,
    constraints=[],
    ent_pos=0.0,
    linewidth=1,
):
    if is_colored == True:
        color = tab_colors[ent_pos]
    if noisy:
        x, y = noisy.get_circle(circle.xCenter, circle.yCenter, circle.radius)
        ax.plot(x, y, color=color, linewidth=linewidth)
    else:
        patch = matplotlib.patches.Circle(
            (circle.xCenter, circle.yCenter),
            circle.radius,
            fill=False,
            linestyle=_get_linestyle(circle),
            color=color,
            linewidth=linewidth,
        )
        ax.add_patch(patch)
    if show_subnodes:
        ax.scatter(circle.xCenter, circle.yCenter, c=color, marker=".", zorder=20, s=4)


def sketch_arc(
    ax,
    arc: Arc,
    color="black",
    show_subnodes=False,
    noisy: noise_models.RenderNoise = None,
    is_colored=False,
    plot_constraints=False,
    constraints=[],
    ent_pos=0.0,
    linewidth=1,
):
    if is_colored == True:
        color = tab_colors[ent_pos]

    angle = math.atan2(arc.yDir, arc.xDir) * 180 / math.pi
    startParam = arc.startParam * 180 / math.pi
    endParam = arc.endParam * 180 / math.pi
    if arc.clockwise:
        startParam, endParam = -endParam, -startParam
    if noisy:
        x, y = noisy.get_arc(
            arc.xCenter, arc.yCenter, arc.radius, angle + startParam, angle + endParam
        )
        ax.plot(x, y, color, linewidth=linewidth)
    else:
        ax.add_patch(
            matplotlib.patches.Arc(
                (arc.xCenter, arc.yCenter),
                2 * arc.radius,
                2 * arc.radius,
                angle=angle,
                theta1=startParam,
                theta2=endParam,
                linestyle=_get_linestyle(arc),
                color=color,
                linewidth=linewidth,
            )
        )
    if show_subnodes:
        ax.scatter(arc.xCenter, arc.yCenter, c=color, marker=".", zorder=40, s=4)
        ax.scatter(*arc.start_point, c=color, marker=".", zorder=40, s=4)
        ax.scatter(*arc.end_point, c=color, marker=".", zorder=40, s=4)


_PLOT_BY_TYPE = {
    Arc: sketch_arc,
    Circle: sketch_circle,
    Line: sketch_line,
    Point: sketch_point,
}


def render_sketch(
    sketch,
    ax=None,
    show_axes=False,
    show_origin=False,
    hand_drawn=False,
    show_subnodes=False,
    show_points=True,
    show_borders=False,
    is_colored=False,
    plot_constraints=False,
    constraint_per_primitive={},
    linewidth=1,
    plot_markers=None,
    render_points=False,
    render_points_resolution=500,
):
    """Renders the given sketch using matplotlib.
    Parameters
    ----------
    sketch : Sketch
        The sketch instance to render
    ax : matplotlib.Axis, optional
        Axis object on which to render the sketch. If None, a new figure is created.
    show_axes : bool
        Indicates whether axis lines should be drawn
    show_origin : bool
        Indicates whether origin point should be drawn
    hand_drawn : bool
        Indicates whether to emulate a hand-drawn appearance
    show_subnodes : bool
        Indicates whether endpoints/centerpoints should be drawn
    show_points : bool
        Indicates whether Point entities should be drawn
    Returns
    -------
    matplotlib.Figure
        If `ax` is not provided, the newly created figure. Otherwise, `None`.
    """
    noisy = None
    if hand_drawn:
        noisy = noise_models.RenderNoise(sketch)
    else:
        if render_points == True:
            noisy = noise_models.RenderPoints(
                sketch, resolution=render_points_resolution
            )
    with nullcontext():
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect="equal")
        else:
            fig = None
        # Eliminate upper and right axes
        if show_borders == False:
            ax.spines["right"].set_color("None")
            ax.spines["top"].set_color("None")
        if not show_axes:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            _ = [line.set_marker("None") for line in ax.get_xticklines()]
            _ = [line.set_marker("None") for line in ax.get_yticklines()]
            # Eliminate lower and left axes
            if show_borders == False:
                ax.spines["left"].set_color("None")
                ax.spines["bottom"].set_color("None")
        if show_origin:
            point_size = mpl.rcParams["lines.markersize"] * 1
            ax.scatter(0, 0, s=point_size, c="black")
        ent_pos = 0
        max_ents = 16
        for ent in sketch.entities.values():
            sketch_fn = _PLOT_BY_TYPE.get(type(ent))
            if isinstance(ent, Point):
                if not show_points:
                    continue
            if sketch_fn is None:
                continue

            constraints = [[], []]
            if plot_constraints == True:
                constraints = [
                    constraint_per_primitive[ent.entityId],
                    constraint_per_primitive[f"{ent.entityId}_pos"],
                ]

                const_types = constraints[0]
                const_pos = constraints[1]
                if len(const_pos) == len(const_types):
                    for i in range(len(const_types)):
                        c = const_types[i]
                        c_pos = const_pos[i]
                        c_img = get_image_by_constraint(c)
                        # ax.plot(line.mid_point[0], line.mid_point[1], c="orange", marker="o")
                        ax = plot_img_at_point(ax, c_img, c_pos)
                        # ax = plot_img_at_point(ax, c_img, line.mid_point)

            sketch_fn(
                ax,
                ent,
                show_subnodes=show_subnodes,
                noisy=noisy,
                is_colored=is_colored,
                plot_constraints=plot_constraints,
                constraints=constraints,
                ent_pos=ent_pos,
                linewidth=linewidth,
            )
            ent_pos += 1

            # if ent_pos == 1:
            #     break
        # Rescale axis limits
        ax.relim()
        ax.autoscale_view()
        return fig


def render_graph(graph, filename, show_node_idxs=False):
    """Renders the given pgv.AGraph to an image file.
    Parameters
    ----------
    graph : pgv.AGraph
        The graph to render
    filename : string
        Where to save the image file
    show_node_idxs: bool
        If true, append node indexes to their labels
    Returns
    -------
    None
    """
    if show_node_idxs:
        for idx, node in enumerate(graph.nodes()):
            node.attr["label"] += " (" + str(idx) + ")"
    graph.layout("dot")
    graph.draw(filename)


__all__ = ["render_sketch", "render_graph"]
