"""Plotly figure and matplotlib thumbnail for ice state visualization.

Renders binary spin configurations as colored arrows on lattice edges:
blue for sigma=+1, red for sigma=-1.
"""
from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from dashboard.components.lattice_figure import (
    COORDINATION_COLORS,
    _color_for_z,
)


_SPIN_PLUS_COLOR = "#3498db"   # blue for sigma = +1
_SPIN_MINUS_COLOR = "#e74c3c"  # red  for sigma = -1
_PERIODIC_ALPHA = 0.4


def _spin_color(s: float) -> str:
    """Return color for spin value."""
    return _SPIN_PLUS_COLOR if s > 0 else _SPIN_MINUS_COLOR


def make_ice_state_figure(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    coordination: np.ndarray,
    sigma: np.ndarray,
    title: Optional[str] = None,
    a1: Optional[Tuple[float, float]] = None,
    a2: Optional[Tuple[float, float]] = None,
    nx_size: Optional[int] = None,
    ny_size: Optional[int] = None,
    boundary: Optional[str] = None,
    show_arrows: bool = True,
) -> go.Figure:
    """Build an interactive Plotly figure of an ice state.

    Edges are colored blue (sigma=+1) or red (sigma=-1). Arrows at edge
    midpoints indicate spin direction: tail->head for +1, head->tail for -1.

    Parameters
    ----------
    positions : (n_vertices, 2)
    edges : list of (u, v)
    coordination : (n_vertices,)
    sigma : (n_edges,) array of +1/-1
    title : optional figure title
    a1, a2, nx_size, ny_size, boundary : periodic edge info
    show_arrows : if True, add arrow annotations at edge midpoints.
        Automatically disabled when n_edges > 200 for performance (Plotly
        annotations are individual SVG elements and become very slow).
    """
    fig = go.Figure()

    # Auto-disable arrows for large lattices (>200 edges → thousands of SVG annotations)
    if show_arrows and len(edges) > 200:
        show_arrows = False

    # Classify edges for periodic rendering
    interior_edges = None
    periodic_edges = None
    use_periodic = (
        boundary == "periodic"
        and a1 is not None
        and a2 is not None
        and nx_size is not None
        and ny_size is not None
    )

    if use_periodic:
        from src.viz.periodic_edges import classify_edges
        interior_edges, periodic_edges, _ = classify_edges(
            positions, edges, a1, a2, nx_size, ny_size, boundary,
        )

    # Draw edges colored by spin
    if interior_edges is not None and periodic_edges is not None:
        # Interior edges: one trace per spin sign for clean legend
        for sign_val, sign_label, color in [
            (1.0, "+1", _SPIN_PLUS_COLOR),
            (-1.0, "-1", _SPIN_MINUS_COLOR),
        ]:
            xs, ys = [], []
            for u, v, idx in interior_edges:
                if sigma[idx] * sign_val > 0:
                    xs.extend([positions[u, 0], positions[v, 0], None])
                    ys.extend([positions[u, 1], positions[v, 1], None])
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"sigma = {sign_label}",
                    legendgroup=f"spin{sign_label}",
                    hoverinfo="skip",
                ))

        # Periodic edge stubs (dashed)
        pxs, pys = [], []
        for pe in periodic_edges:
            su_start, su_end = pe.stub_u
            pxs.extend([su_start[0], su_end[0], None])
            pys.extend([su_start[1], su_end[1], None])
            sv_start, sv_end = pe.stub_v
            pxs.extend([sv_start[0], sv_end[0], None])
            pys.extend([sv_start[1], sv_end[1], None])
        if pxs:
            fig.add_trace(go.Scatter(
                x=pxs, y=pys, mode="lines",
                line=dict(color="#b0b0b0", width=1.5, dash="dash"),
                hoverinfo="skip", showlegend=False,
            ))

        # Arrows on interior edges
        if show_arrows:
            for u, v, idx in interior_edges:
                _add_arrow_annotation(fig, positions, u, v, sigma[idx])

    else:
        # No periodic info: draw all edges
        for sign_val, sign_label, color in [
            (1.0, "+1", _SPIN_PLUS_COLOR),
            (-1.0, "-1", _SPIN_MINUS_COLOR),
        ]:
            xs, ys = [], []
            for i, (u, v) in enumerate(edges):
                if sigma[i] * sign_val > 0:
                    xs.extend([positions[u, 0], positions[v, 0], None])
                    ys.extend([positions[u, 1], positions[v, 1], None])
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"sigma = {sign_label}",
                    legendgroup=f"spin{sign_label}",
                    hoverinfo="skip",
                ))

        if show_arrows:
            for i, (u, v) in enumerate(edges):
                _add_arrow_annotation(fig, positions, u, v, sigma[i])

    # Vertices (small dots, coordination-colored)
    unique_z = sorted(set(int(z) for z in coordination))
    for z in unique_z:
        mask = coordination == z
        idxs = np.where(mask)[0]
        color = _color_for_z(z)
        fig.add_trace(go.Scatter(
            x=positions[idxs, 0], y=positions[idxs, 1],
            mode="markers",
            marker=dict(size=5, color=color, line=dict(width=0.3, color="white")),
            name=f"z = {z}", legendgroup=f"z{z}",
            hoverinfo="text",
            hovertext=[f"v{i} (z={z})" for i in idxs],
        ))

    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center"),
        xaxis=dict(
            scaleanchor="y", scaleratio=1,
            showticklabels=False, showgrid=False, zeroline=False, visible=False,
        ),
        yaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False, visible=False,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="closest",
    )

    return fig


def _add_arrow_annotation(
    fig: go.Figure,
    positions: np.ndarray,
    u: int,
    v: int,
    spin: float,
) -> None:
    """Add a small arrow annotation at the edge midpoint."""
    pu = positions[u]
    pv = positions[v]
    mid = 0.5 * (pu + pv)
    # Direction: tail->head for +1, head->tail for -1
    # B1 convention: tail=u (lower idx), head=v (higher idx)
    if spin > 0:
        dx = pv[0] - pu[0]
        dy = pv[1] - pu[1]
    else:
        dx = pu[0] - pv[0]
        dy = pu[1] - pv[1]

    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-10:
        return
    scale = 0.15 * length
    dx_norm = dx / length * scale
    dy_norm = dy / length * scale

    fig.add_annotation(
        x=mid[0] + dx_norm,
        y=mid[1] + dy_norm,
        ax=mid[0] - dx_norm,
        ay=mid[1] - dy_norm,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=1.5,
        arrowcolor=_spin_color(spin),
    )


def make_ice_state_thumbnail(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    sigma: np.ndarray,
    coordination: np.ndarray,
    a1: Optional[Tuple[float, float]] = None,
    a2: Optional[Tuple[float, float]] = None,
    nx_size: Optional[int] = None,
    ny_size: Optional[int] = None,
    boundary: Optional[str] = None,
    figsize: Tuple[float, float] = (2.5, 2.5),
    dpi: int = 80,
) -> str:
    """Render a small matplotlib thumbnail of an ice state.

    Returns a base64-encoded PNG string suitable for ``html.Img(src=...)``.
    No arrows, just blue/red colored edges + small vertex dots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from src.viz.lattice_drawing import compute_layout_bounds

    use_periodic = (
        boundary == "periodic"
        and a1 is not None
        and a2 is not None
        and nx_size is not None
        and ny_size is not None
    )

    fig_mpl, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if use_periodic:
        from src.viz.lattice_drawing import edge_segments_periodic

        interior_segs, periodic_stubs = edge_segments_periodic(
            positions, edges, a1, a2, nx_size, ny_size, boundary,
        )

        lines_plus, lines_minus = [], []
        for xs, ys, idx in interior_segs:
            seg = np.column_stack((xs, ys))
            if sigma[idx] > 0:
                lines_plus.append(seg)
            else:
                lines_minus.append(seg)

        if lines_plus:
            ax.add_collection(LineCollection(
                lines_plus, colors=_SPIN_PLUS_COLOR, linewidths=1.2, zorder=1))
        if lines_minus:
            ax.add_collection(LineCollection(
                lines_minus, colors=_SPIN_MINUS_COLOR, linewidths=1.2, zorder=1))

        # Periodic stubs: gray dashed
        per_lines = []
        for xs_u, ys_u, xs_v, ys_v, _ in periodic_stubs:
            per_lines.append(np.column_stack((xs_u, ys_u)))
            per_lines.append(np.column_stack((xs_v, ys_v)))
        if per_lines:
            ax.add_collection(LineCollection(
                per_lines, colors="#b0b0b0", linewidths=0.8,
                linestyles="dashed", zorder=1))
    else:
        lines_plus, lines_minus = [], []
        for i, (u, v) in enumerate(edges):
            seg = np.column_stack((
                [positions[u, 0], positions[v, 0]],
                [positions[u, 1], positions[v, 1]],
            ))
            if sigma[i] > 0:
                lines_plus.append(seg)
            else:
                lines_minus.append(seg)

        if lines_plus:
            ax.add_collection(LineCollection(
                lines_plus, colors=_SPIN_PLUS_COLOR, linewidths=1.2, zorder=1))
        if lines_minus:
            ax.add_collection(LineCollection(
                lines_minus, colors=_SPIN_MINUS_COLOR, linewidths=1.2, zorder=1))

    # Small vertex dots
    ax.scatter(positions[:, 0], positions[:, 1], c="black", s=3, zorder=2)

    xmin, xmax, ymin, ymax = compute_layout_bounds(positions, padding=0.3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig_mpl.tight_layout(pad=0.1)

    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_mpl)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
