"""Paper-quality figures for Section 5.8 counting table.

Three standalone drawing functions, each rendering onto a provided
matplotlib Axes:

1. draw_undirected_graph  — edges + coordination-colored vertices
2. draw_reference_orientation — edges with B1 direction arrows
3. draw_ice_state — spin-colored edges with charge annotations
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from matplotlib.collections import LineCollection

from src.viz.lattice_drawing import (
    COORD_COLORS,
    DEFAULT_EDGE_COLOR,
    coord_color,
    compute_layout_bounds,
    edge_segments,
    edge_segments_periodic,
)
from src.viz.periodic_edges import classify_edges

# Colors
_BLUE = "#3498db"
_RED = "#e74c3c"
_EDGE_GRAY = "#7f8c8d"
_PERIODIC_GRAY = "#b0b0b0"
_ARROW_BLACK = "#444444"
_CHARGE_POS_COLOR = "#d35400"   # dark orange for +1 charges
_CHARGE_NEG_COLOR = "#7d3c98"   # dark purple for -1 charges
_VIOLATION_COLOR = "#e74c3c"


# ── Helpers ──────────────────────────────────────────────────────────

def _convention1_flip(edge_dir):
    """Check if a periodic edge's min-image direction needs flipping for Convention 1.

    Convention 1:
      - Horizontal edges → left-to-right (+x)
      - Vertical edges → bottom-to-top (+y)
      - Diagonal edges → lower-index to higher-index (no change)

    Parameters
    ----------
    edge_dir : array-like (2,)
        Min-image direction from u toward v (from stub geometry).

    Returns
    -------
    needs_flip : bool
        True if the edge direction must be negated to match Convention 1.
    """
    dx, dy = float(edge_dir[0]), float(edge_dir[1])
    abs_dx, abs_dy = abs(dx), abs(dy)
    if abs_dx > abs_dy * 1.5:  # predominantly horizontal
        return dx < 0  # flip if pointing left
    elif abs_dy > abs_dx * 1.5:  # predominantly vertical
        return dy < 0  # flip if pointing down
    else:  # diagonal — u < v convention matches Convention 1
        return False


def _get_periodic_info(lat):
    """Return (interior_segs, periodic_stubs) or None for open BC."""
    if lat.boundary != "periodic":
        return None
    a1 = lat.unit_cell.a1
    a2 = lat.unit_cell.a2
    interior_segs, periodic_stubs = edge_segments_periodic(
        lat.positions, lat.edge_list, a1, a2,
        lat.nx_size, lat.ny_size, lat.boundary,
    )
    return interior_segs, periodic_stubs


def _get_classify_info(lat):
    """Return (interior_edges, periodic_edges, is_periodic) or None."""
    if lat.boundary != "periodic":
        return None
    a1 = lat.unit_cell.a1
    a2 = lat.unit_cell.a2
    return classify_edges(
        lat.positions, lat.edge_list, a1, a2,
        lat.nx_size, lat.ny_size, lat.boundary,
    )


def _set_ax_bounds(ax, positions, padding=0.5):
    """Set axis limits, equal aspect, no ticks."""
    xmin, xmax, ymin, ymax = compute_layout_bounds(positions, padding)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_direction_arrow(ax, mid, direction, scale_length, color, lw=1.2):
    """Draw an arrow at *mid* pointing along *direction*.

    Parameters
    ----------
    mid : array-like (2,)
    direction : array-like (2,) — unit direction of flow
    scale_length : float — physical edge length used for arrow sizing
    color : str
    lw : float
    """
    dx, dy = float(direction[0]), float(direction[1])
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-10:
        return
    scale = 0.12 * scale_length
    dx_n = dx / length * scale
    dy_n = dy / length * scale
    ax.annotate(
        "",
        xy=(mid[0] + dx_n, mid[1] + dy_n),
        xytext=(mid[0] - dx_n, mid[1] - dy_n),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
        zorder=4,
    )


# ── 1. Undirected graph ─────────────────────────────────────────────

def draw_undirected_graph(ax, lat):
    """Draw the lattice graph: edges + coordination-colored vertices.

    Open BC: all edges solid gray.
    Periodic BC: interior edges solid gray, wrap-around edges dashed.
    Vertices colored by coordination number with white outline.
    """
    pos = lat.positions
    edges = lat.edge_list
    coord = lat.coordination
    use_periodic = lat.boundary == "periodic"

    if use_periodic:
        pinfo = _get_periodic_info(lat)
        interior_segs, periodic_stubs = pinfo

        # Interior edges — solid gray
        int_lines = [np.column_stack((xs, ys)) for xs, ys, _ in interior_segs]
        if int_lines:
            lc = LineCollection(int_lines, colors=_EDGE_GRAY,
                                linewidths=1.0, zorder=1)
            ax.add_collection(lc)

        # Periodic stubs — dashed lighter gray
        per_lines = []
        for xs_u, ys_u, xs_v, ys_v, _ in periodic_stubs:
            per_lines.append(np.column_stack((xs_u, ys_u)))
            per_lines.append(np.column_stack((xs_v, ys_v)))
        if per_lines:
            lc_per = LineCollection(per_lines, colors=_PERIODIC_GRAY,
                                    linewidths=1.0, linestyles="dashed",
                                    zorder=1)
            ax.add_collection(lc_per)
    else:
        segs = edge_segments(pos, edges)
        lines = [np.column_stack((xs, ys)) for xs, ys in segs]
        if lines:
            lc = LineCollection(lines, colors=_EDGE_GRAY,
                                linewidths=1.0, zorder=1)
            ax.add_collection(lc)

    # Vertices colored by coordination
    unique_z = np.unique(coord)
    vertex_size = max(8, 20 - len(edges) // 20)
    for z in unique_z:
        mask = coord == z
        ax.scatter(pos[mask, 0], pos[mask, 1], c=coord_color(z),
                   s=vertex_size, zorder=2, label=f"z={z}",
                   edgecolors="white", linewidths=0.3)

    ax.legend(loc="upper right", markerscale=0.9, framealpha=0.7, fontsize=8)
    _set_ax_bounds(ax, pos)


# ── 2. Reference B1 orientation ─────────────────────────────────────

def draw_reference_orientation(ax, lat):
    """Draw the lattice with B1 direction arrows on every edge.

    Convention: arrow points from lower-index vertex (tail) to
    higher-index vertex (head), matching B1[tail,e]=-1, B1[head,e]=+1.
    """
    pos = lat.positions
    edges = lat.edge_list
    use_periodic = lat.boundary == "periodic"

    if use_periodic:
        pinfo = _get_periodic_info(lat)
        interior_segs, periodic_stubs = pinfo
        cinfo = _get_classify_info(lat)
        interior_edges, periodic_edges, _ = cinfo

        # Interior edges — solid dark gray lines
        int_lines = [np.column_stack((xs, ys)) for xs, ys, _ in interior_segs]
        if int_lines:
            lc = LineCollection(int_lines, colors=_ARROW_BLACK,
                                linewidths=1.0, zorder=1)
            ax.add_collection(lc)

        # Periodic stubs — dashed dark gray
        per_lines = []
        for xs_u, ys_u, xs_v, ys_v, _ in periodic_stubs:
            per_lines.append(np.column_stack((xs_u, ys_u)))
            per_lines.append(np.column_stack((xs_v, ys_v)))
        if per_lines:
            lc_per = LineCollection(per_lines, colors=_PERIODIC_GRAY,
                                    linewidths=1.0, linestyles="dashed",
                                    zorder=1)
            ax.add_collection(lc_per)

        # Arrows on interior edges: tail(u) → head(v) with u < v
        for u, v, edge_idx in interior_edges:
            pu, pv = pos[u], pos[v]
            mid = 0.5 * (pu + pv)
            direction = pv - pu  # u < v by edge_list convention
            edge_len = np.linalg.norm(direction)
            _draw_direction_arrow(ax, mid, direction, edge_len, "black")

        # Arrows on periodic stubs: Convention 1 direction
        # (horizontal → +x, vertical → +y, diagonal → u<v).
        for pe in periodic_edges:
            su_start, su_end = pe.stub_u
            # stub_u direction = min-image from u toward v
            edge_dir = np.array(su_end) - np.array(su_start)
            edge_len = np.linalg.norm(edge_dir) * 2

            # Apply Convention 1: flip axis-aligned edges to +x or +y
            if _convention1_flip(edge_dir):
                edge_dir = -edge_dir

            mid_u = 0.5 * (np.array(su_start) + np.array(su_end))
            _draw_direction_arrow(ax, mid_u, edge_dir, edge_len, "black")

            sv_start, sv_end = pe.stub_v
            mid_v = 0.5 * (np.array(sv_start) + np.array(sv_end))
            _draw_direction_arrow(ax, mid_v, edge_dir, edge_len, "black")
    else:
        # Open BC: solid dark gray lines
        segs = edge_segments(pos, edges)
        lines = [np.column_stack((xs, ys)) for xs, ys in segs]
        if lines:
            lc = LineCollection(lines, colors=_ARROW_BLACK,
                                linewidths=1.0, zorder=1)
            ax.add_collection(lc)

        # Arrow at midpoint of each edge: lower-index → higher-index
        for u, v in edges:
            pu, pv = pos[u], pos[v]
            mid = 0.5 * (pu + pv)
            direction = pv - pu  # u < v by edge_list convention
            edge_len = np.linalg.norm(direction)
            _draw_direction_arrow(ax, mid, direction, edge_len, "black")

    # Vertices: small black dots
    ax.scatter(pos[:, 0], pos[:, 1], c="black", s=8, zorder=3)
    _set_ax_bounds(ax, pos)


# ── 3. Ice state ────────────────────────────────────────────────────

def draw_ice_state(ax, lat, sigma, B1):
    """Draw an ice-rule state with spin-colored edges and charge annotations.

    Blue (#3498db) for sigma=+1, red (#e74c3c) for sigma=-1.
    Arrows at edge midpoints show spin flow direction.
    Odd-degree vertices annotated with +-1 charge labels.
    """
    pos = lat.positions
    edges = lat.edge_list
    coord = lat.coordination
    use_periodic = lat.boundary == "periodic"

    if use_periodic:
        cinfo = _get_classify_info(lat)
        interior_edges, periodic_edges, _ = cinfo
        _draw_edges_periodic(ax, sigma, pos, interior_edges, periodic_edges)
    else:
        _draw_edges_open(ax, sigma, pos, edges)

    # Vertices: small black dots
    ax.scatter(pos[:, 0], pos[:, 1], c="black", s=12, zorder=3)

    # Charge annotations
    charge = np.asarray(B1 @ sigma.astype(np.float64)).ravel()
    target = coord.astype(np.float64) % 2
    is_violating = np.abs(np.abs(charge) - target) > 0.5

    for vi in range(len(coord)):
        q = int(round(charge[vi]))
        if q == 0:
            continue
        is_viol = is_violating[vi]
        if is_viol:
            txt_color = _VIOLATION_COLOR
        elif q > 0:
            txt_color = _CHARGE_POS_COLOR  # orange for +1
        else:
            txt_color = _CHARGE_NEG_COLOR  # purple for -1
        txt_weight = "bold" if is_viol else "bold"
        txt_size = 7.5
        ax.annotate(
            f"{q:+d}",
            xy=(pos[vi, 0], pos[vi, 1]),
            xytext=(0, 5), textcoords="offset points",
            fontsize=txt_size, fontweight=txt_weight, color=txt_color,
            ha="center", va="bottom", zorder=6,
        )

    _set_ax_bounds(ax, pos)


# ── Edge drawing helpers for ice state ───────────────────────────────

def _draw_edges_open(ax, sigma, positions, edge_list):
    """Draw all edges as solid spin-colored lines with arrows."""
    lines_plus, lines_minus = [], []
    for j, (u, v) in enumerate(edge_list):
        seg = np.array([[positions[u, 0], positions[u, 1]],
                        [positions[v, 0], positions[v, 1]]])
        if sigma[j] > 0:
            lines_plus.append(seg)
        else:
            lines_minus.append(seg)

    if lines_plus:
        ax.add_collection(LineCollection(lines_plus, colors=_BLUE,
                                         linewidths=1.5, zorder=1))
    if lines_minus:
        ax.add_collection(LineCollection(lines_minus, colors=_RED,
                                         linewidths=1.5, zorder=1))

    for j, (u, v) in enumerate(edge_list):
        _draw_spin_arrow_open(ax, positions, u, v, sigma[j])


def _draw_edges_periodic(ax, sigma, positions, interior_edges, periodic_edges):
    """Draw interior edges solid, periodic edges dashed, both spin-colored."""
    # Interior edges
    lines_plus, lines_minus = [], []
    for u, v, edge_idx in interior_edges:
        seg = np.array([[positions[u, 0], positions[u, 1]],
                        [positions[v, 0], positions[v, 1]]])
        if sigma[edge_idx] > 0:
            lines_plus.append(seg)
        else:
            lines_minus.append(seg)

    if lines_plus:
        ax.add_collection(LineCollection(lines_plus, colors=_BLUE,
                                         linewidths=1.5, zorder=1))
    if lines_minus:
        ax.add_collection(LineCollection(lines_minus, colors=_RED,
                                         linewidths=1.5, zorder=1))

    for u, v, edge_idx in interior_edges:
        _draw_spin_arrow_open(ax, positions, u, v, sigma[edge_idx])

    # Periodic stubs — color by Convention 1 sigma (flip on wrap edges
    # where code B1 disagrees with Convention 1).
    stubs_plus, stubs_minus = [], []
    for pe in periodic_edges:
        su_start, su_end = pe.stub_u
        sv_start, sv_end = pe.stub_v
        seg_u = np.array([[su_start[0], su_start[1]],
                          [su_end[0], su_end[1]]])
        seg_v = np.array([[sv_start[0], sv_start[1]],
                          [sv_end[0], sv_end[1]]])
        edge_dir = np.array(su_end) - np.array(su_start)
        effective_sigma = sigma[pe.edge_index]
        if _convention1_flip(edge_dir):
            effective_sigma = -effective_sigma
        if effective_sigma > 0:
            stubs_plus.extend([seg_u, seg_v])
        else:
            stubs_minus.extend([seg_u, seg_v])

    if stubs_plus:
        ax.add_collection(LineCollection(stubs_plus, colors=_BLUE,
                                         linewidths=1.5, linestyles="dashed",
                                         zorder=1))
    if stubs_minus:
        ax.add_collection(LineCollection(stubs_minus, colors=_RED,
                                         linewidths=1.5, linestyles="dashed",
                                         zorder=1))

    # Arrows on periodic stubs: Convention 1 direction with spin flip.
    # For wrap edges where Convention 1 disagrees with code B1 (u<v),
    # we flip the effective sigma so colors are consistent with the
    # reference orientation figure.
    for pe in periodic_edges:
        su_start, su_end = pe.stub_u
        edge_dir = np.array(su_end) - np.array(su_start)
        edge_len = np.linalg.norm(edge_dir) * 2
        spin_val = sigma[pe.edge_index]

        # Apply Convention 1: flip axis-aligned edges to +x or +y
        flip = _convention1_flip(edge_dir)
        if flip:
            edge_dir = -edge_dir
            # Flipping B1 reference direction flips the effective sigma
            spin_val = -spin_val

        # sigma=-1 reverses flow direction
        arrow_dir = edge_dir if spin_val > 0 else -edge_dir

        color = _BLUE if spin_val > 0 else _RED
        mid_u = 0.5 * (np.array(su_start) + np.array(su_end))
        _draw_direction_arrow(ax, mid_u, arrow_dir, edge_len, color, lw=1.0)

        sv_start, sv_end = pe.stub_v
        mid_v = 0.5 * (np.array(sv_start) + np.array(sv_end))
        _draw_direction_arrow(ax, mid_v, arrow_dir, edge_len, color, lw=1.0)


def _draw_spin_arrow_open(ax, positions, u, v, spin_val):
    """Draw spin arrow at midpoint of an open/interior edge."""
    pu = positions[u]
    pv = positions[v]
    mid = 0.5 * (pu + pv)

    # B1 direction is u→v (u < v). sigma=+1 means flow along B1 direction.
    direction = pv - pu
    if spin_val < 0:
        direction = -direction

    edge_len = np.linalg.norm(pv - pu)
    color = _BLUE if spin_val > 0 else _RED
    _draw_direction_arrow(ax, mid, direction, edge_len, color, lw=1.0)
