"""Plotly figures for spectral analysis of Hodge Laplacians.

Provides:
  - ``make_eigenvalue_histogram``  -- histogram of eigenvalues with zero-mode
    annotation.
  - ``make_spectral_overlay``      -- overlaid normalized spectral densities
    from multiple lattice results for side-by-side comparison.
  - ``make_dos_scaling_figure``    -- spectral distribution function (sorted
    eigenvalues vs i/n) across system sizes for thermodynamic limit convergence.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Default color cycle for overlay plots
# ---------------------------------------------------------------------------

_OVERLAY_COLORS = [
    "#3498db",  # blue
    "#e74c3c",  # red
    "#2ecc71",  # green
    "#9b59b6",  # purple
    "#f39c12",  # orange
    "#1abc9c",  # teal
    "#e67e22",  # dark orange
    "#34495e",  # charcoal
    "#16a085",  # green-sea
    "#c0392b",  # pomegranate
]


# ---------------------------------------------------------------------------
# Single eigenvalue histogram
# ---------------------------------------------------------------------------

def make_eigenvalue_histogram(
    eigenvalues: np.ndarray,
    zero_threshold: float = 1e-10,
    title: Optional[str] = None,
    color: str = "#3498db",
) -> go.Figure:
    """Create a histogram of eigenvalues with zero-mode highlighting.

    Parameters
    ----------
    eigenvalues : np.ndarray
        1-D array of eigenvalues (typically from L0 or L1).
    zero_threshold : float
        Eigenvalues below this value are counted as zero (harmonic modes).
    title : str, optional
        Figure title.
    color : str
        Bar fill color.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    n_zero = int(np.sum(np.abs(eigenvalues) < zero_threshold))
    nonzero = eigenvalues[np.abs(eigenvalues) >= zero_threshold]

    fig = go.Figure()

    # Histogram of all eigenvalues
    fig.add_trace(go.Histogram(
        x=eigenvalues,
        nbinsx=80,
        marker_color=color,
        opacity=0.8,
        name="eigenvalues",
        hovertemplate="bin: %{x:.4f}<br>count: %{y}<extra></extra>",
    ))

    # Vertical line at zero to highlight harmonic modes
    if n_zero > 0:
        y_max_est = len(eigenvalues) / 20  # rough guess for annotation height
        fig.add_vline(
            x=0,
            line=dict(color="#e74c3c", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=0,
            y=y_max_est,
            text=f"zero modes: {n_zero}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e74c3c",
            font=dict(color="#e74c3c", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e74c3c",
            borderwidth=1,
            ax=60,
            ay=-30,
        )

    # Layout
    fig.update_layout(
        title=dict(text=title or "Eigenvalue Distribution", x=0.5, xanchor="center"),
        xaxis=dict(
            title="Eigenvalue",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Count",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.05,
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=50),
    )

    return fig


# ---------------------------------------------------------------------------
# Spectral overlay across multiple lattices
# ---------------------------------------------------------------------------

def make_spectral_overlay(
    results_dict: Dict[str, Dict],
    key: str = "L1_eigenvalues",
    title: Optional[str] = None,
) -> go.Figure:
    """Overlay normalized spectral densities from multiple lattice results.

    Parameters
    ----------
    results_dict : dict of {name: result_dict}
        Keys are display names (e.g. ``'square (all)'``). Values are dicts
        containing at least the array specified by *key* (e.g.
        ``'L1_eigenvalues'``).  Missing keys are silently skipped.
    key : str
        Which eigenvalue array to plot from each result dict.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    color_idx = 0
    for name, result in results_dict.items():
        evals = result.get(key)
        if evals is None:
            continue
        evals = np.asarray(evals, dtype=float)
        if len(evals) == 0:
            continue

        color = _OVERLAY_COLORS[color_idx % len(_OVERLAY_COLORS)]
        color_idx += 1

        # Normalized histogram as step line (probability density)
        counts, bin_edges = np.histogram(evals, bins=80, density=True)
        # Step histogram: repeat each count value for left and right bin edge
        step_x = []
        step_y = []
        for i in range(len(counts)):
            step_x.append(bin_edges[i])
            step_y.append(counts[i])
            step_x.append(bin_edges[i + 1])
            step_y.append(counts[i])

        fig.add_trace(go.Scatter(
            x=step_x,
            y=step_y,
            mode="lines",
            line=dict(color=color, width=2),
            name=name,
            hovertemplate=f"{name}<br>eigenvalue: %{{x:.4f}}<br>density: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=title or f"Spectral Density Overlay ({key})",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Eigenvalue",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Normalized Density",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# DOS convergence across system sizes
# ---------------------------------------------------------------------------

# Light-to-dark colour ramp: smaller systems lighter, larger darker.
_SIZE_COLORS = {
    "XS": "#aed6f1",  # light blue
    "S": "#85c1e9",   # medium-light blue
    "M": "#2e86c1",   # blue
    "L": "#1a5276",   # dark blue
    "XL": "#0b2545",  # very dark blue
}

_SIZE_WIDTHS = {"XS": 1.0, "S": 1.0, "M": 1.5, "L": 2.0, "XL": 2.5}

# Skip XS by default — too few eigenvalues for a meaningful density.
_SIZE_ORDER = ["S", "M", "L", "XL"]


def _expected_dim(key: str, result: Dict) -> Optional[int]:
    """Return the expected full-spectrum length for a given eigenvalue key."""
    if key == "L0_eigenvalues":
        return result.get("n_vertices")
    # L1, L1_down, L1_up all have dimension n_edges
    return result.get("n_edges")


def make_dos_scaling_figure(
    results_by_size: Dict[str, Dict],
    key: str = "L1_eigenvalues",
    title: Optional[str] = None,
) -> go.Figure:
    """Spectral distribution function across system sizes.

    Plots sorted eigenvalues vs normalized index (i/n), where n is the
    matrix dimension (n₀ for L0, n₁ for L1).  The x-axis always spans
    [0, 1], making spectra at different sizes directly comparable.
    Convergence of these curves is the integrated density of states
    approaching its thermodynamic limit.

    Only sizes with a *full* eigenvalue spectrum (len == expected matrix
    dimension) are included; partial spectra from sparse solvers and XS
    are silently skipped.

    Parameters
    ----------
    results_by_size : dict of {size_label: result_dict}
        Keyed by size labels (``'XS'``, ``'S'``, etc.).
    key : str
        Which eigenvalue array to plot (e.g. ``'L1_eigenvalues'``).
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Collect sizes with full spectra
    valid: List[tuple] = []  # (size_label, eigenvalues)
    for size in _SIZE_ORDER:
        if size not in results_by_size:
            continue
        result = results_by_size[size]
        evals = result.get(key)
        if evals is None:
            continue
        evals = np.asarray(evals, dtype=float)
        expected = _expected_dim(key, result)
        if expected is not None and len(evals) != expected:
            continue  # partial spectrum — skip
        if len(evals) < 10:
            continue
        valid.append((size, evals))

    fig = go.Figure()

    if len(valid) < 2:
        fig.add_annotation(
            text=(
                "Need full spectra at 2+ sizes "
                "(partial spectra from sparse solver and XS excluded)"
            ),
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#95a5a6"),
        )
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        return fig

    dim_label = "n₀" if key == "L0_eigenvalues" else "n₁"

    for size, evals in valid:
        color = _SIZE_COLORS.get(size, "#2c3e50")
        width = _SIZE_WIDTHS.get(size, 1.5)

        sorted_evals = np.sort(evals)
        n = len(sorted_evals)
        x_norm = np.arange(1, n + 1) / n  # i/n from 1/n to 1

        label = f"{size} ({dim_label}={n:,})"
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=sorted_evals,
            mode="lines",
            line=dict(color=color, width=width),
            name=label,
            hovertemplate=(
                f"{label}<br>"
                f"i/{dim_label}: %{{x:.4f}}<br>"
                "eigenvalue: %{y:.4f}<extra></extra>"
            ),
        ))

    laplacian_label = key.replace("_eigenvalues", "").upper().replace("_", " ")
    fig.update_layout(
        title=dict(
            text=title or f"Spectral Distribution -- {laplacian_label}",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=f"Eigenvalue index / {dim_label}",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            range=[0, 1],
        ),
        yaxis=dict(
            title="Eigenvalue",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            rangemode="tozero",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode="x unified",
    )

    return fig
