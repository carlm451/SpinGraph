"""Eigenvalue histogram page.

Interactive exploration of Laplacian eigenvalue spectra for individual lattices
or overlaid comparisons across the full zoo.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__, path="/spectra", name="Spectra")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LATTICE_NAMES = [
    "square", "kagome", "shakti", "tetris", "santa_fe",
]

LAPLACIAN_OPTIONS = [
    {"label": "L0 (graph Laplacian)", "value": "L0_eigenvalues"},
    {"label": "L1 (Hodge 1-Laplacian)", "value": "L1_eigenvalues"},
    {"label": "L1_down (lower)", "value": "L1_down_eigenvalues"},
    {"label": "L1_up (upper)", "value": "L1_up_eigenvalues"},
]


def _make_empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#95a5a6"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(html.H2("Eigenvalue Spectra", className="mb-3 mt-2"), width=12)
        ),
        dbc.Row(
            [
                # Controls
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Controls", className="card-title"),
                                html.Label("Lattice type", className="mt-2 mb-1"),
                                dcc.Dropdown(
                                    id="spectra-lattice-dropdown",
                                    options=[
                                        {"label": n.replace("_", " ").title(), "value": n}
                                        for n in ALL_LATTICE_NAMES
                                    ],
                                    value="square",
                                    clearable=False,
                                ),
                                html.Label("System size", className="mt-3 mb-1"),
                                dcc.Dropdown(
                                    id="spectra-size-dropdown",
                                    options=[],  # populated by callback
                                    value=None,
                                    clearable=False,
                                ),
                                html.Label("Laplacian", className="mt-3 mb-1"),
                                dcc.Dropdown(
                                    id="spectra-laplacian-dropdown",
                                    options=LAPLACIAN_OPTIONS,
                                    value="L1_eigenvalues",
                                    clearable=False,
                                ),
                                html.Label("Face strategy", className="mt-3 mb-1"),
                                dbc.RadioItems(
                                    id="spectra-strategy-toggle",
                                    options=[
                                        {"label": "All faces", "value": "all"},
                                        {"label": "No faces", "value": "none"},
                                    ],
                                    value="all",
                                    inline=True,
                                ),
                                html.Label("Boundary conditions", className="mt-3 mb-1"),
                                dbc.RadioItems(
                                    id="spectra-boundary-toggle",
                                    options=[
                                        {"label": "Periodic (torus)", "value": "periodic"},
                                        {"label": "Open (disk)", "value": "open"},
                                    ],
                                    value="periodic",
                                    inline=True,
                                ),
                                html.Hr(),
                                dbc.Checklist(
                                    id="spectra-overlay-toggle",
                                    options=[
                                        {"label": "Overlay all lattices", "value": "overlay"},
                                    ],
                                    value=[],
                                    switch=True,
                                ),
                                dbc.Checklist(
                                    id="dos-scaling-toggle",
                                    options=[
                                        {"label": "DOS size convergence", "value": "dos"},
                                    ],
                                    value=[],
                                    switch=True,
                                    className="mt-2",
                                ),
                            ]
                        ),
                    ),
                    width=3,
                ),
                # Figure + info
                dbc.Col(
                    [
                        dcc.Loading(
                            dcc.Graph(
                                id="spectra-figure",
                                style={"height": "550px"},
                            ),
                            type="circle",
                        ),
                        html.Div(id="spectra-info-panel", className="mt-3"),
                        html.Div(id="dos-scaling-section"),
                    ],
                    width=9,
                ),
            ],
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("spectra-size-dropdown", "options"),
    Output("spectra-size-dropdown", "value"),
    Input("spectra-lattice-dropdown", "value"),
    Input("spectra-boundary-toggle", "value"),
)
def update_size_options(lattice_name, boundary):
    """Populate size dropdown based on available precomputed results."""
    if not lattice_name:
        return [], None

    boundary = boundary or "periodic"

    try:
        from dashboard.data_loader import get_available_sizes
        sizes = get_available_sizes(lattice_name, boundary=boundary)
    except Exception:
        sizes = ["XS", "S"]

    options = [{"label": s, "value": s} for s in sizes]
    default = sizes[0] if sizes else None
    return options, default


@callback(
    Output("spectra-figure", "figure"),
    Output("spectra-info-panel", "children"),
    Input("spectra-lattice-dropdown", "value"),
    Input("spectra-size-dropdown", "value"),
    Input("spectra-laplacian-dropdown", "value"),
    Input("spectra-strategy-toggle", "value"),
    Input("spectra-boundary-toggle", "value"),
    Input("spectra-overlay-toggle", "value"),
)
def update_spectra(lattice_name, size_label, laplacian_key, face_strategy, boundary, overlay_toggle):
    """Render eigenvalue histogram or overlay plot."""
    overlay_on = "overlay" in (overlay_toggle or [])
    boundary = boundary or "periodic"

    if not size_label:
        return _make_empty_figure("Select a size to view spectra"), html.P("")

    try:
        from dashboard.data_loader import load_result, get_lattice_names
        from dashboard.components.spectrum_figure import (
            make_eigenvalue_histogram,
            make_spectral_overlay,
        )

        if overlay_on:
            # Overlay mode: load all available lattices at this size/strategy
            results_dict = {}
            try:
                names = get_lattice_names()
            except Exception:
                names = ALL_LATTICE_NAMES

            for name in names:
                try:
                    result = load_result(name, size_label, face_strategy, boundary=boundary)
                    display_name = name.replace("_", " ").title()
                    results_dict[display_name] = result
                except Exception:
                    continue

            if not results_dict:
                return _make_empty_figure("No data for overlay"), html.P("No results found.")

            laplacian_label = laplacian_key.replace("_eigenvalues", "").upper().replace("_", " ")
            title = f"Spectral Overlay -- {laplacian_label} ({face_strategy} faces, size {size_label})"
            fig = make_spectral_overlay(results_dict, key=laplacian_key, title=title)

            info = html.Div(
                [
                    dbc.Alert(
                        f"Showing {len(results_dict)} lattices overlaid at size {size_label}, "
                        f"strategy '{face_strategy}', Laplacian: {laplacian_label}.",
                        color="info",
                        className="mt-2",
                    )
                ]
            )
            return fig, info

        else:
            # Single lattice mode
            result = load_result(lattice_name, size_label, face_strategy, boundary=boundary)
            eigenvalues = result.get(laplacian_key)

            if eigenvalues is None:
                return _make_empty_figure(
                    f"No {laplacian_key} data for {lattice_name} ({size_label}, {face_strategy})"
                ), html.P("Eigenvalue data not available.")

            eigenvalues = np.asarray(eigenvalues, dtype=float)
            laplacian_label = laplacian_key.replace("_eigenvalues", "").upper().replace("_", " ")
            display_name = lattice_name.replace("_", " ").title()
            title = f"{display_name} -- {laplacian_label} spectrum ({size_label}, {face_strategy})"
            fig = make_eigenvalue_histogram(eigenvalues, title=title)

            # Info panel
            n_zero = int(np.sum(np.abs(eigenvalues) < 1e-10))
            beta_1 = result.get("beta_1", "--")
            spectral_gap = result.get(f"{laplacian_key.replace('_eigenvalues', '')}_spectral_gap")
            if spectral_gap is None:
                # Try alternate key patterns
                if laplacian_key == "L0_eigenvalues":
                    spectral_gap = result.get("L0_spectral_gap")
                elif laplacian_key == "L1_eigenvalues":
                    spectral_gap = result.get("L1_spectral_gap")

            info_items = [
                html.Div(
                    [
                        html.H4(
                            f"beta_1 = {beta_1}",
                            className="text-primary d-inline me-4",
                        ),
                        html.Span(
                            f"Zero modes in {laplacian_label}: {n_zero}",
                            className="text-muted",
                        ),
                    ],
                    className="mb-2",
                ),
            ]
            if spectral_gap is not None:
                info_items.append(
                    html.P(
                        f"Spectral gap: {spectral_gap:.6f}",
                        className="text-muted mb-1",
                    )
                )
            info_items.append(
                html.P(
                    f"Total eigenvalues: {len(eigenvalues)} | "
                    f"Max: {float(np.max(eigenvalues)):.4f} | "
                    f"Min nonzero: {float(np.min(eigenvalues[np.abs(eigenvalues) >= 1e-10])):.6f}"
                    if np.any(np.abs(eigenvalues) >= 1e-10) else
                    f"Total eigenvalues: {len(eigenvalues)} (all zero)",
                    className="text-muted",
                    style={"fontSize": "0.85rem"},
                )
            )
            info = html.Div(info_items)

            return fig, info

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}"), html.P(f"Error: {exc}")


@callback(
    Output("dos-scaling-section", "children"),
    Input("spectra-lattice-dropdown", "value"),
    Input("spectra-laplacian-dropdown", "value"),
    Input("spectra-strategy-toggle", "value"),
    Input("spectra-boundary-toggle", "value"),
    Input("dos-scaling-toggle", "value"),
)
def update_dos_scaling(lattice_name, laplacian_key, face_strategy, boundary, dos_toggle):
    """Render DOS convergence plot across all available sizes."""
    if "dos" not in (dos_toggle or []):
        return html.Div()

    if not lattice_name:
        return html.Div()

    boundary = boundary or "periodic"

    try:
        from dashboard.data_loader import load_result, get_available_sizes
        from dashboard.components.spectrum_figure import make_dos_scaling_figure

        sizes = get_available_sizes(lattice_name, boundary=boundary)
        results_by_size = {}
        for size in sizes:
            try:
                results_by_size[size] = load_result(
                    lattice_name, size, face_strategy, boundary=boundary,
                )
            except Exception:
                continue

        if len(results_by_size) < 2:
            return html.P(
                "Not enough sizes with data for this configuration.",
                className="text-muted mt-3",
            )

        display_name = lattice_name.replace("_", " ").title()
        laplacian_label = laplacian_key.replace("_eigenvalues", "").upper().replace("_", " ")
        title = (
            f"DOS Convergence -- {display_name} {laplacian_label} "
            f"({face_strategy} faces, {boundary})"
        )

        fig = make_dos_scaling_figure(
            results_by_size, key=laplacian_key, title=title,
        )

        return html.Div([
            html.Hr(className="mt-4"),
            html.P(
                "Sorted eigenvalues vs normalized index (i/n). "
                "Convergence of curves indicates approach to the "
                "thermodynamic limit integrated density of states. "
                "Only sizes with full spectra are shown (partial spectra "
                "from sparse solvers are excluded).",
                className="text-muted",
                style={"fontSize": "0.85rem"},
            ),
            dcc.Graph(figure=fig, style={"height": "500px"}),
            html.Div(
                [
                    html.H6("Reading the staircase", className="mt-3 mb-2"),
                    html.P([
                        "A flat segment at zero on the y-axis corresponds to ",
                        html.Strong("harmonic modes"),
                        " \u2014 eigenvalues of the Hodge Laplacian L\u2081 that are "
                        "exactly zero. These span ker(L\u2081), whose dimension "
                        "equals the first Betti number \u03b2\u2081. The width of the "
                        "zero plateau as a fraction of the x-axis is \u03b2\u2081/n\u2081, "
                        "the fraction of edge signals that live in the harmonic "
                        "subspace.",
                    ], className="mb-2", style={"fontSize": "0.85rem"}),
                    html.P([
                        "Harmonic signals satisfy L\u2081h = 0: they are "
                        "simultaneously divergence-free and curl-free. Any "
                        "polynomial filter p(L\u2081) leaves them unchanged "
                        "(multiplied by p(0)), so they pass through arbitrarily "
                        "many GNN message-passing layers without attenuation. "
                        "This is the ",
                        html.Strong("topologically protected channel"),
                        " for resisting oversmoothing. Frustrated lattices "
                        "(shakti, tetris) have extensive \u03b2\u2081, meaning a "
                        "finite fraction of all modes are protected \u2014 "
                        "visible as a wide zero plateau in the staircase.",
                    ], className="mb-2", style={"fontSize": "0.85rem"}),
                    html.P([
                        "For the full mathematical treatment (Hodge decomposition, "
                        "spin-ice correspondence, and topological protection theorem), see ",
                        html.A(
                            "TDL \u2194 Spin Ice Correspondence",
                            href="http://127.0.0.1:8000/tdl-spinice-correspondence.html",
                            target="_blank",
                        ),
                        ".",
                    ], className="text-muted", style={"fontSize": "0.8rem"}),
                ],
                className="mt-2 mb-3",
            ),
        ])

    except Exception as exc:
        return html.P(f"Error: {exc}", className="text-danger mt-3")
