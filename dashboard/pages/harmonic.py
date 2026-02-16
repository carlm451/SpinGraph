"""Harmonic mode visualization page.

Visualize individual harmonic eigenvectors on the lattice, with edges colored
by the component value of the selected mode.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__, path="/harmonic", name="Harmonic")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LATTICE_NAMES = [
    "square", "kagome", "shakti", "tetris", "santa_fe",
]

SIZE_CONFIGS = {"XS": (4, 4), "S": (10, 10), "M": (20, 20), "L": (50, 50)}


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
            dbc.Col(
                [
                    html.H2("Harmonic Mode Visualization", className="mb-2 mt-2"),
                    html.P([
                        "Harmonic modes are eigenvectors of the ",
                        html.Strong("Hodge 1-Laplacian L\u2081 = B\u2081\u1d40B\u2081 + B\u2082B\u2082\u1d40"),
                        " with eigenvalue exactly zero. They live in ker(L\u2081) and are ",
                        "simultaneously ",
                        html.Em("divergence-free"),
                        " (B\u2081h = 0, no net flow at any vertex) and ",
                        html.Em("curl-free"),
                        " (B\u2082\u1d40h = 0, zero circulation around every face). "
                        "The number of linearly independent harmonic modes equals "
                        "\u03b2\u2081 (first Betti number), which counts independent "
                        "cycles not bounded by faces.",
                    ], className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                    html.P([
                        "Each mode is an ",
                        html.Strong("edge signal"),
                        " \u2014 a vector with one component per edge (dimension n\u2081). "
                        "The visualization colors each edge by its component value: "
                        "red/blue for positive/negative, intensity for magnitude. "
                        "These signals are the topologically protected channel: "
                        "any polynomial filter p(L\u2081) multiplies them by the scalar "
                        "p(0), leaving their spatial pattern unchanged regardless of "
                        "GNN depth. In the spin-ice correspondence, they represent "
                        "divergence-free spin configurations (ice-rule states) that "
                        "cannot be expressed as boundaries of face currents.",
                    ], className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                    html.P([
                        "For the full mathematical derivation, see ",
                        html.A(
                            "TDL \u2194 Spin Ice Correspondence",
                            href="http://127.0.0.1:8000/tdl-spinice-correspondence.html",
                            target="_blank",
                        ),
                        ".",
                    ], className="text-muted mb-3", style={"fontSize": "0.8rem"}),
                ],
                width=12,
            )
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
                                    id="harmonic-lattice-dropdown",
                                    options=[
                                        {"label": n.replace("_", " ").title(), "value": n}
                                        for n in ALL_LATTICE_NAMES
                                    ],
                                    value="square",
                                    clearable=False,
                                ),
                                html.Label("System size", className="mt-3 mb-1"),
                                dcc.Dropdown(
                                    id="harmonic-size-dropdown",
                                    options=[
                                        {"label": s, "value": s}
                                        for s in SIZE_CONFIGS.keys()
                                    ],
                                    value="XS",
                                    clearable=False,
                                ),
                                html.Label("Face strategy", className="mt-3 mb-1"),
                                dbc.RadioItems(
                                    id="harmonic-strategy-toggle",
                                    options=[
                                        {"label": "All faces", "value": "all"},
                                        {"label": "No faces", "value": "none"},
                                    ],
                                    value="all",
                                    inline=True,
                                ),
                                html.Label("Boundary conditions", className="mt-3 mb-1"),
                                dbc.RadioItems(
                                    id="harmonic-boundary-toggle",
                                    options=[
                                        {"label": "Periodic (torus)", "value": "periodic"},
                                        {"label": "Open (disk)", "value": "open"},
                                    ],
                                    value="periodic",
                                    inline=True,
                                ),
                                html.Hr(),
                                html.Label("Harmonic mode index", className="mt-2 mb-1"),
                                dcc.Slider(
                                    id="harmonic-mode-slider",
                                    min=0,
                                    max=0,
                                    step=1,
                                    value=0,
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                html.Div(
                                    id="harmonic-mode-count",
                                    className="text-muted mt-1",
                                    style={"fontSize": "0.85rem"},
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
                                id="harmonic-lattice-figure",
                                style={"height": "600px"},
                            ),
                            type="circle",
                        ),
                        html.Div(id="harmonic-info-panel", className="mt-3"),
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
    Output("harmonic-mode-slider", "max"),
    Output("harmonic-mode-slider", "value"),
    Output("harmonic-mode-count", "children"),
    Input("harmonic-lattice-dropdown", "value"),
    Input("harmonic-size-dropdown", "value"),
    Input("harmonic-strategy-toggle", "value"),
    Input("harmonic-boundary-toggle", "value"),
)
def update_mode_slider(lattice_name, size_label, face_strategy, boundary):
    """Update the mode slider range based on available harmonic modes."""
    if not lattice_name or not size_label:
        return 0, 0, "No data"

    boundary = boundary or "periodic"

    try:
        from dashboard.data_loader import load_result

        result = load_result(lattice_name, size_label, face_strategy, boundary=boundary)
        harmonic_basis = result.get("harmonic_basis")
        beta_1 = result.get("beta_1", 0)

        if harmonic_basis is not None and hasattr(harmonic_basis, 'shape') and len(harmonic_basis.shape) == 2:
            n_modes = harmonic_basis.shape[1]
        else:
            n_modes = 0

        if n_modes == 0:
            return 0, 0, f"beta_1 = {beta_1}, no harmonic basis stored"

        max_idx = n_modes - 1
        return max_idx, 0, f"beta_1 = {beta_1}, {n_modes} harmonic modes available"

    except Exception as exc:
        return 0, 0, f"Could not load data: {exc}"


@callback(
    Output("harmonic-lattice-figure", "figure"),
    Output("harmonic-info-panel", "children"),
    Input("harmonic-lattice-dropdown", "value"),
    Input("harmonic-size-dropdown", "value"),
    Input("harmonic-strategy-toggle", "value"),
    Input("harmonic-boundary-toggle", "value"),
    Input("harmonic-mode-slider", "value"),
)
def update_harmonic_figure(lattice_name, size_label, face_strategy, boundary, mode_index):
    """Render the lattice with edges colored by the selected harmonic mode."""
    if not lattice_name or not size_label:
        return _make_empty_figure("Select a lattice and size"), html.P("")

    boundary = boundary or "periodic"

    try:
        from dashboard.data_loader import load_result, load_lattice_for_viz
        from dashboard.components.lattice_figure import make_lattice_figure

        result = load_result(lattice_name, size_label, face_strategy, boundary=boundary)
        harmonic_basis = result.get("harmonic_basis")
        beta_1 = result.get("beta_1", 0)

        if harmonic_basis is None or not hasattr(harmonic_basis, 'shape') or len(harmonic_basis.shape) < 2:
            return _make_empty_figure(
                f"No harmonic basis available for {lattice_name} ({size_label}, {face_strategy})"
            ), html.P(f"beta_1 = {beta_1}, but no harmonic eigenvectors were stored.")

        n_modes = harmonic_basis.shape[1]
        if n_modes == 0:
            return _make_empty_figure("beta_1 = 0: no harmonic modes"), html.P("")

        # Clamp mode index
        mode_idx = max(0, min(int(mode_index or 0), n_modes - 1))
        harmonic_vector = harmonic_basis[:, mode_idx]

        # Build lattice geometry
        nx_val, ny_val = SIZE_CONFIGS.get(size_label, (4, 4))
        viz_data = load_lattice_for_viz(lattice_name, nx_val, ny_val, boundary=boundary)

        faces_to_show = viz_data["face_list"] if face_strategy == "all" else None

        display_name = lattice_name.replace("_", " ").title()
        title = f"{display_name} -- Harmonic mode {mode_idx} ({size_label}, {face_strategy})"

        fig = make_lattice_figure(
            positions=viz_data["positions"],
            edges=viz_data["edge_list"],
            coordination=viz_data["coordination"],
            faces=faces_to_show,
            harmonic_vector=harmonic_vector,
            title=title,
            a1=viz_data.get("a1"),
            a2=viz_data.get("a2"),
            nx_size=viz_data.get("nx_size"),
            ny_size=viz_data.get("ny_size"),
            boundary=viz_data.get("boundary"),
        )

        # Info panel
        max_val = float(np.max(harmonic_vector))
        min_val = float(np.min(harmonic_vector))
        norm = float(np.linalg.norm(harmonic_vector))

        info = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6("beta_1", className="card-subtitle text-muted"),
                                        html.H4(str(beta_1), className="card-title text-primary"),
                                    ]
                                ),
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6("Mode index", className="card-subtitle text-muted"),
                                        html.H4(
                                            f"{mode_idx} / {n_modes - 1}",
                                            className="card-title",
                                        ),
                                    ]
                                ),
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6("Max component", className="card-subtitle text-muted"),
                                        html.H4(f"{max_val:.6f}", className="card-title"),
                                    ]
                                ),
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6("Min component", className="card-subtitle text-muted"),
                                        html.H4(f"{min_val:.6f}", className="card-title"),
                                    ]
                                ),
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6("L2 norm", className="card-subtitle text-muted"),
                                        html.H4(f"{norm:.6f}", className="card-title"),
                                    ]
                                ),
                            ),
                            width=2,
                        ),
                    ],
                ),
            ]
        )

        return fig, info

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}"), html.P(f"Error: {exc}")
