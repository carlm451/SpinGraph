"""Interactive lattice gallery page.

Visualize lattice geometry with coordination coloring, face shading, and
lattice statistics for any lattice type and system size.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__, path="/gallery", name="Gallery")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZE_CONFIGS = {"XS": (4, 4), "S": (10, 10), "M": (20, 20), "L": (50, 50)}

ALL_LATTICE_NAMES = [
    "square", "kagome", "shakti", "tetris", "santa_fe",
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
            dbc.Col(html.H2("Lattice Gallery", className="mb-3 mt-2"), width=12)
        ),
        dbc.Row(
            [
                # Controls column
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Controls", className="card-title"),
                                html.Label("Lattice type", className="mt-2 mb-1"),
                                dcc.Dropdown(
                                    id="gallery-lattice-dropdown",
                                    options=[
                                        {"label": name.replace("_", " ").title(), "value": name}
                                        for name in ALL_LATTICE_NAMES
                                    ],
                                    value="square",
                                    clearable=False,
                                ),
                                html.Label("System size", className="mt-3 mb-1"),
                                dcc.Dropdown(
                                    id="gallery-size-dropdown",
                                    options=[
                                        {"label": label, "value": label}
                                        for label in SIZE_CONFIGS.keys()
                                    ],
                                    value="XS",
                                    clearable=False,
                                ),
                                html.Label("Face strategy", className="mt-3 mb-1"),
                                dbc.RadioItems(
                                    id="gallery-strategy-toggle",
                                    options=[
                                        {"label": "All faces", "value": "all"},
                                        {"label": "No faces", "value": "none"},
                                    ],
                                    value="all",
                                    inline=True,
                                ),
                                html.Label("Boundary conditions", className="mt-3 mb-1"),
                                dbc.RadioItems(
                                    id="gallery-boundary-toggle",
                                    options=[
                                        {"label": "Periodic (torus)", "value": "periodic"},
                                        {"label": "Open (disk)", "value": "open"},
                                    ],
                                    value="periodic",
                                    inline=True,
                                ),
                            ]
                        ),
                    ),
                    width=3,
                ),
                # Figure column
                dbc.Col(
                    dcc.Loading(
                        dcc.Graph(
                            id="gallery-lattice-figure",
                            style={"height": "600px"},
                        ),
                        type="circle",
                    ),
                    width=6,
                ),
                # Stats column
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Lattice Statistics", className="card-title"),
                                html.Div(id="gallery-stats-panel"),
                            ]
                        ),
                    ),
                    width=3,
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
    Output("gallery-lattice-figure", "figure"),
    Output("gallery-stats-panel", "children"),
    Input("gallery-lattice-dropdown", "value"),
    Input("gallery-size-dropdown", "value"),
    Input("gallery-strategy-toggle", "value"),
    Input("gallery-boundary-toggle", "value"),
)
def update_gallery(lattice_name, size_label, face_strategy, boundary):
    """Rebuild the lattice figure and stats when any control changes."""
    if not lattice_name or not size_label:
        return _make_empty_figure("Select a lattice and size"), html.P("--")

    boundary = boundary or "periodic"

    try:
        from dashboard.data_loader import load_lattice_for_viz, load_result
        from dashboard.components.lattice_figure import make_lattice_figure

        nx_val, ny_val = SIZE_CONFIGS[size_label]

        # Build lattice geometry
        viz_data = load_lattice_for_viz(lattice_name, nx_val, ny_val, boundary=boundary)

        faces_to_show = viz_data["face_list"] if face_strategy == "all" else None

        bc_label = "periodic" if boundary == "periodic" else "open"
        title = f"{lattice_name.replace('_', ' ').title()} ({size_label}, {face_strategy}, {bc_label})"
        fig = make_lattice_figure(
            positions=viz_data["positions"],
            edges=viz_data["edge_list"],
            coordination=viz_data["coordination"],
            faces=faces_to_show,
            title=title,
            a1=viz_data.get("a1"),
            a2=viz_data.get("a2"),
            nx_size=viz_data.get("nx_size"),
            ny_size=viz_data.get("ny_size"),
            boundary=viz_data.get("boundary"),
        )

        # Try to load precomputed spectral results for stats
        stats_children = []
        try:
            result = load_result(lattice_name, size_label, face_strategy, boundary=boundary)
            stats_items = [
                ("Vertices (n0)", result.get("n_vertices", viz_data["n_vertices"])),
                ("Edges (n1)", result.get("n_edges", viz_data["n_edges"])),
                ("Faces (n2)", result.get("n_faces", viz_data["n_faces"])),
                ("beta_0", result.get("beta_0", "--")),
                ("beta_1", result.get("beta_1", "--")),
                ("beta_2", result.get("beta_2", "--")),
                ("L0 spectral gap", f"{result['L0_spectral_gap']:.4f}" if result.get("L0_spectral_gap") else "--"),
                ("L1 spectral gap", f"{result['L1_spectral_gap']:.4f}" if result.get("L1_spectral_gap") else "--"),
                ("Euler consistent", result.get("euler_consistent", "--")),
                ("Chain complex valid", result.get("chain_complex_valid", "--")),
            ]
        except Exception:
            # No precomputed result -- show geometry stats only
            stats_items = [
                ("Vertices (n0)", viz_data["n_vertices"]),
                ("Edges (n1)", viz_data["n_edges"]),
                ("Faces (n2)", viz_data["n_faces"]),
            ]

        # Coordination distribution
        coord_dist = viz_data.get("coordination_distribution", {})
        coord_str = ", ".join(f"z={k}: {v}" for k, v in sorted(coord_dist.items()))

        for label, value in stats_items:
            badge_color = "success" if value is True else ("danger" if value is False else "primary")
            if isinstance(value, bool):
                display = "Yes" if value else "No"
            else:
                display = str(value)

            stats_children.append(
                html.Div(
                    [
                        html.Strong(f"{label}: ", style={"fontSize": "0.85rem"}),
                        dbc.Badge(display, color=badge_color, className="ms-1"),
                    ],
                    className="mb-2",
                )
            )

        # Add coordination distribution
        stats_children.append(html.Hr())
        stats_children.append(
            html.Div(
                [
                    html.Strong("Coordination distribution:", style={"fontSize": "0.85rem"}),
                    html.Br(),
                    html.Span(coord_str if coord_str else "--", style={"fontSize": "0.85rem"}),
                ],
                className="mb-2",
            )
        )

        return fig, stats_children

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}"), html.P(f"Error loading data: {exc}")
