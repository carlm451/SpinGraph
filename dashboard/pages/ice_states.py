"""Ice manifold state sampling and gallery page.

Sample binary spin configurations from the ice manifold and display them
in a responsive thumbnail gallery with a click-to-expand modal.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, ALL, ctx

dash.register_page(__name__, path="/ice-states", name="Ice States")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LATTICE_NAMES = [
    "square", "kagome", "shakti", "tetris", "santa_fe",
]

SIZE_CONFIGS = {"XS": (4, 4), "S": (10, 10), "M": (20, 20)}


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
        # Data store
        dcc.Store(id="ice-states-store", data=None),

        dbc.Row(
            dbc.Col(
                [
                    html.H2("Ice Manifold States", className="mb-2 mt-2"),
                    html.P([
                        "Ice states are edge signals in ",
                        html.Strong("ker(B\u2081)"),
                        " \u2014 binary spin configurations (\u00b11 per edge) "
                        "that satisfy the ice rule at every vertex: the net "
                        "flow into each vertex is zero (or as close to zero as "
                        "the coordination allows). Mathematically, B\u2081\u03c3 = 0, "
                        "where B\u2081 is the vertex\u2013edge incidence matrix. The ice "
                        "manifold depends ",
                        html.Strong("only on the graph"),
                        " (vertices and edges) \u2014 it knows nothing about faces "
                        "or the boundary operator B\u2082.",
                    ], className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                    html.P([
                        "The dimension of the ice manifold is n\u2081 \u2212 n\u2080 + 1 "
                        "(for connected graphs). This is always large \u2014 "
                        "a finite fraction of all edge configurations. By the ",
                        html.Strong("Hodge decomposition"),
                        ", the ice manifold splits into two orthogonal parts: ",
                        "ker(B\u2081) = im(B\u2082) \u2295 ker(L\u2081). That is, every "
                        "ice state is either a ",
                        html.Em("curl component"),
                        " (boundary of some face current, in im(B\u2082)) or a ",
                        html.Em("harmonic mode"),
                        " (in ker(L\u2081)), or a sum of both.",
                    ], className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                    html.P([
                        "The key distinction: ",
                        html.Strong("ice states depend only on the graph"),
                        " and are unchanged by face-filling strategy. ",
                        html.Strong("Harmonic modes"),
                        " are the subset of ice states that are also curl-free "
                        "(B\u2082\u1d40h = 0), and their count \u03b2\u2081 depends on which "
                        "faces are filled. Filling more faces enlarges im(B\u2082), "
                        "converting some harmonic modes into curl components "
                        "and shrinking \u03b2\u2081 \u2014 but the total ice manifold "
                        "dimension stays the same. See the ",
                        html.A(
                            "Harmonic page",
                            href="/harmonic",
                        ),
                        " for visualization of the harmonic subspace, and the ",
                        html.A(
                            "TDL \u2194 Spin Ice Correspondence",
                            href="http://127.0.0.1:8000/tdl-spinice-correspondence.html",
                            target="_blank",
                        ),
                        " for the full derivation.",
                    ], className="text-muted mb-3", style={"fontSize": "0.8rem"}),
                ],
                width=12,
            )
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
                                    id="ice-lattice-dropdown",
                                    options=[
                                        {"label": n.replace("_", " ").title(), "value": n}
                                        for n in ALL_LATTICE_NAMES
                                    ],
                                    value="square",
                                    clearable=False,
                                ),
                                html.Label("System size", className="mt-3 mb-1"),
                                dcc.Dropdown(
                                    id="ice-size-dropdown",
                                    options=[
                                        {"label": s, "value": s}
                                        for s in SIZE_CONFIGS.keys()
                                    ],
                                    value="XS",
                                    clearable=False,
                                ),
                                html.Label("Boundary conditions", className="mt-3 mb-1"),
                                dbc.RadioItems(
                                    id="ice-boundary-toggle",
                                    options=[
                                        {"label": "Periodic (torus)", "value": "periodic"},
                                        {"label": "Open (disk)", "value": "open"},
                                    ],
                                    value="periodic",
                                    inline=True,
                                ),
                                html.Hr(),
                                dbc.Button(
                                    "Sample States",
                                    id="ice-sample-btn",
                                    color="primary",
                                    className="w-100 mb-2",
                                ),
                                dbc.Button(
                                    "Sample More",
                                    id="ice-sample-more-btn",
                                    color="secondary",
                                    outline=True,
                                    className="w-100",
                                ),
                            ]
                        ),
                    ),
                    width=3,
                ),
                # Main content column
                dbc.Col(
                    [
                        # Stats row
                        html.Div(id="ice-stats-row", className="mb-3"),
                        # Gallery
                        dcc.Loading(
                            html.Div(
                                id="ice-gallery-container",
                                style={"minHeight": "200px"},
                            ),
                            type="circle",
                        ),
                    ],
                    width=9,
                ),
            ],
        ),
        # Modal for full-size figure
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(id="ice-modal-title")),
                dbc.ModalBody(
                    dcc.Graph(
                        id="ice-modal-figure",
                        style={"height": "650px"},
                    ),
                ),
            ],
            id="ice-state-modal",
            size="xl",
            is_open=False,
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("ice-states-store", "data"),
    Input("ice-sample-btn", "n_clicks"),
    State("ice-lattice-dropdown", "value"),
    State("ice-size-dropdown", "value"),
    State("ice-boundary-toggle", "value"),
    prevent_initial_call=True,
)
def sample_states(n_clicks, lattice_name, size_label, boundary):
    """Build lattice, sample ice states, store in dcc.Store."""
    if not lattice_name or not size_label:
        return dash.no_update

    boundary = boundary or "periodic"

    from dashboard.data_loader import load_lattice_for_viz
    from src.topology.incidence import build_B1
    from src.topology.ice_sampling import (
        sample_ice_states, pauling_estimate, verify_ice_state,
    )

    nx_val, ny_val = SIZE_CONFIGS[size_label]
    viz_data = load_lattice_for_viz(lattice_name, nx_val, ny_val, boundary=boundary)

    B1 = build_B1(viz_data["n_vertices"], viz_data["edge_list"])
    coordination = viz_data["coordination"]

    edge_list = viz_data["edge_list"]
    states = sample_ice_states(B1, coordination, n_samples=12, n_flips_between=20,
                               edge_list=edge_list)

    # Compute stats
    n_edges = viz_data["n_edges"]
    n_vertices = viz_data["n_vertices"]
    ice_dim = n_edges - n_vertices + 1
    pauling_est = pauling_estimate(coordination)

    # Verify states
    valid_count = sum(1 for s in states if verify_ice_state(B1, s, coordination))

    return {
        "lattice_name": lattice_name,
        "size_label": size_label,
        "boundary": boundary,
        "nx": nx_val,
        "ny": ny_val,
        "states": [s.tolist() for s in states],
        "ice_dim": ice_dim,
        "pauling_est": float(pauling_est),
        "n_valid": valid_count,
        "n_edges": n_edges,
        "n_vertices": n_vertices,
    }


@callback(
    Output("ice-states-store", "data", allow_duplicate=True),
    Input("ice-sample-more-btn", "n_clicks"),
    State("ice-states-store", "data"),
    prevent_initial_call=True,
)
def sample_more_states(n_clicks, store_data):
    """Generate more states and append to existing store."""
    if not store_data or not store_data.get("states"):
        return dash.no_update

    from dashboard.data_loader import load_lattice_for_viz
    from src.topology.incidence import build_B1
    from src.topology.ice_sampling import (
        sample_ice_states, verify_ice_state,
    )

    lattice_name = store_data["lattice_name"]
    nx_val = store_data["nx"]
    ny_val = store_data["ny"]
    boundary = store_data["boundary"]

    viz_data = load_lattice_for_viz(lattice_name, nx_val, ny_val, boundary=boundary)
    B1 = build_B1(viz_data["n_vertices"], viz_data["edge_list"])
    coordination = viz_data["coordination"]

    # Use the last stored state as a starting point for continued sampling
    existing_states = store_data["states"]
    last_sigma = np.array(existing_states[-1], dtype=np.float64)

    # Sample more via loop flips from the last state
    edge_list = viz_data["edge_list"]
    new_states = sample_ice_states(B1, coordination, n_samples=8, n_flips_between=20,
                                   edge_list=edge_list)

    all_states = existing_states + [s.tolist() for s in new_states]
    valid_count = sum(
        1 for s in all_states
        if verify_ice_state(B1, np.array(s), coordination)
    )

    store_data["states"] = all_states
    store_data["n_valid"] = valid_count
    return store_data


@callback(
    Output("ice-stats-row", "children"),
    Output("ice-gallery-container", "children"),
    Input("ice-states-store", "data"),
)
def render_gallery(store_data):
    """Render stats badges and thumbnail gallery from stored states."""
    if not store_data or not store_data.get("states"):
        stats = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Ice dim", className="card-subtitle text-muted"),
                html.H4("--", className="card-title text-primary"),
            ])), width=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Pauling est", className="card-subtitle text-muted"),
                html.H4("--", className="card-title text-primary"),
            ])), width=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("States found", className="card-subtitle text-muted"),
                html.H4("--", className="card-title text-primary"),
            ])), width=2),
        ])
        gallery = html.P(
            "Click 'Sample States' to generate ice-rule-satisfying configurations.",
            className="text-muted mt-4",
        )
        return stats, gallery

    from dashboard.data_loader import load_lattice_for_viz
    from dashboard.components.ice_state_figure import make_ice_state_thumbnail

    lattice_name = store_data["lattice_name"]
    nx_val = store_data["nx"]
    ny_val = store_data["ny"]
    boundary = store_data["boundary"]
    states = store_data["states"]
    ice_dim = store_data["ice_dim"]
    pauling_est = store_data["pauling_est"]

    viz_data = load_lattice_for_viz(lattice_name, nx_val, ny_val, boundary=boundary)

    # Stats row
    if pauling_est is None or not np.isfinite(pauling_est):
        pauling_str = "huge"
    elif pauling_est < 1e6:
        pauling_str = f"~{pauling_est:.0f}"
    else:
        pauling_str = f"~{pauling_est:.2e}"
    stats = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Ice dim", className="card-subtitle text-muted"),
            html.H4(str(ice_dim), className="card-title text-primary"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Pauling est", className="card-subtitle text-muted"),
            html.H4(pauling_str, className="card-title text-primary"),
        ])), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("States found", className="card-subtitle text-muted"),
            html.H4(str(len(states)), className="card-title text-primary"),
        ])), width=2),
    ])

    # Generate thumbnails
    positions = viz_data["positions"]
    edges = viz_data["edge_list"]
    coordination = viz_data["coordination"]

    cards = []
    for k, sigma_list in enumerate(states):
        sigma = np.array(sigma_list, dtype=np.float64)
        thumb_src = make_ice_state_thumbnail(
            positions, edges, sigma, coordination,
            a1=viz_data.get("a1"),
            a2=viz_data.get("a2"),
            nx_size=viz_data.get("nx_size"),
            ny_size=viz_data.get("ny_size"),
            boundary=boundary,
        )

        # Hamming distance from seed (state 0)
        if k > 0:
            seed = np.array(states[0], dtype=np.float64)
            hamming = int(np.sum(sigma != seed))
            subtitle = f"Hamming d={hamming}"
        else:
            subtitle = "Seed state"

        card = dbc.Col(
            dbc.Card(
                [
                    html.Div(
                        html.Img(
                            src=thumb_src,
                            style={
                                "width": "100%",
                                "cursor": "pointer",
                                "borderRadius": "4px",
                            },
                        ),
                        id={"type": "ice-thumb", "index": k},
                        n_clicks=0,
                    ),
                    dbc.CardBody(
                        [
                            html.P(
                                f"State #{k + 1}",
                                className="card-title mb-0",
                                style={"fontWeight": "bold", "fontSize": "0.85rem"},
                            ),
                            html.P(
                                subtitle,
                                className="text-muted mb-0",
                                style={"fontSize": "0.75rem"},
                            ),
                        ],
                        style={"padding": "0.4rem 0.6rem"},
                    ),
                ],
                className="mb-3",
            ),
            width=3,
        )
        cards.append(card)

    gallery = dbc.Row(cards)
    return stats, gallery


@callback(
    Output("ice-state-modal", "is_open"),
    Output("ice-modal-title", "children"),
    Output("ice-modal-figure", "figure"),
    Input({"type": "ice-thumb", "index": ALL}, "n_clicks"),
    State("ice-states-store", "data"),
    prevent_initial_call=True,
)
def open_modal(n_clicks_list, store_data):
    """Open modal with full interactive ice state figure on thumbnail click."""
    if not store_data or not store_data.get("states"):
        return False, "", _make_empty_figure()

    # Determine which thumbnail was clicked
    triggered = ctx.triggered_id
    if triggered is None or not isinstance(triggered, dict):
        return False, "", _make_empty_figure()

    idx = triggered.get("index", 0)

    # Check that a click actually happened (not just initial render)
    if all(c == 0 or c is None for c in n_clicks_list):
        return False, "", _make_empty_figure()

    from dashboard.data_loader import load_lattice_for_viz
    from dashboard.components.ice_state_figure import make_ice_state_figure
    from src.topology.incidence import build_B1
    from src.topology.ice_sampling import verify_ice_state

    lattice_name = store_data["lattice_name"]
    nx_val = store_data["nx"]
    ny_val = store_data["ny"]
    boundary = store_data["boundary"]
    states = store_data["states"]

    if idx >= len(states):
        return False, "", _make_empty_figure()

    sigma = np.array(states[idx], dtype=np.float64)
    viz_data = load_lattice_for_viz(lattice_name, nx_val, ny_val, boundary=boundary)

    B1 = build_B1(viz_data["n_vertices"], viz_data["edge_list"])
    charge = np.asarray(B1 @ sigma).ravel()
    is_valid = verify_ice_state(B1, sigma, viz_data["coordination"])

    # Hamming distance from seed
    if idx > 0:
        seed = np.array(states[0], dtype=np.float64)
        hamming = int(np.sum(sigma != seed))
    else:
        hamming = 0

    display_name = lattice_name.replace("_", " ").title()
    n_plus = int(np.sum(sigma > 0))
    n_minus = int(np.sum(sigma < 0))
    size_label = store_data["size_label"]
    bc_label = "periodic" if boundary == "periodic" else "open"
    title_text = (
        f"{display_name} ({size_label}, {bc_label}) -- "
        f"State #{idx + 1} | +1:{n_plus} -1:{n_minus} | "
        f"Hamming={hamming} | Valid={is_valid}"
    )

    fig = make_ice_state_figure(
        positions=viz_data["positions"],
        edges=viz_data["edge_list"],
        coordination=viz_data["coordination"],
        sigma=sigma,
        title=title_text,
        a1=viz_data.get("a1"),
        a2=viz_data.get("a2"),
        nx_size=viz_data.get("nx_size"),
        ny_size=viz_data.get("ny_size"),
        boundary=boundary,
        show_arrows=True,
    )

    modal_title = f"State #{idx + 1} -- {display_name} ({size_label}, {bc_label})"
    return True, modal_title, fig
