"""Beta_1 scaling page.

Log-log plots of beta_1 vs system size (N = n_vertices) for selected lattice
types, with optional power-law fits.
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

dash.register_page(__name__, path="/scaling", name="Scaling")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LATTICE_NAMES = [
    "square", "kagome", "shakti", "tetris", "santa_fe",
]

# Color palette for lattices
LATTICE_COLORS = {
    "square": "#3498db",
    "kagome": "#e74c3c",
    "shakti": "#2ecc71",
    "tetris": "#9b59b6",
    "santa_fe": "#1abc9c",
}


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
                html.H2("Beta_1 Scaling Analysis", className="mb-3 mt-2"),
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
                                html.Label("Select lattices", className="mt-2 mb-1"),
                                dcc.Checklist(
                                    id="scaling-lattice-checklist",
                                    options=[
                                        {
                                            "label": html.Span(
                                                name.replace("_", " ").title(),
                                                style={"marginLeft": "5px"},
                                            ),
                                            "value": name,
                                        }
                                        for name in ALL_LATTICE_NAMES
                                    ],
                                    value=ALL_LATTICE_NAMES.copy(),
                                    labelStyle={"display": "block", "marginBottom": "4px"},
                                ),
                                html.Hr(),
                                html.Label("Face strategy", className="mt-2 mb-1"),
                                dbc.RadioItems(
                                    id="scaling-strategy-toggle",
                                    options=[
                                        {"label": "All faces", "value": "all"},
                                        {"label": "No faces", "value": "none"},
                                    ],
                                    value="all",
                                    inline=True,
                                ),
                                html.Label("Boundary conditions", className="mt-3 mb-1"),
                                dbc.RadioItems(
                                    id="scaling-boundary-toggle",
                                    options=[
                                        {"label": "Periodic (torus)", "value": "periodic"},
                                        {"label": "Open (disk)", "value": "open"},
                                    ],
                                    value="periodic",
                                    inline=True,
                                ),
                                html.Hr(),
                                dbc.Checklist(
                                    id="scaling-fit-toggle",
                                    options=[
                                        {"label": "Show power-law fits", "value": "fit"},
                                    ],
                                    value=["fit"],
                                    switch=True,
                                ),
                            ]
                        ),
                    ),
                    width=3,
                ),
                # Plot
                dbc.Col(
                    [
                        dcc.Loading(
                            dcc.Graph(
                                id="scaling-figure",
                                style={"height": "600px"},
                            ),
                            type="circle",
                        ),
                        html.Div(id="scaling-fit-info", className="mt-3"),
                    ],
                    width=9,
                ),
            ],
        ),
        # ── Ice Manifold Section ──────────────────────────────────────
        html.Hr(className="mt-4"),
        dbc.Row(
            dbc.Col(
                [
                    html.H2("Ice Manifold Dimension", className="mb-2 mt-2"),
                    html.P(
                        "The ice manifold is the divergence-free edge signal space: "
                        "dim(ker B\u2081\u1d40) = n_edges \u2212 n_vertices + 1. "
                        "This is independent of face strategy (depends only on the graph). "
                        "Both boundary conditions are overlaid on each plot.",
                        className="text-muted",
                    ),
                ],
                width=12,
            ),
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        dcc.Graph(
                            id="ice-manifold-figure",
                            style={"height": "550px"},
                        ),
                        type="circle",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dcc.Loading(
                        dcc.Graph(
                            id="ice-fraction-figure",
                            style={"height": "550px"},
                        ),
                        type="circle",
                    ),
                    width=6,
                ),
            ],
            className="mt-2",
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("scaling-figure", "figure"),
    Output("scaling-fit-info", "children"),
    Input("scaling-lattice-checklist", "value"),
    Input("scaling-strategy-toggle", "value"),
    Input("scaling-boundary-toggle", "value"),
    Input("scaling-fit-toggle", "value"),
)
def update_scaling(selected_lattices, face_strategy, boundary, fit_toggle):
    """Build log-log beta_1 vs N scaling plot."""
    if not selected_lattices:
        return _make_empty_figure("Select at least one lattice"), html.P("")

    show_fit = "fit" in (fit_toggle or [])
    boundary = boundary or "periodic"

    try:
        from dashboard.data_loader import get_available_results

        results = get_available_results()

        fig = go.Figure()
        fit_info_rows = []

        for lattice_name in selected_lattices:
            # Gather all sizes for this lattice + strategy + boundary
            entries = [
                r for r in results
                if r["lattice_name"] == lattice_name
                and r["face_strategy"] == face_strategy
                and r.get("boundary", "periodic") == boundary
            ]

            if not entries:
                continue

            # Sort by system size (n_vertices)
            entries.sort(key=lambda e: e["n_vertices"])

            n_vals = np.array([e["n_vertices"] for e in entries], dtype=float)
            b1_vals = np.array([e["beta_1"] for e in entries], dtype=float)

            color = LATTICE_COLORS.get(lattice_name, "#7f8c8d")
            display_name = lattice_name.replace("_", " ").title()

            # Data points
            fig.add_trace(go.Scatter(
                x=n_vals,
                y=b1_vals,
                mode="markers+lines",
                marker=dict(size=10, color=color, symbol="circle"),
                line=dict(color=color, width=2),
                name=display_name,
                hovertemplate=(
                    f"{display_name}<br>"
                    "N = %{x}<br>"
                    "beta_1 = %{y}<br>"
                    "<extra></extra>"
                ),
            ))

            # Power-law fit: beta_1 ~ a * N^alpha
            if show_fit and len(n_vals) >= 3:
                # Filter out zero beta_1 for log fit
                mask = b1_vals > 0
                if np.sum(mask) >= 2:
                    log_n = np.log10(n_vals[mask])
                    log_b = np.log10(b1_vals[mask])
                    coeffs = np.polyfit(log_n, log_b, 1)
                    alpha = coeffs[0]
                    log_a = coeffs[1]
                    a = 10 ** log_a

                    # Fit line
                    n_fit = np.logspace(
                        np.log10(n_vals.min() * 0.8),
                        np.log10(n_vals.max() * 1.2),
                        50,
                    )
                    b1_fit = a * n_fit ** alpha

                    fig.add_trace(go.Scatter(
                        x=n_fit,
                        y=b1_fit,
                        mode="lines",
                        line=dict(color=color, width=1, dash="dash"),
                        showlegend=False,
                        hovertemplate=(
                            f"{display_name} fit<br>"
                            f"alpha = {alpha:.3f}<br>"
                            "<extra></extra>"
                        ),
                    ))

                    fit_info_rows.append(
                        html.Tr([
                            html.Td(display_name),
                            html.Td(f"{alpha:.3f}"),
                            html.Td(f"{a:.4f}"),
                            html.Td(f"beta_1 ~ {a:.3f} * N^{alpha:.3f}"),
                        ])
                    )
            elif show_fit and len(n_vals) >= 2:
                fit_info_rows.append(
                    html.Tr([
                        html.Td(display_name),
                        html.Td("--"),
                        html.Td("--"),
                        html.Td(f"Only {len(n_vals)} data points (need 3+ for fit)"),
                    ])
                )

        # Layout
        fig.update_layout(
            title=dict(
                text=f"beta_1 Scaling (face strategy: {face_strategy})",
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="N (vertices)",
                type="log",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title="beta_1",
                type="log",
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
            hovermode="closest",
        )

        # Fit info table
        if fit_info_rows:
            fit_table = dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Lattice"),
                        html.Th("Exponent (alpha)"),
                        html.Th("Prefactor (a)"),
                        html.Th("Fit expression"),
                    ])),
                    html.Tbody(fit_info_rows),
                ],
                bordered=True,
                hover=True,
                size="sm",
                className="mt-2",
            )
            fit_info = html.Div([
                html.H5("Power-law fit: beta_1 ~ a * N^alpha", className="mt-3"),
                fit_table,
            ])
        else:
            fit_info = html.P(
                "Not enough data points for power-law fits (need 3+ sizes per lattice).",
                className="text-muted",
            )

        return fig, fit_info

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}"), html.P(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Ice manifold callbacks
# ---------------------------------------------------------------------------

_BC_STYLES = {"periodic": dict(dash=None), "open": dict(dash="dash")}
_BC_SYMBOLS = {"periodic": "circle", "open": "square"}


@callback(
    Output("ice-manifold-figure", "figure"),
    Input("scaling-lattice-checklist", "value"),
    Input("scaling-fit-toggle", "value"),
)
def update_ice_manifold(selected_lattices, fit_toggle):
    """Log-log ice manifold dim vs n_vertices, both BCs overlaid."""
    if not selected_lattices:
        return _make_empty_figure("Select at least one lattice")

    show_fit = "fit" in (fit_toggle or [])

    try:
        from dashboard.data_loader import get_ice_manifold_data

        ice_data = get_ice_manifold_data()
        fig = go.Figure()

        for lattice_name in selected_lattices:
            color = LATTICE_COLORS.get(lattice_name, "#7f8c8d")
            display = lattice_name.replace("_", " ").title()

            for bc in ["periodic", "open"]:
                subset = sorted(
                    [e for e in ice_data
                     if e["lattice_name"] == lattice_name
                     and e["boundary"] == bc],
                    key=lambda e: e["n_vertices"],
                )
                if len(subset) < 2:
                    continue

                n_vals = np.array([e["n_vertices"] for e in subset], dtype=float)
                dim_vals = np.array([e["ice_manifold_dim"] for e in subset], dtype=float)

                fig.add_trace(go.Scatter(
                    x=n_vals, y=dim_vals,
                    mode="markers+lines",
                    marker=dict(size=8, color=color, symbol=_BC_SYMBOLS[bc]),
                    line=dict(color=color, width=2, **_BC_STYLES[bc]),
                    name=f"{display} ({bc})",
                    hovertemplate=(
                        f"{display} ({bc})<br>"
                        "N = %{x}<br>"
                        "ice dim = %{y}<br>"
                        "<extra></extra>"
                    ),
                ))

                if show_fit and np.sum(dim_vals > 0) >= 2:
                    mask = dim_vals > 0
                    log_n = np.log10(n_vals[mask])
                    log_d = np.log10(dim_vals[mask])
                    coeffs = np.polyfit(log_n, log_d, 1)
                    n_fit = np.logspace(
                        np.log10(n_vals.min() * 0.8),
                        np.log10(n_vals.max() * 1.2), 50,
                    )
                    d_fit = 10 ** coeffs[1] * n_fit ** coeffs[0]
                    fig.add_trace(go.Scatter(
                        x=n_fit, y=d_fit,
                        mode="lines",
                        line=dict(color=color, width=1, dash="dot"),
                        showlegend=False,
                    ))

        fig.update_layout(
            title=dict(text="Ice Manifold Dimension Scaling", x=0.5, xanchor="center"),
            xaxis=dict(title="N (vertices)", type="log", showgrid=True,
                       gridcolor="rgba(200,200,200,0.3)"),
            yaxis=dict(title="Ice manifold dim", type="log", showgrid=True,
                       gridcolor="rgba(200,200,200,0.3)"),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#cccccc", borderwidth=1),
            margin=dict(l=60, r=30, t=50, b=50),
            hovermode="closest",
        )
        return fig

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}")


@callback(
    Output("ice-fraction-figure", "figure"),
    Input("scaling-lattice-checklist", "value"),
)
def update_ice_fraction(selected_lattices):
    """Ice manifold fraction (dim / n_edges) vs n_vertices, both BCs overlaid."""
    if not selected_lattices:
        return _make_empty_figure("Select at least one lattice")

    try:
        from dashboard.data_loader import get_ice_manifold_data

        ice_data = get_ice_manifold_data()
        fig = go.Figure()

        for lattice_name in selected_lattices:
            color = LATTICE_COLORS.get(lattice_name, "#7f8c8d")
            display = lattice_name.replace("_", " ").title()

            for bc in ["periodic", "open"]:
                subset = sorted(
                    [e for e in ice_data
                     if e["lattice_name"] == lattice_name
                     and e["boundary"] == bc],
                    key=lambda e: e["n_vertices"],
                )
                if len(subset) < 2:
                    continue

                n_vals = np.array([e["n_vertices"] for e in subset], dtype=float)
                frac_vals = np.array([e["ice_manifold_fraction"] for e in subset])

                fig.add_trace(go.Scatter(
                    x=n_vals, y=frac_vals,
                    mode="markers+lines",
                    marker=dict(size=8, color=color, symbol=_BC_SYMBOLS[bc]),
                    line=dict(color=color, width=2, **_BC_STYLES[bc]),
                    name=f"{display} ({bc})",
                    hovertemplate=(
                        f"{display} ({bc})<br>"
                        "N = %{x}<br>"
                        "fraction = %{y:.4f}<br>"
                        "<extra></extra>"
                    ),
                ))

        fig.update_layout(
            title=dict(text="Ice Manifold Fraction vs System Size", x=0.5,
                       xanchor="center"),
            xaxis=dict(title="N (vertices)", type="log", showgrid=True,
                       gridcolor="rgba(200,200,200,0.3)"),
            yaxis=dict(title="Ice manifold dim / n_edges", showgrid=True,
                       gridcolor="rgba(200,200,200,0.3)",
                       range=[0, 1.05]),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#cccccc", borderwidth=1),
            margin=dict(l=60, r=30, t=50, b=50),
            hovermode="closest",
        )
        return fig

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}")
