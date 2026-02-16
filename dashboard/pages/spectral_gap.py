"""Spectral Gap Scaling page.

Finite-size scaling analysis of the L1_down spectral gap (monopole gap)
using physics-standard conventions: Delta vs 1/L^2 extrapolation,
Delta*L^2 diagnostic plot, and a prefactor comparison across lattice types.
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

dash.register_page(__name__, path="/spectral-gap", name="Spectral Gap")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LATTICE_NAMES = [
    "square", "kagome", "shakti", "tetris", "santa_fe",
]

LATTICE_COLORS = {
    "square": "#3498db",
    "kagome": "#e74c3c",
    "shakti": "#2ecc71",
    "tetris": "#9b59b6",
    "santa_fe": "#1abc9c",
}

_BC_STYLES = {"periodic": dict(dash=None), "open": dict(dash="dash")}
_BC_SYMBOLS = {"periodic": "circle", "open": "square"}


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
                    html.H2("Spectral Gap Scaling", className="mb-2 mt-2"),
                    html.P(
                        [
                            "The monopole excitation energy is controlled by the spectral gap: ",
                            html.Em(
                                "E_min = (\u03b5/2) \u00b7 \u0394 \u00b7 min(c\u1d62\u00b2)"
                            ),
                            " (eq. 31). For any 2D periodic lattice the gap closes as "
                            "\u0394 ~ c/L\u00b2 where L is the linear system size "
                            "(number of unit cells per side). This is universal \u2014 "
                            "the physics distinguishing lattice types lives in the ",
                            html.Strong("prefactor c"),
                            ", which sets the effective stiffness against "
                            "long-wavelength monopole excitations.",
                        ],
                        className="text-muted",
                    ),
                    html.P(
                        [
                            html.Strong("Reference value: "),
                            "A 1D periodic chain (cycle graph C",
                            html.Sub("N"),
                            ") has \u0394 = 2(1 \u2212 cos 2\u03c0/L) \u2192 4\u03c0\u00b2/L\u00b2, "
                            "giving c = 4\u03c0\u00b2 \u2248 39.5. "
                            "The 2D square lattice matches this exactly (the gap is set by the "
                            "same longest-wavelength mode along one axis of the torus). "
                            "Frustrated lattices are dramatically softer: "
                            "tetris has c \u2248 1.0, only ",
                            html.Strong("2.5%"),
                            " of the 4\u03c0\u00b2 baseline \u2014 a 40\u00d7 reduction "
                            "in monopole excitation stiffness purely from topology.",
                        ],
                        className="text-muted",
                        style={"fontSize": "0.9rem"},
                    ),
                ],
                width=12,
            ),
        ),
        dbc.Row(
            [
                # Controls sidebar
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Controls", className="card-title"),
                                html.Label("Select lattices", className="mt-2 mb-1"),
                                dcc.Checklist(
                                    id="gap-lattice-checklist",
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
                                    labelStyle={
                                        "display": "block",
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.Hr(),
                                html.Small(
                                    "\u0394 = smallest nonzero eigenvalue of "
                                    "L\u2081_down = B\u2081\u1d40B\u2081, "
                                    "which controls the vertex-charge energy "
                                    "in the Nisoli Hamiltonian. "
                                    "(L\u2080, L\u2081, and L\u2081_down share "
                                    "the same gap by the SVD equivalence of "
                                    "BB\u1d40 and B\u1d40B.)",
                                    className="text-muted",
                                ),
                                html.Hr(),
                                dbc.Checklist(
                                    id="gap-fit-toggle",
                                    options=[
                                        {
                                            "label": "Show linear fits",
                                            "value": "fit",
                                        },
                                    ],
                                    value=["fit"],
                                    switch=True,
                                ),
                            ]
                        ),
                    ),
                    width=3,
                ),
                # Plots column
                dbc.Col(
                    [
                        # Plot 1: Delta vs 1/L^2
                        dcc.Loading(
                            dcc.Graph(
                                id="gap-vs-invL-figure",
                                style={"height": "480px"},
                            ),
                            type="circle",
                        ),
                        html.Div(id="gap-fit-info", className="mt-3"),
                        html.Hr(className="mt-3"),
                        # Plot 2: Delta * L^2 vs L
                        html.H4(
                            "Diagnostic: c* = \u0394\u00b7L\u00b2 / 4\u03c0\u00b2 vs L",
                            className="mt-3 mb-2",
                        ),
                        html.P(
                            "If \u0394 ~ c/L\u00b2, the normalized product "
                            "c* = \u0394\u00b7L\u00b2/(4\u03c0\u00b2) is constant. "
                            "c* = 1 for the square lattice (equal to the 1D chain). "
                            "Horizontal lines confirm 1/L\u00b2 scaling; "
                            "upward/downward trends indicate a different exponent "
                            "or finite-size corrections.",
                            className="text-muted",
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id="diagnostic-gap-figure",
                                style={"height": "480px"},
                            ),
                            type="circle",
                        ),
                        html.Hr(className="mt-3"),
                        # Plot 3: Prefactor bar chart
                        html.H4(
                            "Lattice Stiffness Comparison",
                            className="mt-3 mb-2",
                        ),
                        html.P(
                            "Normalized stiffness c* = c/(4\u03c0\u00b2) averaged over all "
                            "available sizes. c* = 1 for the square lattice "
                            "(matching the 1D chain analytical result). "
                            "Frustrated lattices have c* \u226a 1, indicating softer "
                            "resistance to monopole excitations.",
                            className="text-muted",
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id="prefactor-bar-figure",
                                style={"height": "400px"},
                            ),
                            type="circle",
                        ),
                    ],
                    width=9,
                ),
            ],
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Helper: collect gap data
# ---------------------------------------------------------------------------

def _collect_gap_series(selected_lattices):
    """Return dict: (lattice_name, bc) -> sorted list of (L, N, n_edges, gap, size_label).

    Uses the L1_down spectral gap (face-independent, deduplicated).
    L = nx (linear system size = unit cells per side).
    Only returns series with at least 2 data points.
    """
    from dashboard.data_loader import get_spectral_gap_data

    gap_data = get_spectral_gap_data()

    series = {}
    seen_keys = set()

    for entry in gap_data:
        lattice = entry["lattice_name"]
        if lattice not in selected_lattices:
            continue

        bc = entry["boundary"]
        gap_val = entry.get("L1_down_spectral_gap")
        nx = entry.get("nx")
        if gap_val is None or nx is None:
            continue

        # L1_down gap is face-independent; deduplicate
        dedup_key = (lattice, entry["size_label"], bc)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        key = (lattice, bc)
        if key not in series:
            series[key] = []
        series[key].append((
            int(nx),
            entry["n_vertices"],
            entry["n_edges"],
            gap_val,
            entry["size_label"],
        ))

    return {
        k: sorted(v, key=lambda t: t[0])
        for k, v in series.items()
        if len(v) >= 2
    }


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

_GAP_TITLE = "\u0394(L\u2081_down)"
_4PI2 = 4 * np.pi ** 2  # exact gap prefactor for 1D ring / 2D square


@callback(
    Output("gap-vs-invL-figure", "figure"),
    Output("gap-fit-info", "children"),
    Input("gap-lattice-checklist", "value"),
    Input("gap-fit-toggle", "value"),
)
def update_gap_vs_invL(selected_lattices, fit_toggle):
    """Delta vs 1/L^2 extrapolation plot.

    For 2D lattices the gap scales as Delta ~ c/L^2, so plotting against
    1/L^2 linearises the data.  A gapped system would extrapolate to a
    finite y-intercept; gapless passes through the origin.
    """
    if not selected_lattices:
        return _make_empty_figure("Select at least one lattice"), html.P("")

    show_fit = "fit" in (fit_toggle or [])

    try:
        all_series = _collect_gap_series(selected_lattices)

        fig = go.Figure()
        fit_info_rows = []

        for lattice_name in selected_lattices:
            color = LATTICE_COLORS.get(lattice_name, "#7f8c8d")
            display = lattice_name.replace("_", " ").title()

            for bc in ["periodic", "open"]:
                key = (lattice_name, bc)
                if key not in all_series:
                    continue

                pts = all_series[key]
                L_vals = np.array([p[0] for p in pts], dtype=float)
                gap_vals = np.array([p[3] for p in pts], dtype=float)
                size_labels = [p[4] for p in pts]
                inv_L2 = 1.0 / L_vals ** 2

                fig.add_trace(go.Scatter(
                    x=inv_L2,
                    y=gap_vals,
                    mode="markers+lines",
                    marker=dict(size=10, color=color, symbol=_BC_SYMBOLS[bc]),
                    line=dict(color=color, width=2, **_BC_STYLES[bc]),
                    name=f"{display} ({bc})",
                    customdata=list(zip(
                        L_vals.astype(int), size_labels,
                    )),
                    hovertemplate=(
                        f"{display} ({bc})<br>"
                        "L = %{customdata[0]}<br>"
                        "1/L\u00b2 = %{x:.5g}<br>"
                        "\u0394 = %{y:.6g}<br>"
                        "size: %{customdata[1]}<br>"
                        "<extra></extra>"
                    ),
                ))

                # Linear fit: Delta = a + c/L^2
                # intercept a -> thermodynamic gap (0 for gapless)
                # slope c -> the prefactor (lattice stiffness)
                # Exclude XS (L<10) from fit — strong finite-size
                # corrections at the smallest size bias the slope.
                fit_mask = L_vals >= 10
                inv_L2_fit = inv_L2[fit_mask]
                gap_fit_vals = gap_vals[fit_mask]
                if show_fit and len(inv_L2_fit) >= 2:
                    coeffs = np.polyfit(inv_L2_fit, gap_fit_vals, 1)
                    slope, intercept = coeffs
                    inv_fit = np.linspace(0, inv_L2.max() * 1.1, 50)
                    gap_fit = slope * inv_fit + intercept

                    fig.add_trace(go.Scatter(
                        x=inv_fit,
                        y=gap_fit,
                        mode="lines",
                        line=dict(color=color, width=1, dash="dot"),
                        showlegend=False,
                        hovertemplate=(
                            f"{display} ({bc}) fit<br>"
                            f"c = {slope:.4g}<br>"
                            f"intercept = {intercept:.4g}<br>"
                            "<extra></extra>"
                        ),
                    ))

                    cstar_fit = slope / _4PI2
                    rel = abs(intercept / slope) if abs(slope) > 0 else 0
                    fit_info_rows.append(
                        html.Tr([
                            html.Td(display),
                            html.Td(bc),
                            html.Td(f"{intercept:.4g}"),
                            html.Td(f"{slope:.4g}"),
                            html.Td(f"{cstar_fit:.4g}"),
                            html.Td(
                                "gapped"
                                if rel > 0.05
                                else "gapless (\u0394 \u2192 0)"
                            ),
                        ])
                    )

        fig.update_layout(
            title=dict(
                text=(
                    f"{_GAP_TITLE} vs 1/L\u00b2  "
                    "(thermodynamic extrapolation)"
                ),
                x=0.5, xanchor="center",
            ),
            xaxis=dict(
                title="1 / L\u00b2  (L = unit cells per side)",
                type="log",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title="\u0394 (spectral gap)",
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
                        html.Th("Boundary"),
                        html.Th("Intercept (\u0394\u221e)"),
                        html.Th("Stiffness (c)"),
                        html.Th("c* = c/4\u03c0\u00b2"),
                        html.Th("Verdict"),
                    ])),
                    html.Tbody(fit_info_rows),
                ],
                bordered=True,
                hover=True,
                size="sm",
                className="mt-2",
            )
            fit_info = html.Div([
                html.H5(
                    "Linear fit: \u0394 = \u0394\u221e + c/L\u00b2  "
                    "(\u0394\u221e \u2248 0 \u21d2 gapless)",
                    className="mt-3",
                ),
                fit_table,
            ])
        else:
            fit_info = html.P(
                "Need 3+ sizes per lattice for fits.",
                className="text-muted",
            )

        return fig, fit_info

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}"), html.P(f"Error: {exc}")


@callback(
    Output("diagnostic-gap-figure", "figure"),
    Input("gap-lattice-checklist", "value"),
)
def update_diagnostic_gap(selected_lattices):
    """Delta * L^2 vs L -- the diagnostic / prefactor plot.

    If Delta ~ c/L^2, this is a horizontal line at height c.
    """
    if not selected_lattices:
        return _make_empty_figure("Select at least one lattice")

    try:
        all_series = _collect_gap_series(selected_lattices)

        fig = go.Figure()

        for lattice_name in selected_lattices:
            color = LATTICE_COLORS.get(lattice_name, "#7f8c8d")
            display = lattice_name.replace("_", " ").title()

            for bc in ["periodic", "open"]:
                key = (lattice_name, bc)
                if key not in all_series:
                    continue

                pts = all_series[key]
                L_vals = np.array([p[0] for p in pts], dtype=float)
                gap_vals = np.array([p[3] for p in pts], dtype=float)
                size_labels = [p[4] for p in pts]

                cstar = gap_vals * L_vals ** 2 / _4PI2

                fig.add_trace(go.Scatter(
                    x=L_vals,
                    y=cstar,
                    mode="markers+lines",
                    marker=dict(size=10, color=color, symbol=_BC_SYMBOLS[bc]),
                    line=dict(color=color, width=2, **_BC_STYLES[bc]),
                    name=f"{display} ({bc})",
                    customdata=list(zip(size_labels, gap_vals)),
                    hovertemplate=(
                        f"{display} ({bc})<br>"
                        "L = %{x}<br>"
                        "c* = %{y:.4g}<br>"
                        "\u0394 = %{customdata[1]:.6g}<br>"
                        "size: %{customdata[0]}<br>"
                        "<extra></extra>"
                    ),
                ))

        # Reference line at c* = 1 (square / 1D chain)
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#95a5a6",
            line_width=1,
        )

        fig.update_layout(
            title=dict(
                text=f"c* = \u0394\u00b7L\u00b2 / 4\u03c0\u00b2 vs L  ({_GAP_TITLE})",
                x=0.5, xanchor="center",
            ),
            xaxis=dict(
                title="L (unit cells per side)",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title="c* = c / 4\u03c0\u00b2  (normalized stiffness)",
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
        return fig

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}")


@callback(
    Output("prefactor-bar-figure", "figure"),
    Input("gap-lattice-checklist", "value"),
)
def update_prefactor_bar(selected_lattices):
    """Bar chart of fitted prefactor c from Delta = c/L^2."""
    if not selected_lattices:
        return _make_empty_figure("Select at least one lattice")

    try:
        all_series = _collect_gap_series(selected_lattices)

        lattice_labels = []
        bar_vals_periodic = []
        bar_vals_open = []

        for lattice_name in selected_lattices:
            display = lattice_name.replace("_", " ").title()
            lattice_labels.append(display)

            for bc, bar_list in [
                ("periodic", bar_vals_periodic),
                ("open", bar_vals_open),
            ]:
                key = (lattice_name, bc)
                if key not in all_series or len(all_series[key]) < 2:
                    bar_list.append(0)
                    continue

                pts = all_series[key]
                L_vals = np.array([p[0] for p in pts], dtype=float)
                gap_vals = np.array([p[3] for p in pts], dtype=float)
                # c* = mean(Delta * L^2 / 4pi^2), excluding XS (L<10)
                mask = L_vals >= 10
                if mask.sum() == 0:
                    mask = np.ones(len(L_vals), dtype=bool)
                cstar = float(np.mean(
                    gap_vals[mask] * L_vals[mask] ** 2 / _4PI2
                ))
                bar_list.append(cstar)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=lattice_labels,
            y=bar_vals_periodic,
            name="Periodic",
            marker_color=[
                LATTICE_COLORS.get(n, "#7f8c8d") for n in selected_lattices
            ],
            marker_line_width=0,
            hovertemplate="c* = %{y:.4g}<extra>periodic</extra>",
        ))

        fig.add_trace(go.Bar(
            x=lattice_labels,
            y=bar_vals_open,
            name="Open",
            marker_color=[
                LATTICE_COLORS.get(n, "#7f8c8d") for n in selected_lattices
            ],
            marker_pattern_shape="/",
            marker_line_width=0,
            opacity=0.6,
            hovertemplate="c* = %{y:.4g}<extra>open</extra>",
        ))

        # Reference line at c* = 1 (square / 1D chain exact value)
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#95a5a6",
            line_width=1,
        )

        fig.update_layout(
            title=dict(
                text=(
                    "Normalized stiffness c*  (\u0394 = c* \u00b7 4\u03c0\u00b2 / L\u00b2)  "
                    f"\u2014 {_GAP_TITLE}"
                ),
                x=0.5, xanchor="center",
            ),
            xaxis=dict(title="Lattice"),
            yaxis=dict(
                title="c* = c / 4\u03c0\u00b2  (1 = square lattice)",
                type="log",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
            ),
            barmode="group",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#cccccc",
                borderwidth=1,
            ),
            margin=dict(l=60, r=30, t=50, b=50),
        )
        return fig

    except Exception as exc:
        return _make_empty_figure(f"Error: {exc}")
