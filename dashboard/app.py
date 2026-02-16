"""Main Dash application for the ASI Spectral Catalog dashboard.

Multi-page app with sidebar navigation using dash-bootstrap-components.
"""
from __future__ import annotations

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="ASI Spectral Catalog",
)

server = app.server  # for gunicorn

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "250px",
    "padding": "2rem 1rem",
    "backgroundColor": "#2c3e50",
    "color": "white",
    "overflowY": "auto",
}

CONTENT_STYLE = {
    "marginLeft": "250px",
    "padding": "2rem 2rem",
}

# Build nav links from registered pages
def _build_nav_links():
    """Build sidebar navigation links from Dash page registry."""
    links = []
    # Define desired order
    page_order = ["Overview", "Gallery", "Spectra", "Scaling", "Spectral Gap", "Ice States", "Harmonic", "Tables"]

    # Get pages from registry
    pages_by_name = {}
    for page in dash.page_registry.values():
        pages_by_name[page["name"]] = page

    for name in page_order:
        if name in pages_by_name:
            page = pages_by_name[name]
            links.append(
                dbc.NavLink(
                    page["name"],
                    href=page["relative_path"],
                    active="exact",
                    style={"color": "rgba(255,255,255,0.8)", "fontSize": "1rem"},
                    className="mb-1",
                )
            )

    # Add any remaining pages not in the explicit order
    for name, page in pages_by_name.items():
        if name not in page_order:
            links.append(
                dbc.NavLink(
                    page["name"],
                    href=page["relative_path"],
                    active="exact",
                    style={"color": "rgba(255,255,255,0.8)", "fontSize": "1rem"},
                    className="mb-1",
                )
            )

    return links


sidebar = html.Div(
    [
        html.H4(
            "ASI Spectral Catalog",
            style={"color": "white", "fontWeight": "bold", "marginBottom": "0.5rem"},
        ),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.3)"}),
        html.P(
            "Spin Ice Topology + Oversmoothing",
            style={"color": "rgba(255,255,255,0.5)", "fontSize": "0.85rem"},
        ),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.3)"}),
        dbc.Nav(
            _build_nav_links(),
            vertical=True,
            pills=True,
        ),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.3)", "marginTop": "2rem"}),
        html.P(
            "Stage 1: Spectral Catalog",
            style={
                "color": "rgba(255,255,255,0.4)",
                "fontSize": "0.75rem",
                "marginTop": "1rem",
            },
        ),
    ],
    style=SIDEBAR_STYLE,
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        sidebar,
        html.Div(
            dash.page_container,
            style=CONTENT_STYLE,
        ),
    ],
    fluid=True,
    style={"padding": 0},
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
