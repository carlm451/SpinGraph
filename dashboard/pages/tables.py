"""Summary tables page with CSV download.

Three interactive DataTable views:
1. Zoo Summary -- per-lattice structural properties
2. Spectral Properties -- full spectral results at size S
3. Beta_1 Scaling -- beta_1 values at each available size
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table

dash.register_page(__name__, path="/tables", name="Tables")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LATTICE_NAMES = [
    "square", "kagome", "shakti", "tetris", "santa_fe",
]

# Zoo summary: static reference data from Morrison et al.
ZOO_SUMMARY = [
    {
        "lattice": "square",
        "vertices_per_cell": 1,
        "edges_per_cell": 2,
        "coordination": "4",
        "frustration": "Geometric",
    },
    {
        "lattice": "kagome",
        "vertices_per_cell": 3,
        "edges_per_cell": 6,
        "coordination": "3",
        "frustration": "Geometric",
    },
    {
        "lattice": "shakti",
        "vertices_per_cell": 16,
        "edges_per_cell": 24,
        "coordination": "2,3,4",
        "frustration": "Vertex, maximal",
    },
    {
        "lattice": "tetris",
        "vertices_per_cell": 6,
        "edges_per_cell": 9,
        "coordination": "2,3,4",
        "frustration": "Vertex, maximal",
    },
    {
        "lattice": "santa_fe",
        "vertices_per_cell": 6,
        "edges_per_cell": 9,
        "coordination": "2,3,4",
        "frustration": "Both types",
    },
]


def _safe_fmt(val, fmt=".4f"):
    """Format a numeric value safely."""
    if val is None or val == "--":
        return "--"
    try:
        return f"{float(val):{fmt}}"
    except (ValueError, TypeError):
        return str(val)


# ---------------------------------------------------------------------------
# Data loaders (lazy, wrapped in try/except)
# ---------------------------------------------------------------------------

def _load_zoo_table():
    """Build zoo summary with beta_1 values from catalog."""
    try:
        from dashboard.data_loader import get_available_results
        results = get_available_results()
    except Exception:
        results = []

    # Index results by (lattice, strategy) -> list of entries
    result_map = {}
    for r in results:
        key = (r["lattice_name"], r["face_strategy"])
        if key not in result_map:
            result_map[key] = []
        result_map[key].append(r)

    rows = []
    for entry in ZOO_SUMMARY:
        name = entry["lattice"]
        # Get beta_1 at smallest available size for each strategy
        b1_all = "--"
        b1_none = "--"
        for r in result_map.get((name, "all"), []):
            b1_all = r.get("beta_1", "--")
            break
        for r in result_map.get((name, "none"), []):
            b1_none = r.get("beta_1", "--")
            break

        rows.append({
            **entry,
            "beta_1_all": b1_all,
            "beta_1_none": b1_none,
        })

    return rows


def _load_spectral_table():
    """Build spectral properties table at size S, including both boundary conditions."""
    try:
        from dashboard.data_loader import get_available_results, load_result
        results = get_available_results()
    except Exception:
        return []

    rows = []
    for entry in results:
        if entry["size_label"] != "S":
            continue
        boundary = entry.get("boundary", "periodic")
        try:
            full = load_result(
                entry["lattice_name"],
                entry["size_label"],
                entry["face_strategy"],
                boundary=boundary,
            )
        except Exception:
            continue

        rows.append({
            "lattice": entry["lattice_name"],
            "strategy": entry["face_strategy"],
            "boundary": boundary,
            "n0": full.get("n_vertices", "--"),
            "n1": full.get("n_edges", "--"),
            "n2": full.get("n_faces", "--"),
            "beta_0": full.get("beta_0", "--"),
            "beta_1": full.get("beta_1", "--"),
            "beta_2": full.get("beta_2", "--"),
            "L0_gap": _safe_fmt(full.get("L0_spectral_gap"), ".6f"),
            "L1_gap": _safe_fmt(full.get("L1_spectral_gap"), ".6f"),
            "compute_time": _safe_fmt(full.get("compute_time_seconds"), ".3f"),
        })

    return rows


def _load_scaling_table():
    """Build beta_1 scaling table -- beta_1 at each available size."""
    try:
        from dashboard.data_loader import get_available_results
        results = get_available_results()
    except Exception:
        return []

    size_order = ["XS", "S", "M", "L", "XL"]

    # Group by (lattice, strategy, boundary)
    grouped = {}
    for r in results:
        boundary = r.get("boundary", "periodic")
        key = (r["lattice_name"], r["face_strategy"], boundary)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r["size_label"]] = r.get("beta_1", "--")

    rows = []
    for (lattice, strategy, boundary), size_dict in sorted(grouped.items()):
        row = {"lattice": lattice, "strategy": strategy, "boundary": boundary}
        for s in size_order:
            row[f"beta_1_{s}"] = size_dict.get(s, "--")
        rows.append(row)

    return rows


def _rows_to_csv(rows):
    """Convert list of dicts to CSV string."""
    if not rows:
        return ""
    import csv
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(html.H2("Summary Tables", className="mb-3 mt-2"), width=12)
        ),

        # --- Table 1: Zoo Summary ---
        dbc.Row(
            dbc.Col(
                [
                    html.H4("1. Lattice Zoo Summary", className="mt-3 mb-2"),
                    html.P(
                        "Structural properties per lattice type. beta_1 values "
                        "shown from the smallest available catalog entry.",
                        className="text-muted",
                        style={"fontSize": "0.85rem"},
                    ),
                    dbc.Button(
                        "Download CSV",
                        id="zoo-download-btn",
                        color="secondary",
                        size="sm",
                        className="mb-2",
                    ),
                    dcc.Download(id="zoo-download"),
                    html.Div(id="zoo-table-container"),
                ],
                width=12,
            ),
        ),
        html.Hr(),

        # --- Table 2: Spectral Properties at Size S ---
        dbc.Row(
            dbc.Col(
                [
                    html.H4("2. Spectral Properties at Size S", className="mt-3 mb-2"),
                    html.P(
                        "Full spectral results at system size S (10x10 unit cells).",
                        className="text-muted",
                        style={"fontSize": "0.85rem"},
                    ),
                    dbc.Button(
                        "Download CSV",
                        id="spectral-download-btn",
                        color="secondary",
                        size="sm",
                        className="mb-2",
                    ),
                    dcc.Download(id="spectral-download"),
                    html.Div(id="spectral-table-container"),
                ],
                width=12,
            ),
        ),
        html.Hr(),

        # --- Table 3: Beta_1 Scaling ---
        dbc.Row(
            dbc.Col(
                [
                    html.H4("3. Beta_1 Scaling", className="mt-3 mb-2"),
                    html.P(
                        "beta_1 at each available system size, grouped by lattice and face strategy.",
                        className="text-muted",
                        style={"fontSize": "0.85rem"},
                    ),
                    dbc.Button(
                        "Download CSV",
                        id="scaling-table-download-btn",
                        color="secondary",
                        size="sm",
                        className="mb-2",
                    ),
                    dcc.Download(id="scaling-table-download"),
                    html.Div(id="scaling-table-container"),
                ],
                width=12,
            ),
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks: Populate tables on page load
# ---------------------------------------------------------------------------

@callback(
    Output("zoo-table-container", "children"),
    Input("zoo-download-btn", "n_clicks"),  # dummy trigger for initial load
)
def render_zoo_table(_):
    """Render the zoo summary DataTable."""
    rows = _load_zoo_table()
    if not rows:
        return html.P("No catalog data available.", className="text-muted")

    columns = [
        {"name": "Lattice", "id": "lattice"},
        {"name": "Verts/cell", "id": "vertices_per_cell"},
        {"name": "Edges/cell", "id": "edges_per_cell"},
        {"name": "Coordination", "id": "coordination"},
        {"name": "Frustration", "id": "frustration"},
        {"name": "beta_1 (all)", "id": "beta_1_all"},
        {"name": "beta_1 (none)", "id": "beta_1_none"},
    ]

    return dash_table.DataTable(
        data=rows,
        columns=columns,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontWeight": "bold",
            "fontSize": "0.85rem",
        },
        style_cell={
            "fontSize": "0.85rem",
            "padding": "8px",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#f8f9fa",
            }
        ],
    )


@callback(
    Output("spectral-table-container", "children"),
    Input("spectral-download-btn", "n_clicks"),
)
def render_spectral_table(_):
    """Render the spectral properties DataTable."""
    rows = _load_spectral_table()
    if not rows:
        return html.P(
            "No size-S results in the catalog. Run the catalog for size S first.",
            className="text-muted",
        )

    columns = [
        {"name": "Lattice", "id": "lattice"},
        {"name": "Strategy", "id": "strategy"},
        {"name": "Boundary", "id": "boundary"},
        {"name": "n0", "id": "n0"},
        {"name": "n1", "id": "n1"},
        {"name": "n2", "id": "n2"},
        {"name": "beta_0", "id": "beta_0"},
        {"name": "beta_1", "id": "beta_1"},
        {"name": "beta_2", "id": "beta_2"},
        {"name": "L0 gap", "id": "L0_gap"},
        {"name": "L1 gap", "id": "L1_gap"},
        {"name": "Time (s)", "id": "compute_time"},
    ]

    return dash_table.DataTable(
        data=rows,
        columns=columns,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontWeight": "bold",
            "fontSize": "0.85rem",
        },
        style_cell={
            "fontSize": "0.85rem",
            "padding": "8px",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#f8f9fa",
            }
        ],
    )


@callback(
    Output("scaling-table-container", "children"),
    Input("scaling-table-download-btn", "n_clicks"),
)
def render_scaling_table(_):
    """Render the beta_1 scaling DataTable."""
    rows = _load_scaling_table()
    if not rows:
        return html.P("No catalog data available.", className="text-muted")

    columns = [
        {"name": "Lattice", "id": "lattice"},
        {"name": "Strategy", "id": "strategy"},
        {"name": "Boundary", "id": "boundary"},
        {"name": "beta_1 (XS)", "id": "beta_1_XS"},
        {"name": "beta_1 (S)", "id": "beta_1_S"},
        {"name": "beta_1 (M)", "id": "beta_1_M"},
        {"name": "beta_1 (L)", "id": "beta_1_L"},
        {"name": "beta_1 (XL)", "id": "beta_1_XL"},
    ]

    return dash_table.DataTable(
        data=rows,
        columns=columns,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontWeight": "bold",
            "fontSize": "0.85rem",
        },
        style_cell={
            "fontSize": "0.85rem",
            "padding": "8px",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#f8f9fa",
            }
        ],
    )


# ---------------------------------------------------------------------------
# Download callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("zoo-download", "data"),
    Input("zoo-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_zoo_csv(n_clicks):
    """Generate CSV download for zoo summary table."""
    rows = _load_zoo_table()
    csv_str = _rows_to_csv(rows)
    return dict(content=csv_str, filename="lattice_zoo_summary.csv")


@callback(
    Output("spectral-download", "data"),
    Input("spectral-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_spectral_csv(n_clicks):
    """Generate CSV download for spectral properties table."""
    rows = _load_spectral_table()
    csv_str = _rows_to_csv(rows)
    return dict(content=csv_str, filename="spectral_properties_S.csv")


@callback(
    Output("scaling-table-download", "data"),
    Input("scaling-table-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_scaling_csv(n_clicks):
    """Generate CSV download for beta_1 scaling table."""
    rows = _load_scaling_table()
    csv_str = _rows_to_csv(rows)
    return dict(content=csv_str, filename="beta1_scaling.csv")
