"""Overview / landing page for the ASI Spectral Catalog dashboard."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

dash.register_page(__name__, path="/", name="Overview")

# ---------------------------------------------------------------------------
# Lattice zoo quick-reference data
# ---------------------------------------------------------------------------

LATTICE_TABLE_DATA = [
    {
        "Lattice": "Square",
        "Coordination": "All z=4",
        "Frustration": "Geometric only",
        "Key Property": "Clean baseline, ordered ground state",
    },
    {
        "Lattice": "Kagome",
        "Coordination": "All z=3",
        "Frustration": "Geometric",
        "Key Property": "Extensive degeneracy, flat bands, Dirac cones",
    },
    {
        "Lattice": "Shakti",
        "Coordination": "Mixed z=2,3,4",
        "Frustration": "Vertex-frustrated, maximal",
        "Key Property": "Extensive degeneracy, topological order",
    },
    {
        "Lattice": "Tetris",
        "Coordination": "Mixed z=2,3,4",
        "Frustration": "Vertex-frustrated, maximal",
        "Key Property": 'Sliding phase -- ordered/disordered mix',
    },
    {
        "Lattice": "Santa Fe",
        "Coordination": "Mixed z=2,3,4",
        "Frustration": "Both types",
        "Key Property": "Polymer-like strings of unhappy vertices",
    },
]

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2("ASI Spectral Catalog", className="mb-3 mt-2"),
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Markdown(
                    """
This dashboard provides an interactive exploration of the **spectral properties**
of eight artificial spin ice (ASI) lattice topologies. The goal is to catalog
how the **Hodge 1-Laplacian** spectrum -- and in particular the **first Betti
number** (the dimension of the harmonic subspace) -- varies across lattice
types, system sizes, and face-filling strategies.

### Why this matters

Signals in the harmonic subspace of the Hodge 1-Laplacian satisfy **L_1 h = 0**.
This means they are **mathematically immune** to Laplacian-based smoothing: no
matter how many message-passing layers are applied, the harmonic component passes
through unchanged. Lattices with large harmonic subspaces (high beta_1) therefore
provide a **topologically protected channel** that can resist the oversmoothing
problem that limits practical GNN depth.

### The lattice zoo

The five lattice types below were introduced or cataloged by Morrison, Nelson
& Nisoli (2013). They span a range of coordination patterns and frustration
mechanisms, giving us a principled design space for testing the topology-oversmoothing
connection.

### Key mathematical concepts

- **B_1** (vertex-edge incidence) and **B_2** (edge-face incidence) are the
  boundary operators of the simplicial complex.
- The **Hodge 1-Laplacian** decomposes as L_1 = B_1^T B_1 + B_2 B_2^T.
- The **Hodge decomposition** splits edge signals into gradient, curl, and
  harmonic components. Only the harmonic component sits in ker(L_1).
- **beta_1** = dim(ker(L_1)) counts independent cycles not bounded by faces.
- **Face strategy "all"** fills all minimal faces, minimizing beta_1.
  **"none"** fills no faces, maximizing beta_1.
                    """,
                    style={"fontSize": "0.95rem"},
                ),
                width=12,
            ),
        ),
        html.Hr(),
        dbc.Row(
            dbc.Col(
                html.H4("Lattice Zoo Quick Reference", className="mb-3"),
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Table(
                    # Header
                    [html.Thead(html.Tr([
                        html.Th("Lattice"),
                        html.Th("Coordination"),
                        html.Th("Frustration"),
                        html.Th("Key Property"),
                    ]))]
                    +
                    # Body
                    [html.Tbody([
                        html.Tr([
                            html.Td(row["Lattice"]),
                            html.Td(row["Coordination"]),
                            html.Td(row["Frustration"]),
                            html.Td(row["Key Property"]),
                        ])
                        for row in LATTICE_TABLE_DATA
                    ])],
                    bordered=True,
                    hover=True,
                    striped=True,
                    responsive=True,
                    size="sm",
                    className="mt-2",
                ),
                width=12,
            ),
        ),
        dbc.Row(
            dbc.Col(
                dcc.Markdown(
                    """
---
**Navigation:** Use the sidebar to explore the interactive gallery, eigenvalue
spectra, scaling analysis, harmonic mode visualizations, and summary data tables.
                    """,
                    style={"fontSize": "0.9rem", "color": "#7f8c8d"},
                ),
                width=12,
            ),
        ),
    ],
    fluid=True,
)
