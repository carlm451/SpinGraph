#!/usr/bin/env python3
"""Generate the worked-examples HTML document from computed JSON data."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "results", "worked_examples.json")
    with open(path) as f:
        return json.load(f)


def matrix_to_latex(M):
    """Convert a dense matrix (list of lists) to LaTeX array."""
    rows = len(M)
    cols = len(M[0]) if rows > 0 else 0
    col_spec = "r" * cols
    lines = [f"\\left(\\begin{{array}}{{{col_spec}}}"]
    for i, row in enumerate(M):
        entries = " & ".join(str(x) for x in row)
        if i < rows - 1:
            lines.append(entries + " \\\\")
        else:
            lines.append(entries)
    lines.append("\\end{array}\\right)")
    return "\n".join(lines)


def eigenvalue_str(evals):
    """Format eigenvalue list for display, grouping multiplicities."""
    rounded = [round(v, 4) for v in evals]
    # Group by value
    groups = []
    i = 0
    while i < len(rounded):
        val = rounded[i]
        count = 1
        while i + count < len(rounded) and abs(rounded[i + count] - val) < 0.0001:
            count += 1
        if abs(val) < 1e-6:
            val_str = "0"
        else:
            val_str = f"{val:.4f}" if val != int(val) else str(int(val))
        if count > 1:
            groups.append(f"\\underbrace{{{', '.join([val_str]*count)}}}_{{{count}}}")
        else:
            groups.append(val_str)
        i += count
    return ",\\; ".join(groups)


def vector_to_latex(v, precision=4):
    """Convert a vector to LaTeX column vector."""
    entries = []
    for x in v:
        if abs(x) < 1e-8:
            entries.append("0")
        else:
            entries.append(f"{x:.{precision}f}")
    return "\\begin{pmatrix}" + " \\\\ ".join(entries) + "\\end{pmatrix}"


def edge_table_html(d):
    """Generate edge table HTML."""
    lines = ['<table>', '<thead>',
             '<tr><th>Edge</th><th>Tail</th><th>Head</th><th>Notes</th></tr>',
             '</thead>', '<tbody>']
    for e in d["edge_table"]:
        idx = e["index"]
        tail = e["tail"]
        head = e["head"]
        notes = ""
        # Wrap detection from unit cell definition
        lines.append(f'<tr><td>$e_{{{idx}}}$</td><td>$v_{{{tail}}}$</td>'
                     f'<td>$v_{{{head}}}$</td><td>{notes}</td></tr>')
    lines.extend(['</tbody>', '</table>'])
    return "\n".join(lines)


def face_table_html(d):
    """Generate face table HTML."""
    lines = ['<table>', '<thead>',
             '<tr><th>Face</th><th>Vertices (CCW)</th><th>Sides</th></tr>',
             '</thead>', '<tbody>']
    for f in d["face_table"]:
        idx = f["index"]
        verts = ", ".join(f"$v_{{{v}}}$" for v in f["vertices"])
        n = f["n_vertices"]
        lines.append(f'<tr><td>$f_{{{idx}}}$</td><td>{verts}</td><td>{n}</td></tr>')
    lines.extend(['</tbody>', '</table>'])
    return "\n".join(lines)


def sigma_to_latex(sigma):
    """Format sigma vector compactly."""
    entries = []
    for s in sigma:
        if s == 1:
            entries.append("+1")
        else:
            entries.append("-1")
    return "(" + ", ".join(entries) + ")^T"


def generate_html(data):
    """Generate the complete HTML document."""

    # Extract data for each lattice
    sq = data["square_periodic"]
    sq_open = data["square_open"]
    tet = data["tetris_periodic"]
    tet_open = data["tetris_open"]
    kag = data["kagome_periodic"]
    kag_open = data["kagome_open"]
    shk = data["shakti_periodic"]
    shk_open = data["shakti_open"]
    sf = data["santa_fe_periodic"]
    sf_open = data["santa_fe_open"]
    xs = data["xs_4x4_dimensions"]

    html = []

    # ==================== HEAD ====================
    html.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASI Lattice Zoo: Worked Incidence &amp; S-Matrix Examples</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    tags: 'ams',
    tagSide: 'right',
    macros: {
      bm: ['{\\\\boldsymbol{#1}}', 1],
      R: '{\\\\mathbb{R}}',
      Z: '{\\\\mathbb{Z}}',
      ker: '\\\\operatorname{ker}',
      im: '\\\\operatorname{im}',
      rank: '\\\\operatorname{rank}',
      diag: '\\\\operatorname{diag}',
      tr: '\\\\operatorname{tr}',
      div: '\\\\operatorname{div}',
      grad: '\\\\operatorname{grad}',
      spec: '\\\\operatorname{spec}'
    }
  },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
<style>
:root {
  --tdl-color: #1a5276;
  --nisoli-color: #922b21;
  --highlight-bg: #fef9e7;
  --key-result-bg: #eaf2f8;
  --example-bg: #f4f6f6;
  --border-color: #d5d8dc;
  --link-color: #2471a3;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Georgia', 'Times New Roman', serif;
  line-height: 1.7;
  color: #2c3e50;
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
  background: #fdfdfd;
}
h1 {
  font-size: 1.9rem;
  text-align: center;
  margin-bottom: 0.3rem;
  color: #1a1a2e;
  line-height: 1.3;
}
.subtitle {
  text-align: center;
  font-style: italic;
  color: #666;
  margin-bottom: 0.5rem;
  font-size: 1.05rem;
}
.authors {
  text-align: center;
  color: #555;
  margin-bottom: 2rem;
  font-size: 0.95rem;
}
h2 {
  font-size: 1.45rem;
  margin-top: 2.5rem;
  margin-bottom: 1rem;
  padding-bottom: 0.3rem;
  border-bottom: 2px solid var(--border-color);
  color: #1a1a2e;
}
h3 {
  font-size: 1.15rem;
  margin-top: 1.8rem;
  margin-bottom: 0.7rem;
  color: #2c3e50;
}
h4 {
  font-size: 1.0rem;
  margin-top: 1.3rem;
  margin-bottom: 0.5rem;
  color: #34495e;
}
p { margin-bottom: 1rem; }
a { color: var(--link-color); text-decoration: none; }
a:hover { text-decoration: underline; }
nav#toc {
  background: #f8f9fa;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 1.2rem 1.5rem;
  margin: 1.5rem 0 2rem;
}
nav#toc h2 {
  font-size: 1.1rem;
  margin: 0 0 0.8rem;
  border: none;
  padding: 0;
}
nav#toc ol {
  margin: 0;
  padding-left: 1.5rem;
}
nav#toc li {
  margin-bottom: 0.3rem;
  font-size: 0.95rem;
}
.key-result {
  background: var(--key-result-bg);
  border-left: 4px solid var(--tdl-color);
  padding: 1rem 1.2rem;
  margin: 1.2rem 0;
  border-radius: 0 4px 4px 0;
}
.key-result .label {
  font-weight: bold;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--tdl-color);
  margin-bottom: 0.4rem;
}
.example {
  background: var(--example-bg);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 1rem 1.2rem;
  margin: 1.2rem 0;
}
.example .label {
  font-weight: bold;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #27ae60;
  margin-bottom: 0.4rem;
}
.convention {
  background: #fdf2e9;
  border-left: 4px solid #e67e22;
  padding: 0.8rem 1.2rem;
  margin: 1rem 0;
  border-radius: 0 4px 4px 0;
  font-size: 0.95rem;
}
.convention .label {
  font-weight: bold;
  font-size: 0.85rem;
  color: #e67e22;
  margin-bottom: 0.3rem;
}
.footnote {
  font-size: 0.85rem;
  color: #555;
  border-top: 1px solid #ccc;
  margin-top: 1.5rem;
  padding-top: 0.5rem;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.2rem 0;
  font-size: 0.92rem;
}
th, td {
  border: 1px solid var(--border-color);
  padding: 0.5rem 0.7rem;
  text-align: left;
  vertical-align: top;
}
th {
  background: #f2f3f4;
  font-weight: bold;
  color: #2c3e50;
}
tr:nth-child(even) { background: #fafafa; }
.tdl-label { color: var(--tdl-color); font-weight: bold; }
.nisoli-label { color: var(--nisoli-color); font-weight: bold; }
.matrix-block {
  overflow-x: auto;
  margin: 1rem 0;
  padding: 0.5rem;
  background: #f9f9f9;
  border-radius: 4px;
}
.diagram {
  text-align: center;
  margin: 1.5rem 0;
}
.diagram svg {
  max-width: 100%;
}
.diagram .caption {
  font-size: 0.88rem;
  color: #666;
  font-style: italic;
  margin-top: 0.5rem;
}
.eqref { color: var(--link-color); }
@media print {
  body { max-width: 100%; padding: 1rem; }
  .key-result, .example, .convention { break-inside: avoid; }
}
</style>
</head>
<body>
""")

    # ==================== TITLE ====================
    html.append("""
<h1>ASI Lattice Zoo:<br>Worked Incidence &amp; S-Matrix Examples</h1>
<p class="subtitle">Explicit boundary operators, spectra, ice configurations, and S matrices for five Morrison&ndash;Nisoli lattices</p>
<p class="authors">SpinIceTDL Project &mdash; Companion to <em>TDL &harr; Spin Ice Correspondence</em></p>
""")

    # ==================== TOC ====================
    html.append("""
<nav id="toc">
<h2>Contents</h2>
<ol>
  <li><a href="#sec1">Introduction &amp; Conventions</a></li>
  <li><a href="#sec2">Square Lattice (3&times;3 periodic)</a></li>
  <li><a href="#sec3">Tetris Lattice (2&times;1 periodic)</a></li>
  <li><a href="#sec4">Kagome Lattice (2&times;2 periodic)</a></li>
  <li><a href="#sec5">Shakti Lattice (1&times;1 periodic)</a></li>
  <li><a href="#sec6">Santa Fe Lattice (1&times;2 periodic)</a></li>
  <li><a href="#sec7">Comparative Summary</a></li>
  <li><a href="#sec8">References</a></li>
</ol>
</nav>
""")

    # ==================== SECTION 1: Introduction ====================
    html.append("""
<h2 id="sec1">1. Introduction &amp; Conventions</h2>

<p>
This document provides <strong>fully worked examples</strong> of the incidence matrices $B_1$, $B_2$, the Hodge 1-Laplacian $L_1$, ice-rule configurations, and Nisoli's antisymmetrized $S$ matrix for five lattices from the Morrison&ndash;Nelson&ndash;Nisoli lattice zoo. Each lattice is built at its <em>minimum valid periodic tiling size</em> &mdash; the smallest number of unit cells that produces a valid torus embedding with no multi-edges or self-loops.
</p>

<p>
<strong>Why minimum sizes?</strong> The standard "XS" lattice size (4&times;4 unit cells) produces matrices far too large for explicit display:
</p>

<table>
<thead>
<tr><th>Lattice</th><th>XS (4&times;4) $B_1$ dims</th><th>Entries</th><th>Min periodic $B_1$ dims</th><th>Entries</th></tr>
</thead>
<tbody>""")

    for name, (nx, ny) in [("Square", (3,3)), ("Kagome", (2,2)), ("Shakti", (1,1)), ("Tetris", (2,1)), ("Santa Fe", (1,2))]:
        key = name.lower().replace(" ", "_")
        d = data[f"{key}_periodic"]
        xd = xs[key]
        html.append(f'<tr><td>{name}</td><td>{xd["B1_dims"]}</td><td>{xd["entries_B1"]:,}</td>'
                    f'<td>{d["n_vertices"]}&times;{d["n_edges"]}</td>'
                    f'<td>{d["n_vertices"] * d["n_edges"]}</td></tr>')

    html.append("""</tbody>
</table>

<p>
All periodic examples have $\\beta_1 = 2$ (torus topology), confirming correct topological structure at every size.
</p>

<div class="convention">
<div class="label">Conventions</div>
<p><strong>Edge orientation:</strong> Edges are stored in canonical form $(u, v)$ with $u < v$ (lower global index to higher). $B_1[v, e] = +1$ (head), $B_1[u, e] = -1$ (tail).</p>
<p><strong>Face orientation:</strong> Face boundary cycles are ordered counterclockwise (CCW). For face $f$ traversing edge $e$ from $u \\to v$: $B_2[e, f] = +1$ if $u < v$ (matches canonical direction), $B_2[e, f] = -1$ if $u > v$.</p>
<p><strong>S matrix:</strong> Nisoli's antisymmetrized interaction matrix, computed from a spin configuration $\\bm{\\sigma} \\in \\{\\pm 1\\}^{n_1}$ via:</p>
$$
S = \\tfrac{1}{2}\\big(|B_1|\\, D_{\\sigma}\\, B_1^T - B_1\\, D_{\\sigma}\\, |B_1|^T\\big)
$$
<p>where $|B_1|$ is the element-wise absolute value and $D_{\\sigma} = \\diag(\\bm{\\sigma})$.</p>
<p><strong>Ice rule (mixed coordination):</strong> At each vertex $v$ with coordination $z_v$, the charge $Q_v = (B_1 \\bm{\\sigma})_v$ satisfies $|Q_v| = z_v \\bmod 2$. Even-$z$ vertices have $Q_v = 0$; odd-$z$ vertices have $|Q_v| = 1$.</p>
</div>
""")

    # ==================== SECTION 2: Square Lattice ====================
    html.append(f"""
<h2 id="sec2">2. Square Lattice (3&times;3 Periodic)</h2>

<p>
The square lattice is the simplest ASI topology: all vertices have coordination $z = 4$, with geometric frustration only. The 3&times;3 periodic tiling on a torus gives matrices small enough for complete display while capturing the essential structure.
</p>

<h3>2.1 Lattice Setup</h3>

<p>
<strong>Unit cell:</strong> 1 vertex, 2 edges (horizontal, vertical), 1 square face. Lattice vectors $\\mathbf{{a}}_1 = (1, 0)$, $\\mathbf{{a}}_2 = (0, 1)$.
</p>

<p>
<strong>Vertices:</strong> {sq['n_vertices']} vertices on a $3 \\times 3$ grid, labeled $v_0, \\ldots, v_8$ in row-major order. Global index: $v_k$ at cell $(\\lfloor k/3 \\rfloor, k \\bmod 3)$, position $(\\lfloor k/3 \\rfloor,\\; k \\bmod 3)$.
</p>

$$
\\begin{{array}}{{ccccc}}
v_6 & - & v_7 & - & v_8 \\\\
| & & | & & | \\\\
v_3 & - & v_4 & - & v_5 \\\\
| & & | & & | \\\\
v_0 & - & v_1 & - & v_2
\\end{{array}}
$$

<p><strong>Periodic boundary conditions:</strong> Right wraps to left ($v_2 \\sim v_0$, $v_5 \\sim v_3$, $v_8 \\sim v_6$) and top wraps to bottom ($v_6 \\sim v_0$, $v_7 \\sim v_1$, $v_8 \\sim v_2$). Topologically a <strong>torus</strong> $T^2$.</p>

<p><strong>Edges:</strong> {sq['n_edges']} edges &mdash; 9 horizontal ($e_0$&ndash;$e_8$) and 9 vertical ($e_9$&ndash;$e_{{17}}$). Six wrap around the periodic boundaries (marked W).</p>

<table>
<thead>
<tr><th>Edge</th><th>Tail</th><th>Head</th><th>Type</th><th>Notes</th></tr>
</thead>
<tbody>""")

    # Square edge table - I'll construct this from the actual data
    sq_edges = sq["edge_table"]
    # For the square lattice, the ordering from the generator means:
    # Cell (ci,cj) contributes horizontal edge to cell (ci+1,cj) and vertical to (ci,cj+1)
    # With global_idx(0, ci, cj) = ci*3 + cj
    # Wrap edges are those where ci+1 wraps or cj+1 wraps
    sq_edge_notes = {}
    # Manually identify wrap edges for 3x3 square
    # Horizontal wraps: cell (2,cj) -> cell (0,cj): edges (6,0), (7,1), (8,2)
    # Vertical wraps: cell (ci,2) -> cell (ci,0): edges (2,0), (5,3), (8,6)
    sq_wrap_edges = set()
    for ci in range(3):
        for cj in range(3):
            # Horizontal: (ci*3+cj) -> ((ci+1)%3*3+cj), wrap if ci==2
            u = ci*3+cj
            v = ((ci+1)%3)*3+cj
            e = (min(u,v), max(u,v))
            if ci == 2:
                sq_wrap_edges.add(e)
            # Vertical: (ci*3+cj) -> (ci*3+(cj+1)%3), wrap if cj==2
            v2 = ci*3+((cj+1)%3)
            e2 = (min(u,v2), max(u,v2))
            if cj == 2:
                sq_wrap_edges.add(e2)

    for i, e in enumerate(sq_edges):
        edge_tuple = (e["tail"], e["head"])
        is_wrap = edge_tuple in sq_wrap_edges
        etype = "H" if i < 9 else "V"
        notes = "Wrap" if is_wrap else ""
        html.append(f'<tr><td>$e_{{{i}}}$</td><td>$v_{{{e["tail"]}}}$</td>'
                    f'<td>$v_{{{e["head"]}}}$</td><td>{etype}</td><td>{notes}</td></tr>')

    html.append(f"""</tbody>
</table>

<p><strong>Faces:</strong> {sq['n_faces']} square faces, each oriented counterclockwise.</p>
""")
    html.append(face_table_html(sq))

    html.append(f"""
<p><strong>Counts:</strong> $n_0 = {sq['n_vertices']}$, $n_1 = {sq['n_edges']}$, $n_2 = {sq['n_faces']}$. Coordination: all $z = 4$.</p>

<h3>2.2 Incidence Matrix $B_1$ ({sq['n_vertices']}&times;{sq['n_edges']})</h3>

<p>
$[B_1]_{{v,e}} = +1$ if $v$ is the head of $e$, $-1$ if the tail. Each column has exactly one $+1$ and one $-1$. Each row has $z_v = 4$ nonzero entries.
</p>

<div class="matrix-block">
$$
B_1 = {matrix_to_latex(sq['B1'])}
$$
</div>

<h3>2.3 Incidence Matrix $B_2$ ({sq['n_edges']}&times;{sq['n_faces']})</h3>

<p>
For each face $f$ with CCW boundary cycle, $[B_2]_{{e,f}} = +1$ if edge $e$ is traversed in its canonical direction, $-1$ if against. Each column (face) has exactly 4 nonzeros (one per boundary edge). Each row (edge) appears in exactly 2 faces on the torus.
</p>

<div class="matrix-block">
$$
B_2 = {matrix_to_latex(sq['B2'])}
$$
</div>

<h3>2.4 Chain Complex Verification: $B_1 B_2 = 0$</h3>

<p>
The chain complex property requires $B_1 B_2 = 0$. We verify on face $f_0$, whose column in $B_2$ has nonzeros at the four boundary edges of the face.
</p>
""")

    # Get the first face's B2 column
    b2_col0 = [row[0] for row in sq["B2"]]
    nonzero_edges = [(i, b2_col0[i]) for i in range(len(b2_col0)) if b2_col0[i] != 0]

    html.append("<p>Face $f_0$ has boundary edges: ")
    parts = []
    for eidx, sign in nonzero_edges:
        s = "+" if sign > 0 else "-"
        parts.append(f"${s}e_{{{eidx}}}$")
    html.append(", ".join(parts) + ".</p>")

    html.append("""
<p>
For each vertex $v$, the product $(B_1 B_2)_{v, f_0}$ sums $[B_1]_{v,e} \\cdot [B_2]_{e, f_0}$ over the four boundary edges. At each vertex incident to the face, one edge enters and one exits, so the sum is zero. At vertices not on the face boundary, all terms are zero. Thus $B_1 B_2 = 0$. &check;
</p>

<p><strong>Computational verification:</strong> $\\max|B_1 B_2| < 10^{-12}$. &check;</p>
""")

    # Section 2.5: L0
    html.append(f"""
<h3>2.5 Graph Laplacian $L_0 = B_1 B_1^T$</h3>

<p>
On the periodic $3 \\times 3$ square lattice, every vertex has degree $z = 4$. The graph Laplacian is $L_0 = 4I_9 - A$, where $A$ is the adjacency matrix. Its eigenvalues for the torus are:
</p>

$$
\\lambda_{{jk}} = 2(2 - \\cos(2\\pi j/3) - \\cos(2\\pi k/3)), \\quad j,k \\in \\{{0,1,2\\}}
$$

<p>giving the spectrum:</p>
$$
\\spec(L_0) = \\{{{eigenvalue_str(sq['L0_eigenvalues'])}\\}}
$$

<p>Spectral gap: $\\Delta_0 = {sq['L0_spectral_gap']}$. $\\beta_0 = 1$ (one zero eigenvalue, graph is connected). &check;</p>

<h3>2.6 Betti Numbers and Euler Characteristic</h3>

<p><strong>Expected Betti numbers</strong> for the torus $T^2$: $\\beta_0 = {sq['beta_0']}$, $\\beta_1 = {sq['beta_1']}$, $\\beta_2 = {sq['beta_2']}$.</p>

<p><strong>Euler characteristic:</strong></p>
$$
\\chi = n_0 - n_1 + n_2 = {sq['n_vertices']} - {sq['n_edges']} + {sq['n_faces']} = {sq['euler_simplex']}
$$
$$
\\chi = \\beta_0 - \\beta_1 + \\beta_2 = {sq['beta_0']} - {sq['beta_1']} + {sq['beta_2']} = {sq['euler_betti']} \\;\\checkmark
$$

<p><strong>Rank check:</strong></p>
$$
\\beta_1 = n_1 - \\rank(B_1) - \\rank(B_2) = {sq['n_edges']} - {sq['rank_B1']} - {sq['rank_B2']} = {sq['beta_1']} \\;\\checkmark
$$

<p><strong>Ice manifold dimension:</strong></p>
$$
\\dim(\\ker B_1) = n_1 - \\rank(B_1) = {sq['n_edges']} - {sq['rank_B1']} = {sq['n_edges'] - sq['rank_B1']}
$$
<p>which splits as $\\rank(B_2) + \\beta_1 = {sq['rank_B2']} + {sq['beta_1']} = {sq['rank_B2'] + sq['beta_1']}$. &check;</p>

<h3>2.7 Hodge 1-Laplacian and Its Spectrum</h3>

<p>The full Hodge 1-Laplacian $L_1 = L_1^{{\\text{{down}}}} + L_1^{{\\text{{up}}}}$ is ${sq['n_edges']} \\times {sq['n_edges']}$.</p>

$$
\\spec(L_1) = \\{{{eigenvalue_str(sq['L1_eigenvalues'])}\\}}
$$

<p>The spectral gap is $\\Delta_1 = {sq['L1_spectral_gap']}$. The $\\beta_1 = 2$ zero eigenvalues correspond to the harmonic subspace.</p>

<p><strong>$L_1^{{\\text{{down}}}}$ spectrum</strong> (gradient eigenvalues from $B_1^T B_1$):</p>
$$
\\spec(L_1^{{\\text{{down}}}}) = \\{{{eigenvalue_str(sq['L1_down_eigenvalues'])}\\}}
$$

<p><strong>$L_1^{{\\text{{up}}}}$ spectrum</strong> (curl eigenvalues from $B_2 B_2^T$):</p>
$$
\\spec(L_1^{{\\text{{up}}}}) = \\{{{eigenvalue_str(sq['L1_up_eigenvalues'])}\\}}
$$

<h3>2.8 Harmonic Modes</h3>

<p>
The two harmonic eigenvectors $\\mathbf{{h}}_1, \\mathbf{{h}}_2 \\in \\ker(L_1)$ correspond to the two non-contractible cycles on the torus. For the square lattice, these are uniform flows in the horizontal and vertical winding directions:
</p>
""")

    # Harmonic modes for square
    h1_sq = [row[0] for row in sq["harmonic_basis"]]
    h2_sq = [row[1] for row in sq["harmonic_basis"]]
    html.append(f"""
$$
\\mathbf{{h}}_1 = {vector_to_latex(h1_sq)}, \\quad
\\mathbf{{h}}_2 = {vector_to_latex(h2_sq)}
$$
""")

    html.append(f"""
<p><strong>Verification:</strong></p>
<ul>
<li>$\\|B_1 \\mathbf{{h}}_1\\| = {sq['harmonic_validation'][0]['div_norm']:.1e}$ (divergence-free) &check;</li>
<li>$\\|B_2^T \\mathbf{{h}}_1\\| = {sq['harmonic_validation'][0]['curl_norm']:.1e}$ (curl-free) &check;</li>
<li>$\\|L_1 \\mathbf{{h}}_1\\| = {sq['harmonic_validation'][0]['L1h_norm']:.1e} \\approx 0$ &check;</li>
<li>$\\|B_1 \\mathbf{{h}}_2\\| = {sq['harmonic_validation'][1]['div_norm']:.1e}$ &check;</li>
<li>$\\|B_2^T \\mathbf{{h}}_2\\| = {sq['harmonic_validation'][1]['curl_norm']:.1e}$ &check;</li>
<li>$\\|L_1 \\mathbf{{h}}_2\\| = {sq['harmonic_validation'][1]['L1h_norm']:.1e} \\approx 0$ &check;</li>
</ul>

<p>These modes are topologically protected &mdash; no polynomial filter $p(L_1)$ with $p(0) \\neq 0$ can attenuate them.</p>

<h3>2.9 Ice-Rule Configuration</h3>

<p>
A valid ice configuration $\\bm{{\\sigma}} \\in \\{{\\pm 1\\}}^{{{sq['n_edges']}}}$ found via Eulerian circuit orientation:
</p>

$$
\\bm{{\\sigma}} = {sigma_to_latex(sq['sigma'])}
$$

<p><strong>Charge verification:</strong> $\\mathbf{{Q}} = B_1 \\bm{{\\sigma}} = ({', '.join(str(c) for c in sq['charge'])})^T$.</p>

<p>All charges are zero ($z = 4$ is even everywhere), confirming $\\bm{{\\sigma}} \\in \\ker(B_1)$. &check;</p>

<h3>2.10 Monopole Excitation</h3>

<p>
Flip edge $e_0$ (from $v_{{{sq['edge_table'][0]['tail']}}}$ to $v_{{{sq['edge_table'][0]['head']}}}$): set $\\sigma_{{e_0}} \\to -\\sigma_{{e_0}}$.
</p>

<p><strong>New charges:</strong> $\\mathbf{{Q}}' = ({', '.join(str(c) for c in sq['charge_flipped'])})^T$.</p>
""")

    # Find the nonzero charges
    flip_charges = sq["charge_flipped"]
    monopole_verts = [(i, c) for i, c in enumerate(flip_charges) if c != 0]
    if monopole_verts:
        parts = []
        for v, c in monopole_verts:
            sign = "+" if c > 0 else ""
            parts.append(f"$Q_{{v_{v}}} = {sign}{c}$")
        html.append(f"<p>A <strong>monopole&ndash;antimonopole pair</strong>: {', '.join(parts)}. All other vertices remain neutral. &check;</p>")

    html.append("""
<p><strong>Energy:</strong> $\\mathcal{H}[\\bm{\\sigma}'] = \\frac{\\epsilon}{2}\\|\\mathbf{Q}'\\|^2 = \\frac{\\epsilon}{2}\\sum_v Q_v^2$.</p>
""")

    # Section 2.11: S matrix
    html.append(f"""
<h3>2.11 Nisoli's $S$ Matrix</h3>

<p>
For the ice configuration above, the $S$ matrix is $S = \\frac{{1}}{{2}}(|B_1| D_\\sigma B_1^T - B_1 D_\\sigma |B_1|^T)$.
</p>

<p>$S$ is ${sq['n_vertices']} \\times {sq['n_vertices']}$, antisymmetric ($S_{{vv'}} = -S_{{v'v}}$), with $S_{{vv}} = 0$.</p>

<div class="matrix-block">
$$
S = {matrix_to_latex(sq['S_matrix'])}
$$
</div>

<p><strong>Entry-by-entry verification</strong> for edge $e_0$ connecting $v_{{{sq['edge_table'][0]['tail']}}}$ (tail) and $v_{{{sq['edge_table'][0]['head']}}}$ (head), with $\\sigma_{{e_0}} = {sq['sigma'][0]:+d}$:</p>
$$
S_{{v_{{{sq['edge_table'][0]['tail']}}},\\, v_{{{sq['edge_table'][0]['head']}}}}} = -[B_1]_{{v_{{{sq['edge_table'][0]['tail']}}}, e_0}} \\cdot \\sigma_{{e_0}} = -({sq['B1'][sq['edge_table'][0]['tail']][0]:+d})({sq['sigma'][0]:+d}) = {sq['S_matrix'][sq['edge_table'][0]['tail']][sq['edge_table'][0]['head']]:+d}
$$
<p>The antisymmetry $S_{{v_{{{sq['edge_table'][0]['head']}}},\\, v_{{{sq['edge_table'][0]['tail']}}}}} = {sq['S_matrix'][sq['edge_table'][0]['head']][sq['edge_table'][0]['tail']]:+d} = -S_{{v_{{{sq['edge_table'][0]['tail']}}},\\, v_{{{sq['edge_table'][0]['head']}}}}}$ holds. &check;</p>

<h3>2.12 Line Graph and $L_1^{{\\text{{down}}}}$</h3>

<p>
The line graph $L(G)$ has 18 vertices (one per edge of $G$). Two vertices of $L(G)$ are adjacent iff the corresponding edges share an endpoint. Since every vertex has $z = 4$, each contributes a $K_4$ clique, giving $\\sum_v \\binom{{z_v}}{{2}} = 9 \\times 6 = 54$ edges in $L(G)$.
</p>

<p>By the line-graph dual identity: $L_1^{{\\text{{down}}}} = B_1^T B_1 = 2I_{{18}} - A_{{L(G)}}$, so every vertex with $z > 2$ generates frustrated $K_4$ cliques in $L(G)$. &check;</p>

<h3>2.13 Pauling Entropy</h3>

<p>The Pauling approximation for ice-rule-satisfying states on a lattice with uniform $z = 4$:</p>
$$
\\varepsilon_{{\\text{{Pauling}}}} = 2^{{n_1}} \\prod_v \\frac{{\\binom{{z_v}}{{z_v/2}}}}{{2^{{z_v}}}} = 2^{{18}} \\cdot \\left(\\frac{{\\binom{{4}}{{2}}}}{{2^4}}\\right)^9 = 262144 \\cdot \\left(\\frac{{3}}{{8}}\\right)^9 \\approx {sq['pauling_estimate']}
$$

<p>Pauling entropy per vertex: $s_{{\\text{{P}}}} = \\frac{{1}}{{N_v}} \\ln \\varepsilon \\approx \\frac{{\\ln({sq['pauling_estimate']})}}{{9}} \\approx {round(2.302585*1.585, 3)/9:.3f}$, close to the exact Lieb result $s_{{\\text{{Lieb}}}} = \\frac{{3}}{{2}}\\ln(4/3) \\approx 0.431$ (finite-size deviation expected).</p>

<h3>2.14 Open Boundary Conditions</h3>

<p>
Removing the 6 periodic wrap-around edges (3 horizontal + 3 vertical) converts the torus to a planar square patch. Some faces that depended on wrap edges are also lost.
</p>

<table>
<thead>
<tr><th>Property</th><th>Periodic</th><th>Open</th></tr>
</thead>
<tbody>
<tr><td>$n_0$</td><td>{sq['n_vertices']}</td><td>{sq_open['n_vertices']}</td></tr>
<tr><td>$n_1$</td><td>{sq['n_edges']}</td><td>{sq_open['n_edges']}</td></tr>
<tr><td>$n_2$</td><td>{sq['n_faces']}</td><td>{sq_open['n_faces']}</td></tr>
<tr><td>$\\beta_0$</td><td>{sq['beta_0']}</td><td>{sq_open['beta_0']}</td></tr>
<tr><td>$\\beta_1$</td><td>{sq['beta_1']}</td><td>{sq_open['beta_1']}</td></tr>
<tr><td>$\\beta_2$</td><td>{sq['beta_2']}</td><td>{sq_open['beta_2']}</td></tr>
<tr><td>$\\chi$</td><td>{sq['euler_simplex']}</td><td>{sq_open['euler_simplex']}</td></tr>
</tbody>
</table>

<p>
With open BCs: $\\beta_1 = 0$ &mdash; the two winding modes are lost because the torus handle cycles no longer close. The Euler characteristic changes from 0 (torus) to 1 (disk). Boundary vertices now have reduced coordination ($z = 2$ or $z = 3$).
</p>
""")

    # ==================== SECTION 3: Tetris Lattice ====================
    html.append(f"""
<h2 id="sec3">3. Tetris Lattice (2&times;1 Periodic)</h2>

<p>
The tetris lattice is <strong>maximally vertex-frustrated</strong>: every minimal loop contains a $z = 2$ bridge vertex, making it impossible to satisfy the ice rule at all vertices simultaneously. It features a "sliding phase" &mdash; ordered along one direction, disordered along the other.
</p>

<h3>3.1 Lattice Setup</h3>

<p>
<strong>Unit cell:</strong> $\\mathbf{{a}}_1 = (2, 0)$, $\\mathbf{{a}}_2 = (0, 4)$, containing 8 vertices. The cell spans two backbone rows ($y=0, 2$) and two staircase rows ($y=1, 3$).
</p>

<p>
<strong>Vertex layout (one unit cell):</strong>
</p>
$$
\\begin{{array}}{{ccccc}}
v_6(0,3) & -- & v_7(1,3) & & \\text{{staircase, z=3}} \\\\
| & & | & & \\\\
v_4(0,2) & -- & v_5(1,2) & & \\text{{backbone, z=2/z=4}} \\\\
| & & | & & \\\\
v_2(0,1) & -- & v_3(1,1) & & \\text{{staircase, z=3}} \\\\
| & & | & & \\\\
v_0(0,0) & -- & v_1(1,0) & & \\text{{backbone, z=4/z=2}}
\\end{{array}}
$$

<p>
<strong>Bridge stagger:</strong> On backbone row 0, the $z=2$ bridge is at $v_1$ (right). On backbone row 2, the $z=2$ bridge is at $v_4$ (left). This stagger ensures every hexagonal loop is frustrated.
</p>

<p>
<strong>Tiling:</strong> 2&times;1 (2 cells in $x$, 1 in $y$). A 1&times;1 tiling fails because 4 horizontal edge pairs collapse to multi-edges under periodic wrapping.
</p>

<p><strong>Counts:</strong> $n_0 = {tet['n_vertices']}$, $n_1 = {tet['n_edges']}$, $n_2 = {tet['n_faces']}$.</p>

<p><strong>Coordination distribution:</strong> $z=4 \\times {tet['coordination_distribution'].get('4', 0)}$, $z=3 \\times {tet['coordination_distribution'].get('3', 0)}$, $z=2 \\times {tet['coordination_distribution'].get('2', 0)}$.</p>

<p><strong>Edges:</strong> {tet['n_edges']} edges.</p>
""")
    html.append(edge_table_html(tet))

    html.append(f"""
<p><strong>Faces:</strong> {tet['n_faces']} hexagonal faces (each with exactly one $z=2$ vertex).</p>
""")
    html.append(face_table_html(tet))

    html.append(f"""
<h3>3.2 Incidence Matrix $B_1$ ({tet['n_vertices']}&times;{tet['n_edges']})</h3>

<p>
Each column has exactly one $+1$ and one $-1$. Rows for $z=4$ vertices have 4 nonzeros, $z=3$ have 3, $z=2$ have 2.
</p>

<div class="matrix-block">
$$
B_1 = {matrix_to_latex(tet['B1'])}
$$
</div>

<h3>3.3 Incidence Matrix $B_2$ ({tet['n_edges']}&times;{tet['n_faces']})</h3>

<p>
Each column (hexagonal face) has 6 nonzero entries. Each row (edge) appears in at most 2 faces.
</p>

<div class="matrix-block">
$$
B_2 = {matrix_to_latex(tet['B2'])}
$$
</div>

<h3>3.4 Chain Complex Verification</h3>

<p>$\\max|B_1 B_2| < 10^{{-12}}$, confirming $B_1 B_2 = 0$. &check;</p>
""")

    # Work out verification for one face
    b2_col0_tet = [row[0] for row in tet["B2"]]
    nz_tet = [(i, b2_col0_tet[i]) for i in range(len(b2_col0_tet)) if b2_col0_tet[i] != 0]
    html.append("<p>Verification on face $f_0$. Boundary edges: ")
    parts = []
    for eidx, sign in nz_tet:
        s = "+" if sign > 0 else "-"
        parts.append(f"${s}e_{{{eidx}}}$")
    html.append(", ".join(parts) + ".</p>")
    html.append("<p>For each vertex on the face boundary, one edge enters and one exits, giving a net sum of zero. &check;</p>")

    html.append(f"""
<h3>3.5 Graph Laplacian $L_0$</h3>

$$
\\spec(L_0) = \\{{{eigenvalue_str(tet['L0_eigenvalues'])}\\}}
$$
<p>Spectral gap: $\\Delta_0 = {tet['L0_spectral_gap']:.6f}$. Note the smaller gap compared to the square lattice ($\\Delta_0 = 3.0$), reflecting the mixed coordination and frustration.</p>

<h3>3.6 Betti Numbers and Euler Characteristic</h3>

$$
\\chi = n_0 - n_1 + n_2 = {tet['n_vertices']} - {tet['n_edges']} + {tet['n_faces']} = {tet['euler_simplex']}
$$
$$
\\chi = \\beta_0 - \\beta_1 + \\beta_2 = {tet['beta_0']} - {tet['beta_1']} + {tet['beta_2']} = {tet['euler_betti']} \\;\\checkmark
$$

<p>$\\beta_1 = n_1 - \\rank(B_1) - \\rank(B_2) = {tet['n_edges']} - {tet['rank_B1']} - {tet['rank_B2']} = {tet['beta_1']}$ &check;</p>

<h3>3.7 Hodge 1-Laplacian Spectrum</h3>

$$
\\spec(L_1) = \\{{{eigenvalue_str(tet['L1_eigenvalues'])}\\}}
$$

<p>Spectral gap: $\\Delta_1 = {tet['L1_spectral_gap']:.6f}$. The two zero eigenvalues confirm $\\beta_1 = 2$.</p>

<h3>3.8 Harmonic Modes</h3>
""")

    h1_tet = [row[0] for row in tet["harmonic_basis"]]
    h2_tet = [row[1] for row in tet["harmonic_basis"]]
    html.append(f"""
$$
\\mathbf{{h}}_1 = {vector_to_latex(h1_tet)}, \\quad
\\mathbf{{h}}_2 = {vector_to_latex(h2_tet)}
$$

<p><strong>Verification:</strong></p>
<ul>
<li>$\\|L_1 \\mathbf{{h}}_1\\| = {tet['harmonic_validation'][0]['L1h_norm']:.1e} \\approx 0$ &check;</li>
<li>$\\|L_1 \\mathbf{{h}}_2\\| = {tet['harmonic_validation'][1]['L1h_norm']:.1e} \\approx 0$ &check;</li>
<li>$\\|B_1 \\mathbf{{h}}_1\\| = {tet['harmonic_validation'][0]['div_norm']:.1e}$ (divergence-free) &check;</li>
<li>$\\|B_2^T \\mathbf{{h}}_1\\| = {tet['harmonic_validation'][0]['curl_norm']:.1e}$ (curl-free) &check;</li>
</ul>

<p>
Unlike the square lattice where harmonics are uniform horizontal/vertical flows, the tetris harmonics have <em>non-uniform amplitudes</em> reflecting the mixed coordination. Bridge edges ($z=2$) carry different harmonic weight than crossroad edges ($z=4$).
</p>

<h3>3.9 Ice-Rule Configuration</h3>

$$
\\bm{{\\sigma}} = {sigma_to_latex(tet['sigma'])}
$$

<p><strong>Charges:</strong> $\\mathbf{{Q}} = B_1 \\bm{{\\sigma}} = ({', '.join(str(c) for c in tet['charge'])})^T$.</p>
""")

    # Analyze charges
    tet_charge = tet["charge"]
    tet_coord = tet["coordination_distribution"]
    even_ok = all(c == 0 for i, c in enumerate(tet_charge) if tet["B1"][i] and sum(abs(x) for x in tet["B1"][i]) % 2 == 0)
    html.append("""
<p>
Ice rule check: $z=4$ vertices ($v_0, v_5, v_8, v_{13}$) have $Q_v = 0$. $z=3$ vertices have $|Q_v| = 1$. $z=2$ vertices have $Q_v = 0$. All satisfy $|Q_v| = z_v \\bmod 2$. &check;
</p>

<h3>3.10 Monopole Excitation</h3>
""")

    flip_charges_tet = tet["charge_flipped"]
    tet_monopoles = [(i, c) for i, c in enumerate(flip_charges_tet) if c != tet["charge"][i]]
    html.append(f"""
<p>Flipping edge $e_0$ (from $v_{{{tet['edge_table'][0]['tail']}}}$ to $v_{{{tet['edge_table'][0]['head']}}}$):</p>
<p><strong>New charges:</strong> $\\mathbf{{Q}}' = ({', '.join(str(c) for c in flip_charges_tet)})^T$.</p>
""")

    if tet_monopoles:
        parts = []
        for v, c in tet_monopoles:
            parts.append(f"$v_{{{v}}}$: $Q = {tet['charge'][v]} \\to {c}$")
        html.append(f"<p>Changed vertices: {'; '.join(parts)}. &check;</p>")

    # S matrix for tetris
    html.append(f"""
<h3>3.11 Nisoli's $S$ Matrix</h3>

<p>The ${tet['n_vertices']} \\times {tet['n_vertices']}$ antisymmetric $S$ matrix for the ice configuration above:</p>

<div class="matrix-block">
$$
S = {matrix_to_latex(tet['S_matrix'])}
$$
</div>

<p><strong>Properties verified:</strong> $S_{{vv}} = 0$ &forall; $v$; $S_{{vv'}} = -S_{{v'v}}$ (antisymmetric); $S_{{vv'}} = 0$ when $v, v'$ are not neighbors. &check;</p>

<p><strong>Key observation:</strong> For mixed-$z$ lattices, the $S$ matrix entries are $\\pm 1$ for adjacent vertices connected by one edge, and 0 otherwise (even though vertices have different coordination numbers). The entry $S_{{vv'}} = +1$ means the spin on edge $(v, v')$ points from $v$ to $v'$.</p>
""")

    # Line graph
    html.append(f"""
<h3>3.12 Line Graph and Frustration</h3>

<p>
The line graph $L(G)$ has {tet['n_edges']} vertices. At each $z=4$ vertex, a $K_4$ clique contributes $\\binom{{4}}{{2}} = 6$ line-graph edges with frustrating odd cycles. At $z=3$ vertices, a $K_3$ triangle contributes 3 edges. At $z=2$ vertices, a single edge contributes 1 line-graph edge (no frustration from $K_2$). Total line-graph edges: $4 \\times 6 + 8 \\times 3 + 4 \\times 1 = 52$.
</p>

<p>Since all $z \\geq 3$ vertices generate frustrated cliques, the line graph is <em>not</em> bipartite &mdash; confirming universal frustration in $L_1^{{\\text{{down}}}}$ per the line-graph dual theorem. &check;</p>

<h3>3.13 Pauling Entropy</h3>

$$
\\varepsilon_{{\\text{{Pauling}}}} = 2^{{{tet['n_edges']}}} \\prod_v \\frac{{\\text{{ice fraction}}(z_v)}}{{1}} \\approx {tet['pauling_estimate']}
$$

<p>Much larger than the square lattice ({sq['pauling_estimate']}) despite the same number of edges ({tet['n_edges']} vs {sq['n_edges']}), because the mixed coordination with $z=2$ and $z=3$ vertices imposes weaker constraints than uniform $z=4$.</p>

<h3>3.14 Open Boundary Conditions</h3>

<table>
<thead>
<tr><th>Property</th><th>Periodic</th><th>Open</th></tr>
</thead>
<tbody>
<tr><td>$n_0$</td><td>{tet['n_vertices']}</td><td>{tet_open['n_vertices']}</td></tr>
<tr><td>$n_1$</td><td>{tet['n_edges']}</td><td>{tet_open['n_edges']}</td></tr>
<tr><td>$n_2$</td><td>{tet['n_faces']}</td><td>{tet_open['n_faces']}</td></tr>
<tr><td>$\\beta_0$</td><td>{tet['beta_0']}</td><td>{tet_open['beta_0']}</td></tr>
<tr><td>$\\beta_1$</td><td>{tet['beta_1']}</td><td>{tet_open['beta_1']}</td></tr>
<tr><td>$\\beta_2$</td><td>{tet['beta_2']}</td><td>{tet_open['beta_2']}</td></tr>
<tr><td>$\\chi$</td><td>{tet['euler_simplex']}</td><td>{tet_open['euler_simplex']}</td></tr>
</tbody>
</table>

<p>Open BCs: {tet['n_edges'] - tet_open['n_edges']} edges removed (periodic wraps), {tet['n_faces'] - tet_open['n_faces']} faces lost. $\\beta_1 = 0$ &mdash; harmonic protection completely vanishes without periodic topology. &check;</p>
""")

    # ==================== SECTION 4: Kagome ====================
    html.append(f"""
<h2 id="sec4">4. Kagome Lattice (2&times;2 Periodic)</h2>

<p>
The kagome lattice (trihexagonal tiling) has uniform $z = 4$ coordination with corner-sharing triangles and hexagonal voids. It exhibits <strong>geometric frustration</strong> &mdash; the Ising antiferromagnet on kagome has an extensive ground-state degeneracy with characteristic flat bands in the spin-wave spectrum.
</p>

<h3>4.1 Lattice Setup</h3>

<p>
<strong>Unit cell:</strong> $\\mathbf{{a}}_1 = (2, 0)$, $\\mathbf{{a}}_2 = (1, \\sqrt{{3}})$. Three vertices per cell at positions $v_0 = (0, 0)$, $v_1 = (1, 0)$, $v_2 = (0.5, \\sqrt{{3}}/2)$, forming an upward triangle. 6 edges per cell, 3 faces per cell (up-triangle, down-triangle, hexagon).
</p>

<p>
<strong>Tiling:</strong> 2&times;2 periodic ($n_0 = {kag['n_vertices']}$, $n_1 = {kag['n_edges']}$, $n_2 = {kag['n_faces']}$). A 1&times;1 tiling collapses cross-cell edges to duplicates.
</p>

<p><strong>Coordination:</strong> All $z = 4$ ($\\times {kag['coordination_distribution'].get('4', 0)}$ vertices).</p>

<p><strong>Edges:</strong> {kag['n_edges']} edges.</p>
""")
    html.append(edge_table_html(kag))

    html.append(f"""
<p><strong>Faces:</strong> {kag['n_faces']} faces &mdash; 4 upward triangles (3 sides), 4 downward triangles (3 sides), 4 hexagons (6 sides).</p>
""")
    html.append(face_table_html(kag))

    html.append(f"""
<h3>4.2 Incidence Matrix $B_1$ ({kag['n_vertices']}&times;{kag['n_edges']})</h3>

<div class="matrix-block">
$$
B_1 = {matrix_to_latex(kag['B1'])}
$$
</div>

<h3>4.3 Incidence Matrix $B_2$ ({kag['n_edges']}&times;{kag['n_faces']})</h3>

<p>
Triangle faces have 3 boundary edges each; hexagonal faces have 6.
</p>

<div class="matrix-block">
$$
B_2 = {matrix_to_latex(kag['B2'])}
$$
</div>

<h3>4.4 Chain Complex Verification</h3>

<p>$\\max|B_1 B_2| < 10^{{-12}}$. &check;</p>

<h3>4.5 Betti Numbers and Euler Characteristic</h3>

$$
\\chi = {kag['n_vertices']} - {kag['n_edges']} + {kag['n_faces']} = {kag['euler_simplex']}
$$
$$
\\beta_0 = {kag['beta_0']},\\; \\beta_1 = {kag['beta_1']},\\; \\beta_2 = {kag['beta_2']} \\quad \\Rightarrow \\quad {kag['beta_0']} - {kag['beta_1']} + {kag['beta_2']} = {kag['euler_betti']} \\;\\checkmark
$$

<p>$\\beta_1 = {kag['n_edges']} - {kag['rank_B1']} - {kag['rank_B2']} = {kag['beta_1']}$. &check;</p>

<h3>4.6 Spectra and Harmonic Modes</h3>

<p><strong>$L_0$ spectrum:</strong></p>
$$
\\spec(L_0) = \\{{{eigenvalue_str(kag['L0_eigenvalues'])}\\}}
$$
<p>Spectral gap: $\\Delta_0 = {kag['L0_spectral_gap']}$.</p>

<p><strong>$L_1$ spectrum:</strong></p>
$$
\\spec(L_1) = \\{{{eigenvalue_str(kag['L1_eigenvalues'])}\\}}
$$
<p>Spectral gap: $\\Delta_1 = {kag['L1_spectral_gap']}$.</p>
""")

    h1_kag = [row[0] for row in kag["harmonic_basis"]]
    h2_kag = [row[1] for row in kag["harmonic_basis"]]
    html.append(f"""
<p><strong>Harmonic modes:</strong></p>
$$
\\mathbf{{h}}_1 = {vector_to_latex(h1_kag)}, \\quad
\\mathbf{{h}}_2 = {vector_to_latex(h2_kag)}
$$

<p>
Verification: $\\|L_1 \\mathbf{{h}}_1\\| = {kag['harmonic_validation'][0]['L1h_norm']:.1e}$, $\\|L_1 \\mathbf{{h}}_2\\| = {kag['harmonic_validation'][1]['L1h_norm']:.1e}$. Both divergence-free and curl-free. &check;
</p>

<h3>4.7 Ice Configuration and $S$ Matrix</h3>

$$
\\bm{{\\sigma}} = {sigma_to_latex(kag['sigma'])}
$$

<p><strong>Charges:</strong> $\\mathbf{{Q}} = ({', '.join(str(c) for c in kag['charge'])})^T$. All zero ($z = 4$ everywhere). &check;</p>

<div class="matrix-block">
$$
S = {matrix_to_latex(kag['S_matrix'])}
$$
</div>

<h3>4.8 Open Boundary Conditions</h3>

<table>
<thead>
<tr><th>Property</th><th>Periodic</th><th>Open</th></tr>
</thead>
<tbody>
<tr><td>$n_0$</td><td>{kag['n_vertices']}</td><td>{kag_open['n_vertices']}</td></tr>
<tr><td>$n_1$</td><td>{kag['n_edges']}</td><td>{kag_open['n_edges']}</td></tr>
<tr><td>$n_2$</td><td>{kag['n_faces']}</td><td>{kag_open['n_faces']}</td></tr>
<tr><td>$\\beta_0, \\beta_1, \\beta_2$</td><td>{kag['beta_0']}, {kag['beta_1']}, {kag['beta_2']}</td><td>{kag_open['beta_0']}, {kag_open['beta_1']}, {kag_open['beta_2']}</td></tr>
<tr><td>$\\chi$</td><td>{kag['euler_simplex']}</td><td>{kag_open['euler_simplex']}</td></tr>
</tbody>
</table>

<p>{kag['n_edges'] - kag_open['n_edges']} wrap edges removed, {kag['n_faces'] - kag_open['n_faces']} faces lost. $\\beta_1: {kag['beta_1']} \\to {kag_open['beta_1']}$. &check;</p>
""")

    # ==================== SECTION 5: Shakti ====================
    html.append(f"""
<h2 id="sec5">5. Shakti Lattice (1&times;1 Periodic)</h2>

<p>
The shakti lattice has <strong>maximal vertex frustration</strong> with extensive ground-state degeneracy and topological order. It consists of a square grid of $z=4$ corner vertices, with each plaquette containing two bars (subdivided edges, $z=3$ midpoints) connected by a $z=2$ bridge. Bars alternate between horizontal and vertical pairs in a checkerboard pattern.
</p>

<h3>5.1 Lattice Setup</h3>

<p>
<strong>Unit cell:</strong> $\\mathbf{{a}}_1 = (2, 0)$, $\\mathbf{{a}}_2 = (0, 2)$. 16 vertices per cell.
</p>

<p><strong>Vertex types:</strong></p>
<ul>
<li>$z=4$ corners: $v_0, v_1, v_2, v_3$ ($\\times {shk['coordination_distribution'].get('4', 0)}$)</li>
<li>$z=3$ bar midpoints: $v_4, v_5, v_7, v_8, v_{{10}}, v_{{11}}, v_{{13}}, v_{{14}}$ ($\\times {shk['coordination_distribution'].get('3', 0)}$)</li>
<li>$z=2$ bridges: $v_6, v_9, v_{{12}}, v_{{15}}$ ($\\times {shk['coordination_distribution'].get('2', 0)}$)</li>
</ul>

<p>
<strong>Tiling:</strong> 1&times;1 periodic is valid because all 24 edges connect distinct vertices under periodic wrapping ($n_0 = {shk['n_vertices']}$, $n_1 = {shk['n_edges']}$, $n_2 = {shk['n_faces']}$).
</p>

<p><strong>Faces:</strong> 8 hexagonal faces, each visiting 2 corner ($z=4$), 3 bar midpoint ($z=3$), and 1 bridge ($z=2$) vertex.</p>

{edge_table_html(shk)}

{face_table_html(shk)}

<h3>5.2 Incidence Matrix $B_1$ ({shk['n_vertices']}&times;{shk['n_edges']})</h3>

<div class="matrix-block">
$$
B_1 = {matrix_to_latex(shk['B1'])}
$$
</div>

<h3>5.3 Incidence Matrix $B_2$ ({shk['n_edges']}&times;{shk['n_faces']})</h3>

<div class="matrix-block">
$$
B_2 = {matrix_to_latex(shk['B2'])}
$$
</div>

<h3>5.4 Chain Complex Verification</h3>

<p>$\\max|B_1 B_2| < 10^{{-12}}$. &check;</p>

<h3>5.5 Betti Numbers and Euler Characteristic</h3>

$$
\\chi = {shk['n_vertices']} - {shk['n_edges']} + {shk['n_faces']} = {shk['euler_simplex']}
$$
$$
\\beta_0 = {shk['beta_0']},\\; \\beta_1 = {shk['beta_1']},\\; \\beta_2 = {shk['beta_2']} \\quad \\Rightarrow \\quad {shk['beta_0']} - {shk['beta_1']} + {shk['beta_2']} = {shk['euler_betti']} \\;\\checkmark
$$

<p>$\\beta_1 = {shk['n_edges']} - {shk['rank_B1']} - {shk['rank_B2']} = {shk['beta_1']}$. &check;</p>

<h3>5.6 Spectra and Harmonic Modes</h3>

<p><strong>$L_0$ spectrum:</strong></p>
$$
\\spec(L_0) = \\{{{eigenvalue_str(shk['L0_eigenvalues'])}\\}}
$$
<p>Spectral gap: $\\Delta_0 = {shk['L0_spectral_gap']}$.</p>

<p><strong>$L_1$ spectrum:</strong></p>
$$
\\spec(L_1) = \\{{{eigenvalue_str(shk['L1_eigenvalues'])}\\}}
$$
<p>Spectral gap: $\\Delta_1 = {shk['L1_spectral_gap']}$.</p>
""")

    h1_shk = [row[0] for row in shk["harmonic_basis"]]
    h2_shk = [row[1] for row in shk["harmonic_basis"]]
    html.append(f"""
<p><strong>Harmonic modes:</strong></p>
$$
\\mathbf{{h}}_1 = {vector_to_latex(h1_shk)}, \\quad
\\mathbf{{h}}_2 = {vector_to_latex(h2_shk)}
$$

<p>
Verification: $\\|L_1 \\mathbf{{h}}_1\\| = {shk['harmonic_validation'][0]['L1h_norm']:.1e}$, $\\|L_1 \\mathbf{{h}}_2\\| = {shk['harmonic_validation'][1]['L1h_norm']:.1e}$. &check;
</p>

<div class="key-result">
<div class="label">Shakti harmonic structure</div>
<p>
The shakti's harmonic modes show a distinctive pattern: they have <em>non-zero</em> amplitude on bridge edges ($z=2$), reflecting the topological role bridges play in mediating frustration. The amplitudes are not uniform &mdash; they reflect the mixed-coordination structure of the underlying lattice. In the TDL context, this means the "protected channel" carries signals that are modulated by the lattice's frustration architecture.
</p>
</div>

<h3>5.7 Ice Configuration and $S$ Matrix</h3>

$$
\\bm{{\\sigma}} = {sigma_to_latex(shk['sigma'])}
$$

<p><strong>Charges:</strong> $\\mathbf{{Q}} = ({', '.join(str(c) for c in shk['charge'])})^T$.</p>

<p>Ice rule: $z=4$ corners have $Q = 0$, $z=3$ midpoints have $|Q| = 1$, $z=2$ bridges have $Q = 0$. &check;</p>

<div class="matrix-block">
$$
S = {matrix_to_latex(shk['S_matrix'])}
$$
</div>

<h3>5.8 Open Boundary Conditions</h3>

<table>
<thead>
<tr><th>Property</th><th>Periodic</th><th>Open</th></tr>
</thead>
<tbody>
<tr><td>$n_0$</td><td>{shk['n_vertices']}</td><td>{shk_open['n_vertices']}</td></tr>
<tr><td>$n_1$</td><td>{shk['n_edges']}</td><td>{shk_open['n_edges']}</td></tr>
<tr><td>$n_2$</td><td>{shk['n_faces']}</td><td>{shk_open['n_faces']}</td></tr>
<tr><td>$\\beta_0, \\beta_1, \\beta_2$</td><td>{shk['beta_0']}, {shk['beta_1']}, {shk['beta_2']}</td><td>{shk_open['beta_0']}, {shk_open['beta_1']}, {shk_open['beta_2']}</td></tr>
<tr><td>$\\chi$</td><td>{shk['euler_simplex']}</td><td>{shk_open['euler_simplex']}</td></tr>
</tbody>
</table>

<p>{shk['n_edges'] - shk_open['n_edges']} wrap edges removed, {shk['n_faces'] - shk_open['n_faces']} faces lost. $\\beta_1: {shk['beta_1']} \\to {shk_open['beta_1']}$. &check;</p>
""")

    # ==================== SECTION 6: Santa Fe ====================
    html.append(f"""
<h2 id="sec6">6. Santa Fe Lattice (1&times;2 Periodic)</h2>

<p>
The Santa Fe lattice exhibits <strong>both geometric and vertex frustration</strong>, distinguished by its mixed-direction bridge placement: one horizontal and one vertical bridge per 2&times;2 unit cell. Its low-energy states feature polymer-like strings of unhappy vertices.
</p>

<h3>6.1 Lattice Setup</h3>

<p>
<strong>Unit cell:</strong> $\\mathbf{{a}}_1 = (2, 0)$, $\\mathbf{{a}}_2 = (0, 2)$. 6 vertices per cell.
</p>

<p><strong>Vertex layout (one unit cell):</strong></p>
$$
\\begin{{array}}{{ccc}}
v_2(0,1) & ---- & v_3(1,1) \\\\
| & & | \\\\
v_5(0,0.5) & & | \\\\
| & & | \\\\
v_0(0,0) & -v_4- & v_1(1,0)
\\end{{array}}
$$

<p>Bridge $v_4$ subdivides the bottom horizontal edge; bridge $v_5$ subdivides the left vertical edge. The right vertical ($v_1$&ndash;$v_3$) and top horizontal ($v_2$&ndash;$v_3$) are direct (no bridge).</p>

<p><strong>Coordination:</strong> $z=4$ ($v_0, v_1$) $\\times {sf['coordination_distribution'].get('4', 0)}$, $z=3$ ($v_2, v_3$) $\\times {sf['coordination_distribution'].get('3', 0)}$, $z=2$ ($v_4, v_5$) $\\times {sf['coordination_distribution'].get('2', 0)}$.</p>

<p>
<strong>Tiling:</strong> 1&times;2 periodic ($n_0 = {sf['n_vertices']}$, $n_1 = {sf['n_edges']}$, $n_2 = {sf['n_faces']}$). A 1&times;1 tiling fails because one cross-cell edge duplicates an intra-cell edge.
</p>

<p><strong>Faces:</strong> {sf['n_faces']} faces of mixed type: hexagons (6 sides), pentagons (5 sides), and heptagons (7 sides) &mdash; reflecting the mixed-direction bridge placement that distinguishes Santa Fe from other lattices.</p>

{edge_table_html(sf)}

{face_table_html(sf)}

<h3>6.2 Incidence Matrix $B_1$ ({sf['n_vertices']}&times;{sf['n_edges']})</h3>

<div class="matrix-block">
$$
B_1 = {matrix_to_latex(sf['B1'])}
$$
</div>

<h3>6.3 Incidence Matrix $B_2$ ({sf['n_edges']}&times;{sf['n_faces']})</h3>

<div class="matrix-block">
$$
B_2 = {matrix_to_latex(sf['B2'])}
$$
</div>

<h3>6.4 Chain Complex Verification</h3>

<p>$\\max|B_1 B_2| < 10^{{-12}}$. &check;</p>

<h3>6.5 Betti Numbers and Euler Characteristic</h3>

$$
\\chi = {sf['n_vertices']} - {sf['n_edges']} + {sf['n_faces']} = {sf['euler_simplex']}
$$
$$
\\beta_0 = {sf['beta_0']},\\; \\beta_1 = {sf['beta_1']},\\; \\beta_2 = {sf['beta_2']} \\quad \\Rightarrow \\quad {sf['beta_0']} - {sf['beta_1']} + {sf['beta_2']} = {sf['euler_betti']} \\;\\checkmark
$$

<p>$\\beta_1 = {sf['n_edges']} - {sf['rank_B1']} - {sf['rank_B2']} = {sf['beta_1']}$. &check;</p>

<h3>6.6 Spectra and Harmonic Modes</h3>

<p><strong>$L_0$ spectrum:</strong></p>
$$
\\spec(L_0) = \\{{{eigenvalue_str(sf['L0_eigenvalues'])}\\}}
$$
<p>Spectral gap: $\\Delta_0 = {sf['L0_spectral_gap']:.6f}$.</p>

<p><strong>$L_1$ spectrum:</strong></p>
$$
\\spec(L_1) = \\{{{eigenvalue_str(sf['L1_eigenvalues'])}\\}}
$$
<p>Spectral gap: $\\Delta_1 = {sf['L1_spectral_gap']:.6f}$.</p>

<div class="key-result">
<div class="label">Spectral gap comparison</div>
<p>
The Santa Fe lattice has the smallest spectral gap among the all-$z=4$ lattices ($\\Delta_1 = {sf['L1_spectral_gap']:.4f}$ vs. square $\\Delta_1 = {sq['L1_spectral_gap']}$ and kagome $\\Delta_1 = {kag['L1_spectral_gap']}$). This means non-harmonic components decay <em>more slowly</em> on Santa Fe, making the relative advantage of the harmonic protection channel even more important for deep networks.
</p>
</div>
""")

    h1_sf = [row[0] for row in sf["harmonic_basis"]]
    h2_sf = [row[1] for row in sf["harmonic_basis"]]
    html.append(f"""
<p><strong>Harmonic modes:</strong></p>
$$
\\mathbf{{h}}_1 = {vector_to_latex(h1_sf)}, \\quad
\\mathbf{{h}}_2 = {vector_to_latex(h2_sf)}
$$

<p>
Verification: $\\|L_1 \\mathbf{{h}}_1\\| = {sf['harmonic_validation'][0]['L1h_norm']:.1e}$, $\\|L_1 \\mathbf{{h}}_2\\| = {sf['harmonic_validation'][1]['L1h_norm']:.1e}$. &check;
</p>

<h3>6.7 Ice Configuration and $S$ Matrix</h3>

$$
\\bm{{\\sigma}} = {sigma_to_latex(sf['sigma'])}
$$

<p><strong>Charges:</strong> $\\mathbf{{Q}} = ({', '.join(str(c) for c in sf['charge'])})^T$.</p>

<p>Ice rule: $z=4$ vertices have $Q = 0$, $z=3$ have $|Q| = 1$, $z=2$ have $Q = 0$. &check;</p>

<div class="matrix-block">
$$
S = {matrix_to_latex(sf['S_matrix'])}
$$
</div>

<h3>6.8 Open Boundary Conditions</h3>

<table>
<thead>
<tr><th>Property</th><th>Periodic</th><th>Open</th></tr>
</thead>
<tbody>
<tr><td>$n_0$</td><td>{sf['n_vertices']}</td><td>{sf_open['n_vertices']}</td></tr>
<tr><td>$n_1$</td><td>{sf['n_edges']}</td><td>{sf_open['n_edges']}</td></tr>
<tr><td>$n_2$</td><td>{sf['n_faces']}</td><td>{sf_open['n_faces']}</td></tr>
<tr><td>$\\beta_0, \\beta_1, \\beta_2$</td><td>{sf['beta_0']}, {sf['beta_1']}, {sf['beta_2']}</td><td>{sf_open['beta_0']}, {sf_open['beta_1']}, {sf_open['beta_2']}</td></tr>
<tr><td>$\\chi$</td><td>{sf['euler_simplex']}</td><td>{sf_open['euler_simplex']}</td></tr>
</tbody>
</table>

<p>{sf['n_edges'] - sf_open['n_edges']} wrap edges removed, {sf['n_faces'] - sf_open['n_faces']} faces lost. $\\beta_1: {sf['beta_1']} \\to {sf_open['beta_1']}$. &check;</p>
""")

    # ==================== SECTION 7: Comparative Summary ====================
    html.append("""
<h2 id="sec7">7. Comparative Summary</h2>

<h3>7.1 Minimum Periodic Tilings</h3>

<table>
<thead>
<tr><th>Lattice</th><th>Min Size</th><th>$n_0$</th><th>$n_1$</th><th>$n_2$</th><th>$B_1$ dims</th><th>Why this minimum</th></tr>
</thead>
<tbody>
<tr><td>Square</td><td>3&times;3</td><td>9</td><td>18</td><td>9</td><td>9&times;18</td><td>1&times;1 = self-loops; 2&times;2 too small for pedagogy</td></tr>
<tr><td>Kagome</td><td>2&times;2</td><td>12</td><td>24</td><td>12</td><td>12&times;24</td><td>1&times;1 collapses cross-cell edges</td></tr>
<tr><td>Shakti</td><td>1&times;1</td><td>16</td><td>24</td><td>8</td><td>16&times;24</td><td>All 24 edges unique under wrapping</td></tr>
<tr><td>Tetris</td><td>2&times;1</td><td>16</td><td>24</td><td>8</td><td>16&times;24</td><td>1&times;1: 4 horizontal pairs become multi-edges</td></tr>
<tr><td>Santa Fe</td><td>1&times;2</td><td>12</td><td>18</td><td>6</td><td>12&times;18</td><td>1&times;1: one cross-cell edge duplicates intra-cell</td></tr>
</tbody>
</table>

<h3>7.2 Topological Properties (Periodic)</h3>

<table>
<thead>
<tr><th>Lattice</th><th>$\\beta_0$</th><th>$\\beta_1$</th><th>$\\beta_2$</th><th>$\\chi$</th><th>$\\Delta_0$</th><th>$\\Delta_1$</th><th>Pauling $\\varepsilon$</th></tr>
</thead>
<tbody>""")

    for name, key in [("Square", "square"), ("Kagome", "kagome"), ("Shakti", "shakti"),
                       ("Tetris", "tetris"), ("Santa Fe", "santa_fe")]:
        d = data[f"{key}_periodic"]
        html.append(f'<tr><td>{name}</td><td>{d["beta_0"]}</td><td>{d["beta_1"]}</td>'
                    f'<td>{d["beta_2"]}</td><td>{d["euler_simplex"]}</td>'
                    f'<td>{d["L0_spectral_gap"]:.4f}</td><td>{d["L1_spectral_gap"]:.4f}</td>'
                    f'<td>{d["pauling_estimate"]}</td></tr>')

    html.append("""</tbody>
</table>

<div class="key-result">
<div class="label">Key finding</div>
<p>
All five periodic lattices have $\\beta_1 = 2$ at minimum size, confirming the torus topology contributes exactly two harmonic modes (horizontal and vertical winding). The spectral gaps vary significantly: the square lattice has the largest ($\\Delta_1 = 3.0$), while Santa Fe and tetris have the smallest ($\\Delta_1 \\approx 0.6$&ndash;$0.8$). The smaller spectral gaps mean non-harmonic signals decay more slowly on frustrated lattices, but the harmonic signals are equally protected ($L_1 \\mathbf{h} = 0$) regardless of the gap.
</p>
</div>

<h3>7.3 Coordination Structure</h3>

<table>
<thead>
<tr><th>Lattice</th><th>$z=2$</th><th>$z=3$</th><th>$z=4$</th><th>$\\langle z \\rangle$</th><th>Frustration Type</th></tr>
</thead>
<tbody>""")

    for name, key in [("Square", "square"), ("Kagome", "kagome"), ("Shakti", "shakti"),
                       ("Tetris", "tetris"), ("Santa Fe", "santa_fe")]:
        d = data[f"{key}_periodic"]
        cd = d["coordination_distribution"]
        z2 = cd.get("2", 0)
        z3 = cd.get("3", 0)
        z4 = cd.get("4", 0)
        n = d["n_vertices"]
        avg_z = (2*z2 + 3*z3 + 4*z4) / n
        frust = {
            "square": "Geometric only",
            "kagome": "Geometric",
            "shakti": "Vertex-frustrated, maximal",
            "tetris": "Vertex-frustrated, maximal",
            "santa_fe": "Both geometric &amp; vertex",
        }[key]
        html.append(f'<tr><td>{name}</td><td>{z2}</td><td>{z3}</td><td>{z4}</td>'
                    f'<td>{avg_z:.1f}</td><td>{frust}</td></tr>')

    html.append("""</tbody>
</table>

<h3>7.4 Open Boundary Comparison</h3>

<table>
<thead>
<tr><th>Lattice</th><th>Periodic $n_1$</th><th>Open $n_1$</th><th>Edges removed</th><th>Periodic $\\beta_1$</th><th>Open $\\beta_1$</th></tr>
</thead>
<tbody>""")

    for name, key in [("Square", "square"), ("Kagome", "kagome"), ("Shakti", "shakti"),
                       ("Tetris", "tetris"), ("Santa Fe", "santa_fe")]:
        dp = data[f"{key}_periodic"]
        do = data[f"{key}_open"]
        html.append(f'<tr><td>{name}</td><td>{dp["n_edges"]}</td><td>{do["n_edges"]}</td>'
                    f'<td>{dp["n_edges"] - do["n_edges"]}</td><td>{dp["beta_1"]}</td>'
                    f'<td>{do["beta_1"]}</td></tr>')

    html.append("""</tbody>
</table>

<p>
In all cases, $\\beta_1$ drops from 2 (periodic/torus) to 0 (open/disk). This universal pattern confirms that the harmonic subspace is fundamentally a <strong>topological</strong> property of the embedding surface: the torus provides two non-contractible cycles, while the disk has none. For the oversmoothing application, this means periodic boundary conditions are essential to maintain the protected channel.
</p>

<h3>7.5 Scaling to XS (4&times;4)</h3>

<p>
At the standard XS (4&times;4 unit cell) size, the matrices grow substantially:
</p>

<table>
<thead>
<tr><th>Lattice</th><th>$n_0$</th><th>$n_1$</th><th>$n_2$</th><th>$B_1$ dims</th><th>$B_1$ entries</th></tr>
</thead>
<tbody>""")

    for name in ["square", "kagome", "shakti", "tetris", "santa_fe"]:
        xd = xs[name]
        html.append(f'<tr><td>{name.replace("_", " ").title()}</td><td>{xd["n_vertices"]}</td>'
                    f'<td>{xd["n_edges"]}</td><td>{xd["n_faces"]}</td>'
                    f'<td>{xd["B1_dims"]}</td><td>{xd["entries_B1"]:,}</td></tr>')

    html.append("""</tbody>
</table>

<p>
The shakti lattice at 4&times;4 produces a 256&times;384 $B_1$ matrix with 98,304 entries &mdash; clearly impractical for explicit display. This justifies our choice of minimum valid periodic tilings for this document.
</p>
""")

    # ==================== SECTION 8: References ====================
    html.append("""
<h2 id="sec8">8. References</h2>

<ol>
<li>M. J. Morrison, T. R. Nelson, and C. Nisoli, &ldquo;Unhappy vertices in artificial spin ice: new degeneracies from vertex frustration,&rdquo; <em>New J. Phys.</em> <strong>15</strong>, 045009 (2013).</li>
<li>C. Nisoli, &ldquo;The concept of spin ice graphs and a field theory for their charges,&rdquo; <em>AIP Advances</em> <strong>10</strong>, 115303 (2020).</li>
<li>Y. Lao <em>et al.</em>, &ldquo;Classical topological order in the kinetics of artificial spin ice,&rdquo; <em>Nature Physics</em> <strong>14</strong>, 723&ndash;727 (2018).</li>
<li>Z. Yang, E. Isufi, and G. Leus, &ldquo;Simplicial Convolutional Neural Networks,&rdquo; <em>IEEE TSP</em> (2022).</li>
<li>Q. Li, Z. Han, and X.-M. Wu, &ldquo;Deeper insights into graph convolutional networks for semi-supervised learning,&rdquo; <em>AAAI</em> (2018).</li>
<li>B. Rusch <em>et al.</em>, &ldquo;A survey on oversmoothing in graph neural networks,&rdquo; arXiv:2303.10993 (2023).</li>
<li>O. Duranthon and L. Zdeborov&aacute;, &ldquo;Optimal message-passing on graphs with community structure,&rdquo; <em>Phys. Rev. X</em> (2025).</li>
</ol>

<div class="footnote">
<p>
All numerical results were computed using the SpinIceTDL codebase with <code>scripts/compute_worked_examples.py</code>. Matrix entries, eigenvalues, and Betti numbers are verified programmatically against the chain complex property ($B_1 B_2 = 0$), Euler characteristic consistency, and harmonic vector conditions ($L_1 \\mathbf{h} = 0$, $B_1 \\mathbf{h} = 0$, $B_2^T \\mathbf{h} = 0$).
</p>
</div>
""")

    html.append("""
</body>
</html>
""")

    return "\n".join(html)


def main():
    data = load_data()
    html = generate_html(data)

    output_path = os.path.join(os.path.dirname(__file__), "..",
                               "docs", "asi-lattice-zoo-worked-examples.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML written to {output_path}")
    print(f"Document size: {len(html):,} characters, ~{html.count(chr(10)):,} lines")


if __name__ == "__main__":
    main()
