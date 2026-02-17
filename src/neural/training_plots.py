"""Diagnostic plots for Mode A (LoopMPVAN) training runs.

Generates four figure panels from saved training run data:
  Panel 1: Training curves (loss, entropy, KL, ESS)
  Panel 2: Sampling quality (state frequency, Hamming distribution)
  Panel 3: Sample gallery (ice states with spin arrows + monopole markers)
  Panel 4: Summary text card
  Individual sample PNGs with ice-rule validation
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

# Publication style (matches src/viz/matplotlib_figures.py)
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colors
_BLUE = "#3498db"
_RED = "#e74c3c"
_GRAY = "#95a5a6"
_MONOPOLE_POS = "#ff6b00"   # orange for Q > target (positive monopole)
_MONOPOLE_NEG = "#9b59b6"   # purple for Q < target (negative monopole / vacancy)
_VERTEX_OK = "black"


# ── Ice rule validation ───────────────────────────────────────────────

@dataclass
class VertexChargeInfo:
    """Per-vertex ice rule diagnostics for a single spin configuration."""

    charge: np.ndarray          # (n_vertices,) signed vertex charge Q_v = B1 @ sigma
    coordination: np.ndarray    # (n_vertices,) vertex degree z_v
    target_abs_charge: np.ndarray  # (n_vertices,) = z_v mod 2 (0 for even, 1 for odd)
    is_violating: np.ndarray    # (n_vertices,) bool — True if |Q_v| != target
    n_violations: int
    total_vertices: int


def compute_vertex_charges(
    sigma: np.ndarray,
    edge_list: np.ndarray,
    coordination: np.ndarray,
) -> VertexChargeInfo:
    """Compute vertex charges and identify ice-rule violations.

    Ice rule: |Q_v| should equal z_v mod 2.
      - Even-degree vertices (z=2,4): |Q_v| = 0 (equal in/out)
      - Odd-degree vertices (z=3):    |Q_v| = 1 (one excess)

    A vertex with |Q_v| != z_v mod 2 is a monopole excitation.

    Parameters
    ----------
    sigma : array (n_edges,) of +1/-1
    edge_list : array (n_edges, 2) of vertex indices
    coordination : array (n_vertices,)

    Returns
    -------
    VertexChargeInfo
    """
    n_vertices = len(coordination)
    n_edges = len(edge_list)

    # Build B1 directly: B1[head, e] = +1, B1[tail, e] = -1
    # Convention: edge (u, v) with u < v has tail=u, head=v
    rows = []
    cols = []
    vals = []
    for e_idx in range(n_edges):
        u, v = int(edge_list[e_idx, 0]), int(edge_list[e_idx, 1])
        tail, head = (u, v) if u < v else (v, u)
        rows.extend([tail, head])
        cols.extend([e_idx, e_idx])
        vals.extend([-1, 1])

    B1 = sparse.csc_matrix(
        (vals, (rows, cols)), shape=(n_vertices, n_edges)
    )

    charge = np.asarray(B1 @ sigma.astype(np.float64)).ravel()
    target = coordination.astype(np.float64) % 2
    is_violating = np.abs(np.abs(charge) - target) > 0.5

    return VertexChargeInfo(
        charge=charge,
        coordination=coordination,
        target_abs_charge=target,
        is_violating=is_violating,
        n_violations=int(np.sum(is_violating)),
        total_vertices=n_vertices,
    )


# ── Drawing helpers ───────────────────────────────────────────────────

def _draw_spin_arrow(ax, positions, edge, spin_val):
    """Draw a small arrow at the midpoint of an edge indicating spin direction.

    Replicates the style from src/viz/matplotlib_figures.py.
    """
    u, v = edge
    pu = positions[u]
    pv = positions[v]
    mid = 0.5 * (pu + pv)

    if spin_val > 0:
        dx = pv[0] - pu[0]
        dy = pv[1] - pu[1]
    else:
        dx = pu[0] - pv[0]
        dy = pu[1] - pv[1]

    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-10:
        return

    scale = 0.12 * length
    dx_n = dx / length * scale
    dy_n = dy / length * scale

    color = _BLUE if spin_val > 0 else _RED
    ax.annotate(
        "",
        xy=(mid[0] + dx_n, mid[1] + dy_n),
        xytext=(mid[0] - dx_n, mid[1] - dy_n),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        zorder=2,
    )


def plot_training_curves(
    loss_history: np.ndarray,
    entropy_history: np.ndarray,
    kl_history: np.ndarray,
    ess_history: np.ndarray,
    eval_epochs: np.ndarray,
    n_reachable_states: int,
    batch_size: int,
    output_path: str,
) -> str:
    """Panel 1: 2x2 grid of training curves.

    (a) Loss vs epoch (linear scale — loss goes negative at convergence
        because the entropy bonus term dominates the near-zero policy gradient)
    (b) Entropy vs epoch (target = ln(n_reachable_states))
    (c) KL divergence vs epoch (target = 0)
    (d) ESS vs epoch (target = batch_size)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    epochs_loss = np.arange(1, len(loss_history) + 1)
    epochs_ent = np.arange(1, len(entropy_history) + 1)

    # (a) Loss (linear scale — expected to go negative as entropy bonus dominates)
    ax = axes[0, 0]
    ax.plot(epochs_loss, loss_history, color=_BLUE, linewidth=0.8, alpha=0.8)
    ax.axhline(0, color=_GRAY, linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Policy Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (b) Entropy
    ax = axes[0, 1]
    ax.plot(epochs_ent, entropy_history, color=_BLUE, linewidth=0.8, alpha=0.8)
    if n_reachable_states > 0:
        target_entropy = np.log(n_reachable_states)
        ax.axhline(target_entropy, color=_GRAY, linestyle="--", linewidth=1.2,
                    label=f"ln({n_reachable_states}) = {target_entropy:.2f}")
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.set_title("(b) Entropy", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (c) KL divergence
    ax = axes[1, 0]
    ax.plot(eval_epochs, kl_history, color=_BLUE, marker="o", markersize=3, linewidth=1.0)
    ax.axhline(0, color=_GRAY, linestyle="--", linewidth=1.2, label="KL = 0 (target)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL divergence")
    ax.set_title("(c) KL(empirical || uniform)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) ESS
    ax = axes[1, 1]
    ax.plot(eval_epochs, ess_history, color=_BLUE, marker="o", markersize=3, linewidth=1.0)
    ax.axhline(batch_size, color=_GRAY, linestyle="--", linewidth=1.2,
               label=f"Ideal ESS = {batch_size}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ESS")
    ax.set_title("(d) Effective Sample Size", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path, "panel1_training_curves.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_sampling_quality(
    final_samples: np.ndarray,
    exact_states: Optional[np.ndarray],
    output_path: str,
) -> str:
    """Panel 2: Sampling quality diagnostics.

    (a) State frequency histogram (if exact_states available)
    (b) Pairwise Hamming distance distribution
    """
    has_exact = exact_states is not None and len(exact_states) > 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # (a) State frequency histogram
    ax = axes[0]
    if has_exact:
        n_exact = len(exact_states)
        n_samples = len(final_samples)

        # Map states to indices and count
        state_to_idx = {}
        for i in range(n_exact):
            key = tuple(exact_states[i].astype(np.int8))
            state_to_idx[key] = i

        counts = np.zeros(n_exact, dtype=np.int64)
        for s in final_samples:
            key = tuple(s.astype(np.int8))
            idx = state_to_idx.get(key)
            if idx is not None:
                counts[idx] += 1

        # Sort by frequency for visual clarity
        sorted_counts = np.sort(counts)[::-1]
        uniform_expected = n_samples / n_exact

        ax.barh(range(n_exact), sorted_counts, color=_BLUE, alpha=0.7, height=1.0)
        ax.axvline(uniform_expected, color=_GRAY, linestyle="--", linewidth=1.5,
                   label=f"Uniform = {uniform_expected:.1f}")
        ax.set_xlabel("Sample count")
        ax.set_ylabel("State (sorted by frequency)")
        ax.set_title("(a) State Frequency", fontweight="bold")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No exact states\n(enumeration unavailable)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_title("(a) State Frequency", fontweight="bold")

    # (b) Hamming distance distribution
    ax = axes[1]
    n_edges = final_samples.shape[1]
    # Compute pairwise normalized Hamming distances
    dots = final_samples @ final_samples.T
    hamming = (n_edges - dots) / (2.0 * n_edges)
    mask = np.triu(np.ones(hamming.shape, dtype=bool), k=1)
    distances = hamming[mask]

    ax.hist(distances, bins=30, color=_BLUE, alpha=0.7, density=True, edgecolor="white")
    ax.axvline(0.5, color=_GRAY, linestyle="--", linewidth=1.5,
               label="Expected (uniform) ~ 0.5")
    ax.set_xlabel("Normalized Hamming distance")
    ax.set_ylabel("Density")
    ax.set_title("(b) Hamming Distance Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Sampling Quality", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path, "panel2_sampling_quality.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def _draw_ice_state_on_ax(
    ax,
    sigma: np.ndarray,
    positions: np.ndarray,
    edge_list: np.ndarray,
    coordination: np.ndarray,
    show_violations: bool = True,
):
    """Draw a single ice state with spin-colored edges, arrows, and monopole markers.

    Parameters
    ----------
    ax : matplotlib Axes
    sigma : array (n_edges,) of +1/-1
    positions : array (n_vertices, 2)
    edge_list : array (n_edges, 2)
    coordination : array (n_vertices,)
    show_violations : bool
        If True, compute vertex charges and highlight monopoles.
    """
    edges = [(int(e[0]), int(e[1])) for e in edge_list]

    # Draw edges colored by spin
    lines_plus, lines_minus = [], []
    for j, (u, v) in enumerate(edges):
        seg = np.array([[positions[u, 0], positions[u, 1]],
                        [positions[v, 0], positions[v, 1]]])
        if sigma[j] > 0:
            lines_plus.append(seg)
        else:
            lines_minus.append(seg)

    if lines_plus:
        ax.add_collection(LineCollection(lines_plus, colors=_BLUE,
                                         linewidths=1.5, zorder=1))
    if lines_minus:
        ax.add_collection(LineCollection(lines_minus, colors=_RED,
                                         linewidths=1.5, zorder=1))

    # Draw spin arrows
    for j, (u, v) in enumerate(edges):
        _draw_spin_arrow(ax, positions, (u, v), sigma[j])

    # Compute vertex charges and annotate
    if show_violations:
        vci = compute_vertex_charges(sigma, edge_list, coordination)

        # Draw all vertices as small black dots first
        ax.scatter(positions[:, 0], positions[:, 1],
                   c=_VERTEX_OK, s=12, zorder=3)

        # Overdraw monopole vertices (ice-rule violators) with large colored markers
        viol_idx = np.where(vci.is_violating)[0]
        for vi in viol_idx:
            q = vci.charge[vi]
            color = _MONOPOLE_POS if q > 0 else _MONOPOLE_NEG
            excess = abs(abs(q) - vci.target_abs_charge[vi])
            size = 80 + 40 * excess
            ax.scatter(positions[vi, 0], positions[vi, 1],
                       c=color, s=size, zorder=5, edgecolors="black",
                       linewidths=1.5, marker="o")

        # Text labels for ALL nonzero-charge vertices
        #   - Expected nonzero (|Q|==target, e.g. ±1 on z=3): dark gray
        #   - Ice-rule violation (|Q|>target): red, bold
        _CHARGE_EXPECTED = "#555555"   # dark gray for expected ±1
        _CHARGE_VIOLATION = "#e74c3c"  # red for violations

        for vi in range(len(vci.charge)):
            q = int(vci.charge[vi])
            if q == 0:
                continue
            is_viol = vci.is_violating[vi]
            txt_color = _CHARGE_VIOLATION if is_viol else _CHARGE_EXPECTED
            txt_weight = "bold" if is_viol else "normal"
            txt_size = 7.5 if is_viol else 6.5
            ax.annotate(
                f"{q:+d}",
                xy=(positions[vi, 0], positions[vi, 1]),
                xytext=(0, 5), textcoords="offset points",
                fontsize=txt_size, fontweight=txt_weight, color=txt_color,
                ha="center", va="bottom", zorder=6,
            )
    else:
        ax.scatter(positions[:, 0], positions[:, 1], c=_VERTEX_OK, s=8, zorder=3)

    # Set bounds
    pad = 0.5
    xmin, ymin = positions.min(axis=0) - pad
    xmax, ymax = positions.max(axis=0) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_sample_gallery(
    final_samples: np.ndarray,
    positions: np.ndarray,
    edge_list: np.ndarray,
    coordination: np.ndarray,
    output_path: str,
    n_panels: int = 6,
) -> str:
    """Panel 3: 2x3 grid of sampled ice states with spin arrows + monopole markers.

    Blue edges for sigma=+1, red for sigma=-1, small arrows at midpoints.
    Monopole vertices (ice-rule violations) highlighted with colored markers.
    """
    n_show = min(n_panels, len(final_samples))
    nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for panel_idx in range(n_show):
        ax = axes[panel_idx]
        sigma = final_samples[panel_idx]
        _draw_ice_state_on_ax(ax, sigma, positions, edge_list, coordination)

        vci = compute_vertex_charges(sigma, edge_list, coordination)
        viol_str = f" [{vci.n_violations} monopole{'s' if vci.n_violations != 1 else ''}]" if vci.n_violations > 0 else ""
        ax.set_title(f"Sample {panel_idx + 1}{viol_str}", fontweight="bold")

    # Hide unused panels
    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Sampled Ice States", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fpath = os.path.join(output_path, "panel3_sample_gallery.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_individual_samples(
    final_samples: np.ndarray,
    positions: np.ndarray,
    edge_list: np.ndarray,
    coordination: np.ndarray,
    output_path: str,
    n_samples: int = 6,
) -> list[str]:
    """Generate individual PNG for each sample with ice-rule validation.

    Each plot includes:
      - Spin-colored edges (blue +1, red -1) with directional arrows
      - Ice-rule-obeying vertices as small black dots
      - Monopole vertices highlighted: orange (Q>target), purple (Q<target)
      - Charge label (Q=+N) next to each monopole
      - Title with sample index and violation count

    Parameters
    ----------
    final_samples : array (n_total, n_edges) of +1/-1
    positions : array (n_vertices, 2)
    edge_list : array (n_edges, 2)
    coordination : array (n_vertices,)
    output_path : str
        Directory to save individual PNGs.
    n_samples : int
        Number of samples to plot (from the start of final_samples).

    Returns
    -------
    paths : list of str
    """
    os.makedirs(output_path, exist_ok=True)
    n_show = min(n_samples, len(final_samples))
    paths = []

    for idx in range(n_show):
        sigma = final_samples[idx]
        vci = compute_vertex_charges(sigma, edge_list, coordination)

        fig, ax = plt.subplots(figsize=(7, 7))
        _draw_ice_state_on_ax(ax, sigma, positions, edge_list, coordination)

        # Title with validation summary
        if vci.n_violations == 0:
            title = f"Sample {idx + 1}  —  valid ice state"
            title_color = "black"
        else:
            title = f"Sample {idx + 1}  —  {vci.n_violations} monopole{'s' if vci.n_violations != 1 else ''}"
            title_color = _MONOPOLE_POS

        ax.set_title(title, fontweight="bold", color=title_color, fontsize=12)

        # Legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=_BLUE, lw=2, label="$\\sigma = +1$"),
            Line2D([0], [0], color=_RED, lw=2, label="$\\sigma = -1$"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_VERTEX_OK,
                   markersize=5, label="Vertex ($Q_v$=0)"),
            Line2D([0], [0], marker="$\\pm$", color="#555555", markersize=6,
                   linestyle="None", label="$Q_v$ expected ($z$ mod 2)"),
        ]
        if vci.n_violations > 0:
            handles.extend([
                Line2D([0], [0], marker="o", color="w", markerfacecolor=_MONOPOLE_POS,
                       markeredgecolor="black", markersize=8, label="Monopole ($|Q|>$ target)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=_MONOPOLE_NEG,
                       markeredgecolor="black", markersize=8, label="Monopole ($|Q|<$ target)"),
                Line2D([0], [0], marker="$\\pm$", color=_RED, markersize=6,
                       linestyle="None", label="$Q_v$ violation (red)"),
            ])
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

        # Annotation box with per-vertex charge stats
        charge_text = (
            f"$\\langle |Q_v| \\rangle$ = {np.mean(np.abs(vci.charge)):.2f}\n"
            f"Violations: {vci.n_violations}/{vci.total_vertices}"
        )
        ax.text(0.02, 0.02, charge_text, transform=ax.transAxes,
                fontsize=8, verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        fig.tight_layout()
        fname = f"sample_{idx + 1:03d}.png"
        fpath = os.path.join(output_path, fname)
        fig.savefig(fpath)
        plt.close(fig)
        paths.append(fpath)

        logger.info(
            f"  Sample {idx + 1}: {vci.n_violations}/{vci.total_vertices} violations, "
            f"<|Q|>={np.mean(np.abs(vci.charge)):.3f}"
        )

    return paths


def plot_summary_card(
    metadata: dict,
    kl_history: np.ndarray,
    hamming_history: np.ndarray,
    ess_history: np.ndarray,
    final_samples: np.ndarray,
    exact_states: Optional[np.ndarray],
    output_path: str,
) -> str:
    """Panel 4: Summary text card with key statistics."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    # Compute final metrics
    final_kl = kl_history[-1] if len(kl_history) > 0 else float("nan")
    final_hamming = hamming_history[-1] if len(hamming_history) > 0 else float("nan")
    final_ess = ess_history[-1] if len(ess_history) > 0 else float("nan")

    # State coverage
    n_reachable = metadata.get("n_reachable_states", -1)
    if exact_states is not None and len(exact_states) > 0:
        state_set = set()
        for s in final_samples:
            state_set.add(tuple(s.astype(np.int8)))
        n_unique = len(state_set)
        coverage_str = f"{n_unique}/{n_reachable} = {n_unique / n_reachable:.4f}"
    else:
        n_unique = len(set(tuple(s.astype(np.int8)) for s in final_samples))
        coverage_str = f"{n_unique} unique states"

    lines = [
        ("Lattice", f"{metadata['lattice_name']} ({metadata['boundary']} BC)"),
        ("Size", f"{metadata['size_label']}  ({metadata['n_vertices']} vertices, {metadata['n_edges']} edges)"),
        ("beta_1", str(metadata["beta_1"])),
        ("Reachable states", str(n_reachable)),
        ("", ""),
        ("Model params", str(metadata["n_model_params"])),
        ("Layers / equ_dim / inv_dim", f"{metadata['n_layers']} / {metadata['equ_dim']} / {metadata['inv_dim']}"),
        ("", ""),
        ("Epochs", str(metadata["n_epochs"])),
        ("Batch size", str(metadata["batch_size"])),
        ("Learning rate", str(metadata["lr"])),
        ("Training time", f"{metadata['train_time_s']:.1f}s"),
        ("Enumeration time", f"{metadata['enum_time_s']:.1f}s"),
        ("", ""),
        ("Final KL", f"{final_kl:.6f}"),
        ("Final Hamming", f"{final_hamming:.4f}"),
        ("Final ESS", f"{final_ess:.1f}"),
        ("State coverage", coverage_str),
    ]

    y = 0.95
    for label, value in lines:
        if label == "" and value == "":
            y -= 0.02
            continue
        ax.text(0.05, y, label, fontsize=10, fontweight="bold",
                transform=ax.transAxes, verticalalignment="top",
                fontfamily="monospace")
        ax.text(0.45, y, value, fontsize=10,
                transform=ax.transAxes, verticalalignment="top",
                fontfamily="monospace")
        y -= 0.05

    fig.suptitle("Training Run Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path, "panel4_summary.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_gradient_diagnostics(
    grad_norm_history: np.ndarray,
    advantage_var_history: np.ndarray,
    output_path: str,
) -> str:
    """Panel 5: Gradient diagnostics (C2).

    (a) Gradient norm (post-clip) vs epoch
    (b) Advantage variance vs epoch
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    epochs = np.arange(1, len(grad_norm_history) + 1)

    # (a) Gradient norm
    ax = axes[0]
    ax.plot(epochs, grad_norm_history, color=_BLUE, linewidth=0.6, alpha=0.7)
    # Smoothed trend (rolling mean, window=50)
    if len(grad_norm_history) >= 50:
        kernel = np.ones(50) / 50
        smoothed = np.convolve(grad_norm_history, kernel, mode="valid")
        ax.plot(np.arange(25, 25 + len(smoothed)), smoothed,
                color=_RED, linewidth=1.5, label="50-epoch avg")
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient norm (post-clip)")
    ax.set_title("(a) Gradient Norm", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (b) Advantage variance
    ax = axes[1]
    ax.plot(epochs[:len(advantage_var_history)], advantage_var_history,
            color=_BLUE, linewidth=0.6, alpha=0.7)
    if len(advantage_var_history) >= 50:
        kernel = np.ones(50) / 50
        smoothed = np.convolve(advantage_var_history, kernel, mode="valid")
        ax.plot(np.arange(25, 25 + len(smoothed)), smoothed,
                color=_RED, linewidth=1.5, label="50-epoch avg")
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Var(advantage)")
    ax.set_title("(b) Advantage Variance", fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Gradient Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fpath = os.path.join(output_path, "panel5_gradient_diagnostics.png")
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def generate_all_panels(
    run_data: dict,
    output_dir: str,
    n_individual: int = 6,
) -> list[str]:
    """Generate all diagnostic panels + individual sample PNGs.

    Parameters
    ----------
    run_data : dict
        Output of load_training_run().
    output_dir : str
        Directory to save PNGs into.
    n_individual : int
        Number of individual sample PNGs to generate.

    Returns
    -------
    paths : list of str
        Paths to generated figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = run_data["metadata"]
    paths = []

    # Panel 1: Training curves
    paths.append(plot_training_curves(
        loss_history=run_data["loss_history"],
        entropy_history=run_data["entropy_history"],
        kl_history=run_data["kl_history"],
        ess_history=run_data["ess_history"],
        eval_epochs=run_data["eval_epochs"],
        n_reachable_states=metadata.get("n_reachable_states", -1),
        batch_size=metadata["batch_size"],
        output_path=output_dir,
    ))

    # Panel 2: Sampling quality
    paths.append(plot_sampling_quality(
        final_samples=run_data["final_samples"],
        exact_states=run_data.get("exact_states"),
        output_path=output_dir,
    ))

    # Panel 3: Sample gallery (with monopole highlighting)
    paths.append(plot_sample_gallery(
        final_samples=run_data["final_samples"],
        positions=run_data["positions"],
        edge_list=run_data["edge_list"],
        coordination=run_data["coordination"],
        output_path=output_dir,
    ))

    # Panel 4: Summary card
    paths.append(plot_summary_card(
        metadata=metadata,
        kl_history=run_data["kl_history"],
        hamming_history=run_data["hamming_history"],
        ess_history=run_data["ess_history"],
        final_samples=run_data["final_samples"],
        exact_states=run_data.get("exact_states"),
        output_path=output_dir,
    ))

    # Panel 5: Gradient diagnostics (skip if data unavailable from old runs)
    grad_norm = run_data.get("grad_norm_history", np.array([]))
    adv_var = run_data.get("advantage_var_history", np.array([]))
    if len(grad_norm) > 0 and len(adv_var) > 0:
        paths.append(plot_gradient_diagnostics(
            grad_norm_history=grad_norm,
            advantage_var_history=adv_var,
            output_path=output_dir,
        ))

    # Individual sample PNGs with ice-rule validation
    samples_dir = os.path.join(output_dir, "samples")
    logger.info(f"Generating {n_individual} individual sample plots...")
    sample_paths = plot_individual_samples(
        final_samples=run_data["final_samples"],
        positions=run_data["positions"],
        edge_list=run_data["edge_list"],
        coordination=run_data["coordination"],
        output_path=samples_dir,
        n_samples=n_individual,
    )
    paths.extend(sample_paths)

    return paths
