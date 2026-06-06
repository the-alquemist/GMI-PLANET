"""
pling_igraph_analysis.py
=======================
Graph-theoretic centrality analysis for pling plasmid community networks.

Computes four centrality measures on both the full (hub-included) and
hub-removed igraph objects produced by pling_igraph_viz.py:

    1. Degree Centrality      – how many direct neighbours a node has
    2. Betweenness Centrality – how often a node lies on shortest paths
    3. Closeness Centrality   – how close a node is to all others
    4. Eigenvector Centrality – being connected to well-connected nodes

For each measure two visualisations are produced:
    a. Network graph – same three-phase layout as the reference figures,
       nodes coloured by centrality value (light = low, dark = high)
    b. Degree measure: additionally a 6-panel distribution plot using the
       powerlaw package (raw / linear-bins / log-bins × linear / log scale)

All outputs go to:
    <out_dir>

Usage
-----
    python3 pling_igraph_analysis.py            # uses defaults below
    python3 pling_igraph_analysis.py --dpi 300

Dependencies
------------
    pip install igraph matplotlib pandas numpy powerlaw
    (pling_igraph_viz.py must be importable from the same directory)
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
import numpy as np
import pandas as pd
import powerlaw

# pling_igraph_viz.py must be on the path
sys.path.insert(0, str(Path(__file__).parent))
from pling_igraph_viz import build_pling_graph, hybrid_layout

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Centrality analysis for pling plasmid community graphs."
    )
    p.add_argument("--json",    required=True,
                   help="Path to community JSON from pling output")
    p.add_argument("--typing",  required=True,
                   help="Path to typing.tsv")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for analysis results")
    p.add_argument("--community", type=int, default=0)
    p.add_argument("--top-k",  type=int, default=10,
                   help="Number of top-ranked nodes to label in each graph")
    p.add_argument("--dpi",    type=int, default=200)
    return p.parse_args()


# ── Graph loading ─────────────────────────────────────────────────────────────

def load_graphs(json_path: str, typing_path: str):
    """Load pling JSON and typing table, return (g_full, g_no_hubs).

    Both graphs carry the vertex attributes set by build_pling_graph:
        name, color, is_hub, subcom
    and edge attributes: sd, td, lbl.
    They are completely independent igraph.Graph objects ready for
    any further graph-theoretic computation.
    """
    with open(json_path) as fh:
        raw = json.load(fh)
    raw_nodes = raw["elements"]["nodes"]
    raw_edges = raw["elements"]["edges"]

    typing = pd.read_csv(typing_path, sep="\t")
    p2t    = dict(zip(typing["plasmid"], typing["type"]))

    color_to_sc: dict = {}
    for n in raw_nodes:
        nid = n["data"]["id"]
        if nid in p2t:
            color_to_sc[n["data"].get("color", "")] = p2t[nid]

    g_full    = build_pling_graph(raw_nodes, raw_edges, p2t, color_to_sc,
                                  exclude_hubs=False)
    g_no_hubs = build_pling_graph(raw_nodes, raw_edges, p2t, color_to_sc,
                                  exclude_hubs=True)

    print(f"Full graph:    {g_full.vcount():3d} nodes, {g_full.ecount():4d} edges "
          f"({sum(g_full.vs['is_hub'])} hubs)")
    print(f"No-hub graph:  {g_no_hubs.vcount():3d} nodes, {g_no_hubs.ecount():4d} edges "
          f"({g_no_hubs.connected_components().n} components)")
    return g_full, g_no_hubs


# ── Centrality computations ───────────────────────────────────────────────────

def degree_centrality(g: ig.Graph, normalized: bool = True) -> list[float]:
    """Degree centrality measures direct neighbourhood size.

    Return raw integer counts when `normalized` is False, or d(v)/(N-1)
    so values are comparable across graphs of different sizes.
    """
    degrees = g.degree()
    if not normalized:
        return degrees
    n = g.vcount()
    if n <= 1:
        return [0.0 for _ in degrees]
    return [d / (n - 1) for d in degrees]


def betweenness_centrality(g: ig.Graph, normalized: bool = True) -> list[float]:
    """Betweenness centrality measures shortest-path brokerage.

    Raw values count how often a node lies on geodesics; normalization
    divides by (n-1)(n-2)/2 to keep results on a 0-1 scale.
    """
    raw = g.betweenness()
    if not normalized:
        return raw
    n = g.vcount()
    denom = (n - 1) * (n - 2) / 2
    if denom <= 0:
        return [0.0 for _ in raw]
    return [b / denom for b in raw]


def closeness_centrality(g: ig.Graph, normalized: bool = True) -> list[float]:
    """Closeness centrality measures how close a node is to others.

    Raw values use the reachable set only (1 / sum_dist); the
    normalized form applies the Wasserman–Faust correction
    n_reach² / ((n-1) * sum_dist) for disconnected graphs.
    """
    n = g.vcount()
    # g.distances() returns an n×n list of lists; inf marks unreachable nodes.
    all_dist = g.distances()
    result = []
    for v in range(n):
        reach = [d for u, d in enumerate(all_dist[v]) if u != v and d != float("inf")]
        if not reach:
            # Isolated node: no reachable neighbours, so closeness is 0.
            result.append(0.0)
            continue
        n_reach = len(reach)
        sum_dist = sum(reach)
        if not normalized:
            result.append(1 / sum_dist)
        else:
            result.append(n_reach ** 2 / ((n - 1) * sum_dist))
    return result


def eigenvector_centrality(g: ig.Graph, normalized: bool = True) -> list[float]:
    """Eigenvector centrality measures influence via well-connected neighbours.

    The values are computed only on the largest connected component;
    vertices in smaller components get 0.0. The `normalized` flag is kept
    only for API consistency with the other centrality helpers.
    """
    components = g.connected_components()
    if len(components) == 0:
        return []

    membership = components.membership
    largest_component = max(set(membership), key=membership.count)
    giant_vertices = [idx for idx, component_id in enumerate(membership)
                      if component_id == largest_component]

    if not giant_vertices:
        return [0.0 for _ in range(g.vcount())]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        giant_values = g.subgraph(giant_vertices).eigenvector_centrality()

    values = [0.0 for _ in range(g.vcount())]
    for vertex_index, value in zip(giant_vertices, giant_values):
        values[vertex_index] = value
    return values


# ── Degree-distribution / power-law helpers ───────────────────────────────────

def raw_frequency(data: list) -> tuple:
    """Exact degree → frequency counts (no binning)."""
    distinct = sorted(set(data))
    counts   = [data.count(v) for v in distinct]
    return distinct, counts


def bin_frequency(data: list, logarithmic_bins: bool = False) -> tuple:
    """Degree distribution using the powerlaw package binning.

    Parameters
    ----------
    data             : list of integer degree values
    logarithmic_bins : if True use log-spaced bins, else linear bins

    Returns
    -------
    x : bin midpoints (float array)
    y : probability density in each bin (float array)
    """
    x, y = powerlaw.pdf(data, linear_bins=not logarithmic_bins)
    x = x[:-1]   # last element is the right edge of the final bin — discard
    return x, y


# ── Visualisation helpers ─────────────────────────────────────────────────────

# Colour scheme matching Examples.ipynb
_CMAPS = {
    "degree":       cm.Greens,
    "betweenness":  cm.Oranges,
    "closeness":    cm.Blues,
    "eigenvector":  cm.Purples,
}

_MEASURE_LABELS = {
    "degree":       "Degree Centrality",
    "betweenness":  "Betweenness Centrality",
    "closeness":    "Closeness Centrality",
    "eigenvector":  "Eigenvector Centrality",
}


def render_centrality_graph(g: ig.Graph,
                             coords: np.ndarray,
                             values: list[float],
                             measure: str,
                             graph_label: str,
                             community_idx: int,
                             top_k: int,
                             out_path: Path,
                             dpi: int = 200) -> None:
    """Render the pling layout with nodes coloured by a centrality measure.

    Visual design
    -------------
    • Edges: light gray, thin — structure in the background.
    • Nodes: coloured by centrality value using the measure-specific
      colormap (light = low, dark = high).  Hub nodes keep their star
      shape in the full graph.
    • Labels: only the top-K nodes by centrality are named, so the most
      important plasmids stand out without label clutter.
    • Colorbar: right side of the figure, showing the full value range.

    Parameters
    ----------
    g            : igraph.Graph with vertex attributes (name, is_hub)
    coords       : np.ndarray (N, 2) from hybrid_layout()
    values       : centrality value per vertex (same order as g.vs)
    measure      : key into _CMAPS / _MEASURE_LABELS
    graph_label  : short string appended to title, e.g. "Full graph"
    community_idx: community number for title
    top_k        : number of highest-value nodes to label
    out_path     : output PNG path
    dpi          : output resolution
    """
    N        = g.vcount()
    names    = g.vs["name"]
    is_hubs  = g.vs["is_hub"]
    cmap     = _CMAPS[measure]
    val_arr  = np.array(values)

    # Colourmap normalisation — protect against flat distributions
    vmin, vmax = val_arr.min(), val_arr.max()
    if vmax == vmin:
        vmax = vmin + 1e-9
    norm   = Normalize(vmin=vmin, vmax=vmax)
    colors = [to_hex(cmap(norm(v))) for v in val_arr]

    # Identify top-K nodes for labelling
    top_indices = set(np.argsort(val_arr)[-top_k:].tolist())

    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Edges (all rendered first so nodes sit on top) ───────────────────────
    for e in g.es:
        u, v = e.source, e.target
        ax.plot([coords[u, 0], coords[v, 0]],
                [coords[u, 1], coords[v, 1]],
                color="lightgray", lw=0.5, alpha=0.6, zorder=1)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    hub_idx    = [i for i in range(N) if     is_hubs[i]]
    circle_idx = [i for i in range(N) if not is_hubs[i]]

    # Regular nodes → circles, sized and coloured by centrality
    if circle_idx:
        # Node size: scale slightly with value so high-centrality nodes pop
        sizes = [30 + 120 * norm(val_arr[i]) for i in circle_idx]
        ax.scatter(coords[circle_idx, 0], coords[circle_idx, 1],
                   s=sizes,
                   c=[colors[i] for i in circle_idx],
                   edgecolors="white", linewidths=0.5,
                   zorder=3, alpha=0.95)

    # Hub nodes → stars, larger, coloured by centrality
    if hub_idx:
        hub_sizes = [80 + 200 * norm(val_arr[i]) for i in hub_idx]
        ax.scatter(coords[hub_idx, 0], coords[hub_idx, 1],
                   s=hub_sizes, marker="*",
                   c=[colors[i] for i in hub_idx],
                   edgecolors="black", linewidths=0.8,
                   zorder=4)

    # ── Labels for top-K nodes only ───────────────────────────────────────────
    for i in top_indices:
        x, y = coords[i]
        ax.text(x, y, names[i],
                fontsize=6.5, fontweight="bold",
                color="black", ha="center", va="bottom", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.0,
                                            foreground="white")])

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm  = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=35)
    cbar.set_label(_MEASURE_LABELS[measure], fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        f"Community {community_idx} – {_MEASURE_LABELS[measure]}\n"
        f"{graph_label}  |  top-{top_k} nodes labelled  |  "
        f"{N} plasmids · {g.ecount()} edges",
        fontsize=13, pad=10, fontweight="bold"
    )

    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def render_degree_distribution(g: ig.Graph,
                                graph_label: str,
                                community_idx: int,
                                out_path: Path,
                                dpi: int = 200) -> None:
    """Six-panel degree distribution plot following Examples.ipynb style.

    Panels (2 rows × 3 columns)
    ---------------------------
    Row 1 (linear y-axis):
        Col 1 – raw frequency (no bins)
        Col 2 – frequency with linear bins
        Col 3 – frequency with log bins
    Row 2 (log-log axes):
        Same three representations on log-log scale.

    A power-law distribution appears as a straight line on a log-log plot
    with log-spaced bins (bottom-right panel).

    Parameters
    ----------
    g            : igraph.Graph
    graph_label  : e.g. "Full graph" or "No-hub graph"
    community_idx: community number for title
    out_path     : output PNG path
    dpi          : output resolution
    """
    degrees = g.degree()
    # powerlaw package requires positive values; remove isolated nodes (deg=0)
    data = [d for d in degrees if d > 0]

    if len(data) < 3:
        print(f"  Skipping degree distribution for {graph_label} "
              f"(too few non-isolated nodes)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    # Column definitions: (title, function, kwargs, log-log)
    configs = [
        # Row 0 (linear scale)
        ("Raw frequency",                  raw_frequency,  {},                      False),
        ("Linear bins",                    bin_frequency,  {},                      False),
        ("Log bins",                       bin_frequency,  {"logarithmic_bins": True}, False),
        # Row 1 (log-log)
        ("Raw frequency  (log-log)",       raw_frequency,  {},                      True),
        ("Linear bins  (log-log)",         bin_frequency,  {},                      True),
        ("Log bins  (log-log)",            bin_frequency,  {"logarithmic_bins": True}, True),
    ]
    colors_row = ["#d62728", "#1f77b4", "#1f77b4",
                  "#d62728", "#1f77b4", "#1f77b4"]
    markers    = ["v", "o", "o", "^", "x", "o"]

    for idx, (title, fn, kwargs, loglog) in enumerate(configs):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        try:
            x, y = fn(data, **kwargs)
            # Filter zeros/NaNs that break log scale
            pairs = [(xi, yi) for xi, yi in zip(x, y)
                     if xi > 0 and yi > 0 and not math.isnan(yi)]
            if not pairs:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
            else:
                xs, ys = zip(*pairs)
                ax.scatter(xs, ys, marker=markers[idx],
                           c=colors_row[idx], s=30, alpha=0.8)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error:\n{exc}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Degree", fontsize=9)
        ax.set_ylabel("Frequency / density", fontsize=9)
        ax.tick_params(labelsize=8)
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")

    fig.suptitle(
        f"Community {community_idx} – Degree distribution\n{graph_label}",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_top_k(g: ig.Graph, values: list[float], measure: str,
                graph_label: str, k: int = 10) -> None:
    """Print the top-K plasmids ranked by a centrality measure."""
    names    = g.vs["name"]
    is_hubs  = g.vs["is_hub"]
    ranked   = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    print(f"\n  Top-{k} by {_MEASURE_LABELS[measure]}  [{graph_label}]")
    print(f"  {'Rank':>4}  {'Plasmid':<30}  {'Value':>8}  Hub")
    print(f"  {'-'*4}  {'-'*30}  {'-'*8}  ---")
    for rank, idx in enumerate(ranked[:k], 1):
        hub_flag = "★" if is_hubs[idx] else ""
        print(f"  {rank:>4}  {names[idx]:<30}  {values[idx]:>8.4f}  {hub_flag}")


def write_summary(out_dir: Path, community_idx: int, graphs: list,
                  raw_results: dict, top_k: int = 10) -> None:
    """Write a plain-text summary of the reported centrality results.

    The summary is raw for degree, betweenness, and eigenvector, but uses
    the normalized WF closeness values because the graph is disconnected.
    """
    out_path = out_dir / f"summary_community_{community_idx}.txt"
    with open(out_path, "w") as fh:
        fh.write(f"Community {community_idx} centrality summary\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("Note: closeness is reported using WF-normalized values only.\n\n")
        fh.write("Note: eigenvector centrality is only meaningful within the largest connected component.\n\n")

        for g, coords, label, suffix in graphs:
            fh.write(f"Graph: {label} ({suffix})\n")
            fh.write(f"Nodes: {g.vcount()}   Edges: {g.ecount()}\n\n")

            for measure in ("degree", "betweenness", "closeness", "eigenvector"):
                vals = raw_results.get(measure, {}).get(suffix)
                if vals is None:
                    continue
                arr = np.array(vals)
                if arr.size == 0:
                    fh.write(f"{measure}: no data\n\n")
                    continue
                maxv = float(arr.max())
                minv = float(arr.min())
                max_abs = float(np.abs(arr).max())
                fh.write(f"{measure}: min={minv:.6g}  max={maxv:.6g}  max_abs={max_abs:.6g}\n")

                # Top-K listing
                names = g.vs["name"]
                is_hubs = g.vs["is_hub"]
                ranked = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
                fh.write(f"Top-{top_k}:\n")
                for rank, idx in enumerate(ranked[:top_k], 1):
                    hub_flag = "★" if is_hubs[idx] else ""
                    fh.write(f"  {rank:>2}. {names[idx]:<30}  {vals[idx]:>10.6f}  {hub_flag}\n")
                fh.write("\n")

    print(f"  Summary saved → {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Container for computed centrality values:
    # raw_results[measure][suffix] = raw list (absolute values)
    # norm_results[measure][suffix] = normalized list (for visualization)
    measures = ("degree", "betweenness", "closeness", "eigenvector")
    raw_results = {m: {} for m in measures}
    norm_results = {m: {} for m in measures}

    # ── Load both igraph objects ───────────────────────────────────────────────
    print("Loading graphs …")
    g_full, g_no_hubs = load_graphs(args.json, args.typing)

    # ── Compute layouts (same three-phase layout as reference figures) ─────────
    print("\nComputing layouts …")
    coords_full    = hybrid_layout(g_full,    seed=42)
    coords_no_hubs = hybrid_layout(g_no_hubs, seed=42)

    # ── Define the analysis pairs ─────────────────────────────────────────────
    # Each entry: (graph, coords, short_label, file_suffix)
    graphs = [
        (g_full,    coords_full,    "Full graph (with hub nodes)", "full"),
        (g_no_hubs, coords_no_hubs, "No-hub graph",               "no_hubs"),
    ]

    # ── 1. Degree centrality + power-law distribution ─────────────────────────
    print("\n── Degree Centrality ──────────────────────────────────────────")
    for g, coords, label, suffix in graphs:
        # Degree: compute via helper (raw and normalized)
        vals_raw = degree_centrality(g, normalized=False)
        vals_norm = degree_centrality(g, normalized=True)
        raw_results["degree"][suffix] = vals_raw
        norm_results["degree"][suffix] = vals_norm
        print_top_k(g, vals_norm, "degree", label, k=args.top_k)

        render_centrality_graph(
            g, coords, vals_norm,
            measure="degree", graph_label=label,
            community_idx=args.community, top_k=args.top_k,
            out_path=out_dir / f"degree_centrality_{suffix}.png",
            dpi=args.dpi,
        )
        render_degree_distribution(
            g, graph_label=label,
            community_idx=args.community,
            out_path=out_dir / f"degree_distribution_{suffix}.png",
            dpi=args.dpi,
        )

    # ── 2. Betweenness centrality ─────────────────────────────────────────────
    print("\n── Betweenness Centrality ─────────────────────────────────────")
    for g, coords, label, suffix in graphs:
        # Betweenness: helper handles raw vs normalized
        vals_raw = betweenness_centrality(g, normalized=False)
        vals_norm = betweenness_centrality(g, normalized=True)
        raw_results["betweenness"][suffix] = vals_raw
        norm_results["betweenness"][suffix] = vals_norm
        print_top_k(g, vals_norm, "betweenness", label, k=args.top_k)

        render_centrality_graph(
            g, coords, vals_norm,
            measure="betweenness", graph_label=label,
            community_idx=args.community, top_k=args.top_k,
            out_path=out_dir / f"betweenness_centrality_{suffix}.png",
            dpi=args.dpi,
        )

    # ── 3. Closeness centrality ───────────────────────────────────────────────
    print("\n── Closeness Centrality ───────────────────────────────────────")
    for g, coords, label, suffix in graphs:
        # Closeness: raw within-reachable-set vs WF-normalized
        vals_raw = closeness_centrality(g, normalized=False)
        vals_norm = closeness_centrality(g, normalized=True)
        # On disconnected graphs the raw reachable-set value is not stable
        # for cross-graph validation, so keep the WF-normalized values here.
        raw_results["closeness"][suffix] = vals_norm
        norm_results["closeness"][suffix] = vals_norm
        print_top_k(g, vals_norm, "closeness", label, k=args.top_k)

        render_centrality_graph(
            g, coords, vals_norm,
            measure="closeness", graph_label=label,
            community_idx=args.community, top_k=args.top_k,
            out_path=out_dir / f"closeness_centrality_{suffix}.png",
            dpi=args.dpi,
        )

    # ── 4. Eigenvector centrality ─────────────────────────────────────────────
    print("\n── Eigenvector Centrality ─────────────────────────────────────")
    print("  Note: for the no-hub graph (11 components) eigenvector centrality")
    print("  is only meaningful within the largest connected component (129 nodes).")
    for g, coords, label, suffix in graphs:
        vals_norm = eigenvector_centrality(g, normalized=True)
        vals_raw = vals_norm
        raw_results["eigenvector"][suffix] = vals_raw
        norm_results["eigenvector"][suffix] = vals_norm
        print_top_k(g, vals_norm, "eigenvector", label, k=args.top_k)

        render_centrality_graph(
            g, coords, vals_norm,
            measure="eigenvector", graph_label=label,
            community_idx=args.community, top_k=args.top_k,
            out_path=out_dir / f"eigenvector_centrality_{suffix}.png",
            dpi=args.dpi,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    files = sorted(out_dir.glob("*.png"))
    print(f"\n{'─'*60}")
    print(f"All outputs saved to: {out_dir}")
    print(f"Total files: {len(files)}")
    for f in files:
        print(f"  {f.name}")

    # Write a plain-text summary with min/max/max-abs and top-K lists
    write_summary(out_dir, args.community, graphs, raw_results, top_k=args.top_k)


if __name__ == "__main__":
    main()
