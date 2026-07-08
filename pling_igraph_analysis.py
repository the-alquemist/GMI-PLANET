"""
pling_igraph_analysis.py
=======================
Graph-theoretic centrality analysis for pling plasmid community networks.

Computes six graph measures on both the full (hub-included) and
hub-removed igraph objects produced by pling_igraph_viz.py:

    1. Degree Centrality            – how many direct neighbours a node has
    2. Betweenness Centrality       – how often a node lies on shortest paths
    3. Closeness Centrality         – how close a node is to all others
    4. Eigenvector Centrality       – being connected to well-connected nodes
    5. Clustering Coefficient       – local triangle density around each node;
                                      global value (transitivity) reported in summary
    6. Average Path Length          – mean shortest path over all reachable pairs;
                                      full pairwise-distance histogram also saved

For each node-level measure a network visualisation is produced:
    • Nodes coloured by the measure value (light = low, dark = high)
    • Top-K nodes labelled; hub nodes shown as stars in the full graph
    • Degree additionally gets a 6-panel power-law distribution plot

Average path length and global clustering coefficient are scalars —
they are reported in the text summary and as a path-length histogram.

All outputs go to:
    <out_dir>

Usage
-----
    python3 pling_igraph_analysis.py

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

# Core nodes, may highlight that they share a certain pool of genes, but not necessarily a skeleton.
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

# Bridge nodes, while understood as useful for gene propagation, may be more accurately described as the most shared genes between nodes/communities.
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

# Centric nodes, reach the most distant extremes of the graph in a minimum number of steps, share a minimum global set of genes for the entire graph.
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

# Influence nodes, still somewhat abstract with plasmids, check
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


# ── Clustering coefficient & average path length ──────────────────────────────

# Identifys plasmids that are similar to each other, most likely by Incompatibility Group (check)
# In the fuchsia cluster, if the distance is very short they may correspond to the same plasmid.
def local_clustering_coefficient(g: ig.Graph) -> list[float]:
    """Local clustering coefficient for every vertex.

    Uses igraph's ``transitivity_local_undirected()``, which counts the
    fraction of a node's neighbour pairs that are themselves connected:

        C(v) = (triangles through v) / (connected triples centred on v)

    Nodes with degree < 2 cannot form triangles and receive NaN from igraph;
    we replace those with 0.0 so colourmap normalisation works correctly.
    """
    raw = g.transitivity_local_undirected()
    return [0.0 if (v is None or math.isnan(v)) else v for v in raw]


def global_clustering_coefficient(g: ig.Graph) -> float:
    """Global clustering coefficient (graph-level transitivity).

    Uses igraph's ``transitivity_undirected()``, which counts closed
    triplets over all triplets in the graph — equivalent to the ratio of
    (3 × triangles) / (connected triples).  Returns a single float in [0, 1].
    """
    return g.transitivity_undirected()


def average_path_length_metric(g: ig.Graph) -> float:
    """Mean shortest-path length over all reachable vertex pairs.

    Uses igraph's ``average_path_length(directed=False, unconn=True)``.
    When ``unconn=True`` igraph skips pairs with no path (infinite distance),
    which is the correct treatment for the disconnected no-hub graph — it
    computes the average within and across components for reachable pairs only.
    """
    return g.average_path_length(directed=False, unconn=True)


def pairwise_path_lengths(g: ig.Graph) -> list[int]:
    """Collect all finite pairwise shortest-path lengths (upper triangle only).

    Used to draw the path-length distribution histogram.  Self-distances (0)
    and unreachable pairs (inf) are excluded.
    """
    dist_matrix = g.distances()   # n×n list-of-lists; inf = unreachable
    n = g.vcount()
    lengths = []
    for i in range(n):
        for j in range(i + 1, n):   # upper triangle → each pair once
            d = dist_matrix[i][j]
            if d != float("inf"):
                lengths.append(int(d))
    return lengths


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


def raw_histogram(data: list, n_bins: int = None) -> tuple:
    """Histogram for continuous data, returning bin-centre / count pairs.

    Used as the "raw frequency" panel for continuous centrality values,
    in place of the exact-count approach used for discrete degree values.

    Parameters
    ----------
    data   : list of positive float values
    n_bins : number of bins; defaults to the square-root rule (sqrt of n)

    Returns
    -------
    centres : list of bin midpoints
    counts  : list of integer counts per bin
    """
    if n_bins is None:
        n_bins = max(5, int(math.sqrt(len(data))))
    counts, edges = np.histogram(data, bins=n_bins)
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    return centres, list(counts)


def render_measure_distribution(values: list,
                                 measure_name: str,
                                 x_label: str,
                                 graph_label: str,
                                 community_idx: int,
                                 out_path: Path,
                                 discrete: bool = False,
                                 dpi: int = 200) -> None:
    """Six-panel distribution plot for any graph measure — degree or centrality.

    Replaces both the old ``render_degree_distribution`` and the continuous-only
    version: pass ``discrete=True`` for integer degree values (uses exact counts
    in the raw panel), or ``discrete=False`` (default) for continuous centrality
    values (uses a sqrt-rule histogram in the raw panel).

    Panels (2 rows × 3 columns)
    ---------------------------
    Row 1 (linear y-axis):
        Col 1 – raw frequency / histogram
        Col 2 – probability density, linear bins  (powerlaw package)
        Col 3 – probability density, log-spaced bins
    Row 2 (log-log axes):
        Same three representations on log-log scale.

    When the positive data spans < 1 log-decade the log-bin panels are
    suppressed and replaced with an annotation explaining why.

    Parameters
    ----------
    values       : per-node values (integers for degree, floats for centrality)
    measure_name : display name used in the title
    x_label      : x-axis label
    graph_label  : e.g. "Full graph" or "No-hub graph"
    community_idx: community number for title
    out_path     : output PNG path
    discrete     : True → use exact counts (degree); False → histogram (centrality)
    dpi          : output resolution
    """
    positive = [v for v in values if v > 0]

    if len(positive) < 5:
        print(f"  Skipping {measure_name} distribution for {graph_label} "
              f"(fewer than 5 positive values)")
        return

    # Detect whether the data spans enough range for log-binning to be useful.
    # Log-spaced bins require at least ~1 decade (10x range) to produce
    # more than 2-3 meaningful bins.  Connected-graph closeness, for example,
    # typically spans only 0.5 decades — log panels would be nearly empty.
    log_span = math.log10(max(positive) / min(positive))
    log_ok   = log_span >= 1.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    # Col 0: raw_frequency for ALL measures.
    # For degree (discrete): groups identical integer values → meaningful counts.
    # For continuous measures (betweenness, closeness, eigenvector): floats that
    # happen to be identical across nodes still group correctly, giving one point
    # per unique value rather than collapsing everything into sqrt(n) histogram
    # bins.  Betweenness has max_count=19, closeness WF max_count=21, so the
    # y-axis carries real information rather than being a flat line at y=1.
    raw_fn    = raw_frequency
    raw_title = "Raw frequency"
    configs = [
        (raw_title,                    raw_fn,         {},                         False),
        ("Linear bins",                bin_frequency,  {},                         False),
        ("Log bins",                   bin_frequency,  {"logarithmic_bins": True}, False),
        (raw_title + "  (log-log)",    raw_fn,         {},                         True),
        ("Linear bins  (log-log)",     bin_frequency,  {},                         True),
        ("Log bins  (log-log)",        bin_frequency,  {"logarithmic_bins": True}, True),
    ]
    colors_row = ["#d62728", "#1f77b4", "#1f77b4",
                  "#d62728", "#1f77b4", "#1f77b4"]
    markers    = ["v", "o", "o", "^", "x", "o"]

    for idx, (title, fn, kwargs, loglog) in enumerate(configs):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        # Log-bin columns (col 2) need sufficient data range.
        is_logbin_col = (col == 2)
        if is_logbin_col and not log_ok:
            ax.text(0.5, 0.5,
                    f"Log bins unavailable:\ndata spans only "
                    f"{log_span:.2f} decades\n(<1 decade required)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888888",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel("Frequency / density", fontsize=9)
            ax.tick_params(labelsize=8)
            if loglog:
                ax.set_xscale("log")
                ax.set_yscale("log")
            continue

        try:
            x, y = fn(positive, **kwargs)
            # Drop zeros / NaNs that break log scale
            pairs = [(xi, yi) for xi, yi in zip(x, y)
                     if xi > 0 and yi > 0
                     and not (isinstance(yi, float) and math.isnan(yi))]
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
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Frequency / density", fontsize=9)
        ax.tick_params(labelsize=8)
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")

    n_zero    = sum(1 for v in values if v == 0)
    n_positive = len(positive)
    fig.suptitle(
        f"Community {community_idx} – {measure_name} distribution\n"
        f"{graph_label}  |  {n_positive} non-zero values"
        + (f"  ({n_zero} zeros excluded)" if n_zero else ""),
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# Colour scheme matching Examples.ipynb; clustering uses Reds (distinct from others)
_CMAPS = {
    "degree":       cm.Greens,
    "betweenness":  cm.Oranges,
    "closeness":    cm.Blues,
    "eigenvector":  cm.Purples,
    "clustering":   cm.Reds,
}

_MEASURE_LABELS = {
    "degree":       "Degree Centrality",
    "betweenness":  "Betweenness Centrality",
    "closeness":    "Closeness Centrality",
    "eigenvector":  "Eigenvector Centrality",
    "clustering":   "Local Clustering Coefficient",
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


def render_path_length_distribution(g: ig.Graph,
                                     graph_label: str,
                                     community_idx: int,
                                     apl: float,
                                     out_path: Path,
                                     dpi: int = 200) -> None:
    """Histogram of all pairwise shortest-path lengths.

    Shows the full distribution of distances between reachable node pairs,
    with a vertical dashed line marking the average path length (APL).
    For disconnected graphs only reachable pairs are included (inf excluded),
    so the no-hub graph shows within- and across-component distances for
    pairs that can actually communicate.

    Parameters
    ----------
    g             : igraph.Graph
    graph_label   : e.g. "Full graph" or "No-hub graph"
    community_idx : community number for title
    apl           : pre-computed average path length (scalar)
    out_path      : output PNG path
    dpi           : output resolution
    """
    lengths = pairwise_path_lengths(g)

    if not lengths:
        print(f"  Skipping path-length distribution for {graph_label} "
              f"(no reachable pairs)")
        return

    max_len = max(lengths)
    bins    = range(1, max_len + 2)   # one bar per integer distance

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    ax.hist(lengths, bins=bins, align="left", rwidth=0.75,
            color="#2166ac", edgecolor="white", linewidth=0.5, alpha=0.85)

    # Vertical line for the average path length
    ax.axvline(apl, color="#d62728", linewidth=2.0, linestyle="--",
               label=f"Average path length = {apl:.3f}")
    ax.legend(fontsize=10)

    ax.set_xlabel("Shortest path length", fontsize=11)
    ax.set_ylabel("Number of reachable pairs", fontsize=11)
    ax.set_xticks(range(1, max_len + 1))
    ax.tick_params(labelsize=9)
    ax.set_title(
        f"Community {community_idx} – Pairwise path-length distribution\n"
        f"{graph_label}  |  {len(lengths):,} reachable pairs",
        fontsize=13, fontweight="bold"
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def render_combined_centrality_distribution(
        centrality_values: dict[str, list],
        graph_label: str,
        community_idx: int,
        out_path: Path,
        dpi: int = 200) -> None:
    """Single log-log panel showing the distribution of all four centrality
    measures overlaid on the same axes, using log-spaced bins.

    Parameters
    ----------
    centrality_values : dict mapping measure key → per-node value list
                        (degree, betweenness, closeness, eigenvector)
    graph_label       : e.g. "Full graph" or "No-hub graph"
    community_idx     : community number for title
    out_path          : output PNG path
    dpi               : output resolution
    """
    # Representative colour from each colormap (evaluated at 0.75 = dark shade)
    measure_styles = {
        "degree":      (_CMAPS["degree"](0.75),     "Degree",      "o"),
        "betweenness": (_CMAPS["betweenness"](0.75),"Betweenness", "s"),
        "closeness":   (_CMAPS["closeness"](0.75),  "Closeness (WF)", "^"),
        "eigenvector": (_CMAPS["eigenvector"](0.75),"Eigenvector", "D"),
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    plotted_any = False

    for measure, (color, label, marker) in measure_styles.items():
        vals = centrality_values.get(measure)
        if vals is None:
            continue

        # Keep only positive values, then normalise to [0, 1]
        positive = [v for v in vals if v > 0]
        if len(positive) < 5:
            continue
        v_max = max(positive)
        normed = [v / v_max for v in positive]

        # Log-span check: need ≥ 1 decade for log bins to be meaningful
        log_span = math.log10(max(normed) / min(normed))
        if log_span < 1.0:
            # Fall back to linear bins so the measure still appears
            try:
                x, y = bin_frequency(normed, logarithmic_bins=False)
            except Exception:
                continue
        else:
            try:
                x, y = bin_frequency(normed, logarithmic_bins=True)
            except Exception:
                continue

        pairs = [(xi, yi) for xi, yi in zip(x, y)
                 if xi > 0 and yi > 0
                 and not (isinstance(yi, float) and math.isnan(yi))]
        if not pairs:
            continue

        xs, ys = zip(*pairs)
        ax.scatter(xs, ys, marker=marker, color=color, s=45, alpha=0.85,
                   label=label, zorder=3)
        # Thin connecting line to show trend
        ax.plot(xs, ys, color=color, lw=0.8, alpha=0.4, zorder=2)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"  Skipping combined distribution for {graph_label} "
              f"(insufficient data for all measures)")
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Normalised centrality value  [0, 1]", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, framealpha=0.85, title="Measure",
              title_fontsize=9)
    ax.set_title(
        f"Community {community_idx} – Centrality distributions (log-log, log bins)\n"
        f"{graph_label}  |  values normalised per measure to [0, 1]",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


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
                  raw_results: dict, scalar_results: dict,
                  top_k: int = 10) -> None:
    """Write a plain-text summary of all graph measures.

    Node-level measures (degree, betweenness, closeness, eigenvector,
    clustering) are reported with min / max / mean / std and a top-K ranking.

    Graph-level scalars (global clustering coefficient, average path length)
    are reported once per graph since they do not have per-node values.
    """
    out_path = out_dir / f"summary_community_{community_idx}.txt"
    with open(out_path, "w") as fh:
        fh.write(f"Community {community_idx} – Graph Analysis Summary\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("Notes\n")
        fh.write("-----\n")
        fh.write("• Closeness uses Wasserman-Faust normalization for disconnected graphs.\n")
        fh.write("• Eigenvector centrality is only meaningful within the largest\n")
        fh.write("  connected component; smaller-component nodes are set to 0.\n")
        fh.write("• Local clustering coefficient is 0 for nodes with degree < 2.\n")
        fh.write("• Average path length excludes unreachable pairs (inf distances).\n\n")

        for g, coords, label, suffix in graphs:
            fh.write("─" * 60 + "\n")
            fh.write(f"Graph: {label}\n")
            fh.write(f"  Nodes : {g.vcount()}\n")
            fh.write(f"  Edges : {g.ecount()}\n")
            n_comp = g.connected_components().n
            fh.write(f"  Components : {n_comp}\n")
            if n_comp > 1:
                sizes = sorted(g.connected_components().sizes(), reverse=True)
                fh.write(f"  Component sizes : {sizes}\n")
            fh.write("\n")

            # ── Graph-level scalars ───────────────────────────────────────────
            global_cc = scalar_results.get("global_clustering", {}).get(suffix)
            apl       = scalar_results.get("average_path_length", {}).get(suffix)
            if global_cc is not None:
                fh.write(f"  Global Clustering Coefficient (transitivity) : {global_cc:.6f}\n")
            if apl is not None:
                fh.write(f"  Average Path Length (reachable pairs)         : {apl:.6f}\n")
            fh.write("\n")

            # ── Node-level measures ───────────────────────────────────────────
            node_measures = ("degree", "betweenness", "closeness",
                             "eigenvector", "clustering")
            for measure in node_measures:
                vals = raw_results.get(measure, {}).get(suffix)
                if vals is None:
                    continue
                arr = np.array(vals, dtype=float)
                if arr.size == 0:
                    fh.write(f"  {measure}: no data\n\n")
                    continue

                fh.write(f"  {_MEASURE_LABELS[measure]}\n")
                fh.write(f"    min     = {arr.min():.6f}\n")
                fh.write(f"    max     = {arr.max():.6f}\n")
                fh.write(f"    max_abs = {np.abs(arr).max():.6f}\n")
                fh.write(f"    mean    = {arr.mean():.6f}\n")
                fh.write(f"    std     = {arr.std():.6f}\n")

                names   = g.vs["name"]
                is_hubs = g.vs["is_hub"]
                ranked  = sorted(range(len(vals)), key=lambda i: vals[i],
                                 reverse=True)
                fh.write(f"    Top-{top_k}:\n")
                for rank, idx in enumerate(ranked[:top_k], 1):
                    hub_flag = " ★" if is_hubs[idx] else ""
                    fh.write(f"      {rank:>2}. {names[idx]:<30}"
                             f"  {vals[idx]:>10.6f}{hub_flag}\n")
                fh.write("\n")

    print(f"  Summary saved → {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = parse_args()
    out_dir     = Path(args.out_dir)
    graphs_dir  = out_dir / "graphs"    # network visualisations
    metrics_dir = out_dir / "metrics"   # distribution plots
    graphs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Container for computed centrality values:
    # raw_results[measure][suffix]    = per-node list (absolute / WF values)
    # norm_results[measure][suffix]   = normalized list (for visualisation)
    # scalar_results[measure][suffix] = single float (graph-level scalars)
    measures = ("degree", "betweenness", "closeness", "eigenvector", "clustering")
    raw_results    = {m: {} for m in measures}
    norm_results   = {m: {} for m in measures}
    scalar_results = {"global_clustering": {}, "average_path_length": {}}

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
            out_path=graphs_dir / f"degree_centrality_{suffix}.png",
            dpi=args.dpi,
        )
        render_measure_distribution(
            values=g.degree(), measure_name="Degree",
            x_label="Degree", graph_label=label,
            community_idx=args.community,
            out_path=metrics_dir / f"degree_distribution_{suffix}.png",
            discrete=True, dpi=args.dpi,
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
            out_path=graphs_dir / f"betweenness_centrality_{suffix}.png",
            dpi=args.dpi,
        )
        # Six-panel distribution for raw (unnormalised) betweenness scores.
        # Raw integer-like counts expose the heavy tail more clearly than
        # normalised values, which compress everything into [0, 1].
        render_measure_distribution(
            values       = vals_raw,
            measure_name = "Betweenness Centrality",
            x_label      = "Raw betweenness score",
            graph_label  = label,
            community_idx= args.community,
            out_path     = metrics_dir / f"betweenness_distribution_{suffix}.png",
            dpi          = args.dpi,
        )

    # ── 3. Closeness centrality ───────────────────────────────────────────────
    print("\n── Closeness Centrality ───────────────────────────────────────")
    for g, coords, label, suffix in graphs:
        # Closeness: raw within-reachable-set vs WF-normalized
        vals_raw = closeness_centrality(g, normalized=False)
        vals_norm = closeness_centrality(g, normalized=True)
        # On disconnected graphs the raw reachable-set value is not stable
        # for cross-graph validation, so we keep the WF-normalized values here.
        raw_results["closeness"][suffix] = vals_norm
        norm_results["closeness"][suffix] = vals_norm
        print_top_k(g, vals_norm, "closeness", label, k=args.top_k)

        render_centrality_graph(
            g, coords, vals_norm,
            measure="closeness", graph_label=label,
            community_idx=args.community, top_k=args.top_k,
            out_path=graphs_dir / f"closeness_centrality_{suffix}.png",
            dpi=args.dpi,
        )
        # Six-panel distribution using WF-normalised values.
        # WF normalization is used (not raw within-component closeness) because
        # it gives comparable, artefact-free values across both graphs.
        render_measure_distribution(
            values       = vals_norm,
            measure_name = "Closeness Centrality (Wasserman-Faust)",
            x_label      = "WF-normalised closeness",
            graph_label  = label,
            community_idx= args.community,
            out_path     = metrics_dir / f"closeness_distribution_{suffix}.png",
            dpi          = args.dpi,
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
            out_path=graphs_dir / f"eigenvector_centrality_{suffix}.png",
            dpi=args.dpi,
        )
        # Six-panel distribution using only values from the largest connected
        # component.  Nodes in smaller components receive near-zero floating-point
        # values (e.g. 3e-19) due to numerical precision — NOT meaningful
        # eigenvector scores.  We identify them explicitly via component membership
        # rather than a magic threshold, which is cleaner and more robust.
        comps_ev   = g.connected_components()
        main_id_ev = comps_ev.sizes().index(max(comps_ev.sizes()))
        ev_major_only = [vals_norm[i] for i in range(g.vcount())
                         if comps_ev.membership[i] == main_id_ev]
        n_excluded    = g.vcount() - len(ev_major_only)
        ev_label      = (f"{label}  –  major component only"
                         f"  ({n_excluded} small-component nodes excluded)"
                         if n_excluded else label)
        render_measure_distribution(
            values       = ev_major_only,
            measure_name = "Eigenvector Centrality",
            x_label      = "Eigenvector centrality score",
            graph_label  = ev_label,
            community_idx= args.community,
            out_path     = metrics_dir / f"eigenvector_distribution_{suffix}.png",
            dpi          = args.dpi,
        )

    # ── 5. Clustering coefficient ─────────────────────────────────────────────
    print("\n── Clustering Coefficient ─────────────────────────────────────")
    for g, coords, label, suffix in graphs:
        # Local CC: per-node value (NaN-safe; nodes with deg < 2 → 0)
        vals_local = local_clustering_coefficient(g)
        raw_results["clustering"][suffix]  = vals_local
        norm_results["clustering"][suffix] = vals_local
        print_top_k(g, vals_local, "clustering", label, k=args.top_k)

        # Global CC: single transitivity value for the graph
        g_cc = global_clustering_coefficient(g)
        scalar_results["global_clustering"][suffix] = g_cc
        print(f"\n  Global clustering coefficient [{label}]: {g_cc:.4f}")

        render_centrality_graph(
            g, coords, vals_local,
            measure="clustering", graph_label=label,
            community_idx=args.community, top_k=args.top_k,
            out_path=graphs_dir / f"clustering_coefficient_{suffix}.png",
            dpi=args.dpi,
        )

    # ── 6. Average path length ────────────────────────────────────────────────
    print("\n── Average Path Length ────────────────────────────────────────")
    print("  (computed over reachable pairs only — disconnected pairs excluded)")
    for g, coords, label, suffix in graphs:
        apl = average_path_length_metric(g)
        scalar_results["average_path_length"][suffix] = apl
        print(f"\n  Average path length [{label}]: {apl:.4f}")

        render_path_length_distribution(
            g, graph_label=label,
            community_idx=args.community,
            apl=apl,
            out_path=metrics_dir / f"path_length_distribution_{suffix}.png",
            dpi=args.dpi,
        )

    # ── 7. Combined centrality distribution (1 × 1 log-log panel) ────────────
    print("\n── Combined Centrality Distribution ───────────────────────────")
    for g, coords, label, suffix in graphs:
        combined_vals = {
            "degree":      norm_results["degree"][suffix],
            "betweenness": norm_results["betweenness"][suffix],
            "closeness":   norm_results["closeness"][suffix],
            "eigenvector": norm_results["eigenvector"][suffix],
        }
        render_combined_centrality_distribution(
            centrality_values = combined_vals,
            graph_label       = label,
            community_idx     = args.community,
            out_path          = metrics_dir / f"combined_distribution_{suffix}.png",
            dpi               = args.dpi,
        )

    # ── 8. Export graphs ──────────────────────────────────────────────────────
    print("\n── Exporting graphs ───────────────────────────────────────────")
    for g, _, label, suffix in graphs:
        for fmt, writer in [("graphml", g.write_graphml),
                             ("gml",     g.write_gml)]:
            p = out_dir / f"community_{args.community}_{suffix}.{fmt}"
            writer(str(p))
            print(f"  Saved → {p.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    # Write a plain-text summary with min/max/max-abs and top-K lists
    write_summary(out_dir, args.community, graphs, raw_results,
                  scalar_results, top_k=args.top_k)

if __name__ == "__main__":
    main()