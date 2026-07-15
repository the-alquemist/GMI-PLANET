"""
pling_igraph_consensus.py
==========================
Composite centrality ranking for pling plasmid community networks.

Replaces the previous binary top-N consensus score (0–5 integer) with a
continuous composite centrality score derived from z-scores, preserving
the full information in each centrality measure.

Algorithm
---------
For each of the four centrality measures (degree, betweenness, closeness,
eigenvector):

  1. Optionally apply np.log1p() to the raw values before z-scoring.
     → Use this when a measure is highly right-skewed (e.g. betweenness,
       where a few hub nodes have values orders of magnitude above the rest).
       log1p(x) = log(1 + x) compresses the long tail while keeping the
       ranking order intact and handling zeros safely (log1p(0) = 0).
       Without this transformation, extreme outliers dominate the z-score
       and most nodes are assigned very similar negative scores.

  2. Compute the z-score:  z = (x - mean) / std
     → Centres each measure at 0 and scales to unit variance so that
       measures on different scales (degree 0–0.5, betweenness 0–0.3,
       closeness 0.15–0.48, eigenvector 0–1) contribute equally.

  3. Average the four z-scores to obtain the composite centrality score.
     → A node scoring +2 on all four measures gets composite = +2.
     → A node with mixed high/low scores gets a moderate composite.
     → Nodes below average on all measures receive a negative composite.

  4. Rank nodes from highest (rank 1) to lowest composite score.

Visual encoding
---------------
  Node fill   → biological resistance category (most-specific wins):
                  red    = KPC + NDM double resistance
                  yellow = Klebsiella beta-lactamase
                  green  = Any beta-lactamase (mobilisable)
                  gray   = No resistance annotation
  Node size   → composite score shifted to [0,1] and scaled to [15, 245]
                so even the lowest-scoring node is visible
  Node label  → rank number (1 = highest composite score);
                shown for top-K nodes and all bio-annotated nodes
  Node border → thick black for rank ≤ top 10% of N, thin white otherwise
  Edge colour → black if any endpoint carries resistance genes, gray otherwise

Outputs (written to <out-dir>/)
---------------------------------
  hub_graphs/
      composite_ranking.png
  no_hub_graphs/
      composite_ranking.png
  summary.txt           — full ranked table for both graph variants

Usage
-----
    python3 pling_igraph_consensus.py \\
        --json    community_0.json \\
        --typing  typing.tsv \\
        --out-dir output/ \\
        --community 0 \\
        --top-k 20 \\
        --use-log1p \\
        --dpi 200

Dependencies
------------
    pip install igraph matplotlib pandas numpy scipy
    pling_igraph_viz.py and pling_igraph_analysis.py must be importable
    from the same directory.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from pling_igraph_viz import build_pling_graph, hybrid_layout
from pling_igraph_analysis import (
    degree_centrality,
    betweenness_centrality,
    closeness_centrality,
    eigenvector_centrality,
)


# ── Gene lists ────────────────────────────────────────────────────────────────

ALL_BETALACTAMASE_POSITIVE = [
    "pVA2682_52","pIH359_51","pIH359_23","pIH359_178","pIH359_36",
    "pVA1393_45","pVA1393_58","pVA1505_46","pVA1505_61","pVA2589_99",
    "pVA2589_21","pVA2682_199","pVA2817_112","pVA2817_162","pVA2817_210",
    "pVA527_45","pVA527_58","pVA544_51","pVA544_8","pVA569_47","pVA585_53",
    "pVA754_27","pVA754_57","pVA754_184","pVA767_46","pVA767_57",
    "pVA1626_87","pVA1626_45","pVA1626_57","pVA1626_171","pVA1686_54",
    "pVA1686_57","pVA1825_54","pVA1825_57","pVA3495_148","pVA3495_43",
    "pVA436_54","pVA436_52","pVA634_102","pVA634_54","pVA634_271",
    "pVA634_25","pVA634_75","pReSeq_VA61_41","pReSeq_VA61_126_B",
    "pVA1101_21","pVA1722_46","pVA1722_58","pVA1835_58_A","pVA1835_58_B",
    "pVA2067_45","pVA2067_271","pVA528_44","pVA528_281","pVA605_91",
    "pVA605_296","pVA692_45","pVA692_270","pVA765_45","pVA765_275",
    "pVA791_57","pVA791_270","pVA524_46","pVA524_58","pIH290_84",
    "pIH290_7","pIH290_60",
]
KLEBSIELLA_BETALACTAMASE_POSITIVE = [
    "pVA2682_52","pIH359_51","pIH359_23","pIH359_178","pIH359_36",
    "pVA1393_45","pVA1393_58","pVA1505_46","pVA1505_61","pVA2589_99",
    "pVA2589_21","pVA2682_199","pVA2817_112","pVA2817_162","pVA2817_210",
    "pVA527_45","pVA527_58","pVA544_51","pVA544_8","pVA569_47","pVA585_53",
    "pVA754_27","pVA754_57","pVA754_184","pVA767_46","pVA767_57",
]
KPC_NDM_DOUBLE_RESISTANCE = ["pVA2682_52"]

_SET_KPC   = set(KPC_NDM_DOUBLE_RESISTANCE)
_SET_KLEBS = set(KLEBSIELLA_BETALACTAMASE_POSITIVE)
_SET_ALL   = set(ALL_BETALACTAMASE_POSITIVE)

# The four measures used in the composite score.
# Clustering coefficient is deliberately excluded: it measures local
# neighbourhood density rather than network-wide influence, and its
# z-score can dominate the composite in densely cliqued subcommunities.
MEASURES = ["degree", "betweenness", "closeness", "eigenvector"]

# ── Visual constants ──────────────────────────────────────────────────────────

BIO_COLOR = {
    "kpc_ndm":    "#FF0000",
    "klebsiella": "#FFAA00",
    "all_beta":   "#FFF700",
    "none":       "#BBBBBB",
}
BIO_LABEL = {
    "kpc_ndm":    "KPC + NDM double resistance",
    "klebsiella": "Klebsiella beta-lactamase",
    "all_beta":   "Beta-lactamase (mobilisable)",
    "none":       "No resistance annotation",
}

EDGE_COLOR_BIO  = "#000000"
EDGE_COLOR_NONE = "#CCCCCC"
NODE_ALPHA      = 0.92


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Composite z-score centrality ranking for pling networks."
    )
    p.add_argument("--json",      required=True)
    p.add_argument("--typing",    required=True)
    p.add_argument("--out-dir",   required=True)
    p.add_argument("--community", type=int, default=0)
    p.add_argument("--top-k",     type=int, default=20,
                   help="Number of highest-ranked nodes to label in the figure "
                        "(default 20). All bio-annotated nodes are also labelled "
                        "regardless of rank.")
    p.add_argument("--use-log1p", action="store_true", default=False,
                   help="Apply np.log1p() to each centrality measure before "
                        "computing z-scores. Recommended when betweenness or "
                        "degree are highly right-skewed (most nodes near zero, "
                        "a few hubs with very high values). log1p(x) = log(1+x) "
                        "compresses the tail while preserving ranking order and "
                        "handling zeros safely.")
    p.add_argument("--dpi",       type=int, default=200)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def bio_category(name: str) -> str:
    """Return the most-specific biological resistance category for a node."""
    if name in _SET_KPC:   return "kpc_ndm"
    if name in _SET_KLEBS: return "klebsiella"
    if name in _SET_ALL:   return "all_beta"
    return "none"


def compute_centralities(g: ig.Graph) -> dict[str, list[float]]:
    """Compute the four centrality measures used in the composite score.

    All values are normalised to [0, 1] so that z-scores are computed on
    comparable scales before log1p transformation (if requested).

    Returns a dict: measure_name → per-node value list (same order as g.vs).
    """
    return {
        "degree":      degree_centrality(g,      normalized=True),
        "betweenness": betweenness_centrality(g, normalized=True),
        "closeness":   closeness_centrality(g,   normalized=True),
        "eigenvector": eigenvector_centrality(g, normalized=True),
    }


def compute_composite_score(g: ig.Graph,
                              centralities: dict[str, list[float]],
                              use_log1p: bool = False) -> pd.DataFrame:
    """Build a DataFrame with z-scores and composite centrality ranking.

    Steps
    -----
    1. Collect raw centraliy values into a DataFrame (one column per measure,
       one row per node).

    2. Optionally apply np.log1p() column-wise BEFORE z-scoring.
       When to use log1p:
         - Betweenness centrality is almost always right-skewed: most nodes
           lie on very few shortest paths, but hub nodes concentrate enormous
           betweenness. Without log1p, a single hub's z-score can exceed +10
           while 80% of nodes cluster near -0.3, compressing their differences.
         - Degree is similarly skewed in scale-free networks.
         - Closeness and eigenvector are typically more symmetric.
       log1p(x) = log(1 + x) is preferred over log(x) because it is safe
       for x = 0 (log1p(0) = 0) and is a smooth monotone transformation that
       preserves the original ranking order.

    3. Compute z-scores column-wise:  z = (x - mean(x)) / std(x)
       Each measure is now centred at 0 with unit variance, so all four
       contribute equally to the composite regardless of their original scale.

    4. Composite score = mean of the four z-scores.
       A node that ranks highly on all four measures gets a large positive
       composite; a node below average on all four gets a negative composite.
       Nodes with mixed high/low specialised scores get a moderate composite.

    5. Rank nodes from highest (rank 1) to lowest composite score.
       Ties are broken by the pandas default (average rank).

    Parameters
    ----------
    g             : igraph.Graph with vertex attribute 'name'
    centralities  : output of compute_centralities()
    use_log1p     : if True, apply np.log1p() before z-scoring

    Returns
    -------
    pd.DataFrame sorted by composite_score descending, with columns:
        node_id
        degree, betweenness, closeness, eigenvector  (raw normalised values)
        degree_z, betweenness_z, closeness_z, eigenvector_z  (z-scores)
        composite_score   (mean of four z-scores)
        rank              (1 = highest composite score)
        bio_category
    """
    names = g.vs["name"]
    N     = g.vcount()

    # ── Step 1: collect raw values ────────────────────────────────────────────
    df = pd.DataFrame({
        "node_id": names,
        **{m: centralities[m] for m in MEASURES},
    })

    # ── Step 2: optional log1p transformation ─────────────────────────────────
    # Apply before z-scoring so the z-scores reflect the compressed distribution.
    # The raw values in 'degree', 'betweenness', etc. columns are kept as-is;
    # we work on a separate transformed copy for z-score computation.
    values_for_zscore = df[MEASURES].copy()
    if use_log1p:
        # log1p is applied element-wise; values are already in [0,1] so the
        # transformation is mild but still compresses the right tail.
        values_for_zscore = values_for_zscore.apply(np.log1p)

    # ── Step 3: z-scores ──────────────────────────────────────────────────────
    # scipy.stats.zscore uses ddof=0 (population std) by default, which is
    # correct here — we are characterising the full node population, not a sample.
    for m in MEASURES:
        col = values_for_zscore[m].values
        std = col.std()
        if std == 0:
            # Degenerate case: all nodes have identical values → z-score = 0
            df[f"{m}_z"] = 0.0
        else:
            df[f"{m}_z"] = stats.zscore(col, ddof=0)

    # ── Step 4: composite score ───────────────────────────────────────────────
    z_cols = [f"{m}_z" for m in MEASURES]
    df["composite_score"] = df[z_cols].mean(axis=1)

    # ── Step 5: rank ──────────────────────────────────────────────────────────
    # ascending=False → highest composite gets rank 1
    df["rank"] = df["composite_score"].rank(ascending=False,
                                            method="average").astype(int)

    # ── Biological annotation ─────────────────────────────────────────────────
    df["bio_category"] = df["node_id"].map(bio_category)

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_figure(g: ig.Graph,
                  coords: np.ndarray,
                  df: pd.DataFrame,
                  community_idx: int,
                  graph_label: str,
                  top_k: int,
                  use_log1p: bool,
                  out_path: Path,
                  dpi: int = 200) -> None:
    """Render the composite centrality figure.

    Node size and label both reflect the continuous composite z-score:
      - Size is linearly scaled from the minimum to maximum composite score
        so every gradation of centrality is visually distinct.
      - Label shows the integer rank (1 = most central) for the top-K nodes
        and for all bio-annotated nodes, regardless of rank.
      - Border is thick black for nodes in the top 10% by rank.
    """
    names  = g.vs["name"]
    is_hub = g.vs["is_hub"]
    N      = g.vcount()

    # Build lookup tables indexed by node name for fast access
    name_to_row   = df.set_index("node_id")
    composite_arr = np.array([name_to_row.loc[n, "composite_score"]
                               if n in name_to_row.index else 0.0
                               for n in names])
    rank_arr      = [int(name_to_row.loc[n, "rank"])
                     if n in name_to_row.index else N
                     for n in names]
    bio_cats      = [name_to_row.loc[n, "bio_category"]
                     if n in name_to_row.index else "none"
                     for n in names]

    # ── Node sizes: shift composite to [0,1] then scale to [15, 245] ──────────
    # Shift so the lowest-scoring node is still visible (size ≥ 15)
    c_min, c_max = composite_arr.min(), composite_arr.max()
    c_range      = c_max - c_min if c_max > c_min else 1.0
    size_norm    = (composite_arr - c_min) / c_range   # [0, 1]
    sizes        = (15 + 230 * size_norm).tolist()

    # ── Node borders: thick black for top 10% by rank ─────────────────────────
    top10pct_cutoff = max(1, int(N * 0.10))
    def border(i):
        return ("#111111", 2.5) if rank_arr[i] <= top10pct_cutoff else ("white", 0.5)

    # ── Which nodes get a label? ───────────────────────────────────────────────
    # top-K by rank, PLUS any bio-annotated node regardless of rank
    top_k_names = set(df.head(top_k)["node_id"].tolist())
    bio_names   = {n for n, c in zip(names, bio_cats) if c != "none"}
    label_names = top_k_names | bio_names

    # ── Edge colour: black if either endpoint is in a resistance gene list ─────
    in_bio = set(bio_names)
    bio_edges, plain_edges = [], []
    for e in g.es:
        u, v = e.source, e.target
        if names[u] in in_bio or names[v] in in_bio:
            bio_edges.append((u, v))
        else:
            plain_edges.append((u, v))

    fig, ax = plt.subplots(figsize=(26, 22))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Draw edges ────────────────────────────────────────────────────────────
    for u, v in plain_edges:
        ax.plot([coords[u, 0], coords[v, 0]],
                [coords[u, 1], coords[v, 1]],
                color=EDGE_COLOR_NONE, lw=0.4, alpha=0.45, zorder=1)
    for u, v in bio_edges:
        ax.plot([coords[u, 0], coords[v, 0]],
                [coords[u, 1], coords[v, 1]],
                color=EDGE_COLOR_BIO, lw=0.65, alpha=0.55, zorder=2)

    # ── Draw nodes ────────────────────────────────────────────────────────────
    # Render lowest composite first so high-ranking nodes sit on top
    render_order = sorted(range(N), key=lambda i: composite_arr[i])
    circle_idx   = [i for i in render_order if not is_hub[i]]
    hub_idx      = [i for i in render_order if     is_hub[i]]

    def scatter_group(idx_list, marker, zorder):
        if not idx_list:
            return
        bc = [border(i) for i in idx_list]
        ax.scatter(
            coords[idx_list, 0], coords[idx_list, 1],
            s          = [sizes[i]  for i in idx_list],
            c          = [BIO_COLOR[bio_cats[i]] for i in idx_list],
            edgecolors = [b[0]      for b in bc],
            linewidths = [b[1]      for b in bc],
            alpha=NODE_ALPHA, zorder=zorder, marker=marker,
        )

    scatter_group(circle_idx, "o", zorder=5)
    scatter_group(hub_idx,    "*", zorder=6)

    # ── Labels: rank number for top-K + all bio-annotated nodes ───────────────
    for i in range(N):
        name = names[i]
        if name not in label_names:
            continue
        x, y = coords[i]
        rank = rank_arr[i]
        ax.text(x, y, str(rank),
                fontsize=6.0 if rank > top10pct_cutoff else 7.5,
                fontweight="bold" if rank <= top10pct_cutoff else "normal",
                color="black",
                ha="center", va="center", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.5,
                                            foreground="white")])

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = []

    handles.append(mpatches.Patch(fc="none", ec="none",
                                  label="─ Biological category ─"))
    for cat in ["kpc_ndm", "klebsiella", "all_beta", "none"]:
        handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=BIO_COLOR[cat],
                   markeredgecolor="#555555", markeredgewidth=0.5,
                   markersize=10, label=BIO_LABEL[cat])
        )

    handles.append(mpatches.Patch(fc="none", ec="none", label="─ Edges ─"))
    handles.append(Line2D([0], [0], color=EDGE_COLOR_BIO,  lw=2.0,
                          label="Resistance genes present"))
    handles.append(Line2D([0], [0], color=EDGE_COLOR_NONE, lw=2.0,
                          label="No resistance genes"))

    handles.append(mpatches.Patch(fc="none", ec="none", label="─ Score & labels ─"))
    handles.append(mpatches.Patch(fc="#dddddd", ec="none",
                                  label="Node size ∝ composite z-score"))
    handles.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#aaaaaa", markeredgecolor="#111111",
               markeredgewidth=1.8, markersize=10,
               label=f"Thick border = top 10% (rank ≤ {top10pct_cutoff})"))
    handles.append(mpatches.Patch(fc="none", ec="none",
                                  label=f"Label = rank  (top-{top_k} + all bio nodes)"))

    handles.append(mpatches.Patch(fc="none", ec="none", label="─ Markers ─"))
    handles.append(
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="#888888", markeredgecolor="black",
               markeredgewidth=0.8, markersize=13,
               label="Hub (connector) node  ★"))

    leg = ax.legend(handles=handles, loc="lower left", frameon=True,
                    framealpha=0.92, fontsize=7.5, labelspacing=0.42,
                    borderpad=0.9, handletextpad=0.6)
    leg.get_frame().set_edgecolor("#aaaaaa")
    for txt, hdl in zip(leg.get_texts(), leg.legend_handles):
        if isinstance(hdl, mpatches.Patch) and hdl.get_facecolor()[3] == 0:
            txt.set_color("#555555")
            txt.set_style("italic")
            txt.set_fontsize(7.0)

    log1p_note = "  |  log1p applied before z-scoring" if use_log1p else ""
    ax.set_title(
        f"Community {community_idx}  –  Composite Centrality Ranking"
        f"  |  {graph_label}{log1p_note}\n"
        f"Composite = mean z-score of degree, betweenness, closeness, eigenvector  "
        f"|  label = rank  |  size ∝ composite score",
        fontsize=12, pad=12, fontweight="bold"
    )

    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved → {out_path.name}")


# ── Text summary ──────────────────────────────────────────────────────────────

def write_summary(out_path: Path,
                  df_full: pd.DataFrame,
                  df_nh: pd.DataFrame,
                  community_idx: int,
                  use_log1p: bool) -> None:
    """Write the full ranked table for both graph variants."""

    def section(df: pd.DataFrame, label: str) -> list[str]:
        lines = []
        lines.append(f"\n{'━'*72}")
        lines.append(f"  {label}")
        lines.append(f"  {len(df)} nodes")
        lines.append(f"{'━'*72}")

        # Header
        lines.append(
            f"  {'Rank':>4}  {'Node':<30}  "
            f"{'Deg':>6}  {'Bwn':>6}  {'Cls':>6}  {'Eig':>6}  "
            f"{'Deg_z':>6}  {'Bwn_z':>6}  {'Cls_z':>6}  {'Eig_z':>6}  "
            f"{'Comp':>7}  {'Bio':<12}"
        )
        lines.append("  " + "─" * 115)

        for _, row in df.iterrows():
            lines.append(
                f"  {int(row['rank']):>4}  {row['node_id']:<30}  "
                f"{row['degree']:>6.3f}  {row['betweenness']:>6.3f}  "
                f"{row['closeness']:>6.3f}  {row['eigenvector']:>6.3f}  "
                f"{row['degree_z']:>6.3f}  {row['betweenness_z']:>6.3f}  "
                f"{row['closeness_z']:>6.3f}  {row['eigenvector_z']:>6.3f}  "
                f"{row['composite_score']:>7.4f}  {row['bio_category']:<12}"
            )
        return lines

    with open(out_path, "w") as fh:
        fh.write("=" * 72 + "\n")
        fh.write(f"  COMPOSITE CENTRALITY RANKING  —  Community {community_idx}\n")
        fh.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write("=" * 72 + "\n\n")
        fh.write("Method: z-score composite of 4 centrality measures\n")
        fh.write("  Measures : degree (norm) · betweenness (norm) · "
                 "closeness (WF) · eigenvector\n")
        fh.write(f"  log1p    : {'YES — applied before z-scoring' if use_log1p else 'NO'}\n")
        fh.write("  z-score  : (x - mean) / std  per measure across all nodes\n")
        fh.write("  Composite: mean of the four z-scores\n")
        fh.write("  Rank 1   : highest composite score\n\n")
        fh.write("When to use --use-log1p:\n")
        fh.write("  Betweenness and degree are often right-skewed in plasmid\n")
        fh.write("  networks (hub nodes dominate).  log1p(x) compresses the\n")
        fh.write("  tail and gives non-hub nodes more discriminating z-scores.\n")
        fh.write("  Use it when skewness > 2 or when the ranking looks dominated\n")
        fh.write("  by hub nodes across all four measures.\n")

        for lines in [
            section(df_full, "FULL GRAPH (with hub nodes)"),
            section(df_nh,   "NO-HUB GRAPH"),
        ]:
            fh.write("\n".join(lines) + "\n")

        fh.write("\n" + "=" * 72 + "\n")

    print(f"  Summary saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    base     = Path(args.out_dir)
    hub_dir  = base / "hub_graphs"
    nh_dir   = base / "no_hub_graphs"
    hub_dir.mkdir(parents=True, exist_ok=True)
    nh_dir.mkdir(parents=True, exist_ok=True)

    # ── Load both graphs ──────────────────────────────────────────────────────
    print("Loading graphs …")
    with open(args.json) as fh:
        raw = json.load(fh)
    typing = pd.read_csv(args.typing, sep="\t")
    p2t    = dict(zip(typing["plasmid"], typing["type"]))
    c2sc: dict = {}
    for nd in raw["elements"]["nodes"]:
        nid = nd["data"]["id"]
        if nid in p2t:
            c2sc[nd["data"].get("color", "")] = p2t[nid]

    g_full    = build_pling_graph(raw["elements"]["nodes"],
                                  raw["elements"]["edges"], p2t, c2sc,
                                  exclude_hubs=False)
    g_no_hubs = build_pling_graph(raw["elements"]["nodes"],
                                  raw["elements"]["edges"], p2t, c2sc,
                                  exclude_hubs=True)
    print(f"  Full graph:   {g_full.vcount()} nodes, {g_full.ecount()} edges")
    print(f"  No-hub graph: {g_no_hubs.vcount()} nodes, {g_no_hubs.ecount()} edges")

    # ── Layouts ───────────────────────────────────────────────────────────────
    print("Computing layouts …")
    coords_full = hybrid_layout(g_full,    seed=42)
    coords_nh   = hybrid_layout(g_no_hubs, seed=42)

    # ── Centrality + composite score ──────────────────────────────────────────
    print("Computing centrality measures and composite z-scores …")
    cent_full = compute_centralities(g_full)
    cent_nh   = compute_centralities(g_no_hubs)

    df_full = compute_composite_score(g_full,    cent_full, args.use_log1p)
    df_nh   = compute_composite_score(g_no_hubs, cent_nh,   args.use_log1p)

    print(f"  log1p transformation: {'ON' if args.use_log1p else 'OFF'}")
    for label, df in [("Full", df_full), ("No-hub", df_nh)]:
        top3 = df.head(3)["node_id"].tolist()
        print(f"  {label} top-3: {top3}")

    # ── Render one figure per graph variant ───────────────────────────────────
    print("\nRendering …")
    for g, coords, df, label, out_subdir in [
        (g_full,    coords_full, df_full, "Full graph (with hubs)", hub_dir),
        (g_no_hubs, coords_nh,   df_nh,   "No-hub graph",           nh_dir),
    ]:
        render_figure(
            g, coords, df,
            community_idx = args.community,
            graph_label   = label,
            top_k         = args.top_k,
            use_log1p     = args.use_log1p,
            out_path      = out_subdir / "composite_ranking.png",
            dpi           = args.dpi,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(base / "summary.txt", df_full, df_nh,
                  args.community, args.use_log1p)

    print(f"\nAll outputs in: {base}/")
    print(f"  hub_graphs/composite_ranking.png")
    print(f"  no_hub_graphs/composite_ranking.png")
    print(f"  summary.txt")


if __name__ == "__main__":
    main()