"""
pling_consensus_centrality.py
==============================
Consensus centrality score: binary top-N flags across five graph measures,
merged with biological resistance annotations and visualised on the community
network layout.

Four threshold levels are analysed: top 5%, 10%, 15%, 20%.
Both the full graph (with hub nodes) and the no-hub graph are processed.

Output structure
----------------
    <out-dir>/consensus_results/
        hub_graphs/
            consensus_top05pct.png
            consensus_top10pct.png
            consensus_top15pct.png
            consensus_top20pct.png
        no_hub_graphs/
            consensus_top05pct.png
            consensus_top10pct.png
            consensus_top15pct.png
            consensus_top20pct.png
        summary.txt

Visual encoding
---------------
  Node fill    → biological category (most-specific wins):
                   red    = KPC + NDM double resistance
                   yellow = Klebsiella beta-lactamase
                   green  = Any beta-lactamase (mobilisable)
                   gray   = No resistance annotation
  Node size    → consensus score (0 = tiny … 5 = large)
  Node border  → thick black for score ≥ 4, thin white otherwise
  Node label   → consensus score digit; hidden for score = 0
  Edge colour  → dark purple if any endpoint carries resistance genes,
                 light gray otherwise

Usage
-----
    python3 pling_consensus_centrality.py \\
        --json    community_0.json \\
        --typing  typing.tsv \\
        --out-dir output/ \\
        --community 0 --dpi 200

Dependencies
------------
    pip install igraph matplotlib pandas numpy
    pling_igraph_viz.py must be importable from the same directory.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pling_igraph_viz import build_pling_graph, hybrid_layout
from pling_igraph_analysis import (
    degree_centrality,
    betweenness_centrality,      # returns normalised values; ordering preserved
    closeness_centrality,        # Wasserman-Faust normalisation
    eigenvector_centrality,      # warns on disconnected graph — suppressed inside
    local_clustering_coefficient,# NaN (deg < 2) → 0
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

MEASURES = ["degree", "betweenness", "closeness", "eigenvector", "clustering"]

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

EDGE_COLOR_BIO  = "#000000"   # black — at least one endpoint in gene list
EDGE_COLOR_NONE = "#CCCCCC"   # light gray  — no gene-list endpoint
NODE_ALPHA      = 0.92        # uniform alpha for all nodes

SIZE_MAP = {0: 22, 1: 58, 2: 100, 3: 148, 4: 200, 5: 260}

def border_style(score: int) -> tuple[str, float]:
    return ("white", 0.5)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--json",      required=True, help="Path to community JSON file")
    p.add_argument("--typing",    required=True, help="Path to typing TSV file")
    p.add_argument("--out-dir",   required=True, help="Path to output directory")
    p.add_argument("--community", type=int, default=0)
    p.add_argument("--dpi",       type=int, default=200)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def bio_category(name: str) -> str:
    """Most-specific biological category for a node."""
    if name in _SET_KPC:   return "kpc_ndm"
    if name in _SET_KLEBS: return "klebsiella"
    if name in _SET_ALL:   return "all_beta"
    return "none"


def compute_centralities(g: ig.Graph) -> dict[str, list]:
    """Compute all five centrality measures by delegating to the shared
    functions in pling_igraph_analysis.py.

    Using the same implementations as the main analysis script guarantees
    that the values, normalisation choices, and edge-case handling (NaN,
    disconnected graphs, eigenvector warnings) are identical across scripts.

    Measures and their source functions
    ------------------------------------
    degree      : degree_centrality(g)              → d(v)/(N-1), range [0,1]
    betweenness : betweenness_centrality(g)          → normalised to [0,1]
    closeness   : closeness_centrality(g)            → Wasserman-Faust, [0,1]
    eigenvector : eigenvector_centrality(g)          → igraph EV, [0,1]
    clustering  : local_clustering_coefficient(g)    → local transitivity, [0,1]

    Note: betweenness is normalised here (unlike the raw values used in the
    distribution plots).  For computing top-N rankings the ordering is
    identical to raw betweenness, so consensus scores are unaffected.
    """
    return {
        "degree":      degree_centrality(g, normalized=False),
        "betweenness": betweenness_centrality(g, normalized=False),
        "closeness":   closeness_centrality(g, normalized=True),
        "eigenvector": eigenvector_centrality(g, normalized=True),
        "clustering":  local_clustering_coefficient(g),
    }


def top_n_flags(values: list, k: int) -> list[int]:
    """Binary flag: 1 if node ranks in top-k, else 0.

    Ties at the boundary are resolved by including all nodes sharing the
    k-th value, so the number of flagged nodes may slightly exceed k.
    """
    sorted_unique = sorted(set(values), reverse=True)
    covered, threshold = 0, sorted_unique[0]
    for v in sorted_unique:
        covered += values.count(v)
        threshold = v
        if covered >= k:
            break
    return [1 if val >= threshold else 0 for val in values]


def compute_consensus(g: ig.Graph, centralities: dict, pct: int):
    """Return (consensus_list, binary_flags_dict) for one threshold."""
    k     = math.ceil(g.vcount() * pct / 100)
    flags = {m: top_n_flags(centralities[m], k) for m in MEASURES}
    score = [sum(flags[m][i] for m in MEASURES) for i in range(g.vcount())]
    return score, flags


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_figure(g: ig.Graph,
                  coords: np.ndarray,
                  consensus: list[int],
                  pct: int,
                  community_idx: int,
                  graph_label: str,
                  out_path: Path,
                  dpi: int = 200) -> None:
    """Render one consensus centrality figure."""

    names  = g.vs["name"]
    is_hub = g.vs["is_hub"]
    N      = g.vcount()
    k      = math.ceil(N * pct / 100)

    bio_cats  = [bio_category(names[i]) for i in range(N)]
    fills     = [BIO_COLOR[c]                 for c in bio_cats]
    sizes     = [SIZE_MAP[consensus[i]]        for i in range(N)]
    borders   = [border_style(consensus[i])    for i in range(N)]
    edgecols  = [b[0] for b in borders]
    linewidths= [b[1] for b in borders]

    # Pre-compute which nodes are in any gene list (for edge colouring)
    in_bio = {names[i] for i in range(N) if bio_cats[i] != "none"}

    n_score_ge3 = sum(1 for s in consensus if s >= 3)
    n_overlap   = sum(1 for i in range(N)
                      if consensus[i] > 0 and bio_cats[i] != "none")

    fig, ax = plt.subplots(figsize=(26, 22))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Edges ─────────────────────────────────────────────────────────────────
    # Draw non-bio edges first (background), then bio edges on top.
    bio_edges, plain_edges = [], []
    for e in g.es:
        u, v = e.source, e.target
        if names[u] in in_bio or names[v] in in_bio:
            bio_edges.append((u, v))
        else:
            plain_edges.append((u, v))

    for u, v in plain_edges:
        ax.plot([coords[u, 0], coords[v, 0]],
                [coords[u, 1], coords[v, 1]],
                color=EDGE_COLOR_NONE, lw=0.4, alpha=0.45, zorder=1)

    for u, v in bio_edges:
        ax.plot([coords[u, 0], coords[v, 0]],
                [coords[u, 1], coords[v, 1]],
                color=EDGE_COLOR_BIO, lw=0.65, alpha=0.55, zorder=2)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # Render in ascending consensus-score order so high-scoring nodes sit on top.
    render_order = sorted(range(N), key=lambda i: consensus[i])
    circle_idx   = [i for i in render_order if not is_hub[i]]
    hub_idx      = [i for i in render_order if     is_hub[i]]

    from itertools import groupby as _groupby

    def scatter_group(idx_list, marker):
        """Scatter a list of node indices with their individual attributes."""
        if not idx_list:
            return
        ax.scatter(coords[idx_list, 0], coords[idx_list, 1],
                   s=[sizes[i]      for i in idx_list],
                   c=[fills[i]      for i in idx_list],
                   edgecolors=[edgecols[i]   for i in idx_list],
                   linewidths=[linewidths[i] for i in idx_list],
                   alpha=NODE_ALPHA, zorder=5 if marker == "o" else 6,
                   marker=marker)

    scatter_group(circle_idx, "o")
    scatter_group([i for i in hub_idx], "*")

    # ── Labels (score digit — never shown for score=0) ─────────────────────────
    for i in range(N):
        score = consensus[i]
        if score == 0:
            continue

        x, y   = coords[i]
        txt_col = "black"

        ax.text(x, y, str(score),
                fontsize=6.0 if score < 4 else 7.5,
                fontweight="bold" if score >= 3 else "normal",
                color=txt_col,
                ha="center", va="center", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = []

    # Biological categories
    handles.append(mpatches.Patch(fc="none", ec="none",
                                  label="─ Biological category ─"))
    for cat in ["kpc_ndm", "klebsiella", "all_beta", "none"]:
        handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=BIO_COLOR[cat],
                   markeredgecolor="#555555", markeredgewidth=0.5,
                   markersize=10, label=BIO_LABEL[cat])
        )

    # Edge colour
    handles.append(mpatches.Patch(fc="none", ec="none",
                                  label="─ Edges ─"))
    handles.append(
        Line2D([0], [0], color=EDGE_COLOR_BIO,  lw=2.0,
               label="Resistance genes present"))
    handles.append(
        Line2D([0], [0], color=EDGE_COLOR_NONE, lw=2.0,
               label="No resistance genes"))

    # Hub marker
    handles.append(mpatches.Patch(fc="none", ec="none",
                                  label="─ Markers ─"))
    handles.append(
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="#888888", markeredgecolor="black",
               markeredgewidth=0.8, markersize=13,
               label="Hub (connector) node  ★"))
    handles.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#888888", markeredgecolor="#111111",
               markeredgewidth=1.6, markersize=9,
               label="Score ≥ 4  →  thick border"))
    handles.append(mpatches.Patch(fc="none", ec="none",
                                  label="Node label = consensus score (0 unlabelled)"))

    leg = ax.legend(handles=handles, loc="lower left", frameon=True,
                    framealpha=0.92, fontsize=7.5, labelspacing=0.42,
                    borderpad=0.9, handletextpad=0.6)
    leg.get_frame().set_edgecolor("#aaaaaa")
    for txt, hdl in zip(leg.get_texts(), leg.legend_handles):
        if isinstance(hdl, mpatches.Patch) and hdl.get_facecolor()[3] == 0:
            txt.set_color("#555555")
            txt.set_style("italic")
            txt.set_fontsize(7.0)

    score_dist = {s: consensus.count(s) for s in range(6)}
    ax.set_title(
        f"Community {community_idx}  –  Consensus Centrality"
        f"  (top {pct}%,  k={k} per measure)  |  {graph_label}\n"
        f"Nodes with score ≥ 3: {n_score_ge3}   "
        f"Bio-annotated & score > 0: {n_overlap}   "
        f"Score distribution: "
        + "  ".join(f"{s}→{n}" for s, n in score_dist.items()),
        fontsize=12, pad=12, fontweight="bold"
    )

    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved → {out_path.name}")


# ── Text summary ──────────────────────────────────────────────────────────────

def write_summary(out_path: Path,
                  g_full: ig.Graph,
                  g_no_hubs: ig.Graph,
                  cent_full: dict,
                  cent_nh: dict,
                  all_cons_full: dict,
                  all_cons_nh: dict,
                  all_flags_full: dict,
                  all_flags_nh: dict,
                  community_idx: int) -> None:
    """Write a human-readable plain-text summary."""

    def section(g, centralities, all_consensus, all_flags, graph_label):
        lines = []
        names  = g.vs["name"]
        is_hub = g.vs["is_hub"]
        N      = g.vcount()

        lines.append(f"\n{'━'*70}")
        lines.append(f"  {graph_label}")
        lines.append(f"  {g.vcount()} nodes  ·  {g.ecount()} edges")
        lines.append(f"{'━'*70}")

        for pct in [5, 10, 15, 20]:
            k         = math.ceil(N * pct / 100)
            consensus = all_consensus[pct]
            flags     = all_flags[pct]

            dist = {s: consensus.count(s) for s in range(6)}
            lines.append(f"\n  ── Top {pct}%  (k = {k} nodes per measure) ──")
            lines.append("  Score distribution: "
                         + "  ".join(f"{s} --> {dist[s]}" for s in range(6)))

            # Collect nodes with score > 0, sorted descending
            scored = [(names[i], consensus[i], centralities, flags, is_hub[i])
                      for i in range(N) if consensus[i] > 0]
            scored.sort(key=lambda t: t[1], reverse=True)

            # Highlight overlap nodes (bio + score > 0)
            overlaps = [(name, score, bio_category(name))
                        for name, score, *_ in scored
                        if bio_category(name) != "none"]
            if overlaps:
                lines.append(f"  Bio-annotated nodes with score > 0 --> {len(overlaps)}")

            if not scored:
                lines.append("  (no nodes above threshold)")
                continue

            # Header
            lines.append("")
            lines.append(f"  {'Node':<30} {'Score':>5}  "
                         f"{'Deg':>6} {'Bwn(n)':>8} {'Cls':>6} "
                         f"{'Eig':>6} {'CC':>6}  "
                         f"{'Hub':>3}  {'Bio category'}")
            lines.append("  " + "─"*100)

            for name, score, cent, fl, hub in scored:
                i     = names.index(name)
                bio   = bio_category(name)
                hub_m = "★" if hub else ""
                # Which measures is the node top-k in?
                top_m = [m[:3].upper() for m in MEASURES if fl[m][i] == 1]
                lines.append(
                    f"  {name:<30} {score:>5}  "
                    f"{cent['degree'][i]:>6.3f} "
                    f"{cent['betweenness'][i]:>8.5f} "
                    f"{cent['closeness'][i]:>6.3f} "
                    f"{cent['eigenvector'][i]:>6.3f} "
                    f"{cent['clustering'][i]:>6.3f}  "
                    f"{hub_m:>3}  {bio:<12}  "
                    f"[top-{pct}%: {','.join(top_m)}]"
                )

        return lines

    with open(out_path, "w") as fh:
        fh.write("=" * 70 + "\n")
        fh.write(f"  CONSENSUS CENTRALITY SUMMARY  —  Community {community_idx}\n")
        fh.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write("=" * 70 + "\n")
        fh.write("\nMeasures: degree (raw) · betweenness (raw) · closeness (WF)\n")
        fh.write("          eigenvector · local clustering coefficient\n")
        fh.write("Consensus score = sum of binary top-N flags across all 5 measures\n")
        fh.write("Score range: 0 (not top-N in any) … 5 (top-N in all five)\n")
        fh.write("\nBiological categories (most-specific wins for colour):\n")
        fh.write("  RED    kpc_ndm_double_resistance       (1 node)\n")
        fh.write("  YELLOW klebsiella_betalactamase_pos.  (25 nodes)\n")
        fh.write("  GREEN  all_betalactamase_positive      (67 nodes)\n")

        for lines in [
            section(g_full,    cent_full, all_cons_full, all_flags_full,
                    "FULL GRAPH (with hub nodes)"),
            section(g_no_hubs, cent_nh,   all_cons_nh,   all_flags_nh,
                    "NO-HUB GRAPH"),
        ]:
            fh.write("\n".join(lines) + "\n")

        fh.write("\n" + "=" * 70 + "\n")

    print(f"  Summary saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    base = Path(args.out_dir)
    hub_dir   = base / "hub_graphs"
    nohub_dir = base / "no_hub_graphs"
    hub_dir.mkdir(parents=True, exist_ok=True)
    nohub_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  Full graph:    {g_full.vcount()} nodes, {g_full.ecount()} edges")
    print(f"  No-hub graph:  {g_no_hubs.vcount()} nodes, {g_no_hubs.ecount()} edges")

    # ── Layouts ───────────────────────────────────────────────────────────────
    print("Computing layouts …")
    coords_full    = hybrid_layout(g_full,    seed=42)
    coords_no_hubs = hybrid_layout(g_no_hubs, seed=42)

    # ── Centrality measures ───────────────────────────────────────────────────
    print("Computing centrality measures …")
    cent_full = compute_centralities(g_full)
    cent_nh   = compute_centralities(g_no_hubs)

    # ── Consensus for every threshold ─────────────────────────────────────────
    all_cons_full: dict  = {}
    all_flags_full: dict = {}
    all_cons_nh: dict    = {}
    all_flags_nh: dict   = {}

    print("Consensus scores:")
    for pct in [5, 10, 15, 20]:
        cons_f, flags_f = compute_consensus(g_full,    cent_full, pct)
        cons_n, flags_n = compute_consensus(g_no_hubs, cent_nh,   pct)
        all_cons_full[pct]  = cons_f
        all_flags_full[pct] = flags_f
        all_cons_nh[pct]    = cons_n
        all_flags_nh[pct]   = flags_n
        k_f = math.ceil(g_full.vcount()    * pct / 100)
        k_n = math.ceil(g_no_hubs.vcount() * pct / 100)
        dist_f = {s: cons_f.count(s) for s in range(6) if cons_f.count(s)}
        dist_n = {s: cons_n.count(s) for s in range(6) if cons_n.count(s)}
        print(f"  {pct:2d}%  full(k={k_f}): {dist_f}  |  no-hub(k={k_n}): {dist_n}")

    # ── Render figures ────────────────────────────────────────────────────────
    graphs = [
        (g_full,    coords_full,    all_cons_full,  "Full graph (with hubs)",  hub_dir),
        (g_no_hubs, coords_no_hubs, all_cons_nh,    "No-hub graph",            nohub_dir),
    ]
    print("\nRendering …")
    for g, coords, all_cons, label, out_subdir in graphs:
        print(f"  {label}:")
        for pct in [5, 10, 15, 20]:
            render_figure(
                g, coords, all_cons[pct],
                pct=pct, community_idx=args.community,
                graph_label=label,
                out_path=out_subdir / f"consensus_top{pct:02d}pct.png",
                dpi=args.dpi,
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(
        base / "summary.txt",
        g_full, g_no_hubs,
        cent_full, cent_nh,
        all_cons_full, all_cons_nh,
        all_flags_full, all_flags_nh,
        community_idx=args.community,
    )

    print(f"\nAll outputs in: {base}/")
    print(f"  hub_graphs/    → {len(list(hub_dir.glob('*.png')))} figures")
    print(f"  no_hub_graphs/ → {len(list(nohub_dir.glob('*.png')))} figures")
    print(f"  summary.txt")


if __name__ == "__main__":
    main()