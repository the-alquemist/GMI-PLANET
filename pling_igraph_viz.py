"""
pling_igraph_viz.py
===================
Recreates pling's interactive community HTML visualisations as static
high-resolution PNG figures using **igraph** for graph algorithms and
**matplotlib** for rendering.

Background
----------
pling (Plasmid Linking by Inference of Network Graphs) clusters plasmids into
communities and subcommunities based on sequence similarity metrics:
  - sd  (sequence distance)  – continuous [0, 1], lower = more similar
  - td  (topological depth)  – integer hop-count in the plasmid graph

Each community JSON file contains:
  nodes  – plasmid IDs with a colour (subcommunity) and optional is_hub flag
  edges  – pairs of plasmids with sd, td, and d_lbl ("sd / td" label)

Layout strategy — three phases
-------------------------------
A single global Fruchterman-Reingold (FR) run on 150+ nodes produces
elongated, poorly-separated clusters because cross-community edges fight
against tight intra-community structure.  We use a three-phase pipeline:

  Phase 1 – GLOBAL FR
      Run FR on the full graph with heavily up-weighted intra-community edges.
      This gives the correct macro topology (relative positions of blobs) even
      if the internal structure of each blob is poor.

  Phase 2 – LOCAL FR / KK per subcommunity
      For each multi-node subcommunity, extract its intra-cluster edges, run a
      fresh local layout (Kamada-Kawai for small clusters ≤ 15 nodes, FR for
      larger ones), apply a density-adaptive radius, then re-centre the result
      at the Phase-1 centroid (optionally pulled slightly toward its partners).
      Nodes with NO intra-cluster edges keep their Phase-1 global positions.

  Phase 3 – CENTROID REPULSION
      After local layouts are injected some cluster boundaries may overlap —
      particularly when a small isolated cluster was pulled toward a hub that
      already sits inside a large dense cluster.  We run an iterative
      centroid-repulsion pass (like the repulsion term of FR, but at cluster
      level) until all boundaries are separated by a minimum gap.  Crucially,
      every node in a displaced cluster shifts by the same vector so the
      internal clique structure is preserved perfectly.

Usage
-----
    python3 pling_igraph_viz.py \\
        --community 0 \\
        --json      community_0.json \\
        --typing    typing.tsv \\
        --out       community_0_igraph.png \\
        --dpi       200

Dependencies
------------
    pip install igraph matplotlib pandas numpy
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

# igraph provides the graph layout algorithms (FR, Kamada-Kawai, etc.)
import igraph as ig

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — needed for PNG output
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe   # white stroke behind labels for legibility
import numpy as np
import pandas as pd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    All paths default to the pling upload/output convention used in this
    environment; override them when running on different data.
    """
    p = argparse.ArgumentParser(
        description="Recreate pling community visualisations with igraph."
    )
    p.add_argument("--community", type=int, default=0,
                   help="Community index (used in legend labels)")
    p.add_argument("--json",    default="/mnt/user-data/uploads/community_0.json",
                   help="Path to the pling community JSON file")
    p.add_argument("--typing",  default="/mnt/user-data/uploads/typing.tsv",
                   help="Path to typing.tsv  (plasmid → subcommunity mapping)")
    p.add_argument("--out",     default="/mnt/user-data/outputs/community_0_igraph.png",
                   help="Output PNG file path")
    p.add_argument("--dpi",     type=int, default=200,
                   help="Output resolution in DPI")
    return p.parse_args()


# ── Utility ───────────────────────────────────────────────────────────────────

def nat_key(s: str) -> list:
    """Natural-sort key so 'Sub 10' sorts after 'Sub 9', not 'Sub 2'.

    Splits the string into alternating text / integer chunks; integer chunks
    are compared numerically rather than lexicographically.
    """
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


# ── Phase 1: global FR ────────────────────────────────────────────────────────

def build_global_graph(N: int, edge_list: list, edge_td: list,
                       subcom_arr: np.ndarray) -> ig.Graph:
    """Build the full graph with differentiated edge weights for Phase-1 FR.

    Edge-weight design
    ------------------
    FR uses edge weights as spring stiffnesses.  We exploit this to encode
    community structure directly into the physics:

        intra-community  →  weight = max(8, 20/td)   strong spring, tight cluster
        inter-community  →  weight = max(0.1, 0.5/td) weak spring, loose coupling

    td (topological depth) acts as a distance proxy: smaller td = more similar
    plasmids = should sit closer together, so the weight is *inversely*
    proportional to td.

    Returns an igraph.Graph with a "weight" edge attribute.
    """
    g = ig.Graph(directed=False)
    g.add_vertices(N)

    weights = []
    for (u, v), td in zip(edge_list, edge_td):
        same = subcom_arr[u] == subcom_arr[v]
        weights.append(
            max(8.0, 20.0 / max(td, 0.1)) if same
            else max(0.1,  0.5 / max(td, 0.1))
        )

    g.add_edges(edge_list)
    g.es["weight"] = weights
    return g


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def local_layout_coords(active: list, edge_list: list,
                        edge_td: list, n_act: int) -> np.ndarray | None:
    """Compute a unit-normalised local layout for one subcommunity.

    Algorithm choice
    ----------------
    Kamada-Kawai (KK) minimises a spring energy based on graph-theoretic
    shortest paths.  For small, sparse graphs it produces more organic,
    less elongated shapes than FR.

    FR is used for larger clusters (> 15 nodes) where KK becomes slow and
    where there are enough nodes for the force-balance to look good.

    Auto-rotation
    -------------
    Chain-like sub-graphs (two cliques connected by a long-distance edge)
    often land in a horizontal strip because FR/KK have no preferred axis.
    We detect layouts with aspect ratio > 1.8 and rotate 90° so the chain
    runs vertically — this looks far better in the full-community view.

    Returns
    -------
    np.ndarray (n_act, 2) centred at origin with max radius = 1.0,
    or None if there are no intra-cluster edges to layout.
    """
    local_map = {v: i for i, v in enumerate(active)}

    lg = ig.Graph(directed=False)
    lg.add_vertices(n_act)

    loc_edges, loc_w = [], []
    for (u, v), td in zip(edge_list, edge_td):
        if u in local_map and v in local_map:
            loc_edges.append((local_map[u], local_map[v]))
            # Stronger weight for closer plasmids → tighter FR/KK placement
            loc_w.append(max(1.0, 8.0 / max(td, 0.1)))

    lg.add_edges(loc_edges)
    if lg.ecount() == 0:
        return None

    if n_act <= 15:
        layout = lg.layout_kamada_kawai(maxiter=1000)
    else:
        layout = lg.layout_fruchterman_reingold(
            weights=loc_w, niter=1500, grid="nogrid"
        )

    lc = np.array(layout.coords)
    lc -= lc.mean(axis=0)

    # Normalise to unit radius so the caller controls the physical scale
    r = np.max(np.linalg.norm(lc, axis=1))
    if r > 0:
        lc /= r

    # Auto-rotate: if significantly wider than tall, flip to vertical
    sx = lc[:, 0].max() - lc[:, 0].min()
    sy = lc[:, 1].max() - lc[:, 1].min()
    if sx > sy * 1.8:
        lc = np.column_stack([-lc[:, 1], lc[:, 0]])   # 90° CCW

    return lc


def density_adaptive_scale(n_act: int, n_int: int,
                           ref_unit: float) -> float:
    """Density-adaptive local layout radius for one subcommunity.

    Why density-adaptive?
    ---------------------
    FR naturally compresses dense graphs (many springs pulling nodes together)
    and expands sparse ones.  Applying the same scale factor to both would make
    sparse clusters look over-blown relative to dense ones.

    The formula linearly maps density [0, 1] to a scale factor [0.38, 0.62]:

        scale_f = clamp(0.38 + 0.28 × density, max=0.62)

    Calibration against community_0 clusters:
        fuchsia   density=0.85 → scale_f=0.62  ← visually correct, no change
        orange    density=0.26 → scale_f=0.45  ← tighter than naïve 0.62
        indianred density=0.34 → scale_f=0.48
        aqua      density=0.46 → scale_f≈0.51, then capped at 0.28 (small n)

    The returned value is the physical max-node distance from cluster centre.
    """
    max_edges = n_act * (n_act - 1) / 2
    density   = n_int / max_edges if max_edges > 0 else 0.0

    scale_f = min(0.38 + 0.28 * density, 0.62)
    if n_act <= 12:
        scale_f = min(scale_f, 0.28)   # extra cap: small clusters stay compact

    return ref_unit * math.sqrt(n_act) * scale_f


# ── Phase 3: centroid repulsion ───────────────────────────────────────────────

def resolve_cluster_overlaps(final: np.ndarray,
                              sc_members: dict,
                              has_intra: set,
                              ref_unit: float,
                              min_gap_factor: float = 0.5,
                              max_iters: int = 200) -> np.ndarray:
    """Push apart overlapping cluster boundaries while preserving internal structure.

    Why is this needed?
    -------------------
    After Phase 2, the density-adaptive radius of each cluster is known.
    A small isolated cluster (e.g. aqua, radius ≈ 1.7 ref_units) pulled toward
    its hub may end up inside a large dense cluster (e.g. fuchsia, radius ≈ 6.7
    ref_units) because the hub position IS inside that larger cluster.
    Reducing the pull alone cannot fix this: even at pull = 0, the aqua global
    FR centroid may already be within fuchsia's territory.

    Algorithm
    ---------
    Mirrors the repulsion term of FR applied at cluster level:
      for every pair (i, j):
        required = radius_i + radius_j + min_gap
        if dist(centroid_i, centroid_j) < required:
          push both centroids apart along the line between them
    Repeat until all pairs satisfy the constraint or max_iters is reached.

    All nodes in a displaced cluster shift by the same rigid vector so the
    local clique structure computed in Phase 2 is perfectly preserved.

    Parameters
    ----------
    final          : coordinate array (N, 2), modified in-place and returned
    sc_members     : subcommunity → list of global node indices
    has_intra      : set of node indices with at least one intra-cluster edge
    ref_unit       : global reference length (total_span / sqrt(N))
    min_gap_factor : minimum boundary gap = min_gap_factor × ref_unit
    max_iters      : safety cap on repulsion iterations
    """
    min_gap = min_gap_factor * ref_unit

    # Collect only subcommunities with enough active nodes to have a real radius
    active_scs: list   = []
    centroids:  dict   = {}
    radii:      dict   = {}

    for sc, mem in sc_members.items():
        active = [i for i in mem if i in has_intra]
        if len(active) < 3:
            continue
        c = final[active].mean(axis=0)
        r = float(np.max(np.linalg.norm(final[active] - c, axis=1)))
        if r < 1e-9:
            continue
        active_scs.append(sc)
        centroids[sc] = c.copy()
        radii[sc]     = r

    for _ in range(max_iters):
        any_overlap = False
        for i, sci in enumerate(active_scs):
            for scj in active_scs[i + 1:]:
                ci, ri = centroids[sci], radii[sci]
                cj, rj = centroids[scj], radii[scj]
                dist     = float(np.linalg.norm(ci - cj))
                required = ri + rj + min_gap
                if dist >= required or dist < 1e-9:
                    continue

                # Overlap: push clusters apart with SIZE-WEIGHTED forces.
                # Larger clusters absorb more displacement (analogous to higher
                # inertial mass), so big clusters move more while small isolated
                # clusters stay near their natural positions.  This prevents the
                # cascade where fixing one large-large overlap shoves a big cluster
                # into a small nearby cluster, which then needs its own repulsion.
                any_overlap = True
                direction = (ci - cj) / dist
                push      = required - dist          # full separation needed
                ni        = len(sc_members.get(sci, [0]))
                nj        = len(sc_members.get(scj, [0]))
                total_n   = max(ni + nj, 1)
                centroids[sci] += direction * push * (ni / total_n)
                centroids[scj] -= direction * push * (nj / total_n)

        if not any_overlap:
            break   # all boundaries satisfied — done

    # Apply centroid shifts: every node in the cluster moves by the same delta
    for sc in active_scs:
        mem    = sc_members[sc]
        active = [i for i in mem if i in has_intra]
        if not active:
            continue
        old_c = final[active].mean(axis=0)
        shift = centroids[sc] - old_c
        # Shift ALL members (active + passive) so isolated nodes follow their cluster
        for gi in mem:
            final[gi] += shift

    return final


# ── Master layout ─────────────────────────────────────────────────────────────

def hybrid_layout(node_ids: list, node_subcom: dict,
                  edge_list: list, edge_td: list,
                  seed: int = 42) -> np.ndarray:
    """Three-phase hierarchical layout for a pling community graph.

    Parameters
    ----------
    node_ids    : ordered list of plasmid ID strings
    node_subcom : plasmid ID → subcommunity label
    edge_list   : list of (u, v) integer index pairs (0-indexed)
    edge_td     : td integer per edge
    seed        : RNG seed for reproducibility

    Returns
    -------
    np.ndarray (N, 2) — final node coordinates for matplotlib.
    Y is flipped (positive = down) to match the pling HTML orientation.
    """
    np.random.seed(seed)
    random.seed(seed)

    N          = len(node_ids)
    subcom_arr = np.array([node_subcom[n] for n in node_ids])

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    g_global   = build_global_graph(N, edge_list, edge_td, subcom_arr)
    glo_layout = g_global.layout_fruchterman_reingold(
        weights  = g_global.es["weight"],
        niter    = 3000,      # more iterations → stabler convergence
        grid     = "nogrid",  # exact repulsion (slower, better quality)
    )
    glo = np.array(glo_layout.coords)   # shape (N, 2)

    # ref_unit = "average inter-node spacing" in the global layout.
    # Anchors all local radii to a consistent physical scale.
    total_span = max(glo[:, 0].max() - glo[:, 0].min(),
                     glo[:, 1].max() - glo[:, 1].min())
    ref_unit   = total_span / math.sqrt(N)

    graph_centroid = glo.mean(axis=0)

    # ── Per-subcommunity statistics ───────────────────────────────────────────
    sc_members   : dict = defaultdict(list)
    sc_int_count : dict = defaultdict(int)
    sc_ext_count : dict = defaultdict(int)

    for i, sc in enumerate(subcom_arr):
        sc_members[sc].append(i)

    for (u, v) in edge_list:
        su, sv = subcom_arr[u], subcom_arr[v]
        if su == sv:
            sc_int_count[su] += 1
        else:
            sc_ext_count[su] += 1
            sc_ext_count[sv] += 1

    # Mean position of each cluster's external partners in the global layout.
    # Pulling toward partners is more topologically meaningful than pulling
    # toward the overall graph centroid — isolated clusters land near their
    # actual connection point in the graph.
    sc_partner_pos: dict = {}
    for sc in sc_members:
        partners = []
        for (u, v) in edge_list:
            su, sv = subcom_arr[u], subcom_arr[v]
            if su == sc and sv != sc:
                partners.append(glo[v])
            elif sv == sc and su != sc:
                partners.append(glo[u])
        sc_partner_pos[sc] = (np.mean(partners, axis=0)
                              if partners else graph_centroid)

    # Nodes WITH intra-cluster edges participate in local FR/KK.
    # Nodes with only external edges (e.g. a plasmid in a cluster that only
    # connects to hub nodes) keep their Phase-1 position unchanged.
    has_intra: set = set()
    for (u, v) in edge_list:
        if subcom_arr[u] == subcom_arr[v]:
            has_intra.add(u)
            has_intra.add(v)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    final = glo.copy()

    for sc, mem in sc_members.items():
        active = [i for i in mem if i in has_intra]
        n_act  = len(active)
        if n_act < 3:
            continue   # singletons / pairs: global position is fine

        raw_centroid = glo[active].mean(axis=0)
        ext          = sc_ext_count.get(sc, 0)
        target       = sc_partner_pos[sc]

        # Gravity pull: isolated clusters (low ext) pulled toward their partner.
        # Pull is deliberately modest (0.25) because Phase 3 will handle any
        # remaining overlap — a large pull here is the root cause of the
        # aqua-into-fuchsia collision seen in earlier versions.
        isolation = 1.0 / (1.0 + ext * 0.10)
        centroid  = raw_centroid + 0.25 * isolation * (target - raw_centroid)

        lc = local_layout_coords(active, edge_list, edge_td, n_act)
        if lc is None:
            continue

        local_radius = density_adaptive_scale(
            n_act, sc_int_count.get(sc, 0), ref_unit
        )
        lc *= local_radius

        for li, gi in enumerate(active):
            final[gi] = centroid + lc[li]

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    # Iteratively push apart any cluster boundaries that still overlap after
    # Phase 2.  min_gap_factor=0.5 ensures at least half a ref_unit of white
    # space between every pair of cluster boundaries.
    final = resolve_cluster_overlaps(
        final, sc_members, has_intra,
        ref_unit       = ref_unit,
        min_gap_factor = 0.3,
        max_iters      = 200,
    )

    final[:, 1] = -final[:, 1]   # flip y: screen coords (y-down) vs maths (y-up)
    return final


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load pling data, run the three-phase layout, and render to PNG."""
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(args.json) as fh:
        raw = json.load(fh)
    raw_nodes = raw["elements"]["nodes"]
    raw_edges = raw["elements"]["edges"]

    typing = pd.read_csv(args.typing, sep="\t")
    p2t    = dict(zip(typing["plasmid"], typing["type"]))

    # ── Node attributes ───────────────────────────────────────────────────────
    node_ids    = [n["data"]["id"] for n in raw_nodes]

    # Colour is assigned by pling per subcommunity (one colour = one subcommunity)
    node_color  = {n["data"]["id"]: n["data"].get("color", "steelblue")
                   for n in raw_nodes}

    # Hub nodes bridge subcommunities; pling marks them with is_hub=True
    # and draws them as stars — we replicate that here.
    node_is_hub = {n["data"]["id"]: bool(n["data"].get("is_hub", False))
                   for n in raw_nodes}

    # Fallback colour→subcommunity map for nodes absent from typing.tsv
    # (hub nodes and any unclassified plasmids)
    color_to_sc: dict = {}
    for pid, sc in p2t.items():
        if pid in node_color:
            color_to_sc[node_color[pid]] = sc

    def get_sc(nid: str) -> str:
        """Resolve subcommunity label with three-level fallback."""
        if nid in p2t:            return p2t[nid]        # typing.tsv (primary)
        if node_is_hub.get(nid):  return "hub"           # hub connector node
        return color_to_sc.get(node_color.get(nid, ""), "unknown")

    node_subcom = {nid: get_sc(nid) for nid in node_ids}
    node_idx    = {nid: i for i, nid in enumerate(node_ids)}
    subcom_arr  = np.array([node_subcom[n] for n in node_ids])
    sc_sizes    = {sc: int(np.sum(subcom_arr == sc))
                   for sc in np.unique(subcom_arr)}

    # ── Edge parsing ──────────────────────────────────────────────────────────
    edge_list, edge_sd, edge_td, edge_lbl = [], [], [], []
    for e in raw_edges:
        d = e["data"]
        s, t = d["source"], d["target"]
        if s in node_idx and t in node_idx:
            edge_list.append((node_idx[s], node_idx[t]))
            edge_sd.append(float(d.get("sd",    0.0)))
            edge_td.append(int(  d.get("td",    1  )))
            edge_lbl.append(     d.get("d_lbl", ""  ))

    print(f"Community {args.community}: "
          f"{len(node_ids)} nodes, {len(edge_list)} edges")

    # ── Three-phase layout ────────────────────────────────────────────────────
    coords = hybrid_layout(node_ids, node_subcom, edge_list, edge_td, seed=42)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(26, 22))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    td_max = max(edge_td) if edge_td else 1

    # ── Edges ─────────────────────────────────────────────────────────────────
    for (u, v), sd, td, lbl in zip(edge_list, edge_sd, edge_td, edge_lbl):
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        same  = subcom_arr[u] == subcom_arr[v]
        n_sc  = sc_sizes.get(subcom_arr[u], 1)

        if same:
            # Intra-cluster: slightly darker/thicker so clique structure shows
            alpha, lw, col = 0.38, 0.75, "dimgray"
        else:
            # Inter-cluster: faint and thin — topology hint, not dominant
            alpha = max(0.05, 0.22 - 0.16 * (td / td_max))
            lw, col = 0.42, "gray"

        ax.plot([x0, x1], [y0, y1],
                color=col, lw=lw, alpha=alpha,
                zorder=1, solid_capstyle="round")

        # Edge labels (sd / td) shown selectively to avoid clutter:
        # always in small clusters (≤ 30 nodes), sparingly in large ones.
        if lbl:
            show = (same and (n_sc <= 30 or (td <= 4 and sd > 0))) or \
                   (not same and td <= 4 and not (sd == 0.0 and td == 1))
            if show:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, lbl,
                        fontsize=3.0 if n_sc > 30 else 3.6,
                        color="#333333", ha="center", va="center", zorder=2,
                        path_effects=[
                            pe.withStroke(linewidth=1.2, foreground="white")
                        ])

    # ── Nodes ─────────────────────────────────────────────────────────────────
    hub_idx    = [i for i, n in enumerate(node_ids) if     node_is_hub[n]]
    circle_idx = [i for i, n in enumerate(node_ids) if not node_is_hub[n]]

    # Regular nodes → filled circles
    ax.scatter(coords[circle_idx, 0], coords[circle_idx, 1],
               s=85, c=[node_color[node_ids[i]] for i in circle_idx],
               edgecolors="white", linewidths=0.7, zorder=4, alpha=0.93)

    # Hub nodes → larger stars with black outline (matches pling HTML style)
    ax.scatter(coords[hub_idx, 0], coords[hub_idx, 1],
               s=320, marker="*",
               c=[node_color[node_ids[i]] for i in hub_idx],
               edgecolors="black", linewidths=1.1, zorder=5)

    # ── Labels ────────────────────────────────────────────────────────────────
    for i, nid in enumerate(node_ids):
        x, y = coords[i]
        ax.text(x, y, nid,
                fontsize=7.2 if node_is_hub[nid] else 5.5,
                fontweight="bold" if node_is_hub[nid] else "normal",
                color=node_color[nid],
                ha="center", va="bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.9, foreground="white")])

    # ── Legend ────────────────────────────────────────────────────────────────
    sc_col: dict = {}
    for nid in node_ids:
        sc = node_subcom[nid]
        if sc not in sc_col:
            sc_col[sc] = node_color[nid]

    def leg_label(sc: str) -> str:
        sc = sc.replace(f"community_{args.community}_subcommunity_", "Sub ")
        sc = sc.replace("hub", "Hub (connector)")
        sc = sc.replace("unknown", "Unclassified")
        return sc

    handles = [
        plt.Line2D([0], [0],
                   marker="*" if sc == "hub" else "o", color="w",
                   markerfacecolor=sc_col[sc],
                   markeredgecolor="black" if sc == "hub" else sc_col[sc],
                   markersize=10 if sc == "hub" else 8,
                   label=leg_label(sc))
        for sc in sorted(sc_col, key=nat_key)
    ]
    leg = ax.legend(handles=handles,
                    title=f"Community {args.community} – subcommunities",
                    title_fontsize=9, fontsize=7.5,
                    loc="lower left", frameon=True, framealpha=0.88,
                    ncol=2, borderpad=0.9, handletextpad=0.5, labelspacing=0.4)
    leg.get_frame().set_edgecolor("#aaaaaa")

    ax.set_title(
        f"Community {args.community} – Plasmid relatedness network  |  "
        f"Three-phase hierarchical layout  |  "
        f"{len(node_ids)} plasmids · {len(edge_list)} edges",
        fontsize=13, pad=12, fontweight="bold")

    plt.tight_layout(pad=0.5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()