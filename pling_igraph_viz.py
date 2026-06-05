"""
pling_igraph_viz.py
===================
Recreates pling community visualisations as static PNG figures using
**igraph** for graph algorithms and **matplotlib** for rendering.

Two figures are produced per community:
  1. Full graph      – all plasmids including hub (connector) nodes
  2. No-hub graph    – hub nodes and their edges removed; shows the
                       direct within-subcommunity structure more clearly

Both igraph.Graph objects are kept in memory with rich vertex / edge
attributes so callers can run further graph-theoretic analyses on them
(centrality, shortest paths, community detection, etc.).

Background
----------
pling clusters plasmids into communities / subcommunities based on:
  sd  (sequence distance)  – continuous [0, 1], lower = more similar
  td  (topological depth)  – integer hop-count in the plasmid graph

Each community JSON contains:
  nodes  – plasmid IDs with a colour (subcommunity) and optional is_hub flag
  edges  – plasmid pairs with sd, td, and d_lbl ("sd / td" display label)

Three-phase layout strategy
---------------------------
Phase 1 – GLOBAL FR
    Full-graph Fruchterman-Reingold with strongly up-weighted intra-community
    edges.  Gives correct macro topology (blob positions) at low cost.

Phase 2 – LOCAL FR / KK per subcommunity
    Each subcommunity gets its own independent layout (Kamada-Kawai for
    ≤ 15 nodes, FR otherwise) scaled by a density-adaptive radius, then
    re-centred at its Phase-1 centroid plus a mild gravity pull toward its
    external connection partners.  Nodes with no intra-cluster edges keep
    their Phase-1 global positions.

Phase 3 – CENTROID REPULSION (size-weighted)
    Iteratively pushes apart overlapping cluster boundaries.  Forces are
    SIZE-WEIGHTED so larger clusters absorb more displacement, preventing
    the cascading-collision bug where resolving one large overlap shoves a
    big cluster into a small isolated one.

Usage
-----
    python3 pling_igraph_viz.py \\
        --community  0 \\
        --json       community_0.json \\
        --typing     typing.tsv \\
        --out-dir    community_0_outputs/ \\
        --dpi        200

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

import igraph as ig          # graph data structure + layout algorithms
import matplotlib
matplotlib.use("Agg")        # non-interactive backend — needed for PNG output
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe   # white text stroke for legibility
import numpy as np
import pandas as pd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns a Namespace with:
        community  (int)  – community index for legend labels
        json       (str)  – path to the pling community JSON
        typing     (str)  – path to typing.tsv
        out_dir    (str)  – output directory where both PNGs will be written
        dpi        (int)  – output resolution
    """
    p = argparse.ArgumentParser(
        description="Recreate pling community visualisations with igraph."
    )
    p.add_argument("--community",  type=int, default=0)
    p.add_argument("--json",       required=True,
                   help="Path to community JSON from pling output")
    p.add_argument("--typing",     required=True,
                   help="Path to typing.tsv")
    p.add_argument("--out-dir",    required=True,
                   help="Output directory where both full and no-hub PNGs will be written")
    p.add_argument("--dpi",        type=int, default=200)
    return p.parse_args()


# ── Utility ───────────────────────────────────────────────────────────────────

def nat_key(s: str) -> list:
    """Natural-sort key: 'Sub 10' sorts after 'Sub 9', not before 'Sub 2'."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


# ── Graph construction ────────────────────────────────────────────────────────

def build_pling_graph(raw_nodes: list, raw_edges: list,
                      p2t: dict, color_to_sc: dict,
                      exclude_hubs: bool = False) -> ig.Graph:
    """Build a fully attributed igraph.Graph from pling JSON data.

    This is the single source of truth for graph construction.  Both the
    full graph and the no-hub graph are created by this function; the only
    difference is the ``exclude_hubs`` flag.

    Vertex attributes set on the returned graph
    -------------------------------------------
    name    (str)   plasmid ID string
    color   (str)   matplotlib colour name — one colour per subcommunity
    is_hub  (bool)  True for hub connector nodes
    subcom  (str)   subcommunity label (from typing.tsv or colour fallback)

    Edge attributes set on the returned graph
    -----------------------------------------
    sd   (float)  sequence distance [0, 1]
    td   (int)    topological depth (hop count)
    lbl  (str)    display label used in the pling HTML ("sd / td")

    Parameters
    ----------
    raw_nodes    : list of node dicts from JSON["elements"]["nodes"]
    raw_edges    : list of edge dicts from JSON["elements"]["edges"]
    p2t          : dict  plasmid_id → subcommunity label (from typing.tsv)
    color_to_sc  : dict  colour_string → subcommunity label (fallback)
    exclude_hubs : if True, hub nodes and ALL their incident edges are removed

    Returns
    -------
    igraph.Graph  – undirected, with vertex and edge attributes as above
    """

    def get_sc(nid: str, is_hub: bool) -> str:
        """Resolve subcommunity label with three-level fallback."""
        if nid in p2t:   return p2t[nid]          # typing.tsv  (primary)
        if is_hub:        return "hub"             # hub connector
        return color_to_sc.get(                    # colour → subcom (fallback)
            next((n["data"].get("color", "") for n in raw_nodes
                  if n["data"]["id"] == nid), ""),
            "unknown"
        )

    # ── Filter nodes ──────────────────────────────────────────────────────────
    # When exclude_hubs=True we drop hub nodes entirely so the returned graph
    # has no hub vertices at all — callers can be sure g.vs["is_hub"] is all
    # False in that case.
    filtered_nodes = [
        n for n in raw_nodes
        if not (exclude_hubs and n["data"].get("is_hub", False))
    ]
    node_ids  = [n["data"]["id"] for n in filtered_nodes]
    node_idx  = {nid: i for i, nid in enumerate(node_ids)}  # id → vertex index

    # ── Build igraph vertex set ───────────────────────────────────────────────
    g = ig.Graph(directed=False)
    g.add_vertices(len(node_ids))

    colors  = [n["data"].get("color",  "steelblue")      for n in filtered_nodes]
    is_hubs = [bool(n["data"].get("is_hub", False))       for n in filtered_nodes]
    subcoms = [get_sc(n["data"]["id"], bool(n["data"].get("is_hub", False)))
               for n in filtered_nodes]

    g.vs["name"]   = node_ids
    g.vs["color"]  = colors
    g.vs["is_hub"] = is_hubs
    g.vs["subcom"] = subcoms

    # ── Build igraph edge set ─────────────────────────────────────────────────
    # Drop edges whose source or target was removed (hub exclusion).
    edge_list, edge_sd, edge_td, edge_lbl = [], [], [], []
    for e in raw_edges:
        d = e["data"]
        s, t = d["source"], d["target"]
        if s in node_idx and t in node_idx:
            edge_list.append((node_idx[s], node_idx[t]))
            edge_sd.append(float(d.get("sd",    0.0)))
            edge_td.append(int(  d.get("td",    1  )))
            edge_lbl.append(     d.get("d_lbl", ""  ))

    g.add_edges(edge_list)
    g.es["sd"]  = edge_sd
    g.es["td"]  = edge_td
    g.es["lbl"] = edge_lbl

    return g


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def global_fr_layout(g: ig.Graph, subcom_arr: np.ndarray) -> np.ndarray:
    """Phase-1 global Fruchterman-Reingold layout.

    Edge-weight design
    ------------------
    Intra-community edges get high weights (tight springs) so nodes in the
    same subcommunity cluster together.  Inter-community edges get low weights
    so they preserve global topology without distorting local structure.
    td (topological depth) scales each weight inversely: closer plasmids =
    stronger spring.

    Parameters
    ----------
    g           : igraph.Graph (the graph to lay out)
    subcom_arr  : np.ndarray of subcommunity labels, one per vertex (same order)

    Returns
    -------
    np.ndarray (N, 2) – raw FR coordinates
    """
    weights = []
    for e in g.es:
        u, v = e.source, e.target
        td   = e["td"]
        same = subcom_arr[u] == subcom_arr[v]
        weights.append(
            max(8.0, 20.0 / max(td, 0.1)) if same
            else max(0.1,  0.5 / max(td, 0.1))
        )

    layout = g.layout_fruchterman_reingold(
        weights  = weights,
        niter    = 3000,      # more iterations → stabler convergence
        grid     = "nogrid",  # exact repulsion (slower, higher quality)
    )
    return np.array(layout.coords)


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def local_layout_coords(active: list, g: ig.Graph,
                        n_act: int) -> np.ndarray | None:
    """Phase-2 local layout for one subcommunity.

    Algorithm choice
    ----------------
    Kamada-Kawai (KK) for small clusters (≤ 15 nodes): minimises spring
    energy based on graph-theoretic shortest paths, producing more organic
    shapes than FR for sparse/small graphs.

    Fruchterman-Reingold for larger clusters: scales better and respects
    edge weights more clearly at higher node counts.

    Auto-rotation
    -------------
    Chain-like sub-graphs (two cliques joined by a long-distance edge) tend
    to be laid out horizontally by both algorithms.  We detect layouts with
    aspect ratio > 1.8 and rotate 90° so they run vertically instead.

    Returns
    -------
    np.ndarray (n_act, 2) centred at origin, normalised to unit max-radius,
    or None when no intra-cluster edges exist.
    """
    # Build a subgraph containing only the active nodes and intra-cluster edges
    local_map = {v: i for i, v in enumerate(active)}

    lg = ig.Graph(directed=False)
    lg.add_vertices(n_act)

    loc_edges, loc_w = [], []
    for e in g.es:
        u, v = e.source, e.target
        if u in local_map and v in local_map:
            loc_edges.append((local_map[u], local_map[v]))
            # Higher weight = shorter spring = nodes pulled closer together.
            # td=0 (near-identical sequence) → very strong; td=4 → gentle.
            loc_w.append(max(1.0, 8.0 / max(e["td"], 0.1)))

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
    lc -= lc.mean(axis=0)        # centre at origin

    r = np.max(np.linalg.norm(lc, axis=1))
    if r > 0:
        lc /= r                  # normalise to unit max-radius

    # Auto-rotate: if width > 1.8 × height, rotate 90° to make it vertical
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
    FR naturally compresses dense graphs (many springs) and spreads sparse
    ones.  A fixed scale factor would make sparse clusters look over-expanded
    relative to dense ones.  We map density linearly to a scale factor:

        scale_f = clamp(0.38 + 0.28 × density, max=0.62)

    Calibrated on community_0:
        fuchsia   density=0.85 → scale_f=0.62  (visually correct)
        orange    density=0.26 → scale_f=0.45  (tighter, avoids bloat)
        indianred density=0.34 → scale_f=0.48
        aqua      density=0.46 → scale_f=0.51, capped at 0.28 (small n)

    Returns the physical max-node distance from the cluster centre.
    """
    max_edges = n_act * (n_act - 1) / 2
    density   = n_int / max_edges if max_edges > 0 else 0.0

    scale_f = min(0.38 + 0.28 * density, 0.62)
    if n_act <= 12:
        scale_f = min(scale_f, 0.28)   # small clusters: extra compact cap

    return ref_unit * math.sqrt(n_act) * scale_f


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def resolve_cluster_overlaps(final: np.ndarray,
                              sc_members: dict,
                              has_intra: set,
                              ref_unit: float,
                              min_gap_factor: float = 0.3,
                              max_iters: int = 200) -> np.ndarray:
    """Phase-3 size-weighted centroid repulsion.

    Why size-weighted?
    ------------------
    Equal (50/50) repulsion caused a cascading bug: resolving the large
    orange↔fuchsia overlap (7 raw units) pushed fuchsia into aqua's space,
    which then pushed aqua to the canvas edge.

    With SIZE-WEIGHTED forces (larger cluster = more displacement, like
    higher inertial mass in Newtonian mechanics) orange absorbs ~69% of the
    push, fuchsia only ~31%, and fuchsia never reaches aqua.

    Algorithm
    ---------
    For each overlapping pair (i, j):
        required = radius_i + radius_j + min_gap
        push     = required - dist(centroid_i, centroid_j)
        centroid_i += direction × push × (n_i / (n_i + n_j))
        centroid_j -= direction × push × (n_j / (n_i + n_j))
    Repeat until all pairs are satisfied or max_iters is reached.
    Every node in a displaced cluster shifts by the same rigid vector so
    the internal clique structure from Phase 2 is perfectly preserved.

    Parameters
    ----------
    final          : coordinate array (N, 2), modified in-place
    sc_members     : subcommunity label → list of global node indices
    has_intra      : set of node indices with ≥ 1 intra-cluster edge
    ref_unit       : global reference length (total_span / sqrt(N))
    min_gap_factor : minimum boundary gap = min_gap_factor × ref_unit
    max_iters      : safety cap on repulsion iterations
    """
    min_gap = min_gap_factor * ref_unit

    # Collect clusters with enough active nodes to have a meaningful radius
    active_scs: list = []
    centroids:  dict = {}
    radii:      dict = {}

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

                # Size-weighted push: larger cluster moves more
                any_overlap = True
                direction  = (ci - cj) / dist
                push       = required - dist
                ni = len(sc_members.get(sci, [0]))
                nj = len(sc_members.get(scj, [0]))
                total_n = max(ni + nj, 1)
                centroids[sci] += direction * push * (ni / total_n)
                centroids[scj] -= direction * push * (nj / total_n)

        if not any_overlap:
            break   # all boundaries satisfied

    # Apply centroid shifts rigidly — all nodes in a cluster move together
    for sc in active_scs:
        mem    = sc_members[sc]
        active = [i for i in mem if i in has_intra]
        if not active:
            continue
        old_c = final[active].mean(axis=0)
        shift = centroids[sc] - old_c
        for gi in mem:     # shift ALL members (active + passive)
            final[gi] += shift

    return final


# ── Master layout function ────────────────────────────────────────────────────

def hybrid_layout(g: ig.Graph, seed: int = 42) -> np.ndarray:
    """Three-phase hierarchical layout for a pling community (sub)graph.

    Accepts any igraph.Graph that has the vertex attributes set by
    ``build_pling_graph`` (name, color, is_hub, subcom) and edge attributes
    (sd, td, lbl).  Works identically for the full graph and the no-hub graph.

    Parameters
    ----------
    g    : igraph.Graph produced by build_pling_graph()
    seed : RNG seed for reproducibility

    Returns
    -------
    np.ndarray (N, 2) – final node coordinates for matplotlib.
    Y-axis is flipped (positive = down) to match the pling HTML orientation.
    """
    np.random.seed(seed)
    random.seed(seed)

    N          = g.vcount()
    subcom_arr = np.array(g.vs["subcom"])

    # ── Phase 1: global FR ────────────────────────────────────────────────────
    glo = global_fr_layout(g, subcom_arr)   # shape (N, 2)

    # ref_unit = "average inter-node spacing" in the global layout.
    # All local radii are expressed as multiples of ref_unit so that clusters
    # have consistent visual density regardless of total graph size.
    span_x   = glo[:, 0].max() - glo[:, 0].min()
    span_y   = glo[:, 1].max() - glo[:, 1].min()
    ref_unit = max(span_x, span_y) / math.sqrt(N)

    graph_centroid = glo.mean(axis=0)   # overall centre of mass

    # ── Per-subcommunity statistics ───────────────────────────────────────────
    sc_members   : dict = defaultdict(list)
    sc_int_count : dict = defaultdict(int)   # intra-cluster edge count
    sc_ext_count : dict = defaultdict(int)   # inter-cluster edge count

    for i, sc in enumerate(subcom_arr):
        sc_members[sc].append(i)

    for e in g.es:
        u, v = e.source, e.target
        su, sv = subcom_arr[u], subcom_arr[v]
        if su == sv:
            sc_int_count[su] += 1
        else:
            sc_ext_count[su] += 1
            sc_ext_count[sv] += 1

    # Mean position of each cluster's external partners in the global layout.
    # Pulling toward partners (rather than the overall centroid) gives more
    # topologically meaningful placement: isolated clusters land near their
    # actual connection point rather than the graph's geometric centre.
    sc_partner_pos: dict = {}
    for sc in sc_members:
        partners = []
        for e in g.es:
            u, v = e.source, e.target
            su, sv = subcom_arr[u], subcom_arr[v]
            if su == sc and sv != sc:
                partners.append(glo[v])
            elif sv == sc and su != sc:
                partners.append(glo[u])
        sc_partner_pos[sc] = (np.mean(partners, axis=0)
                              if partners else graph_centroid)

    # Nodes WITH at least one intra-cluster edge participate in local Phase-2
    # layout.  Nodes with ONLY external edges keep their Phase-1 position —
    # they are "passive" members that trail their cluster without distorting
    # the local FR/KK result.
    has_intra: set = set()
    for e in g.es:
        u, v = e.source, e.target
        if subcom_arr[u] == subcom_arr[v]:
            has_intra.add(u)
            has_intra.add(v)

    # ── Phase 2: per-subcommunity local layouts ───────────────────────────────
    final = glo.copy()

    for sc, mem in sc_members.items():
        active = [i for i in mem if i in has_intra]
        n_act  = len(active)
        if n_act < 3:
            continue   # singletons / pairs: Phase-1 position is adequate

        raw_centroid = glo[active].mean(axis=0)
        ext          = sc_ext_count.get(sc, 0)
        target       = sc_partner_pos[sc]

        # Gravity pull: moves the cluster centroid toward its external partners.
        # Pull is modest (0.25) — Phase 3 handles residual overlaps so we
        # don't need aggressive pull (which caused the aqua↔fuchsia collision).
        isolation = 1.0 / (1.0 + ext * 0.10)
        centroid  = raw_centroid + 0.25 * isolation * (target - raw_centroid)

        lc = local_layout_coords(active, g, n_act)
        if lc is None:
            continue

        local_radius = density_adaptive_scale(
            n_act, sc_int_count.get(sc, 0), ref_unit
        )
        lc *= local_radius   # lc is unit-normalised from local_layout_coords()

        for li, gi in enumerate(active):
            final[gi] = centroid + lc[li]
        # passive nodes already sit at glo[gi] — no change needed

    # ── Phase 3: centroid repulsion ───────────────────────────────────────────
    final = resolve_cluster_overlaps(
        final, sc_members, has_intra,
        ref_unit       = ref_unit,
        min_gap_factor = 0.3,   # gap = 0.3 × ref_unit between cluster boundaries
        max_iters      = 200,
    )

    final[:, 1] = -final[:, 1]   # flip y: igraph uses y-up, screen uses y-down
    return final


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_figure(g: ig.Graph, coords: np.ndarray,
                  community_idx: int, title_suffix: str,
                  out_path: str, dpi: int = 200) -> None:
    """Render one pling-style network figure to a PNG file.

    Draws edges, then nodes (circles for regular, stars for hubs), then labels.
    Matches the visual style of the pling HTML output.

    Parameters
    ----------
    g             : igraph.Graph (full or no-hub) with all vertex/edge attrs
    coords        : np.ndarray (N, 2) from hybrid_layout()
    community_idx : community index — used in the title and legend header
    title_suffix  : appended to the figure title, e.g. "(no hub nodes)"
    out_path      : output PNG file path
    dpi           : output resolution
    """
    N = g.vcount()
    subcom_arr = np.array(g.vs["subcom"])
    sc_sizes   = {sc: int(np.sum(subcom_arr == sc))
                  for sc in np.unique(subcom_arr)}

    td_vals = g.es["td"] if g.ecount() > 0 else [1]
    td_max  = max(td_vals)

    fig, ax = plt.subplots(figsize=(26, 22))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Edges ─────────────────────────────────────────────────────────────────
    for e in g.es:
        u, v = e.source, e.target
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        td   = e["td"]
        sd   = e["sd"]
        lbl  = e["lbl"]
        same = subcom_arr[u] == subcom_arr[v]
        n_sc = sc_sizes.get(subcom_arr[u], 1)

        if same:
            # Intra-cluster: darker/thicker so internal clique structure shows
            alpha, lw, col = 0.38, 0.75, "dimgray"
        else:
            # Inter-cluster: faint and thin — topology hint, not dominant
            alpha = max(0.05, 0.22 - 0.16 * (td / td_max))
            lw, col = 0.42, "gray"

        ax.plot([x0, x1], [y0, y1],
                color=col, lw=lw, alpha=alpha,
                zorder=1, solid_capstyle="round")

        # Edge labels (sd / td): shown for all edges in small clusters,
        # selectively for large clusters to avoid clutter.
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
    is_hubs = g.vs["is_hub"]
    colors  = g.vs["color"]

    hub_idx    = [i for i in range(N) if     is_hubs[i]]
    circle_idx = [i for i in range(N) if not is_hubs[i]]

    # Regular nodes → filled circles
    if circle_idx:
        ax.scatter(coords[circle_idx, 0], coords[circle_idx, 1],
                   s=85,
                   c=[colors[i] for i in circle_idx],
                   edgecolors="white", linewidths=0.7,
                   zorder=4, alpha=0.93)

    # Hub nodes → larger stars with black outline (matches pling HTML style)
    if hub_idx:
        ax.scatter(coords[hub_idx, 0], coords[hub_idx, 1],
                   s=320, marker="*",
                   c=[colors[i] for i in hub_idx],
                   edgecolors="black", linewidths=1.1,
                   zorder=5)

    # ── Labels ────────────────────────────────────────────────────────────────
    names = g.vs["name"]
    for i in range(N):
        x, y = coords[i]
        ax.text(x, y, names[i],
                fontsize=7.2 if is_hubs[i] else 5.5,
                fontweight="bold" if is_hubs[i] else "normal",
                color=colors[i],
                ha="center", va="bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.9, foreground="white")])

    # ── Legend ────────────────────────────────────────────────────────────────
    sc_col: dict = {}
    for i in range(N):
        sc = subcom_arr[i]
        if sc not in sc_col:
            sc_col[sc] = colors[i]

    def leg_label(sc: str) -> str:
        sc = sc.replace(f"community_{community_idx}_subcommunity_", "Sub ")
        sc = sc.replace("hub",     "Hub (connector)")
        sc = sc.replace("unknown", "Unclassified")
        return sc

    handles = [
        plt.Line2D([0], [0],
                   marker="*" if sc == "hub" else "o",
                   color="w",
                   markerfacecolor=sc_col[sc],
                   markeredgecolor="black" if sc == "hub" else sc_col[sc],
                   markersize=10 if sc == "hub" else 8,
                   label=leg_label(sc))
        for sc in sorted(sc_col, key=nat_key)
    ]
    leg = ax.legend(handles=handles,
                    title=f"Community {community_idx} – subcommunities",
                    title_fontsize=9, fontsize=7.5,
                    loc="lower left", frameon=True, framealpha=0.88,
                    ncol=2, borderpad=0.9, handletextpad=0.5,
                    labelspacing=0.4)
    leg.get_frame().set_edgecolor("#aaaaaa")

    ax.set_title(
        f"Community {community_idx} – Plasmid relatedness network  "
        f"{title_suffix} |  "
        f"Three-phase hierarchical layout  |  "
        f"{N} plasmids · {g.ecount()} edges",
        fontsize=13, pad=12, fontweight="bold"
    )

    plt.tight_layout(pad=0.5)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)   # free memory — important when rendering multiple figures
    print(f"Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Load pling data and produce two figures: full graph and no-hub graph.

    The two igraph.Graph objects (``g_full`` and ``g_no_hubs``) are kept as
    separate, fully attributed data structures so they can be used for further
    graph-theoretic calculations after this function returns.
    """
    args = parse_args()

    # ── Load raw data ─────────────────────────────────────────────────────────
    with open(args.json) as fh:
        raw = json.load(fh)
    raw_nodes = raw["elements"]["nodes"]
    raw_edges = raw["elements"]["edges"]

    typing = pd.read_csv(args.typing, sep="\t")
    p2t    = dict(zip(typing["plasmid"], typing["type"]))

    # Fallback colour → subcommunity map for nodes absent from typing.tsv
    color_to_sc: dict = {}
    for pid, sc in p2t.items():
        node_color = next(
            (n["data"].get("color", "") for n in raw_nodes
             if n["data"]["id"] == pid), ""
        )
        if node_color:
            color_to_sc[node_color] = sc

    # ── Build the two separate igraph objects ─────────────────────────────────
    # g_full    : complete graph — hub nodes act as bridges between subcommunities
    # g_no_hubs : hub-free graph — shows direct within-subcommunity structure;
    #             subcommunities may appear as disconnected components
    #
    # Both graphs carry identical vertex/edge attributes (name, color, is_hub,
    # subcom, sd, td, lbl) and can be used independently for further analysis.

    g_full = build_pling_graph(
        raw_nodes, raw_edges, p2t, color_to_sc,
        exclude_hubs=False
    )
    g_no_hubs = build_pling_graph(
        raw_nodes, raw_edges, p2t, color_to_sc,
        exclude_hubs=True
    )

    print(f"Full graph:     {g_full.vcount():3d} nodes, {g_full.ecount():4d} edges "
          f"({sum(g_full.vs['is_hub'])} hubs)")
    print(f"No-hub graph:   {g_no_hubs.vcount():3d} nodes, {g_no_hubs.ecount():4d} edges")

    # ── Layout — run independently on each graph ──────────────────────────────
    # Each call to hybrid_layout() uses only the edges present in that graph,
    # so the no-hub layout clusters subcommunities more tightly (fewer
    # cross-community edges means less tension pulling blobs apart).
    print("\nComputing layout for full graph …")
    coords_full    = hybrid_layout(g_full,    seed=42)

    print("Computing layout for no-hub graph …")
    coords_no_hubs = hybrid_layout(g_no_hubs, seed=42)

    # ── Render — one figure per graph ─────────────────────────────────────────
    # Write both figures into the single output directory requested by CLI.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out = str(out_dir / f"community_{args.community}.png")
    args.out_nohubs = str(out_dir / f"community_{args.community}_no_hubs.png")

    render_figure(
        g_full, coords_full,
        community_idx = args.community,
        title_suffix  = "| with hub nodes",
        out_path      = args.out,
        dpi           = args.dpi,
    )
    render_figure(
        g_no_hubs, coords_no_hubs,
        community_idx = args.community,
        title_suffix  = "| hub nodes removed",
        out_path      = args.out_nohubs,
        dpi           = args.dpi,
    )

    # ── Summary of the two igraph objects ─────────────────────────────────────
    # These can be used directly for further calculations, e.g.:
    #   g_full.betweenness()          → vertex betweenness centrality
    #   g_no_hubs.clusters()          → connected components without hubs
    #   g_full.shortest_paths()       → all-pairs shortest paths
    #   g_no_hubs.community_fastgreedy() → greedy modularity clustering
    print(f"\nTwo igraph objects available for further analysis:")
    print(f"  g_full    → {g_full.vcount()} vertices, {g_full.ecount()} edges")
    print(f"  g_no_hubs → {g_no_hubs.vcount()} vertices, {g_no_hubs.ecount()} edges")
    print(f"  Vertex attributes: {g_full.vs.attributes()}")
    print(f"  Edge attributes:   {g_full.es.attributes()}")

    # Return both graph objects so callers using this as a module can access them
    return g_full, coords_full, g_no_hubs, coords_no_hubs


if __name__ == "__main__":
    main()