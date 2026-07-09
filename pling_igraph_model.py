"""
pling_igraph_model.py
================================
Abstract graph-level generator for plasmid–plasmid similarity networks
using an asymmetric duplication–divergence model, following:

    Pastor-Satorras, R. et al. (2003)
    "Evolving protein interaction networks through gene duplication"
    Journal of Theoretical Biology 222, 199-210
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2092385/

Algorithm description
---------------------
We model a plasmid similarity network G(V, E) as a purely abstract graph
process where nodes represent plasmids and edges represent similarity
relations emergent entirely from topology — no sequence data is used.
The generator initialises from a small Erdős–Rényi seed graph G₀ with n₀
nodes, created natively via igraph's Graph.Erdos_Renyi(); if the result is
disconnected, its spanning tree (Graph.spanning_tree) is added to guarantee
connectivity.  At each step a source node i is selected uniformly at random
or with probability proportional to degree (degree_bias=True), and a new
node n is added.  In the DUPLICATION step n inherits each edge (i, k) of i
independently with probability σ, and a direct parent–child edge (n, i) is
created with probability p.  In the DIVERGENCE step each retained edge is
independently dissolved with probability (1−σ), and n may form novel edges
to non-neighbours with probability ε.  If n remains isolated after both
steps it is removed (FAILURE condition), enforcing selection pressure toward
connected plasmids.  This process — degree inheritance, stochastic
fragmentation, and isolation removal — reproduces scale-free degree
distributions and short average path lengths without any explicit
preferential-attachment rule.

Model note
----------
σ (edge retention) primarily controls graph density: higher σ → more edges
inherited → denser graph → lower mean degree heterogeneity.  ε (novel-edge
probability) controls the fraction of cross-community "random" edges.

The fitted power-law exponent α is reported as a descriptive statistic,
but it is not used as a control target in this script.

Outputs (written to <out-dir>/duplication_divergence/)
------------------------------------------------------
    generated_network.png   – FR layout, nodes coloured by degree
    metrics/
        degree_distribution.png
        betweenness_distribution.png
        closeness_distribution.png
        eigenvector_distribution.png
        clustering_distribution.png
        path_length_distribution.png
    summary.txt             – scalar statistics and power-law fit

Usage
-----
    # Default run
    python3 pling_igraph_model.py --N 200 --sigma 0.5 --seed 42

Dependencies
------------
    pip install igraph matplotlib numpy
    pling_igraph_analysis.py must be importable from the same directory.
"""

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import igraph as ig
from igraph.statistics import power_law_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
import numpy as np

from pling_igraph_analysis import (
    degree_centrality,
    betweenness_centrality,
    closeness_centrality,
    eigenvector_centrality,
    local_clustering_coefficient,
    global_clustering_coefficient,
    average_path_length_metric,
    render_measure_distribution,
    render_path_length_distribution,
    render_combined_centrality_distribution,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Asymmetric duplication–divergence plasmid similarity "
                    "network generator."
    )
    p.add_argument("--N",            type=int,   default=200,
                   help="Target node count (default 200)")
    p.add_argument("--n0",           type=int,   default=5,
                   help="Seed graph size (default 5)")
    p.add_argument("--p0",           type=float, default=0.6,
                   help="Seed graph ER edge probability (default 0.6)")
    p.add_argument("--sigma",        type=float, default=0.50,
                   help="Edge inheritance probability σ (default 0.50)")
    p.add_argument("--p",            type=float, default=0.20,
                   help="Parent-child edge probability p (default 0.20)")
    p.add_argument("--epsilon",      type=float, default=0.01,
                   help="Novel-edge formation probability ε (default 0.01)")
    p.add_argument("--degree-bias",
                   type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True, metavar="BOOL",
                   help="Select source node ∝ degree: True (default) enables "
                        "implicit preferential attachment; False uses uniform "
                        "random selection.  Pass False/0/no to disable.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility. The generation "
                        "process is stochastic — fixing the seed guarantees "
                        "identical graphs across runs. Change it to explore "
                        "different realisations of the same parameters. "
                        "(default 42)")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for generated network and metrics")
    p.add_argument("--dpi", type=int, default=200,
                   help="Figure resolution in dots per inch. 200 is good for "
                        "screen; use 300-600 for publication. (default 200)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Seed graph
# ─────────────────────────────────────────────────────────────────────────────

def make_seed_graph(n0: int, p0: float) -> ig.Graph:
    """Build a connected seed graph G₀ using igraph's native ER generator.

    Graph.Erdos_Renyi(n, p) samples the classic G(n, p) model where each
    of the C(n,2) potential edges is included independently with probability
    p.  If the result is disconnected we augment with its spanning_tree()
    to guarantee full connectivity before duplication begins.
    """
    g = ig.Graph.Erdos_Renyi(n=n0, p=p0, directed=False, loops=False)
    if not g.is_connected():
        span = g.spanning_tree(return_tree=True)
        for e in span.es:
            u, v = e.source, e.target
            if not g.are_adjacent(u, v):
                g.add_edge(u, v)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Core generative model
# ─────────────────────────────────────────────────────────────────────────────

def _select_source(g: ig.Graph, rng: random.Random,
                   degree_bias: bool) -> int:
    """Select the duplication source node.

    degree_bias=True  → probability ∝ degree: high-degree nodes are
                        duplicated more often, which produces implicit
                        preferential attachment and accelerates hub growth.
    degree_bias=False → uniform random selection.
    """
    n = g.vcount()
    if not degree_bias:
        return rng.randrange(n)
    degrees = g.degree()
    total   = sum(degrees)
    if total == 0:
        return rng.randrange(n)
    r, cumsum = rng.random() * total, 0
    for i, d in enumerate(degrees):
        cumsum += d
        if r <= cumsum:
            return i
    return n - 1


def _step(g: ig.Graph, rng: random.Random,
          sigma: float, p: float, epsilon: float,
          degree_bias: bool) -> None:
    """One duplication–divergence step.  Mutates g in-place.

    Steps
    -----
    1. Select source node i.
    2. Add new node n.
    3. DUPLICATION : inherit each edge (i,k) with probability σ.
    4. PARENT EDGE : add edge (n,i) with probability p.
    5. DIVERGENCE  : dissolve each retained edge with prob (1−σ).
                     Add novel edge (n,j) to non-neighbour j with prob ε.
    6. FAILURE     : remove n if it is isolated after step 5.
    """
    src = _select_source(g, rng, degree_bias)
    g.add_vertex()
    new_v = g.vcount() - 1

    # Snapshot neighbours of src before mutation
    for k in [nb for nb in g.neighbors(src) if nb != new_v]:
        if rng.random() < sigma:
            g.add_edge(new_v, k)

    if rng.random() < p and not g.are_adjacent(new_v, src):
        g.add_edge(new_v, src)

    # Partial divergence: dissolve inherited edges with prob (1−σ)
    to_remove = [g.get_eid(new_v, k)
                 for k in g.neighbors(new_v)
                 if rng.random() < (1.0 - sigma)]
    if to_remove:
        g.delete_edges(to_remove)

    # Novel edges: form with small prob ε to random non-neighbours
    current_nb = set(g.neighbors(new_v)) | {new_v}
    for j in range(new_v):
        if j not in current_nb and rng.random() < epsilon:
            g.add_edge(new_v, j)

    # Failure condition
    if g.degree(new_v) == 0:
        g.delete_vertices(new_v)


def generate_network(N: int, n0: int, p0: float,
                     sigma: float, p: float, epsilon: float,
                     degree_bias: bool, seed: int,
                     verbose: bool = True) -> ig.Graph:
    """Run the full duplication–divergence process until |V| = N.

    Returns an unweighted igraph.Graph with vertex attributes:
        name   : "n0", "n1", … — unique string IDs
        is_hub : False for all (no hub concept in generated graph)
        subcom : "generated" for all (single dummy subcommunity)
        color  : uniform "#4e9fd4"
    and generation parameters stored as graph-level attributes.
    """
    rng = random.Random(seed)
    g   = make_seed_graph(n0, p0)

    eps      = epsilon
    attempts = 0
    max_tries = N * 100

    while g.vcount() < N:
        if attempts >= max_tries:
            eps = max(eps, 0.05)
            if verbose and attempts == max_tries:
                print(f"  [warn] relaxed ε to {eps:.3f} — "
                      f"σ may be too low for target N")
        _step(g, rng, sigma, p, eps, degree_bias)
        attempts += 1

    # Vertex attributes required by analysis rendering functions
    g.vs["name"]   = [f"n{i}" for i in range(g.vcount())]
    g.vs["is_hub"] = [False]        * g.vcount()
    g.vs["subcom"] = ["generated"]  * g.vcount()
    g.vs["color"]  = ["#4e9fd4"]    * g.vcount()

    # Graph-level metadata for traceability
    g["N"]           = N
    g["n0"]          = n0
    g["p0"]          = p0
    g["sigma"]       = sigma
    g["p"]           = p
    g["epsilon"]     = epsilon
    g["degree_bias"] = degree_bias
    g["seed"]        = seed
    g["attempts"]    = attempts

    if verbose:
        print(f"  Generated: {g.vcount()} nodes, {g.ecount()} edges "
              f"({attempts} attempts)  "
              f"mean_degree={np.mean(g.degree()):.2f}")
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Power-law fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_power_law(values: list[float], discrete: bool = False) -> tuple[float, float] | None:
    """Fit a power law to positive values with igraph.

    Returns (alpha, xmin) when the fit succeeds, or None when there are
    too few positive values or igraph cannot determine a fit.
    """
    data = [float(v) for v in values if v > 0]
    if len(data) < 10:
        return None
    method = "discrete" if discrete else "continuous"
    try:
        fit = power_law_fit(data, xmin=None, method=method, p_precision=0.01)
    except Exception:
        return None
    return float(fit.alpha), float(fit.xmin)


# ─────────────────────────────────────────────────────────────────────────────
# Generated network visualisation
# ─────────────────────────────────────────────────────────────────────────────

def render_network(g: ig.Graph, coords: np.ndarray,
                   out_path: Path, dpi: int = 200) -> None:
    """Render the generated network: nodes coloured by degree (Blues).

    Uses the pre-computed FR layout coords.  Edge opacity and width scale
    with the average degree of their two endpoints so hub connections are
    more visually prominent.  No edge weights exist in the generated graph.
    """
    N        = g.vcount()
    degrees  = g.degree()
    deg_arr  = np.array(degrees, dtype=float)

    vmin, vmax = deg_arr.min(), deg_arr.max()
    if vmax == vmin:
        vmax = vmin + 1
    norm   = Normalize(vmin=vmin, vmax=vmax)
    cmap   = cm.Blues
    colors = [to_hex(cmap(norm(d))) for d in degrees]

    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for e in g.es:
        u, v    = e.source, e.target
        avg_deg = (degrees[u] + degrees[v]) / 2.0
        alpha   = 0.12 + 0.55 * norm(avg_deg)
        lw      = 0.30 + 1.00 * norm(avg_deg)
        ax.plot([coords[u, 0], coords[v, 0]],
                [coords[u, 1], coords[v, 1]],
                color="#888888", lw=lw, alpha=float(alpha), zorder=1)

    node_sizes = [max(20, 40 * math.log(d + 2)) for d in degrees]
    ax.scatter(coords[:, 0], coords[:, 1],
               s=node_sizes, c=colors,
               edgecolors="white", linewidths=0.5,
               zorder=3, alpha=0.93)

    # Label the top-10 highest-degree nodes
    top10 = sorted(range(N), key=lambda i: degrees[i], reverse=True)[:10]
    for i in top10:
        x, y = coords[i]
        ax.text(x, y, f"n{i}  (d={degrees[i]})",
                fontsize=5.8, ha="center", va="bottom", zorder=5,
                path_effects=[pe.withStroke(linewidth=1.8,
                                            foreground="white")])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=35)
    cbar.set_label("Node degree", fontsize=11)

    ax.set_title(
        f"Duplication–Divergence Plasmid Similarity Network\n"
        f"N={g['N']}  n₀={g['n0']}  σ={g['sigma']:.3f}  "
        f"p={g['p']}  ε={g['epsilon']:.4f}  "
        f"degree_bias={g['degree_bias']}  seed={g['seed']}\n"
        f"{g.ecount()} edges  ·  density={g.density():.4f}  ·  "
        f"mean_degree={np.mean(degrees):.2f}",
        fontsize=12, pad=10, fontweight="bold"
    )
    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics summary
# ─────────────────────────────────────────────────────────────────────────────

def write_stats(g: ig.Graph, out_path: Path,
                power_law_fits: dict[str, tuple[float, float] | None]) -> None:
    """Human-readable statistics summary."""
    degrees  = g.degree()
    comps    = g.connected_components()
    sizes    = sorted(comps.sizes(), reverse=True)
    cc_local = local_clustering_coefficient(g)
    apl      = average_path_length_metric(g)

    with open(out_path, "w") as fh:
        fh.write("=" * 65 + "\n")
        fh.write("  DUPLICATION–DIVERGENCE NETWORK  —  STATISTICS\n")
        fh.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write("=" * 65 + "\n\n")

        fh.write("── Generation parameters ──────────────────────────────────\n")
        rows = [
            ("N",                  g["N"]),
            ("n0 (seed nodes)",    g["n0"]),
            ("p0 (ER seed prob.)", g["p0"]),
            ("σ (edge retention)", g["sigma"]),
            ("p (parent edge)",    g["p"]),
            ("ε (novel edges)",    g["epsilon"]),
            ("degree_bias",        g["degree_bias"]),
            ("seed",               g["seed"]),
            ("duplication attempts", g["attempts"]),
        ]
        for k, v in rows:
            fh.write(f"  {k:<26}: {v}\n")

        fh.write("\n── Graph structure ─────────────────────────────────────────\n")
        fh.write(f"  Nodes               : {g.vcount()}\n")
        fh.write(f"  Edges               : {g.ecount()}\n")
        fh.write(f"  Density             : {g.density():.5f}\n")
        fh.write(f"  Connected components: {len(sizes)}\n")
        if len(sizes) > 1:
            fh.write(f"  Component sizes     : {sizes[:10]}"
                     f"{'…' if len(sizes) > 10 else ''}\n")

        fh.write("\n── Degree distribution ─────────────────────────────────────\n")
        fh.write(f"  min         : {min(degrees)}\n")
        fh.write(f"  max         : {max(degrees)}\n")
        fh.write(f"  mean        : {np.mean(degrees):.4f}\n")
        fh.write(f"  std         : {np.std(degrees):.4f}\n")
        degree_fit = power_law_fits.get("degree")
        if degree_fit is not None:
            alpha, xmin = degree_fit
            fh.write(f"  Degree power-law fit     : alpha={alpha:.4f}, xmin={xmin:.4f}\n")
        else:
            fh.write("  Degree power-law fit     : (igraph power-law fit unavailable)\n")

        for measure in ("betweenness", "closeness", "eigenvector"):
            fit = power_law_fits.get(measure)
            label = measure.capitalize()
            if fit is not None:
                alpha, xmin = fit
                fh.write(f"  {label} power-law fit     : alpha={alpha:.4f}, xmin={xmin:.4f}\n")
            else:
                fh.write(f"  {label} power-law fit     : (igraph power-law fit unavailable)\n")

        fh.write("\n── Topology ────────────────────────────────────────────────\n")
        fh.write(f"  Global clustering (transitivity) : "
                 f"{global_clustering_coefficient(g):.5f}\n")
        fh.write(f"  Mean local clustering coeff.     : "
                 f"{np.mean(cc_local):.5f}\n")
        fh.write(f"  Average path length              : {apl:.4f}\n")

        fh.write("── Model note ──────────────────────────────────────────────\n")
        fh.write(
            "  σ controls graph density (mean degree) reliably:\n"
            "    σ↑ → more edges inherited → higher mean degree\n"
            "    σ↓ → fewer edges retained → lower mean degree\n"
            "  ε adds random novel edges, also increasing mean degree.\n\n"
            "  Power-law fits are reported as descriptive statistics only;\n"
            "  they are not used as control targets here.\n"
        )
        fh.write("=" * 65 + "\n")

    print(f"  Saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    met_dir = out_dir / "duplication_divergence_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    sigma, epsilon = args.sigma, args.epsilon

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\nGenerating network  "
          f"N={args.N}  σ={sigma:.4f}  p={args.p}  ε={epsilon:.5f} …")
    g = generate_network(
        N=args.N, n0=args.n0, p0=args.p0,
        sigma=sigma, p=args.p, epsilon=epsilon,
        degree_bias=args.degree_bias, seed=args.seed,
    )

    power_law_fits = {
        "degree":      fit_power_law(degree_centrality(g, normalized=False), discrete=True),
        "betweenness": fit_power_law(betweenness_centrality(g, normalized=False)),
        "closeness":   fit_power_law(closeness_centrality(g, normalized=True)),
        "eigenvector": fit_power_law(eigenvector_centrality(g, normalized=True)),
    }
    for measure, fit in power_law_fits.items():
        if fit is not None:
            alpha, xmin = fit
            print(f"  {measure.capitalize()} power-law fit: alpha={alpha:.4f}, xmin={xmin:.4f}")

    # ── Layout: native igraph FR (no pling edge attrs required) ───────────────
    # hybrid_layout from pling_igraph_viz requires 'td' edge attributes
    # that only exist in pling JSON graphs.  For the generated graph we
    # use igraph's layout_fruchterman_reingold directly.
    print("Computing layout …")
    layout_obj = g.layout_fruchterman_reingold(niter=1500, grid="nogrid")
    coords     = np.array(layout_obj.coords)
    coords[:, 1] = -coords[:, 1]   # flip y: screen coords (y-down)

    # ── Generated network figure ──────────────────────────────────────────────
    print("\nRendering …")
    render_network(g, coords, out_path=out_dir / "generated_network.png",
                   dpi=args.dpi)

    # ── Centrality distributions (no network graphs, distributions only) ──────
    graph_label = (f"Generated  N={args.N}  σ={sigma:.3f}  "
                   f"ε={epsilon:.4f}  seed={args.seed}")

    distributions = [
        (degree_centrality,            "Degree",           "degree",      "raw degree score"),
        (betweenness_centrality,       "Betweenness",      "betweenness", "raw betweenness score"),
        (closeness_centrality,         "Closeness (WF)",   "closeness",   "closeness WF"),
        (eigenvector_centrality,       "Eigenvector",      "eigenvector", "eigenvector score"),
    ]

    for fn, measure_name, fname, x_label in distributions:
        if fn == eigenvector_centrality:
            vals = fn(g, normalized=True)
        else:
            vals = fn(g, normalized=False)
        if fname == "degree":
            power_law_fits["degree"] = fit_power_law(vals, discrete=True)
        elif fname == "betweenness":
            power_law_fits["betweenness"] = fit_power_law(vals)
        elif fname == "closeness":
            power_law_fits["closeness"] = fit_power_law(vals)
        elif fname == "eigenvector":
            power_law_fits["eigenvector"] = fit_power_law(vals)
        render_measure_distribution(
            values        = vals,
            measure_name  = measure_name,
            x_label       = x_label,
            graph_label   = graph_label,
            community_idx = 0,
            out_path      = met_dir / f"{fname}_distribution.png",
            dpi           = args.dpi,
        )

    # ── Combined centrality distribution (1 × 1 log-log panel) ───────────────
    # Collect the same four measure values used above into a dict and pass
    # them to the shared function imported from pling_igraph_analysis.
    combined_vals = {
        "degree":      degree_centrality(g,      normalized=True),
        "betweenness": betweenness_centrality(g, normalized=True),
        "closeness":   closeness_centrality(g,   normalized=True),
        "eigenvector": eigenvector_centrality(g, normalized=True),
    }
    render_combined_centrality_distribution(
        centrality_values = combined_vals,
        graph_label       = graph_label,
        community_idx     = 0,
        out_path          = met_dir / "combined_distribution.png",
        dpi               = args.dpi,
    )

    # Path-length histogram
    apl = average_path_length_metric(g)
    render_path_length_distribution(
        g, graph_label=graph_label, community_idx=0, apl=apl,
        out_path=met_dir / "path_length_distribution.png", dpi=args.dpi,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_stats(g, out_dir / "summary.txt",
                power_law_fits = power_law_fits)

    # ── Export graph ──────────────────────────────────────────────────────────
    # GraphML preserves all vertex/edge attributes and is the richer format.
    # GML is a simpler plain-text format with broader tool support (Gephi,
    # Cytoscape, R igraph, NetworkX).  Both are written for maximum portability.
    g.write_graphml(str(out_dir / "generated_graph.graphml"))
    g.write_gml(str(out_dir / "generated_graph.gml"))
    print(f"  Saved → generated_graph.graphml")
    print(f"  Saved → generated_graph.gml")

    print(f"\nOutputs:")
    print(f"  {out_dir}/generated_network.png")
    print(f"  {out_dir}/duplication_divergence_metrics/  "
          f"({len(list(met_dir.glob('*.png')))} distribution plots)")
    print(f"  {out_dir}/summary.txt")
    print(f"  {out_dir}/generated_graph.graphml")
    print(f"  {out_dir}/generated_graph.gml")


if __name__ == "__main__":
    main()