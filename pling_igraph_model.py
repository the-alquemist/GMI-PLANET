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

Parameter optimisation
----------------------
σ (edge retention) primarily controls graph density: higher σ → more edges
inherited → denser graph → lower mean degree heterogeneity.  ε (novel-edge
probability) controls the fraction of cross-community "random" edges.

Optimising σ or ε to match a target power-law exponent α* is problematic
because α is nearly constant across a wide σ range (≈2.6–2.9) in this
model at N=200, with high stochastic variance between seeds.  The fitted α
from powerlaw.Fit is an unreliable optimisation target at these graph sizes.

Instead, optimisation targets the MEAN DEGREE of the generated graph, which
is a monotone, low-variance function of σ and ε and allows reliable
convergence using scipy.optimize.minimize_scalar (Brent's method):

    σ* = argmin |mean_degree(G_σ) − target_mean_degree|

This is the principled approach: σ controls density, density controls
connectivity, and the power-law tail shape is an emergent property that
cannot be reliably steered by a single scalar at small N.  For larger N
(≥1000) the α–σ relationship becomes more reliable and direct α fitting
can be reintroduced.

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

    # Optimise sigma to match a target mean degree (e.g. 7)
    python3 pling_igraph_model.py --optimise sigma --target-degree 7

    # Optimise epsilon to match a target mean degree
    python3 pling_igraph_model.py --optimise epsilon --target-degree 5

Dependencies
------------
    pip install igraph matplotlib numpy scipy powerlaw
    pling_igraph_analysis.py must be importable from the same directory.
"""

import argparse
import math
import random
import warnings
from datetime import datetime
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
import numpy as np

try:
    import powerlaw as pl_pkg
    _HAS_POWERLAW = True
except ImportError:
    _HAS_POWERLAW = False

try:
    from scipy.optimize import minimize_scalar
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

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
    p.add_argument("--degree-bias",  action="store_true", default=True,
                   help="Select source node ∝ degree (default True)")
    p.add_argument("--no-degree-bias", dest="degree_bias",
                   action="store_false")
    p.add_argument("--optimise",     choices=["sigma", "epsilon"],
                   default=None,
                   help="Parameter to auto-tune to match --target-degree")
    p.add_argument("--target-degree", type=float, default=6.0,
                   help="Target mean degree for optimisation (default 6.0)")
    p.add_argument("--opt-reps",     type=int, default=5,
                   help="Graph replicates per optimisation evaluation "
                        "(more = less noise, slower; default 5)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--out-dir",      required=True,
                   help="Output directory for generated network and metrics")
    p.add_argument("--dpi",          type=int,   default=200)
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

def fit_power_law_alpha(degrees: list[int]) -> float | None:
    """Fit a discrete power-law to the positive degree values.

    Uses powerlaw.Fit with discrete=True (Clauset et al. 2009).
    Returns None if powerlaw is unavailable or fewer than 10 data points.

    IMPORTANT CAVEAT (documented in summary):
    At N=200 the fitted α is nearly constant across σ values (≈2.6–2.9)
    with high stochastic variance between seeds.  α should be interpreted
    as a rough descriptor of the tail, not a precise parameter estimate.
    """
    if not _HAS_POWERLAW:
        return None
    data = [d for d in degrees if d > 0]
    if len(data) < 10:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import logging
            logging.disable(logging.CRITICAL)
            fit = pl_pkg.Fit(data, discrete=True, verbose=False)
            logging.disable(logging.NOTSET)
        return float(fit.power_law.alpha)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Parameter optimisation — targets mean degree, not alpha
# ─────────────────────────────────────────────────────────────────────────────

def optimise_parameter(param: str,
                       target_degree: float,
                       N: int, n0: int, p0: float,
                       sigma: float, p: float, epsilon: float,
                       degree_bias: bool, seed: int,
                       n_reps: int = 5) -> tuple[float, float]:
    """Find the σ or ε value that minimises |mean_degree − target_degree|.

    WHY MEAN DEGREE, NOT ALPHA?
    ---------------------------
    The fitted power-law exponent α is nearly constant across the σ range
    [0.2, 0.8] in this model at N=200 (empirically ≈2.6–2.9 with large
    seed-to-seed variance).  Using α as an optimisation target would make
    the minimiser converge to a random value — the signal is too noisy.

    Mean degree, by contrast, is a monotone, low-variance function of σ:
      - σ↑ → more edges inherited → higher mean degree
      - σ↓ → fewer edges retained → lower mean degree
      - ε↑ → more novel edges     → higher mean degree

    This makes mean degree a reliable, well-conditioned optimisation target.

    Strategy
    --------
    For each candidate parameter value, generate n_reps independent graphs
    (different seeds) and average their mean degrees.  Averaging reduces
    stochastic noise before the optimiser (scipy minimize_scalar, Brent's
    method) moves to the next candidate.  Brent's method requires the
    objective to be unimodal in the search interval — mean degree is
    monotone in σ and ε, satisfying this requirement.

    Parameters
    ----------
    param         : "sigma" or "epsilon"
    target_degree : desired mean degree of the generated graph
    n_reps        : number of independent replications per evaluation
                    (increase for less noise, at the cost of speed)

    Returns
    -------
    (best_val, achieved_degree)
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for optimisation.")

    bounds = {"sigma": (0.05, 0.95), "epsilon": (0.001, 0.15)}[param]

    def objective(val: float) -> float:
        mean_degs = []
        for rep in range(n_reps):
            kw = dict(sigma=sigma, epsilon=epsilon)
            kw[param] = val
            g_tmp = generate_network(
                N, n0, p0,
                sigma=kw["sigma"], p=p, epsilon=kw["epsilon"],
                degree_bias=degree_bias,
                seed=seed + rep * 17,   # deterministic but spread seeds
                verbose=False,
            )
            mean_degs.append(np.mean(g_tmp.degree()))
        return abs(np.mean(mean_degs) - target_degree)

    print(f"\n  Optimising {param} to achieve mean degree ≈ {target_degree:.1f}")
    print(f"  Search bounds: {bounds}  |  {n_reps} reps per evaluation")
    result    = minimize_scalar(objective, bounds=bounds, method="bounded",
                                options={"xatol": 1e-3, "maxiter": 40})
    best_val  = float(result.x)

    # Estimate achieved mean degree at the optimum (5 final reps)
    final_degs = []
    for rep in range(5):
        kw = dict(sigma=sigma, epsilon=epsilon)
        kw[param] = best_val
        g_tmp = generate_network(
            N, n0, p0,
            sigma=kw["sigma"], p=p, epsilon=kw["epsilon"],
            degree_bias=degree_bias,
            seed=seed + 1000 + rep,
            verbose=False,
        )
        final_degs.append(np.mean(g_tmp.degree()))
    achieved = float(np.mean(final_degs))

    print(f"  Optimal {param} = {best_val:.4f}  →  "
          f"mean degree = {achieved:.2f}  (target = {target_degree:.1f})")
    return best_val, achieved


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
                alpha_fitted: float | None,
                optimised_param: str | None,
                target_degree: float | None,
                achieved_degree: float | None) -> None:
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

        if optimised_param:
            fh.write(f"\n  Auto-tuned parameter    : {optimised_param}\n")
            fh.write(f"  Target mean degree      : {target_degree:.2f}\n")
            fh.write(f"  Achieved mean degree    : {achieved_degree:.2f}\n")

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
        if alpha_fitted:
            fh.write(f"  Fitted α    : {alpha_fitted:.4f}\n")
            fh.write("  Note: α is an emergent property, not a direct\n")
            fh.write("  function of σ/ε at N=200 — see docstring.\n")
        else:
            fh.write("  Fitted α    : (powerlaw package unavailable)\n")

        fh.write("\n── Topology ────────────────────────────────────────────────\n")
        fh.write(f"  Global clustering (transitivity) : "
                 f"{global_clustering_coefficient(g):.5f}\n")
        fh.write(f"  Mean local clustering coeff.     : "
                 f"{np.mean(cc_local):.5f}\n")
        fh.write(f"  Average path length              : {apl:.4f}\n")

        fh.write("── Optimisation note ───────────────────────────────────────\n")
        fh.write(
            "  σ controls graph density (mean degree) reliably:\n"
            "    σ↑ → more edges inherited → higher mean degree\n"
            "    σ↓ → fewer edges retained → lower mean degree\n"
            "  ε adds random novel edges, also increasing mean degree.\n\n"
            "  The fitted power-law exponent α ≈ 2.6–2.9 across the full\n"
            "  σ range at N=200, with high seed-to-seed variance.  To\n"
            "  study α–σ relationships reliably, use N ≥ 1000.\n"
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
    achieved_degree = None

    # ── Parameter optimisation ────────────────────────────────────────────────
    if args.optimise:
        if not _HAS_SCIPY:
            print("[error] scipy is required for --optimise. Skipping.")
        else:
            best_val, achieved_degree = optimise_parameter(
                param=args.optimise, target_degree=args.target_degree,
                N=args.N, n0=args.n0, p0=args.p0,
                sigma=sigma, p=args.p, epsilon=epsilon,
                degree_bias=args.degree_bias, seed=args.seed,
                n_reps=args.opt_reps,
            )
            if args.optimise == "sigma":
                sigma   = best_val
            else:
                epsilon = best_val

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\nGenerating network  "
          f"N={args.N}  σ={sigma:.4f}  p={args.p}  ε={epsilon:.5f} …")
    g = generate_network(
        N=args.N, n0=args.n0, p0=args.p0,
        sigma=sigma, p=args.p, epsilon=epsilon,
        degree_bias=args.degree_bias, seed=args.seed,
    )

    alpha_fitted = fit_power_law_alpha(g.degree())
    if alpha_fitted:
        print(f"  Fitted power-law α = {alpha_fitted:.4f}")

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
        render_measure_distribution(
            values        = vals,
            measure_name  = measure_name,
            x_label       = x_label,
            graph_label   = graph_label,
            community_idx = 0,
            out_path      = met_dir / f"{fname}_distribution.png",
            dpi           = args.dpi,
        )

    # Path-length histogram
    apl = average_path_length_metric(g)
    render_path_length_distribution(
        g, graph_label=graph_label, community_idx=0, apl=apl,
        out_path=met_dir / "path_length_distribution.png", dpi=args.dpi,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_stats(g, out_dir / "summary.txt",
                alpha_fitted    = alpha_fitted,
                optimised_param = args.optimise,
                target_degree   = args.target_degree if args.optimise else None,
                achieved_degree = achieved_degree)

    g.write_graphml(str(out_dir / "generated_graph.graphml"))
    print(f"  Saved → generated_graph.graphml")

    print(f"\nOutputs:")
    print(f"  {out_dir}/generated_network.png")
    print(f"  {out_dir}/duplication_divergence_metrics/  "
          f"({len(list(met_dir.glob('*.png')))} distribution plots)")
    print(f"  {out_dir}/summary.txt")
    print(f"  {out_dir}/generated_graph.graphml")


if __name__ == "__main__":
    main()