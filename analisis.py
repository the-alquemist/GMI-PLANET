import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import powerlaw
import math


#-- ROUTES --------------------------------------------------------------------
distances_route = "raw_pling_example_output/all_plasmids_distances.tsv"
communities_route = "raw_pling_example_output/dcj_thresh_4_graph/objects/typing.tsv"
hubs_route = "raw_pling_example_output/dcj_thresh_4_graph/objects/hub_plasmids.csv"

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Data loading from all_plasmids_distances.tsv, typing.tsv and hub_plasmids.csv files

    Returns pandas DataFrames for distances and communities, list for hub plasmids """
    
    print("Iniciando carga de datos ...")

    #loading and column renaming
    #DISTANCES
    df_distances = pd.read_csv(distances_route, sep="\t")
    df_distances.columns = ["source", "target", "weight"]

    #COMMUNITIES
    df_communities = pd.read_csv(communities_route, sep="\t")
    df_communities.columns = ["plasmid", "community"]

    #HUBS
    df_hubs = pd.read_csv(hubs_route)
    hubs_list = df_hubs.iloc[:,0].tolist()

    print(f"Aristas: {len(df_distances)} | Comunidades: {len(df_communities)} | Hubs: {len(df_hubs)}")

    return df_distances, df_communities, hubs_list



#-- GRAPH BUILDING ------------------------------------------------------------
def build_graphs(df_dist: pd.DataFrame,
                 df_com: pd.DataFrame,
                 hub_list: list[str]) -> tuple[ig.Graph, ig.Graph]:
    """Graph building with output from load_data()
    
    Returns comlpete graph and filtered graph (without highly connected nodes)"""
    
    print("Construyendo grafos ...")

    edge_tuples = df_dist.itertuples(index=False, name=None)
    graph = ig.Graph.TupleList(edge_tuples, directed=False, edge_attrs=["weight"])

    #Communities DF to Dictionary
    communities_dict = pd.Series(df_com.community.values, index=df_com.plasmid).to_dict()

    for vertex in graph.vs:
        vertex["community"] = communities_dict.get(vertex["name"], "Not Found")

    filtered_graph = graph.copy()

    #find hubs in graph according to list 
    hub_index_list = [v.index for v in filtered_graph.vs if v["name"] in hub_list]

    filtered_graph.delete_vertices(hub_index_list)

    print(f"Grafo -> Nodos: {graph.vcount()} | Edges: {graph.ecount()}")
    print(f"Grafo Filtrado -> Nodos: {filtered_graph.vcount()} | Edges: {filtered_graph.ecount()}")

    return graph, filtered_graph



#-- DEGREE DISTR --------------------------------------------------------------
def degrees_out(graph: ig.Graph) -> tuple[list[int], dict[str, list[int]]]:
    """Extracts global degrees and degrees grouped by community from a given graph
    
    Returns a list of global degrees and a dictionary for degrees by community"""
    global_degrees = graph.degree() #global list

    #dict with classification by community
    community_degrees = {}

    for vertex in graph.vs:
        comm = vertex["community"]
        deg = graph.degree(vertex.index)

        if comm not in community_degrees:
            community_degrees[comm] = []

        community_degrees[comm].append(deg)

    return global_degrees, community_degrees



#-- POWERLAW ------------------------------------------------------------------
def powerlaw_plot(degrees: list[int],
                  title: str,
                  filename: str) -> None:
    """Fits degree distribution to powerlaw and saves a plot"""

    print(f"  ->  Generando Powerlaw Fit: {title}")

    #defree = 0 gets filtered out
    valid_deg = [d for d in degrees if d > 0]

    if len(valid_deg) < 2:
        print("No hay suficientes datos para ajuste")
        return
    
    plt.figure(figsize=(8, 6))

    fit = powerlaw.Fit(valid_deg, discrete=True) #fit data

    #plot
    fit.plot_pdf(color='red', linear_bins=True, marker='o', linestyle='', alpha=0.7, label='Empírico (Linear Bins)')  #lineal
    
    fit.plot_pdf(color='blue', marker='x', linestyle='', alpha=0.7, label='Empírico (Log Bins)')                      #log
    
    #FIT
    fit.power_law.plot_pdf(color='black', linestyle='--', label='PowerLaw Fit Line')

    #Fformatting
    plt.title(title, fontweight='bold')
    plt.xlabel("Grado (k)")
    plt.ylabel("Probabilidad P(k)")

    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, which="both", ls='--', alpha=0.5)
    plt.legend(loc='best')

    #save
    plt.tight_layout()
    plt.savefig(filename, dpi=300) 
    plt.close()



#-- CENTRALITIES --------------------------------------------------------------
def calc_centralities(graph: ig.Graph,
                      graph_name: str) -> pd.DataFrame:
    """Calculate centralities (Degree, Eigenvector, Betwenness, Closeness)
    
    Returns a DF with plasmids and each centrality asociated"""

    print(f"----- Calculando centralidades para: {graph_name} -----")

    names = graph.vs["name"]
    degrees = graph.degree()

    betweenness = graph.betweenness(directed=False)
    closeness = graph.closeness()
    eigenvector = graph.eigenvector_centrality(directed=False)

    clustering_local = graph.transitivity_local_undirected()
    clustering_local_clean = [0 if math.isnan(node) else node for node in clustering_local]


    #all in one DF 
    df_metrics = pd.DataFrame({
        "Plasmid": names,
        "Degree": degrees,
        "Betweenness": betweenness,
        "Closeness": closeness,
        "Eigenvector": eigenvector,
        "Clustering Local": clustering_local_clean
    })

    clustering_global = graph.transitivity_undirected()
    avg_path = graph.average_path_length(directed=False, unconn=True)

    #print 5 highest for each measurement
    print(f"Top 5 nodos Núcleo (Degree) {graph_name}:")
    print(df_metrics.sort_values(by="Degree", ascending=False).head(5)[["Plasmid", "Degree"]].to_string(index=False))

    print(f"Top 5 nodos Núcleo (Eigenvector) {graph_name}:")
    print(df_metrics.sort_values(by="Eigenvector", ascending=False).head(5)[["Plasmid", "Eigenvector"]].to_string(index=False))

    print(f"Top 5 nodos Puente (Betweenness) {graph_name}:")
    print(df_metrics.sort_values(by="Betweenness", ascending=False).head(5)[["Plasmid", "Betweenness"]].to_string(index=False))

    print(f"Top 5 Accesibilidad (Closeness) {graph_name}:")
    print(df_metrics.sort_values(by="Closeness", ascending=False).head(5)[["Plasmid", "Closeness"]].to_string(index=False))

    print(f"Clustering Global: {clustering_global} | Avg Path Length: {avg_path}")

    return df_metrics


#-- TOPOLOGICAL VIS ----------------------------------------------------
def plot_top_centrality(graph: ig.Graph,
                        df: pd.DataFrame,
                        metric: str,
                        filename: str,
                        top: int = 5) -> None:
    """Draws network highlighting nodes according to the metric value associated"""

    if df.empty or metric not in df.columns:
        print(f"No se puede graficar {metric}, no se encontraron datos")
        return
    
    plt.figure(figsize=(10, 10))

    #layout with fruchterman reignold
    layout = graph.layout_fruchterman_reingold()
    coords = np.array(layout.coords)

    #extracting values 
    values = df[metric].values
    #extract plasmid names from df
    names = df["Plasmid"].values if "Plasmid" in df.columns else df.iloc[:, 0].values

    v_min, v_max = min(values), max(values)
    rango = v_max - v_min if v_max != v_min else 1
    values_normalized = (values - v_min) / rango #for [0, 1]

    for arista in graph.es:
        source, target = arista.tuple
        x = [coords[source, 0], coords[target, 0]]
        y = [coords[source, 1], coords[target, 1]]
        plt.plot(x, y, color="#E0E0E0", linewidth=0.8, zorder=1)

    colours = plt.cm.plasma(values_normalized)
    sizes = 60 + (400 * values_normalized)

    plt.scatter(coords[:, 0], coords[:, 1], s=sizes, c=colours, edgecolors='white',
                linewidths=0.5, zorder=2)
    
    top_indexes = np.argsort(values)[-top:]

    for idx in top_indexes:
        plt.text(coords[idx, 0], coords[idx, 1], names[idx], 
                 fontsize=9, ha='center', va='bottom', fontweight='bold', zorder=3,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
        
    #final visualization
    plt.title(f"Distribución Topológica - {metric}", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='white')
    plt.close()






#-- HISTOGRAMS ----------------------------------------------------------------
def plot_hist(df_full: pd.DataFrame,
              df_filtered: pd.DataFrame,
              metric: str,
              title: str,
              filename: str) -> None:
    """Generates comparative histogram for a specific metric between two graphs and it saves
    image"""

    print(f"  ->  Generando histograma: {metric}")

    plt.figure(figsize=(8, 6))

    plt.hist(df_full[metric], bins=30, alpha=0.5, color='blue', label='Grafo Completo')
    if not df_filtered.empty:
        plt.hist(df_filtered[metric], bins=30, alpha=0.5, color='red', label='Gráfico Filtrado (sin Hubs)')

    plt.title(title)
    plt.xlabel(f"{metric}")
    plt.ylabel("Frecuencia (Cant. de plásmidos)")
    #plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    #save
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def gen_all_histograms(df_full: pd.DataFrame,
                       df_filtered: pd.DataFrame) -> None:
    """Runs plot_hist for all comparisons needed"""
    print("----- Generando Histogramas Comparativos")
    plot_hist(df_full, df_filtered, "Betweenness", "Comparación de Betweenness", "output_analisis/hist_betweenness.png")
    plot_hist(df_full, df_filtered, "Closeness", "Comparación de Closeness", "output_analisis/hist_closeness.png")
    plot_hist(df_full, df_filtered, "Eigenvector", "Comparación de Eigenvector", "output_analisis/hist_eigenvector.png")

    print("Histogramas guardados como PNG")



def plot_path_len_hist(graph: ig.Graph,
                       title: str,
                       filename: str) -> None:
    """Calculates and plots histogram for distances of minimal routes between pairs
    of nodes"""
    print(f"   ->  Generando Histograma de Rutas: {title}")

    #distances matrix
    dist_matrix = np.array(graph.distances())

    #get upper triangle avoiding repetition of pairs, k=1 doesn't include diagonal
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

    #filter infinite nodes (not conected)
    valid_paths = upper_tri[np.isfinite(upper_tri)]

    if len(valid_paths) == 0:
        print("No hay rutas para graficar")
        return
    
    plt.figure(figsize=(8, 6))

    max_dist = int(np.max(valid_paths))
    bins = np.arange(1, max_dist + 2) - 0.5

    plt.hist(valid_paths, bins=bins, color="#2ca02c", edgecolor="black", alpha=0.7)

    #calculate avg
    avg_path = np.mean(valid_paths)
    plt.axvline(avg_path, color='red', linestyle='dashed', linewidth=2, 
                label=f'Promedio: {avg_path:.2f} saltos')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Largo de Ruta (saltos)", fontsize=12)
    plt.ylabel("Frecuencia (Pares de plásmidos)", fontsize=12)
    plt.xticks(range(1, max_dist + 1)) 
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='white')
    plt.close()




if __name__ == "__main__":
    df_distances, df_communities, hub_list = load_data()
    full_graph, filtered_graph = build_graphs(df_distances, df_communities, hub_list)

    #Extract degrees for full graph
    print("----- Analizando grafo completo -----")
    full_global_deg, full_comm_deg = degrees_out(full_graph)
    print(f"Grado máximo: {max(full_global_deg)}")
    powerlaw_plot(full_global_deg, "Distribución de grado (grafo completo)", "output_analisis/powerlaw_full.png")

    #Extract degrees for filtered graph
    print("----- Analizando grafo filtrado -----")
    filtered_global_deg, filtered_comm_deg = degrees_out(filtered_graph)
    if len(filtered_global_deg) > 0:
        print(f"Grado máximo: {max(filtered_global_deg)}")
        powerlaw_plot(filtered_global_deg, "Distribucion de grado (sin hubs)", "output_analisis/powerlaw_filtered.png")
    else:
        print("Grafo vacío")

    print("Gráficos de PowerLaw guardados como PNG")

    df_full_metrics = calc_centralities(full_graph, "Grafo Completo")

    if filtered_graph.vcount() > 0:
        df_filtered_metrics = calc_centralities(filtered_graph, "Grafo Filtrado (sin Hubs)")
        gen_all_histograms(df_full_metrics, df_filtered_metrics)
    else:
        gen_all_histograms(df_full_metrics, pd.DataFrame())

    print("Generando redes topológicas ...")
    plot_top_centrality(full_graph, df_full_metrics, "Betweenness", 
                        "output_analisis/topologia_betweenness_full.png", top=5)
    
    plot_top_centrality(full_graph, df_full_metrics, "Degree", 
                        "output_analisis/topologia_degree_full.png", top=5)
    
    if not df_filtered_metrics.empty:
        plot_top_centrality(filtered_graph, df_filtered_metrics, "Betweenness",
                            "output_analisis/topologia_betweenness_filtered.png", top=5)
        
    print("Generando histogramas de rutas (Avg. Path Length) ...")

    plot_path_len_hist(full_graph, "Distribucion de Rutas (Completo)",
                       "output_analisis/path_length_full.png")
    
    if filtered_graph.vcount() > 0:
        plot_path_len_hist(filtered_graph, "Distribucion de rutas (sin hubs)",
                           "output_analisis/path_length_filtered.png")
    






