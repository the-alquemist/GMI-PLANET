import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import powerlaw


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

    fit = powerlaw.Fit(valid_deg, discrete=True, verbose=False) #fit data

    #plot
    plt.figure(figsize=(8, 6))
    fit.plot_pdf(linear_bins=True, color='r', label='Linear-binned PDF')  #linear bins 
    fit.plot_pdf(color='b', label='Log-binned PDF')                        #log bins
    
    #FIT
    fit.power_law.plot_pdf(color='g', linestyle='--', label='PowerLaw Fit Line')

    #Fformatting
    plt.title(title)
    plt.xlabel("Grado (k)")
    plt.ylabel("Probabilidad P(k)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    #save
    plt.savefig(filename, dpi=300, bbox_inches='tight') 
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

    #all in one DF 
    df_metrics = pd.DataFrame({
        "Plasmid": names,
        "Degree": degrees,
        "Betweenness": betweenness,
        "Closeness": closeness,
        "Eigenvector": eigenvector
    })

    #print 5 highest for each measurement
    print(f"Top 5 nodos Núcleo (Degree) {graph_name}:")
    print(df_metrics.sort_values(by="Degree", ascending=False).head(5)[["Plasmid", "Degree"]].to_string(index=False))

    print(f"Top 5 nodos Núcleo (Eigenvector) {graph_name}:")
    print(df_metrics.sort_values(by="Eigenvector", ascending=False).head(5)[["Plasmid", "Eigenvector"]].to_string(index=False))

    print(f"Top 5 nodos Puente (Betweenness) {graph_name}:")
    print(df_metrics.sort_values(by="Betweenness", ascending=False).head(5)[["Plasmid", "Betweenness"]].to_string(index=False))

    print(f"Top 5 Accesibilidad (Closeness) {graph_name}:")
    print(df_metrics.sort_values(by="Closeness", ascending=False).head(5)[["Plasmid", "Closeness"]].to_string(index=False))

    return df_metrics



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




