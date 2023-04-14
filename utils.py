from typing import List

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from tqdm import tqdm

node_cmap = [
    mcolors.rgb2hex(mpl.colormaps["gist_rainbow"](i))
    for i in np.linspace(0, 256, 50, dtype="int")
]
edge_cmap = mpl.colormaps["Set1"]

COLORS = [
    [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
    [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
    [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
    [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
    [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
    [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
    [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
    [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
    [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
    [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0],
]


def get_nodes_attributes(graph: nx.classes.graph.Graph) -> np.ndarray:
    nodes = graph.nodes(data="labels")
    return np.stack(list(dict(nodes).values()), axis=0)


def get_edges_attributes(graph: nx.classes.graph.Graph) -> np.ndarray:
    edges = graph.edges(data="labels")
    return np.array([label for _, _, label in edges])


def relabel(graphs: List[nx.classes.graph.Graph]) -> List[nx.classes.graph.Graph]:
    for graph in graphs:
        for i in range(len(graph)):
            graph.nodes[i]["labels"] = graph.nodes[i]["labels"][0]
        for u, v in graph.edges():
            graph.edges[u, v]["labels"] = graph.edges[u, v]["labels"][0]


def compute_statistics(graphs):
    statistics = (
        []
    )  # main statistics per graph: nb of edges, nodes, max degree, min degree, number of connected components...

    count_nodes = np.zeros(50)  # count of node types in every graph
    count_edges = np.zeros(4)  # count of edge types in every graph

    for index_graph, graph in enumerate(tqdm(graphs)):
        # Retrieve node and edge labels
        node_labels = get_nodes_attributes(graph)
        edge_labels = get_edges_attributes(graph)

        count_node_labels = np.unique(node_labels, return_counts=True)
        count_nodes[count_node_labels[0]] += count_node_labels[1]

        if len(edge_labels) > 0:
            count_edge_labels = np.unique(edge_labels, return_counts=True)
            count_edges[count_edge_labels[0]] += count_edge_labels[1]

        degree_sequence = [graph.degree(node) for node in graph.nodes()]

        statistics.append(
            {
                "Number of nodes": len(node_labels),
                "Number of edges": len(edge_labels),
                "Number of connected components": len(
                    list(nx.connected_components(graph))
                ),
                "Average degree": np.mean(degree_sequence),
                "Maximum degree": np.max(degree_sequence),
                "Main atom type in graph": count_node_labels[0][
                    count_node_labels[1].argmax()
                ],
                "Main edge type in graph": count_edge_labels[0][
                    count_edge_labels[1].argmax()
                ]
                if len(edge_labels) > 0
                else -1,
            }
        )

    statistics = pd.DataFrame.from_records(statistics)
    return statistics, count_nodes, count_edges


def plot_graph(graph: nx.classes.graph.Graph, ax: Axes) -> None:
    nodes = graph.nodes(data="labels")
    edges = graph.edges(data="labels")

    node_labels = np.stack(list(dict(nodes).values()), axis=0)
    edge_labels = np.array([label for _, _, label in edges])

    nx.draw(
        graph,
        node_color=[node_cmap[i] for i in node_labels],
        cmap=node_cmap,
        edge_color=edge_labels,
        edge_cmap=edge_cmap,
        node_size=30,
        ax=ax,
    )
    ax.axis("on")
