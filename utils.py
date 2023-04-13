from typing import List

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes

node_cmap = [
    mcolors.rgb2hex(mpl.colormaps["gist_rainbow"](i))
    for i in np.linspace(0, 256, 50, dtype="int")
]
edge_cmap = mpl.colormaps["Set1"]


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


def plot_graph(graph: nx.classes.graph.Graph, ax: Axes) -> None:
    graph = graphs[graph_index]
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
