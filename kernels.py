import time
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from utils import get_edges_attributes, get_nodes_attributes


class BaseKernel:
    def __init__(self, sigma: Optional[float] = 1):
        self.sigma = sigma
        self.mode = "rbf" if sigma > 0 else "linear"

    def __repr__(self) -> str:
        pretty_mode = "Gaussian" if self.mode == "rbf" else "Linear"
        suffix = "" if self.mode == "linear" else f"(sigma: {self.sigma})"
        return f"{self.type} | Wrapper: {self.mode} {suffix}"

    def compute_K(
        self, phi_a: np.ndarray, phi_b: np.ndarray, normalize: Optional[bool] = False
    ) -> np.ndarray:
        if normalize:
            mask = np.sum(phi_a, axis=1) > 0
            phi_a[mask] = phi_a[mask] / np.sum(phi_a[mask], axis=1)[:, None]

            mask = np.sum(phi_b, axis=1) > 0
            phi_b[mask] = phi_b[mask] / np.sum(phi_b[mask], axis=1)[:, None]

        if self.mode == "linear":
            return phi_a @ phi_b.T

        diff_count_nodes = np.zeros((len(phi_a), len(phi_b)))
        for idx_a in range(len(phi_a)):
            diff_count_nodes[idx_a, :] = (
                -np.linalg.norm(phi_a[idx_a][None] - phi_b, axis=-1) ** 2
            )
        return np.exp(diff_count_nodes / self.sigma)

    def __call__(
        self,
        graphs_a: List[nx.classes.graph.Graph],
        graphs_b: List[nx.classes.graph.Graph],
    ) -> np.ndarray:
        phi_a = self.compute_phi(graphs_a)
        if len(graphs_a) == len(graphs_b) and all(graphs_a == graphs_b):
            phi_b = phi_a.copy()
        ##################
        else:
            phi_b = self.compute_phi(graphs_b)
        ##################
        return self.compute_K(phi_a, phi_b)

    def compute_phi(self, graphs: List[nx.classes.graph.Graph]) -> np.ndarray:
        pass


class EdgeHistKernel(BaseKernel):
    def __init__(self, sigma: Optional[float] = 1) -> None:
        super().__init__(sigma)

        self.type = "Edge Histogram"

    def compute_phi(self, graphs: List[nx.classes.graph.Graph]) -> np.ndarray:
        count_edges = np.zeros((len(graphs), 4))  # count of edge types in every graph
        for index_graph, graph in enumerate(graphs):
            # Retrieve node and edge labels
            edge_labels = get_edges_attributes(graph)

            if len(edge_labels) > 0:
                count_edge_labels = np.unique(edge_labels, return_counts=True)
                count_edges[index_graph, count_edge_labels[0]] += count_edge_labels[1]
        return count_edges


class NodeHistKernel(BaseKernel):
    def __init__(
        self, sigma: Optional[float] = 1, max_nodes: Optional[int] = None
    ) -> None:
        super().__init__(sigma)
        self.max_nodes = max_nodes if max_nodes is not None else 50

        self.type = "Vertex Histogram"

    def compute_phi(self, graphs: List[nx.classes.graph.Graph]) -> np.ndarray:
        count_nodes = np.zeros((len(graphs), 50))  # count of node types in every graph

        for index_graph, graph in enumerate(graphs):
            # Retrieve node and edge labels
            node_labels = get_nodes_attributes(graph)
            count_node_labels = np.unique(node_labels, return_counts=True)
            count_nodes[index_graph, count_node_labels[0]] += count_node_labels[1]

        return count_nodes[:, : self.max_nodes]


class GraphletKernel(BaseKernel):
    def __init__(
        self, sigma: Optional[float] = 1, n_samples: Optional[int] = 100
    ) -> None:
        super().__init__(sigma)
        self.n_samples = n_samples

        self.type = "Graphlet Histogram"

    def compute_phi(self, graphs: List[nx.classes.graph.Graph]) -> np.ndarray:
        graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]

        graphlets[0].add_nodes_from(range(3))

        graphlets[1].add_nodes_from(range(3))
        graphlets[1].add_edge(0, 1)

        graphlets[2].add_nodes_from(range(3))
        graphlets[2].add_edge(0, 1)
        graphlets[2].add_edge(1, 2)

        graphlets[3].add_nodes_from(range(3))
        graphlets[3].add_edge(0, 1)
        graphlets[3].add_edge(1, 2)
        graphlets[3].add_edge(0, 2)
        phi = np.zeros((len(graphs), len(graphlets)))
        for index_graph, graph in enumerate(graphs):
            for _ in range(self.n_samples):
                s = np.random.choice(graph.nodes(), 3)
                subgraph = graph.subgraph(s)
                phi[index_graph] += np.array(
                    [nx.is_isomorphic(g, subgraph) for g in graphlets]
                )
        return phi


class SPKernel(BaseKernel):
    def __init__(
        self, sigma: Optional[float] = 1, max_features: Optional[int] = 35
    ) -> None:
        super().__init__(sigma)
        self.max_features = max_features

        self.type = "Shortest Path Histogram"

    def compute_phi_sp(self, graphs: List[nx.classes.graph.Graph]) -> np.ndarray:
        all_paths = dict()
        sp_counts_train = dict()

        for i, G in enumerate(graphs):
            sp_lengths = dict(nx.shortest_path_length(G))
            sp_counts_train[i] = dict()
            nodes = G.nodes()
            for v1 in nodes:
                for v2 in nodes:
                    if v2 in sp_lengths[v1]:
                        length = sp_lengths[v1][v2]
                        if length in sp_counts_train[i]:
                            sp_counts_train[i][length] += 1
                        else:
                            sp_counts_train[i][length] = 1

                        if length not in all_paths:
                            all_paths[length] = len(all_paths)
        phi = np.zeros((len(graphs), len(all_paths)))
        for i in range(len(graphs)):
            for length in sp_counts_train[i]:
                phi[i, all_paths[length]] = sp_counts_train[i][length]
        return phi


class WLKernel(BaseKernel):
    def __init__(
        self,
        sigma: Optional[float] = 1,
        iterations: Optional[int] = 3,
        max_nodes: Optional[int] = 43,
        normalize: Optional[bool] = True,
        node_attr: Optional[str] = "labels",
        edge_attr: Optional[str] = None,
    ) -> None:
        super().__init__(sigma)
        self.iterations = iterations
        self.max_nodes = max_nodes
        self.normalize = normalize
        self.node_attr = node_attr
        self.edge_attr = edge_attr

        if self.node_attr is not None and self.edge_attr is None:
            attribute = "nodes"
        elif self.edge_attr is not None and self.node_attr is None:
            attribute = "edges"
        else:
            attribute = "edges and nodes"
        self.type = f"Weisfeiler Lehman (on {attribute})"

    def _init_node_labels(self, graph: nx.classes.graph.Graph):
        if self.node_attr:
            return {u: dd[self.node_attr] for u, dd in graph.nodes(data=True)}
        elif self.edge_attr:
            return {u: 0 for u in graph}
        else:
            raise NotImplementedError

    def _neighborhood_aggregate(
        self,
        graph: nx.classes.graph.Graph,
        node: int,
        node_labels: Dict[int, float],
    ):
        """
        Compute new labels for given node by aggregating
        the labels of each node's neighbors.
        """
        label_list = []
        for nbr in graph.neighbors(node):
            prefix = 0 if self.edge_attr is None else graph[node][nbr][self.edge_attr]
            label_list.append(prefix + node_labels[nbr])
        return node_labels[node] + np.sum(label_list)

    def _weisfeiler_lehman_step(self, graph: nx.classes.graph.Graph, labels):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = {}
        for node in graph.nodes():
            label = self._neighborhood_aggregate(graph, node, labels)
            new_labels[node] = label
        return new_labels

    def _weisfeiler_lehman_algo(self, graph: nx.classes.graph.Graph):
        # set initial node labels
        node_labels = self._init_node_labels(graph)
        all_node_labels = [node_labels]
        for _ in range(self.iterations):
            node_labels = self._weisfeiler_lehman_step(graph, node_labels)
            all_node_labels.append(node_labels)
        return all_node_labels

    def compute_phi(
        self,
        graphs: List[nx.classes.graph.Graph],
    ) -> np.ndarray:
        count_nodes = np.zeros(
            (len(graphs), (self.iterations + 1) * self.max_nodes)
        )  # count of node types in every graph

        # for each graph, compute a vector all_node_labels
        # containing the labels of all nodes in the graph during the WL iterations
        # only keep the labels < self.max_nodes. Otherwise, the labels increases exponentially
        for index_graph, graph in enumerate(graphs):
            if not isinstance(graphs[0], nx.classes.graph.Graph):
                graph = graph[0]
            all_node_labels = self._weisfeiler_lehman_algo(
                graph,
            )
            for idx_iteration, node_labels in enumerate(all_node_labels):
                unique_values, count_values = np.unique(
                    list(node_labels.values()), return_counts=True
                )
                mask = unique_values < self.max_nodes
                unique_values = unique_values[mask]
                count_values = count_values[mask]
                count_nodes[
                    index_graph,
                    unique_values.astype(int) + self.max_nodes * idx_iteration,
                ] += count_values
        if self.normalize:
            count_nodes = (
                count_nodes
                / np.array(
                    [
                        len(g[0].nodes)
                        if not isinstance(g, nx.classes.graph.Graph)
                        else len(g.nodes)
                        for g in graphs
                    ]
                )[:, None]
            )
        return count_nodes
