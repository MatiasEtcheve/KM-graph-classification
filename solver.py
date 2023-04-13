from typing import Callable, List, Optional

import cvxopt
import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        kernel: Callable,
        C: float = 1,
        class_weight: Optional[str] = None,
        verbose: Optional[bool] = False,
    ):
        self.kernel = kernel
        self.C = C
        self.class_weight = class_weight
        self.classes_ = np.array([-1, 1])
        self.n_classes = 2
        self.verbose = verbose

        if not self.verbose:
            cvxopt.solvers.options["show_progress"] = False

    def fit(self, X: List[nx.classes.graph.Graph], y: List, sample_weight=None):
        n_samples = len(X)

        K = self.kernel(X, X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), "d")
        b = cvxopt.matrix(0.0)

        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))

        h = np.zeros(2 * n_samples)
        if sample_weight is None:
            if self.class_weight is None:
                self.sample_weight = np.ones(n_samples)
            elif self.class_weight == "balanced":
                self.sample_weight = np.ones(n_samples)
                positive_mask = y == 1
                self.sample_weight[positive_mask] = n_samples / (
                    2 * np.sum(positive_mask)
                )
                self.sample_weight[~positive_mask] = n_samples / (
                    2 * np.sum(~positive_mask)
                )
        h[n_samples:] = self.C * self.sample_weight
        h = cvxopt.matrix(h)
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution["x"])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-4
        self.ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        if self.verbose:
            print(
                "{0} support vectors out of {1} points".format(len(self.a), n_samples)
            )
        # Intercept
        self.b = np.mean(
            -np.sum(self.a[None, :] * self.sv_y[None, :] * K[sv, :][:, sv], axis=0)
        )

    def decision_function(self, X: List[nx.classes.graph.Graph]):
        K = self.kernel(X, self.sv)
        y = np.sum(self.a[None, :] * self.sv_y[None, :] * K, axis=1)
        return y + self.b

    def predict(self, X: List[nx.classes.graph.Graph]):
        return np.sign(self.decision_function(X))
