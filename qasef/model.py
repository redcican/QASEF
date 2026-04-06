"""QASEF: Quality-Aware Spectral Embedding Fusion for Robust Multi-View Clustering.

Implements Algorithm 1 from the paper with four alternating optimization steps:
  Y-update: fast coordinate descent (FCD)
  G-update: Procrustes SVD (Theorem 4.1)
  Q-update: closed-form coordinate descent (Theorem 4.2)
  alpha-update: simplex QP via augmented Lagrangian method
"""

import numpy as np
from sklearn.cluster import KMeans

from .anchor import compute_all_embeddings


class QASEF:
    """Quality-Aware Spectral Embedding Fusion.

    Parameters
    ----------
    n_clusters : int
        Number of clusters c.
    n_anchors : int or None
        Number of anchors m per view. If None, defaults to c + 50.
    k_anchors : int
        Number of nearest anchors per sample for bipartite graph.
    lam : float
        Regularization parameter lambda controlling quality score trust level.
    max_iter : int
        Maximum number of alternating optimization iterations.
    tol : float
        Convergence tolerance on relative objective change.
    quality_mode : str
        One of 'full' (default), 'none' (w/o Q), 'no_reg' (lambda=0),
        'fixed' (freeze Q after initialization), 'similarity' (Sim-Q).
    random_state : int
        Random seed.
    verbose : bool
        Print convergence information.
    """

    def __init__(self, n_clusters: int, n_anchors: int | None = None,
                 k_anchors: int = 5, lam: float = 1.0,
                 max_iter: int = 100, tol: float = 1e-5,
                 quality_mode: str = "full", random_state: int = 0,
                 verbose: bool = False):
        self.n_clusters = n_clusters
        self.n_anchors = n_anchors if n_anchors is not None else n_clusters + 50
        self.k_anchors = k_anchors
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.quality_mode = quality_mode
        self.random_state = random_state
        self.verbose = verbose

        # Results
        self.labels_ = None
        self.Q_ = None
        self.alpha_ = None
        self.G_ = None
        self.objectives_ = []

    def fit(self, views: list[np.ndarray],
            M: np.ndarray | None = None) -> "QASEF":
        """Fit QASEF to multi-view data.

        Parameters
        ----------
        views : list of np.ndarray
            List of V data matrices, each of shape (n, d_v).
        M : np.ndarray of shape (n, V), optional
            Availability mask. If None, all views are assumed available.

        Returns
        -------
        self
        """
        n = views[0].shape[0]
        V = len(views)
        c = self.n_clusters

        if M is None:
            M = np.ones((n, V), dtype=np.float64)

        # --- Spectral embedding extraction ---
        embeddings = compute_all_embeddings(
            views, M, c, self.n_anchors, self.k_anchors,
            random_state=self.random_state
        )

        # --- Initialize variables ---
        alpha = np.ones(V) / V
        Q = M.copy().astype(np.float64)
        G = np.eye(c, dtype=np.float64)

        # Initialize Y via k-means on weighted average embedding
        F_avg = sum(alpha[v] * embeddings[v] for v in range(V))
        km = KMeans(n_clusters=c, n_init=10, random_state=self.random_state)
        km.fit(F_avg)
        labels = km.labels_
        Y = self._labels_to_indicator(labels, c)

        # Quality mode adjustments
        lam = self.lam
        if self.quality_mode == "none":
            Q = M.copy().astype(np.float64)
        elif self.quality_mode == "no_reg":
            lam = 0.0

        # Precompute aligned embeddings: FG[v] = F^(v) @ G, shape (n, c)
        FG = [embeddings[v] @ G for v in range(V)]

        self.objectives_ = []

        for iteration in range(self.max_iter):
            # --- Y-update: FCD (Eq. 15) ---
            W = self._compute_W(FG, Q, alpha, V, n, c)
            Y, labels = self._update_Y(W, Y, c)
            U = self._compute_U(Y)

            # --- G-update: Procrustes SVD (Theorem 4.1) ---
            Z = self._compute_Z(embeddings, Q, alpha, V, n, c)
            G = self._update_G(Z, U)
            FG = [embeddings[v] @ G for v in range(V)]

            # --- Q-update: coordinate descent (Theorem 4.2) ---
            if self.quality_mode == "full":
                Q = self._update_Q(FG, U, alpha, M, lam, n, V, c)
            elif self.quality_mode == "no_reg":
                Q = self._update_Q(FG, U, alpha, M, 0.0, n, V, c)
            elif self.quality_mode == "similarity":
                Q = self._update_Q_similarity(embeddings, M, n, V, c)
            # 'none' and 'fixed': Q stays unchanged

            # --- alpha-update: simplex QP via ALM (Eq. 19) ---
            alpha = self._update_alpha(FG, Q, U, V, n, c)

            # --- Objective ---
            obj = self._objective(FG, Q, alpha, U, M, lam, V, n, c)
            self.objectives_.append(obj)

            if self.verbose:
                print(f"Iter {iteration + 1:3d}: obj = {obj:.6f}")

            if iteration > 0:
                rel_change = abs(self.objectives_[-1] - self.objectives_[-2])
                if self.objectives_[-2] != 0:
                    rel_change /= abs(self.objectives_[-2])
                if rel_change < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break

        self.labels_ = labels
        self.Q_ = Q
        self.alpha_ = alpha
        self.G_ = G
        return self

    @staticmethod
    def _labels_to_indicator(labels: np.ndarray, c: int) -> np.ndarray:
        """Convert label vector to indicator matrix Y."""
        n = labels.shape[0]
        Y = np.zeros((n, c), dtype=np.float64)
        Y[np.arange(n), labels] = 1.0
        return Y

    @staticmethod
    def _compute_U(Y: np.ndarray) -> np.ndarray:
        """Compute U = Y (Y^T Y)^{-1/2}."""
        # Y^T Y is diagonal with cluster sizes
        cluster_sizes = Y.sum(axis=0)  # (c,)
        # Avoid division by zero
        scale = np.where(cluster_sizes > 0, 1.0 / np.sqrt(cluster_sizes), 0.0)
        return Y * scale[np.newaxis, :]

    @staticmethod
    def _compute_W(FG: list[np.ndarray], Q: np.ndarray, alpha: np.ndarray,
                   V: int, n: int, c: int) -> np.ndarray:
        """Compute quality-weighted consensus W = sum_v alpha^(v) diag(q^(v)) F^(v) G."""
        W = np.zeros((n, c), dtype=np.float64)
        for v in range(V):
            W += alpha[v] * (Q[:, v:v+1] * FG[v])
        return W

    @staticmethod
    def _compute_Z(embeddings: list[np.ndarray], Q: np.ndarray,
                   alpha: np.ndarray, V: int, n: int, c: int) -> np.ndarray:
        """Compute Z = sum_v alpha^(v) diag(q^(v)) F^(v) (before G rotation)."""
        Z = np.zeros((n, c), dtype=np.float64)
        for v in range(V):
            Z += alpha[v] * (Q[:, v:v+1] * embeddings[v])
        return Z

    def _update_Y(self, W: np.ndarray, Y: np.ndarray, c: int):
        """Fast coordinate descent for Y-update (Eq. 15).

        Maximize H(Y) = sum_l (w_l^T y_l) / sqrt(y_l^T y_l).
        """
        n = W.shape[0]
        labels = np.argmax(Y, axis=1)
        cluster_sizes = Y.sum(axis=0).copy()
        # Column sums of W weighted by Y membership
        col_sums = W.T @ Y  # (c, c): col_sums[l, :] but we need w_l^T y_l

        # Actually, H(Y) = sum_l w_l^T y_l / sqrt(n_l)
        # where w_l = W[:, l], y_l = Y[:, l], n_l = y_l^T y_l = cluster_size[l]
        # w_l^T y_l = sum_{i: labels[i]=l} W[i, l]
        w_dot_y = np.array([W[labels == l, l].sum() for l in range(c)])

        for i in range(n):
            p = labels[i]
            if cluster_sizes[p] <= 1:
                continue

            # Current contribution from sample i being in cluster p
            # Try moving to each other cluster q
            best_gain = 0.0
            best_q = p

            old_n_p = cluster_sizes[p]
            old_wy_p = w_dot_y[p]

            for q in range(c):
                if q == p:
                    continue

                # Remove i from p
                new_n_p = old_n_p - 1
                new_wy_p = old_wy_p - W[i, p]

                # Add i to q
                new_n_q = cluster_sizes[q] + 1
                new_wy_q = w_dot_y[q] + W[i, q]

                # Objective change for clusters p and q
                old_val = 0.0
                if old_n_p > 0:
                    old_val += old_wy_p / np.sqrt(old_n_p)
                old_val += w_dot_y[q] / np.sqrt(cluster_sizes[q]) if cluster_sizes[q] > 0 else 0.0

                new_val = 0.0
                if new_n_p > 0:
                    new_val += new_wy_p / np.sqrt(new_n_p)
                if new_n_q > 0:
                    new_val += new_wy_q / np.sqrt(new_n_q)

                gain = new_val - old_val
                if gain > best_gain:
                    best_gain = gain
                    best_q = q

            if best_q != p:
                # Move sample i from p to best_q
                w_dot_y[p] -= W[i, p]
                cluster_sizes[p] -= 1
                w_dot_y[best_q] += W[i, best_q]
                cluster_sizes[best_q] += 1
                Y[i, p] = 0.0
                Y[i, best_q] = 1.0
                labels[i] = best_q

        return Y, labels

    @staticmethod
    def _update_G(Z: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Procrustes SVD for G-update (Theorem 4.1).

        G* = Delta @ Lambda^T from SVD(Z^T U) = Delta Pi Lambda^T.
        """
        ZtU = Z.T @ U  # (c, c)
        Delta, _, LambdaT = np.linalg.svd(ZtU)
        return Delta @ LambdaT

    @staticmethod
    def _update_Q(FG: list[np.ndarray], U: np.ndarray,
                  alpha: np.ndarray, M: np.ndarray,
                  lam: float, n: int, V: int, c: int) -> np.ndarray:
        """Coordinate descent for Q-update (Theorem 4.2, Eq. 16-17).

        For each (i, v) with M[i,v]=1:
          q_i^(v) = clip[0,1]( (alpha * r_i^(-v) @ g_i^(v)^T + lam) /
                                (alpha^2 * ||g_i^(v)||^2 + lam) )
        """
        Q = np.zeros((n, V), dtype=np.float64)

        for i in range(n):
            u_i = U[i]  # (c,)

            for v in range(V):
                if M[i, v] < 0.5:
                    Q[i, v] = 0.0
                    continue

                g_iv = FG[v][i]  # (c,)

                # Partial residual: r_i^(-v) = u_i - sum_{v'!=v} alpha^(v') q_i^(v') g_i^(v')
                r_i = u_i.copy()
                for vp in range(V):
                    if vp != v:
                        r_i -= alpha[vp] * Q[i, vp] * FG[vp][i]

                # Closed-form (Eq. 17)
                numer = alpha[v] * np.dot(r_i, g_iv) + lam
                denom = alpha[v] ** 2 * np.dot(g_iv, g_iv) + lam

                if denom > 1e-12:
                    q_tilde = numer / denom
                else:
                    q_tilde = 1.0

                Q[i, v] = np.clip(q_tilde, 0.0, 1.0)

        return Q

    @staticmethod
    def _update_Q_similarity(embeddings: list[np.ndarray], M: np.ndarray,
                             n: int, V: int, c: int) -> np.ndarray:
        """Sim-Q ablation: quality based on inter-view embedding similarity."""
        Q = np.zeros((n, V), dtype=np.float64)

        # Average embedding across available views
        avg = np.zeros((n, c), dtype=np.float64)
        counts = M.sum(axis=1, keepdims=True)
        for v in range(V):
            avg += M[:, v:v+1] * embeddings[v]
        counts = np.maximum(counts, 1)
        avg /= counts

        for v in range(V):
            for i in range(n):
                if M[i, v] < 0.5:
                    Q[i, v] = 0.0
                    continue
                # Cosine similarity between F_i^(v) and average
                f = embeddings[v][i]
                a = avg[i]
                f_norm = np.linalg.norm(f)
                a_norm = np.linalg.norm(a)
                if f_norm > 1e-12 and a_norm > 1e-12:
                    sim = np.dot(f, a) / (f_norm * a_norm)
                    Q[i, v] = np.clip((sim + 1) / 2, 0.0, 1.0)
                else:
                    Q[i, v] = 0.5
        return Q

    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        """Project vector v onto the probability simplex {x >= 0, sum(x) = 1}.

        Algorithm from Duchi et al. (2008).
        """
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
        theta = cssv[rho] / (rho + 1.0)
        return np.maximum(v - theta, 0.0)

    @staticmethod
    def _update_alpha(FG: list[np.ndarray], Q: np.ndarray, U: np.ndarray,
                      V: int, n: int, c: int) -> np.ndarray:
        """Simplex QP for alpha-update via projected gradient descent (Eq. 19).

        min_{1^T alpha = 1, alpha >= 0} alpha^T F_hat alpha - 2 h_hat^T alpha
        where F_hat[v,v'] = Tr(Z_v^T Z_v') and h_hat[v] = Tr(Z_v^T U).
        """
        # Compute Z_v = diag(q^(v)) F^(v) G for each v
        Z_list = []
        for v in range(V):
            Z_v = Q[:, v:v+1] * FG[v]  # (n, c)
            Z_list.append(Z_v)

        # Build F_hat and h_hat
        F_hat = np.zeros((V, V), dtype=np.float64)
        h_hat = np.zeros(V, dtype=np.float64)

        for v in range(V):
            h_hat[v] = np.sum(Z_list[v] * U)
            for vp in range(v, V):
                val = np.sum(Z_list[v] * Z_list[vp])
                F_hat[v, vp] = val
                F_hat[vp, v] = val

        # Lipschitz constant for step size
        L = 2 * np.linalg.eigvalsh(F_hat).max() + 1e-8
        step = 1.0 / L

        alpha = np.ones(V, dtype=np.float64) / V
        for _ in range(500):
            grad = 2 * F_hat @ alpha - 2 * h_hat
            alpha_new = QASEF._project_simplex(alpha - step * grad)

            if np.linalg.norm(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        return alpha

    @staticmethod
    def _objective(FG: list[np.ndarray], Q: np.ndarray, alpha: np.ndarray,
                   U: np.ndarray, M: np.ndarray, lam: float,
                   V: int, n: int, c: int) -> float:
        """Compute the QASEF objective (Eq. 13)."""
        # Data fidelity: || sum_v alpha^(v) diag(q^(v)) F^(v) G - U ||_F^2
        W = np.zeros((n, c), dtype=np.float64)
        for v in range(V):
            W += alpha[v] * (Q[:, v:v+1] * FG[v])
        fidelity = np.sum((W - U) ** 2)

        # Regularization: lambda * sum_{v,i} M[i,v] * (q_i^(v) - 1)^2
        reg = lam * np.sum(M * (Q - 1) ** 2)

        return fidelity + reg

    def fit_predict(self, views: list[np.ndarray],
                    M: np.ndarray | None = None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(views, M)
        return self.labels_
