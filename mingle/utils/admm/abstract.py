import numpy as np

from mingle.utils.graph import degrees_from_W
from mingle.utils.admm.helpers import pairwise_sq_dists_rows, l1_proximal_operator


class AbstractAdmm:

    def __init__(
        self,
        alpha, beta, lambda_l, eps,
        max_epochs, abs_eps, rel_eps,
        adapt_every=25, mu=10.0, tau=2.0,
        rho_min=1e-4, rho_max=1e4,
    ):
        """
        Args:
            alpha: l1 regularisation weight on W
            beta: Degree/connectivity (log-barrier) regularisation weight
            lambda_l: Graph Laplacian smoothness weight
            eps: Small positive constant ensuring log-barrier feasibility
        """
        self.alpha = alpha
        self.beta = beta
        self.lambda_l = lambda_l
        self.eps = eps
        self.max_epochs = max_epochs
        self.abs_eps = abs_eps
        self.rel_eps = rel_eps
        self.adapt_every = adapt_every
        self.mu = mu
        self.tau = tau
        self.rho_min = rho_min
        self.rho_max = rho_max

    ### ---------- UPDATE DECISION VARIABLES ----------

    def _update_W(self, representation, V, d, dual_W, dual_d, rho_W, rho_d):
        """
        Closed form ADMM update for the adjacency matrix W, decoupled row-wise.

        This solves for the first order optimality condition applied on
        the augmented Lagrangian:

            (rho_W * I + rho_d * 11^T) w_i = b_i

        decoupled by row of W, w_i, and where

            b_i = rho_W * V - dual_W - lambda_l * C + (rho_d * d - dual_d) 1^T

        and

            C_{i,j} = || r_i - r_j ||^2_2

        with r the rows of `representation` (the factor exposures B for the
        low-rank model, or the unconstrained Y for the ablation).

        Using Sherman–Morrison for (rho_W I + rho_d 11^T)^{-1}, the update becomes:

            W = (b - alpha * row_sums(b)) / rho_W,

        where

            alpha = rho_d / (rho_W + rho_d * p),

            row_sums(b) = b 1  (i.e., sum across columns per row),

            p = number of nodes/rows

        Args:
            representation: Matrix whose row-similarity defines graph locality (nb_assets, ...)
            V: Auxiliary variable imposing sparsity and structural constraints of graph (nb_assets, nb_assets)
            d: Auxiliary variable imposing connectivity of the adjacency matrix d (nb_assets, 1)
            dual_W: ADMM dual variable associated with W ≈ V (nb_assets, nb_assets)
            dual_d: ADMM dual variable associated with W1 ≈ d (nb_assets, 1)
            rho_W: ADMM penalty parameter controlling adherence to W ≈ V
            rho_d: ADMM penalty parameter controlling adherence to W1 ≈ d

        Returns:
            W: Updated adjacency matrix W (nb_assets, nb_assets)
        """
        p = V.shape[0]

        # Build matrix with norms of row differences
        C = pairwise_sq_dists_rows(representation)

        # Build RHS
        degrees_term = (rho_d * d - dual_d).reshape(p, 1)
        b = rho_W * V - dual_W - self.lambda_l * C + degrees_term

        # Solve for W with the inverse determined by the Sherman-Morrison formula
        alpha = rho_d / (rho_W + rho_d * p)
        row_sums = b.sum(axis=1, keepdims=True)
        W = (b - alpha * row_sums) / rho_W
        return W

    def _update_d(self, W, dual_d, rho_d):
        """
        Closed form ADMM update for the degree regularisation term, d, decoupled element-wise.

            d_i^{k} = arg min_d - β log(ε + d) + rho_d / 2 ||d - W1_i ||^2_2

        The first order optimality condition yields a quadratic equation in d_i

            ρ_d d_i^2
            + (ρ_d ε - ρ_d 1^T w_i^{k+1} - Λ_{d,i}^k) d_i
            - (β + Λ_{d,i}^k ε + ρ_d 1^T w_i^{k+1} ε) = 0.

        whose unique admissible solution is given by its positive root.

        Args:
            W: Updated adjacency matrix W (nb_assets, nb_assets)
            dual_d: ADMM dual variable associated with W1 ≈ d (nb_assets, 1)
            rho_d: ADMM penalty parameter controlling adherence to W1 ≈ d

        Returns:
            d: Updated auxiliary variable imposing graph degree regularisation (nb_tickers, 1)
        """
        s = degrees_from_W(W)
        A = rho_d * self.eps - rho_d * s - dual_d

        disc = A ** 2 + 4.0 * rho_d * (self.beta + dual_d * self.eps + rho_d * s * self.eps)
        disc = np.maximum(disc, 0.0)

        d = (-A + np.sqrt(disc)) / (2.0 * rho_d)
        return d

    def _update_V(self, W, dual_W, rho_W):
        """
        ADMM update for the auxiliary matrix V enforcing sparsity and structure of graph.

        The update is computed in two steps:
            1) Apply element-wise soft-thresholding to promote sparsity,
            2) Project the result onto the set of symmetric matrices with positive weights and zero diagonal.

        This corresponds to applying:

            M = W + (1/ρ_W) Λ_W
            V = Π_C( Prox_{(α/ρ_W) ||·||_1}(M) )

        Args:
            W: Updated adjacency matrix W (nb_assets, nb_assets)
            dual_W: Updated ADMM dual variable associated with W ≈ V (nb_assets, nb_assets)
            rho_W: ADMM penalty parameter controlling adherence to W ≈ V

        Returns:
            V: Updated auxiliary variable satisfying sparsity and structural constraints of graph (nb_assets, nb_assets)
        """
        M = W + (1.0 / rho_W) * dual_W
        tmp_V = l1_proximal_operator(M, self.alpha / rho_W)

        # Project into the symmetric matrix subspace
        V = 0.5 * (tmp_V + tmp_V.T)

        # Ensure graph only captures similarity
        V = np.maximum(V, 0.0)

        # Project onto zero-diagonal matrices
        np.fill_diagonal(V, 0.0)
        return V

    def _update_dual_W(self, W, V, dual_W, rho_W):
        """
        Update of ADMM dual variable enforcing W ≈ V using gradient ascent.

        Args:
            W: Updated adjacency matrix W (nb_assets, nb_assets)
            V: Updated auxiliary variable satisfying sparsity and structural constraints of graph (nb_assets, nb_assets)
            dual_W: ADMM dual variable associated with W ≈ V (nb_assets, nb_assets)
            rho_W: ADMM penalty parameter controlling adherence to W ≈ V

        Returns:
            dual_W: Updated ADMM dual variable associated with W ≈ V (nb_assets, nb_assets)
        """
        return dual_W + rho_W * (W - V)

    def _update_dual_d(self, W, d, dual_d, rho_d):
        """
        Update of ADMM dual variable enforcing W1 ≈ d using gradient ascent.

        Args:
            W: Updated adjacency matrix W (nb_assets, nb_assets)
            d: Updated auxiliary variable imposing structure of graph (nb_assets, 1)
            dual_d: ADMM dual variable associated with W1 ≈ d (nb_assets, 1)
            rho_d: ADMM penalty parameter controlling adherence to W1 ≈ d

        Returns:
            dual_d: Updated ADMM dual variable associated with W1 ≈ d (nb_assets, 1)
        """
        degrees = degrees_from_W(W)
        return dual_d + rho_d * (degrees - d)

    ### ---------- DETERMINE RESIDUALS ----------

    def _get_primal_residual_norm_W(self, W, V):
        return np.linalg.norm(W - V)

    def _get_primal_residual_norm_d(self, W, d):
        degrees = degrees_from_W(W)
        return np.linalg.norm(degrees - d)

    def _get_dual_residual_norm_W(self, prev_V, curr_V, rho_W):
        return np.linalg.norm(rho_W * (curr_V - prev_V))

    def _get_dual_residual_norm_d(self, prev_d, curr_d, rho_d):
        return np.linalg.norm(rho_d * (curr_d - prev_d))

    ### ---------- DETERMINE TOLERANCES ----------

    def _get_primal_tolerance_W(self, W, V):
        n = W.size
        return np.sqrt(n) * self.abs_eps + self.rel_eps * max(np.linalg.norm(W), np.linalg.norm(V))

    def _get_primal_tolerance_d(self, W, d):
        n = d.size
        degrees = degrees_from_W(W)
        return np.sqrt(n) * self.abs_eps + self.rel_eps * max(np.linalg.norm(degrees), np.linalg.norm(d))

    def _get_dual_tolerance_W(self, dual_W):
        n = dual_W.size
        return np.sqrt(n) * self.abs_eps + self.rel_eps * np.linalg.norm(dual_W)

    def _get_dual_tolerance_d(self, dual_d):
        n = dual_d.size
        return np.sqrt(n) * self.abs_eps + self.rel_eps * np.linalg.norm(dual_d)

    ### ---------- BALANCE CONSTRAINT ADHERENCE AND OBJECTIVE MINIMISATION ----------

    def _balance_rho(self, r_norm, s_norm, rho, dual):
        """
        Residual balancing for one ADMM constraint block.

        Args:
            r_norm: Norm of the primal residual (measures constraint violation)
            s_norm: Norm of the dual residual (measures dual feasibility)
            rho: Current ADMM penalty parameter
            dual: ADMM dual variable associated with this constraint

        Returns:
            rho_new: Updated ADMM penalty parameter
            dual: Rescaled dual variable
        """
        rho_new = rho

        # Enforce constraint adherence if violation is large
        if r_norm > self.mu * s_norm: rho_new = min(rho * self.tau, self.rho_max)

        # Loosen constraint regularisation if constraint is adhered to
        elif s_norm > self.mu * r_norm: rho_new = max(rho / self.tau, self.rho_min)

        if rho != rho_new: dual = dual * rho_new / rho

        return rho_new, dual
