import logging
import numpy as np
from scipy.linalg import solve_sylvester

from mingle.utils.graph import degrees_from_W, laplacian_from_W
from mingle.utils.admm.abstract import AbstractAdmm
from mingle.utils.admm.metrics import ADMMStepMetrics
from mingle.utils.admm.helpers import pairwise_sq_dists_rows

AdmmLogger = logging.getLogger(__name__)

class AdmmJointAlgorithm(AbstractAdmm):

    def __init__(
        self,
        nb_factors,
        alpha, beta, lambda_l, eps, delta,
        max_epochs, abs_eps, rel_eps, abs_eps_B, rel_eps_B,
        adapt_every=25, mu=10.0, tau=2.0,
        rho_min=1e-4, rho_max=1e4,
    ):
        """
        Args:
            nb_factors: Number of latent factors in the low-rank exposure model B
            alpha: l1 regularisation weight
            beta: Degree regularisation weight
            lambda_l: Graph Laplacian regularisation weight
            eps: Small positive constant ensuring log-barrier feasibility
            delta: Factor decorrelation weight
            abs_eps_B, rel_eps_B: Convergence tolerances for the B = Q consensus
        """
        super().__init__(
            alpha=alpha, beta=beta, lambda_l=lambda_l, eps=eps,
            max_epochs=max_epochs, abs_eps=abs_eps, rel_eps=rel_eps,
            adapt_every=adapt_every, mu=mu, tau=tau,
            rho_min=rho_min, rho_max=rho_max,
        )
        self.nb_factors = nb_factors
        self.delta = delta
        self.abs_eps_B = abs_eps_B
        self.rel_eps_B = rel_eps_B

    ### ---------- ALGORITHM OPTIMISATION LOOP ----------
    def run(
        self,
        X,
        F, B, Q, W, d, V, dual_B, dual_W, dual_d,
        omega,
        rho_B, rho_W, rho_d,
    ):

        history: list[ADMMStepMetrics] = []
        terminates = False

        for epoch in range(self.max_epochs):

            if epoch and not epoch % 1000: AdmmLogger.info(f"Current epoch: {epoch}")

            prev_Q = np.copy(Q)
            prev_V = np.copy(V)
            prev_d = np.copy(d)

            # Update variables
            F = self._update_F(X, B, F, omega)
            B = self._update_B(X, F, omega, W, Q, dual_B, rho_B)
            Q = self._update_Q(B, dual_B, rho_B)
            W = self._update_W(B, V, d, dual_W, dual_d, rho_W, rho_d)
            d = self._update_d(W, dual_d, rho_d)
            V = self._update_V(W, dual_W, rho_W)
            dual_B = self._update_dual_B(B, Q, dual_B, rho_B)
            dual_W = self._update_dual_W(W, V, dual_W, rho_W)
            dual_d = self._update_dual_d(W, d, dual_d, rho_d)

            # Calculate primal residuals
            primal_residual_B = self._get_primal_residual_norm_B(B, Q)
            primal_residual_W = self._get_primal_residual_norm_W(W, V)
            primal_residual_d = self._get_primal_residual_norm_d(W, d)

            # Calculate primal tolerances
            primal_tolerance_B = self._get_primal_tolerance_B(B, Q)
            primal_tolerance_W = self._get_primal_tolerance_W(W, V)
            primal_tolerance_d = self._get_primal_tolerance_d(W, d)

            # Calculate dual residuals
            dual_residual_B = self._get_dual_residual_norm_B(prev_Q, Q, rho_B)
            dual_residual_W = self._get_dual_residual_norm_W(prev_V, V, rho_W)
            dual_residual_d = self._get_dual_residual_norm_d(prev_d, d, rho_d)

            # Calculate dual tolerances
            dual_tolerance_B = self._get_dual_tolerance_B(dual_B)
            dual_tolerance_W = self._get_dual_tolerance_W(dual_W)
            dual_tolerance_d = self._get_dual_tolerance_d(dual_d)

            if (epoch % self.adapt_every == 0) and epoch > 0:
                rho_B, dual_B = self._balance_rho(
                    primal_residual_B, dual_residual_B, rho_B, dual_B,
                )
                rho_W, dual_W = self._balance_rho(
                    primal_residual_W, dual_residual_W, rho_W, dual_W,
                )
                rho_d, dual_d = self._balance_rho(
                    primal_residual_d, dual_residual_d, rho_d, dual_d,
                )

            # Save metrics for debugging
            metrics = ADMMStepMetrics(
                k=epoch,
                rB=primal_residual_B, rW=primal_residual_W, rd=primal_residual_d,
                sB=dual_residual_B, sW=dual_residual_W, sd=dual_residual_d,
                eps_pri_B=primal_tolerance_B, eps_pri_W=primal_tolerance_W, eps_pri_d=primal_tolerance_d,
                eps_dual_B=dual_tolerance_B, eps_dual_W=dual_tolerance_W, eps_dual_d=dual_tolerance_d,
                nB=np.linalg.norm(B),
                nQ=np.linalg.norm(Q),
                nW=np.linalg.norm(W),
                nV=np.linalg.norm(V),
                nd=np.linalg.norm(d),
                nDualB=np.linalg.norm(dual_B),
                nDualW=np.linalg.norm(dual_W),
                nDuald=np.linalg.norm(dual_d),
                rho_W=rho_W,
                rho_B=rho_B,
                rho_d=rho_d,
            )

            history.append(metrics)

            # Stop early if both primal and dual problems are optimal feasible
            if (
                primal_residual_B <= primal_tolerance_B and
                primal_residual_W <= primal_tolerance_W and
                primal_residual_d <= primal_tolerance_d and
                dual_residual_B <= dual_tolerance_B and
                dual_residual_W <= dual_tolerance_W and
                dual_residual_d <= dual_tolerance_d
            ):
                AdmmLogger.info(f"Algorithm terminates after {epoch} epochs.")
                terminates = True
                return F, B, W, V, history, terminates

        AdmmLogger.info(f"Algorithm fails to terminate in {self.max_epochs} epochs.")
        return F, B, W, V, history, terminates

    ### ---------- ALGORITHM INTIALISATION ----------
    def initialise(self, X, omega):

        # Define B and F as PCA decomposition of X
        sqrt_omega = np.sqrt(omega)
        X_decay = X * sqrt_omega
        U, sigma, Vt = np.linalg.svd(X_decay, full_matrices=False)
        B_ini = U[:,:self.nb_factors]
        F_ini = np.diag(sigma[:self.nb_factors]) @ Vt[:self.nb_factors, :] / sqrt_omega

        # Define the orthogonality auxiliary to B equal to B
        Q_ini = B_ini.copy()

        # Construct initial graph weights from exposure similarity through RBF kernel
        C = pairwise_sq_dists_rows(B_ini)
        sigma2 = np.median(C[C>0])
        W_ini = np.exp(-C / (2 * sigma2))
        np.fill_diagonal(W_ini, 0.0) # Remove self-loops

        # Define the graph-structure (symmetry and non-negativity) auxiliary
        V_ini = W_ini.copy()

        # Define the out-degree auxiliary equal to the adjacency out-degrees
        d_ini = degrees_from_W(W_ini)

        # Initialise dual variables with no memory
        dual_B_ini = np.zeros_like(B_ini)
        dual_W_ini = np.zeros_like(W_ini)
        dual_d_ini = np.zeros_like(d_ini)

        return F_ini, B_ini, Q_ini, W_ini, d_ini, V_ini, dual_B_ini, dual_W_ini, dual_d_ini

    ### ---------- UPDATE ADMM DECISION VARIABLES (LOW-RANK FACTOR MODEL) ----------

    def _update_F(self, X, B, F, omega, nb_steps=3):
        """
        Update the factor matrix F by solving a least-squares subproblem.

        This update corresponds to the iteration solution of

            F^{k+1} = argmin_F || X - B^k F ||_F^2 + || FΩF^T - diag(FΩF^T) ||_F^2

        This is solved through gradient descent, with gradient

            -2 B^T (X-BF) + 4 * delta * off-diagonal(FΩF^T)FΩ

        The step-size is dynamically chosen to upperbound the Hessian.

        Args:
            X: Data matrix (nb_assets, nb_tickers)
            B: Factor exposure matrix (nb_assets, nb_factors)
            F: Factor matrix (nb_factors, nb_tickers)
            omega: Exponential decay (nb_tickers, nb_tickers)

        Returns:
            F: Updated coefficients matrix (nb_factors, nb_tickers)
        """
        for _ in range(nb_steps):

            # Set adaptive learning rate to upperbound through the Lipschitz constant
            lr = 1.0 / (2 + 8 * self.delta * np.linalg.norm(F) ** 2)

            # Calculate residual gradient from ||X - BF||_F^2
            residual = X - B @ F
            reconstruction_grad = - 2 * B.T @ residual

            # Calculate orthogonality gradient from ||FΩF^T - diag(FΩF^T)||_F^2
            F_omega = F * omega
            cov = F_omega @ F.T
            np.fill_diagonal(cov, 0.0)
            orthogonality_grad = 4 * self.delta * cov @ F_omega

            # Perform gradient descent step
            grad = reconstruction_grad + orthogonality_grad
            F = F - lr * grad

        return F

    def _update_B(self, X, F, omega, W, Q, dual_B, rho_B):
        """
        Update the factor exposure matrix B via the ADMM B-subproblem.

        This step minimises a quadratic objective consisting of:
            (1) a data fidelity term ||(X - B F)Ω^0.5||_F^2,
            (2) a graph Laplacian regularisation term λ tr(B^T L B),
            (3) an ADMM quadratic penalty enforcing B ≈ Q.

        Setting the gradient to 0 yields the Sylvester condition:

            4 λ L^k B^{k+1} + 2 B^{k+1} (F^{k+1} Ω F^{k+1}^T)
                = 2 X F^{k+1}^T + ρ_B Q^k - Λ_B^k + 2 X Ω F.T

        which can be solved directly using a Sylvester solver

        Args:
            X: Data matrix (nb_assets, nb_tickers)
            F: Updated coefficients matrix (nb_factors, nb_tickers)
            Omega: Historical observations weights decay
            W: Adjacency matrix over assets (nb_assets, nb_assets)
            Q: Auxiliary variable imposing orthonomality constraint on B (nb_assets, nb_factors)
            dual_B: ADMM dual variable associated with B ≈ Q (nb_assets, nb_factors)
            rho_B: ADMM penalty parameter controlling adherence to B ≈ Q

        Returns:
            B: Updated factor exposure matrix (nb_assets, nb_factors)
        """
        # Determine graph Laplacian
        L = laplacian_from_W(W)

        # Build matrix A, which left-multiplies B^{k+1}
        n = L.shape[0]
        I = np.eye(n, dtype=L.dtype)
        A_right = 4 * self.lambda_l * L + rho_B * I

        # Build matrix C, which right-multiplies B^{k+1}
        F_decay = F * omega
        C_left = 2 * F_decay @ F.T

        # Build RHS
        X_decay = X * omega
        D = 2 * X_decay @ F.T + rho_B * Q - dual_B

        # Solve for B
        B = solve_sylvester(A_right, C_left, D)
        return B

    def _update_Q(self, B, dual_B, rho_B):
        """
        Update the auxiliary matrix Q by projecting onto the Stiefel manifold.

        In the ADMM formulation, Q enforces the orthonormality constraint on B:

            Q^{k+1} = argmin_{Q^T Q = I} (rho_B/2) || B^{k+1} - Q + (1/rho_B) Λ_B^k ||_F^2

        which is equivalent to the Orthogonal Procrustes problem.

        The solution is given by the polar factorisation of (rho_B B + Λ_B):

            Q = U V^T.

        For tall matrices, this is computed efficiently via a thin QR
        followed by an SVD of the reduced factor.

            A = Q * R = Q * ( U * Σ * V^T)

        Args:
            B: Updated factor exposure matrix (nb_assets, nb_factors)
            dual_B: ADMM dual variable associated with B ≈ Q (nb_assets, nb_factors)
            rho_B: ADMM penalty parameter controlling adherence to B ≈ Q

        Returns:
            Q: Updated auxiliary variable imposing orthonomality constraint on B (nb_assets, nb_factors)
        """
        M = rho_B * B + dual_B
        Qr, R = np.linalg.qr(M, mode="reduced")
        Ur, _, Vt = np.linalg.svd(R)
        Q = (Qr @ Ur) @ Vt
        return Q

    def _update_dual_B(self, B, Q, dual_B, rho_B):
        """
        Update of ADMM dual variable enforcing B ≈ Q using gradient ascent.

        Args:
            B: Updated factor exposure matrix (nb_assets, nb_factors)
            Q: Updated auxiliary variable imposing orthonomality constraint on B (nb_assets, nb_factors)
            dual_B: ADMM dual variable associated with B ≈ Q (nb_assets, nb_factors)
            rho_B: ADMM penalty parameter controlling adherence to B ≈ Q

        Returns:
            dual_B: Updated ADMM dual variable associated with B ≈ Q (nb_assets, nb_factors)
        """
        return dual_B + rho_B * (B - Q)

    ### ---------- DETERMINE RESIDUALS (LOW-RANK FACTOR MODEL) ----------

    def _get_primal_residual_norm_B(self, B, Q):
        return np.linalg.norm(B - Q)

    def _get_dual_residual_norm_B(self, prev_Q, curr_Q, rho_B):
        return np.linalg.norm(rho_B * (curr_Q - prev_Q))

    ### ---------- DETERMINE TOLERANCES (LOW-RANK FACTOR MODEL) ----------

    def _get_primal_tolerance_B(self, B, Q):
        n = B.size
        return np.sqrt(n) * self.abs_eps_B + self.rel_eps_B * max(np.linalg.norm(B), np.linalg.norm(Q))

    def _get_dual_tolerance_B(self, dual_B):
        n = dual_B.size
        return np.sqrt(n) * self.abs_eps_B + self.rel_eps_B * np.linalg.norm(dual_B)
