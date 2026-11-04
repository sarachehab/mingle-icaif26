import logging

import numpy as np
from scipy.linalg import solve_sylvester

from mingle.utils.graph import degrees_from_W, laplacian_from_W
from mingle.utils.admm.abstract import AbstractAdmm
from mingle.utils.admm.metrics import ADMMNoLowRankStepMetrics
from mingle.utils.admm.helpers import pairwise_sq_dists_rows

AdmmLogger = logging.getLogger(__name__)


class AdmmNoLowRankAlgorithm(AbstractAdmm):
    """
    Ablation baseline reverting the low-rank factor prior (X ≈ BF, B^T B = I)
    back to the unconstrained representation Y of Dong et al.:

        min_{Y, L}  || (X - Y) Ω^(1/2) ||_F^2 + λ tr(Y^T L Y) + g(L)

    Graph smoothness is therefore defined directly on Y rather than on the
    factor exposures B, and g(L) is instantiated through the same sparsity
    (l1) + log-barrier connectivity penalty used in the full model (inherited
    from AbstractAdmm), so that the only change relative to the full
    algorithm is the reconstruction term. This isolates the effect of the
    low-rank/APT prior for the comparison table.

    Because Y carries no constraint (no orthogonality, no rank restriction),
    its update is a direct closed-form Sylvester solve.
    """

    ### ---------- ALGORITHM OPTIMISATION LOOP ----------
    def run(
        self,
        X,
        Y, W, d, V, dual_W, dual_d,
        omega,
        rho_W, rho_d,
    ):

        history: list[ADMMNoLowRankStepMetrics] = []
        terminates = False

        for epoch in range(self.max_epochs):

            if epoch and not epoch % 1000: AdmmLogger.info(f"Current epoch: {epoch}")

            prev_V = np.copy(V)
            prev_d = np.copy(d)

            # Update variables
            Y = self._update_Y(X, W, omega)
            W = self._update_W(Y, V, d, dual_W, dual_d, rho_W, rho_d)
            d = self._update_d(W, dual_d, rho_d)
            V = self._update_V(W, dual_W, rho_W)
            dual_W = self._update_dual_W(W, V, dual_W, rho_W)
            dual_d = self._update_dual_d(W, d, dual_d, rho_d)

            # Calculate primal residuals
            primal_residual_W = self._get_primal_residual_norm_W(W, V)
            primal_residual_d = self._get_primal_residual_norm_d(W, d)

            # Calculate primal tolerances
            primal_tolerance_W = self._get_primal_tolerance_W(W, V)
            primal_tolerance_d = self._get_primal_tolerance_d(W, d)

            # Calculate dual residuals
            dual_residual_W = self._get_dual_residual_norm_W(prev_V, V, rho_W)
            dual_residual_d = self._get_dual_residual_norm_d(prev_d, d, rho_d)

            # Calculate dual tolerances
            dual_tolerance_W = self._get_dual_tolerance_W(dual_W)
            dual_tolerance_d = self._get_dual_tolerance_d(dual_d)

            if (epoch % self.adapt_every == 0) and epoch > 0:
                rho_W, dual_W = self._balance_rho(
                    primal_residual_W, dual_residual_W, rho_W, dual_W,
                )
                rho_d, dual_d = self._balance_rho(
                    primal_residual_d, dual_residual_d, rho_d, dual_d,
                )

            # Save metrics for debugging
            metrics = ADMMNoLowRankStepMetrics(
                k=epoch,
                rW=primal_residual_W, rd=primal_residual_d,
                sW=dual_residual_W, sd=dual_residual_d,
                eps_pri_W=primal_tolerance_W, eps_pri_d=primal_tolerance_d,
                eps_dual_W=dual_tolerance_W, eps_dual_d=dual_tolerance_d,
                nY=np.linalg.norm(Y),
                nW=np.linalg.norm(W),
                nV=np.linalg.norm(V),
                nd=np.linalg.norm(d),
                nDualW=np.linalg.norm(dual_W),
                nDuald=np.linalg.norm(dual_d),
                rho_W=rho_W,
                rho_d=rho_d,
            )

            history.append(metrics)

            # Stop early if both primal and dual problems are optimal feasible
            if (
                primal_residual_W <= primal_tolerance_W and
                primal_residual_d <= primal_tolerance_d and
                dual_residual_W <= dual_tolerance_W and
                dual_residual_d <= dual_tolerance_d
            ):
                AdmmLogger.info(f"Algorithm terminates after {epoch} epochs.")
                terminates = True
                return Y, W, V, history, terminates

        AdmmLogger.info(f"Algorithm fails to terminate in {self.max_epochs} epochs.")
        return Y, W, V, history, terminates

    ### ---------- ALGORITHM INTIALISATION ----------
    def initialise(self, X, omega):

        # Initialise directly at the raw (recency-weighted), consistent with Dong et al.
        Y_ini = X.copy()

        # Construct initial graph weights from raw-signal similarity through RBF kernel
        C = pairwise_sq_dists_rows(Y_ini)
        sigma2 = np.median(C[C > 0])
        W_ini = np.exp(-C / (2 * sigma2))
        np.fill_diagonal(W_ini, 0.0)  # Remove self-loops

        # Define the graph-structure (symmetry and non-negativity) auxiliary
        V_ini = W_ini.copy()

        # Define the out-degree auxiliary equal to the adjacency out-degrees
        d_ini = degrees_from_W(W_ini)

        # Initialise dual variables with no memory
        dual_W_ini = np.zeros_like(W_ini)
        dual_d_ini = np.zeros_like(d_ini)

        return Y_ini, W_ini, d_ini, V_ini, dual_W_ini, dual_d_ini

    ### ---------- UPDATE DECISION VARIABLES (UNCONSTRAINED REPRESENTATION) ----------

    def _update_Y(self, X, W, omega):
        """
        Update the unconstrained representation Y in closed form.

        This corresponds to the stationarity condition of

            || (X - Y) Ω^(1/2) ||_F^2 + λ tr(Y^T L Y)

        which, with Ω = diag(omega), yields the Sylvester equation

            λ L Y + Y Ω = X Ω

        Args:
            X: Data matrix (nb_assets, nb_tickers)
            W: Adjacency matrix over assets (nb_assets, nb_assets)
            omega: Recency-decay weights (nb_tickers,), broadcasting over columns of X

        Returns:
            Y: Updated unconstrained representation (nb_assets, nb_tickers)
        """
        L = laplacian_from_W(W)

        Omega = np.diag(np.asarray(omega).flatten())
        A = self.lambda_l * L
        Bmat = Omega
        C = X * omega

        Y = solve_sylvester(A, Bmat, C)
        return Y
