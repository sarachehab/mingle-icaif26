from dataclasses import dataclass


@dataclass
class AbstractADMMStepMetrics:
    k: int

    # Primal residual norms
    rW: float
    rd: float

    # Dual residual norms
    sW: float
    sd: float

    # Tolerances
    eps_pri_W: float
    eps_pri_d: float
    eps_dual_W: float
    eps_dual_d: float

    # Norms / scale diagnostics
    nW: float
    nV: float
    nd: float
    nDualW: float
    nDuald: float

    # Balancing ADMM constants
    rho_W: float
    rho_d: float


@dataclass
class ADMMStepMetrics(AbstractADMMStepMetrics):
    # Primal residual norms
    rB: float

    # Dual residual norms
    sB: float

    # Tolerances
    eps_pri_B: float
    eps_dual_B: float

    # Norms / scale diagnostics
    nB: float
    nQ: float
    nDualB: float

    # Balancing ADMM constants
    rho_B: float


@dataclass
class ADMMNoLowRankStepMetrics(AbstractADMMStepMetrics):
    nY: float
