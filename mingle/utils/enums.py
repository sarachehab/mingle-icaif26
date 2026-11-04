from enum import Enum

class BloombergField(str, Enum):
    # NOTE: these are Bloomberg Terminal field mnemonics — my specific data source,
    # not required by the method.
    OPEN   = "px_open"
    CLOSE   = "px_last"
    LOW    = "px_low"
    HIGH   = "px_high"
    VOLUME = "volume"

class Index(str, Enum):
    # NOTE: my specific universe setup. The paper (Section 4) only uses
    # NIKKEI, SPX, and STOXX (top 100 by 2017 volume from each, 300 assets total).
    NIKKEI = "NIKKEI225"
    SPX = "S&P500"
    STOXX = "STOXX600"

class AllocationStrategy(str, Enum):

    # ADMM-based strategies (Ours)
    ADMM_CUT      = "Adaptive CutV on induced covariance"
    ADMM_CONTAGION_CUT = "Contagion cut"

    # Baselines
    HRP        = "HRP on raw covariance"
    MVO        = "MVO on raw covariance"
    EW         = "Equal weight"
    CUT        = "Adaptive CutV on raw covariance"
    FACTOR     = "Adaptive CutV on factor-induced covariance"


ADMM_STRATEGIES = frozenset({
    AllocationStrategy.ADMM_CUT,
    AllocationStrategy.ADMM_CONTAGION_CUT,
})

BASELINE_STRATEGIES = frozenset({
    AllocationStrategy.HRP,
    AllocationStrategy.MVO,
    AllocationStrategy.EW,
    AllocationStrategy.CUT,
    AllocationStrategy.FACTOR,
})

def coerce_strategy(v):
    """Coerce deserialised strategy into AllocationStrategy."""
    if isinstance(v, AllocationStrategy):
        return v
    if isinstance(v, str):
        if v in AllocationStrategy.__members__:
            return AllocationStrategy[v]
        try:
            return AllocationStrategy(v)
        except ValueError:
            raise ValueError(f"Unknown AllocationStrategy: {v!r}") from None
    return v

class CutType(str, Enum):
    CutN = "CutN"
    CutV = "CutV"
