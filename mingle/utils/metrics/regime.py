import numpy as np
import pandas as pd
    
def label_vix_regimes(
    vix: pd.Series,
    calm_thr: float = 20.0,
    crisis_thr: float = 30.0,
) -> pd.Series:
    """
    Map a VIX level series to regime labels.

    Returns a pd.Series of str with the same index:
        "CALM"     — VIX < calm_thr
        "ELEVATED" — calm_thr <= VIX < crisis_thr
        "CRISIS"   — VIX >= crisis_thr
    """
    labels = pd.Series("Elevated", index=vix.index, dtype=str)
    labels[vix < calm_thr] = "Calm"
    labels[vix >= crisis_thr] = "Crisis"
    return labels


def performance_by_regime(
    portfolio_returns: pd.Series,
    risk_free_returns: pd.Series,
    market_returns: pd.Series,
    regimes: pd.Series,
) -> dict:
    """Slice portfolio_returns by VIX regime and compute PerformanceMetrics for each slice."""
    from mingle.utils.evaluation import PerformanceEvaluation

    aligned_regimes = regimes.reindex(portfolio_returns.index).ffill()
    results = {}

    for label in ("Calm", "Elevated", "Crisis"):
        mask = aligned_regimes == label
        ret_slice = portfolio_returns[mask]
        if len(ret_slice) < 2:
            continue
        rf_slice = risk_free_returns.reindex(ret_slice.index)
        mkt_slice = market_returns.reindex(ret_slice.index)
        dummy_weights = pd.DataFrame(
            np.zeros((len(ret_slice), 1)), index=ret_slice.index
        )
        results[label] = PerformanceEvaluation.get_metrics(
            ret_slice, rf_slice, dummy_weights, mkt_slice
        )

    return results
