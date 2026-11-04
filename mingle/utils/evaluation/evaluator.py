import numpy as np
import pandas as pd
from mingle.utils.evaluation.metrics import PerformanceMetrics

class PerformanceEvaluation:

    annualisation_factor = float(252)

    @classmethod
    def align_data(cls, portfolio_returns, risk_free_returns):
        risk_free_returns = risk_free_returns.reindex(portfolio_returns.index)
        return portfolio_returns, risk_free_returns

    @classmethod
    def get_metrics(
        cls,
        portfolio_returns,
        risk_free_returns,
        porfolio_weights,
        market_returns,
        cost_rate=None,
        rolling_window: int = 252,
    ):
        portfolio_returns, risk_free_returns = cls.align_data(portfolio_returns, risk_free_returns)

        if cost_rate is not None:
            portfolio_returns, _ = cls.apply_transaction_costs(portfolio_returns, porfolio_weights, cost_rate)

        portfolio_excess_returns = portfolio_returns - risk_free_returns
        market_returns = market_returns.reindex(portfolio_returns.index)
        market_excess_returns = market_returns - risk_free_returns

        return PerformanceMetrics(
            cagr=cls._cagr(portfolio_returns),
            sharpe_ratio=cls._sharpe_ratio(portfolio_excess_returns),
            annualised_sharpe_ratio=cls._annualised_sharpe_ratio(portfolio_excess_returns),
            sortino=cls._sortino(portfolio_excess_returns),
            annualised_sortino=cls._annualised_sortino(portfolio_excess_returns),
            turnover=cls._turnover(porfolio_weights),
            max_drawdown=cls._max_drawdown(portfolio_returns),
            drawdown=cls._drawdown(portfolio_returns),
            calmar=cls._calmar(portfolio_returns),
            alpha=cls._alpha(portfolio_excess_returns, market_excess_returns),
            beta=cls._beta(portfolio_excess_returns, market_excess_returns),
            rolling_alpha=cls._rolling_alpha(portfolio_excess_returns, market_excess_returns, rolling_window),
            rolling_beta=cls._rolling_beta(portfolio_excess_returns, market_excess_returns, rolling_window),
        )
    
    @classmethod
    def apply_transaction_costs(
        cls,
        portfolio_returns,
        porfolio_weights,
        cost_rate: float,
    ):
        turnover = cls._turnover(porfolio_weights)
        costs = turnover * cost_rate

        net = portfolio_returns.copy()

        idx = net.index.intersection(costs.index)
        if idx.empty:
            raise ValueError(
                "No overlap between return dates and rebalance dates. "
                "*Reminder*: Transaction costs on in-sample returns were never implemented."
            )
        net.loc[idx] = net.loc[idx] - costs.loc[idx]
        return net, costs

    @classmethod
    def _rolling_cagr(cls, portfolio_returns):
        wealth_index = (1 + portfolio_returns).cumprod()
        years_passed = np.arange(1, len(portfolio_returns) + 1) / cls.annualisation_factor
        instant_cagr = wealth_index ** (1 / years_passed) - 1
        return instant_cagr

    @classmethod
    def _cagr(cls, portfolio_returns):
        return cls._rolling_cagr(portfolio_returns).iloc[-1]

    @classmethod
    def _sharpe_ratio(cls, portfolio_excess_returns):
        return portfolio_excess_returns.mean() / portfolio_excess_returns.std()

    @classmethod
    def _annualised_sharpe_ratio(cls, portfolio_excess_returns):
        return cls._sharpe_ratio(portfolio_excess_returns) * np.sqrt(cls.annualisation_factor)

    @classmethod
    def _sortino(cls, portfolio_excess_returns):
        downside_returns = portfolio_excess_returns.copy()
        downside_returns[downside_returns > 0] = 0
        denom = np.sqrt((downside_returns**2).mean())
        return portfolio_excess_returns.mean() / denom
    
    @classmethod               
    def _annualised_sortino(cls, portfolio_excess_returns):
        return cls._sortino(portfolio_excess_returns) * np.sqrt(cls.annualisation_factor)

    @classmethod
    def _turnover(cls, porfolio_weights):
        return 0.5 * porfolio_weights.diff().abs().sum(axis=1).iloc[1:]

    @classmethod
    def _drawdown(cls, portfolio_returns):
        equity = (1.0 + portfolio_returns).cumprod()
        peak = equity.cummax()
        dd = equity / peak - 1.0
        return dd

    @classmethod
    def _max_drawdown(cls, portfolio_returns):
        return cls._drawdown(portfolio_returns).min()

    @classmethod
    def _calmar(cls, portfolio_returns):
        return - cls._cagr(portfolio_returns) / cls._max_drawdown(portfolio_returns)

    @classmethod
    def _beta(cls, portfolio_excess_returns, market_excess_returns):
        aligned = pd.concat([portfolio_excess_returns, market_excess_returns], axis=1).dropna()
        rp, rm = aligned.iloc[:, 0], aligned.iloc[:, 1]
        return float(rp.cov(rm) / rm.var())

    @classmethod
    def _alpha(cls, portfolio_excess_returns, market_excess_returns):
        b = cls._beta(portfolio_excess_returns, market_excess_returns)
        aligned = pd.concat([portfolio_excess_returns, market_excess_returns], axis=1).dropna()
        rp, rm = aligned.iloc[:, 0], aligned.iloc[:, 1]
        return float((rp.mean() - b * rm.mean()) * cls.annualisation_factor)

    @classmethod
    def _rolling_beta(cls, portfolio_excess_returns, market_excess_returns, window: int):
        aligned = pd.concat([portfolio_excess_returns, market_excess_returns], axis=1).dropna()
        rp, rm = aligned.iloc[:, 0], aligned.iloc[:, 1]
        cov = rp.rolling(window).cov(rm)
        var = rm.rolling(window).var()
        return (cov / var).rename("rolling_beta")

    @classmethod
    def _rolling_alpha(cls, portfolio_excess_returns, market_excess_returns, window: int):
        aligned = pd.concat([portfolio_excess_returns, market_excess_returns], axis=1).dropna()
        rp, rm = aligned.iloc[:, 0], aligned.iloc[:, 1]
        rolling_b = cls._rolling_beta(portfolio_excess_returns, market_excess_returns, window)
        rolling_a = (rp.rolling(window).mean() - rolling_b * rm.rolling(window).mean()) * cls.annualisation_factor
        return rolling_a.rename("rolling_alpha")
