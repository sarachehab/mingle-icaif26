from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    cagr: float
    sharpe_ratio: float
    annualised_sharpe_ratio: float
    sortino: float
    annualised_sortino: float
    calmar: float
    turnover: list
    max_drawdown: float
    drawdown: list
    alpha: float
    beta: float
    rolling_alpha: list
    rolling_beta: list

    def __str__(self):
        lines = (
            f"Performance Evaluation\n"
            f"\tCAGR: {self.cagr:.2%}\n"
            f"\tSharpe Ratio: {self.sharpe_ratio:.4f}\n"
            f"\tAnnualised Sharpe Ratio: {self.annualised_sharpe_ratio:.4f}\n"
            f"\tSortino: {self.sortino:.4f}\n"
            f"\tAnnualised Sortino: {self.annualised_sortino:.4f}\n"
            f"\tCalmar: {self.calmar:.4f}\n"
            f"\tTurnover Mean: {self.turnover.mean():.4f}\n"
            f"\tMax Drawdown: {self.max_drawdown:.2%}\n"
            f"\tAlpha: {self.alpha:.4f}\n"
            f"\tBeta: {self.beta:.4f}"
        )
        return lines
