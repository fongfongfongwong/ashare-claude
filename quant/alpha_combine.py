"""
多因子合成模块
- IC 加权
- IC_IR 加权
- 最大化 ICIR 优化
- 等权合成
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats


def _rolling_ic(
    factor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    window: int = 52,
) -> pd.Series:
    """滚动 Rank IC"""
    ic_list = []
    common_dates = factor.index.intersection(fwd_returns.index)

    for i, date in enumerate(common_dates):
        f = factor.loc[date].dropna()
        r = fwd_returns.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 30:
            ic_list.append(np.nan)
            continue
        corr, _ = stats.spearmanr(f.loc[common], r.loc[common])
        ic_list.append(corr)

    return pd.Series(ic_list, index=common_dates, name="IC")


class AlphaCombiner:
    """多因子合成器"""

    def __init__(
        self,
        factors: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        ic_window: int = 52,
        halflife: int = 26,
    ):
        self.factors = factors
        self.returns = returns
        self.ic_window = ic_window
        self.halflife = halflife
        self.factor_names = list(factors.keys())
        self.n_factors = len(self.factor_names)

        # 计算前瞻收益
        self.fwd_returns = returns.shift(-1)

        # 预计算滚动 IC
        self._ic_series = {}
        for name, f in factors.items():
            self._ic_series[name] = _rolling_ic(f, self.fwd_returns, ic_window)

    def _exponential_weights(self, n: int) -> np.ndarray:
        lam = 0.5 ** (1 / self.halflife)
        w = lam ** np.arange(n - 1, -1, -1)
        return w / w.sum()

    def equal_weight(self) -> pd.DataFrame:
        """等权合成"""
        combined = None
        for name in self.factor_names:
            f = self.factors[name]
            if combined is None:
                combined = f.copy() / self.n_factors
            else:
                combined = combined.add(f / self.n_factors, fill_value=0)
        return combined

    def ic_weight(self, date) -> dict[str, float]:
        """
        IC 加权: 用过去窗口期的平均 IC 作为权重
        """
        weights = {}
        for name in self.factor_names:
            ic = self._ic_series[name]
            past_ic = ic.loc[:date].dropna().tail(self.ic_window)
            if len(past_ic) < 10:
                weights[name] = 1.0 / self.n_factors
            else:
                w = self._exponential_weights(len(past_ic))
                weights[name] = np.sum(w * past_ic.values)

        # 归一化 (绝对值之和为1)
        total = sum(abs(v) for v in weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def icir_weight(self, date) -> dict[str, float]:
        """
        ICIR 加权: 用 IC/IC_std 作为权重
        比纯 IC 更稳定
        """
        weights = {}
        for name in self.factor_names:
            ic = self._ic_series[name]
            past_ic = ic.loc[:date].dropna().tail(self.ic_window)
            if len(past_ic) < 10:
                weights[name] = 1.0 / self.n_factors
            else:
                w = self._exponential_weights(len(past_ic))
                ic_mean = np.sum(w * past_ic.values)
                ic_std = np.sqrt(np.sum(w * (past_ic.values - ic_mean) ** 2))
                weights[name] = ic_mean / ic_std if ic_std > 1e-8 else 0
        total = sum(abs(v) for v in weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def max_icir_optimize(self, date) -> dict[str, float]:
        """
        最大化合成因子的 ICIR
        通过优化因子权重使得合成后的 ICIR 最大

        max  w'μ / sqrt(w'Σw)
        s.t. ||w||_1 = 1

        等价于: w* ∝ Σ^{-1} μ
        """
        ic_means = []
        ic_matrix = []
        for name in self.factor_names:
            ic = self._ic_series[name]
            past_ic = ic.loc[:date].dropna().tail(self.ic_window)
            if len(past_ic) < 10:
                ic_means.append(0)
                ic_matrix.append(np.zeros(len(past_ic)))
            else:
                w = self._exponential_weights(len(past_ic))
                ic_means.append(np.sum(w * past_ic.values))
                ic_matrix.append(past_ic.values[-self.ic_window :])

        mu = np.array(ic_means)

        # IC 协方差矩阵
        min_len = min(len(m) for m in ic_matrix)
        if min_len < 10:
            return {name: 1.0 / self.n_factors for name in self.factor_names}

        ic_mat = np.column_stack([m[-min_len:] for m in ic_matrix])
        Sigma = np.cov(ic_mat.T)

        # 正则化
        Sigma += np.eye(self.n_factors) * 1e-6

        try:
            # w* ∝ Σ^{-1} μ
            w_opt = np.linalg.solve(Sigma, mu)
        except np.linalg.LinAlgError:
            w_opt = mu  # fallback to IC weighting

        # 归一化
        total = np.sum(np.abs(w_opt))
        if total > 0:
            w_opt = w_opt / total

        return dict(zip(self.factor_names, w_opt))

    def combine_dynamic(
        self, method: str = "icir"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        动态合成因子 (滚动更新权重)

        Returns:
            combined_factor: 合成后的因子值
            weight_history: 权重历史
        """
        method_map = {
            "equal": lambda d: {n: 1.0 / self.n_factors for n in self.factor_names},
            "ic": self.ic_weight,
            "icir": self.icir_weight,
            "max_icir": self.max_icir_optimize,
        }
        weight_func = method_map[method]

        # 获取所有因子公共时间索引
        all_dates = self.factors[self.factor_names[0]].index
        # 需要足够的预热期
        start_idx = max(self.ic_window + 10, 60)

        combined_rows = {}
        weight_rows = {}

        for i in range(start_idx, len(all_dates)):
            date = all_dates[i]
            weights = weight_func(date)
            weight_rows[date] = weights

            # 加权合成
            row = pd.Series(0.0, index=self.factors[self.factor_names[0]].columns)
            for name, w in weights.items():
                if date in self.factors[name].index:
                    f_row = self.factors[name].loc[date].fillna(0)
                    row = row + w * f_row
            combined_rows[date] = row

        combined_factor = pd.DataFrame(combined_rows).T
        weight_history = pd.DataFrame(weight_rows).T

        return combined_factor, weight_history

    def summary(self, method: str = "icir") -> dict:
        """合成因子的回测摘要"""
        combined, weights = self.combine_dynamic(method)

        # 计算合成因子的 IC
        ic_list = []
        for date in combined.index:
            f = combined.loc[date].dropna()
            r = self.fwd_returns.loc[date].dropna() if date in self.fwd_returns.index else pd.Series()
            common = f.index.intersection(r.index)
            if len(common) < 30:
                ic_list.append(np.nan)
                continue
            corr, _ = stats.spearmanr(f.loc[common], r.loc[common])
            ic_list.append(corr)

        ic = pd.Series(ic_list, index=combined.index).dropna()

        return {
            "method": method,
            "IC_mean": ic.mean(),
            "IC_std": ic.std(),
            "ICIR": ic.mean() / ic.std() if ic.std() > 0 else 0,
            "IC_positive_ratio": (ic > 0).mean(),
            "n_periods": len(ic),
            "avg_weights": weights.mean().to_dict(),
        }
