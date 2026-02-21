"""
因子回测引擎
支持 IC/ICIR 分析、分层回测、多空组合
"""
import numpy as np
import pandas as pd
from scipy import stats


class FactorBacktest:
    """单因子回测"""

    def __init__(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        n_groups: int = 5,
        holding_period: int = 1,
    ):
        self.factor = factor
        self.returns = returns
        self.n_groups = n_groups
        self.holding_period = holding_period

        # 对齐
        common_dates = factor.index.intersection(returns.index)
        common_stocks = factor.columns.intersection(returns.columns)
        self.factor = factor.loc[common_dates, common_stocks]
        # 下期收益 (shift -1)
        self.fwd_returns = returns.loc[common_dates, common_stocks].shift(-holding_period)

    def calc_ic_series(self) -> pd.Series:
        """计算每期 Rank IC (Spearman 相关系数)"""
        ic_list = []
        for date in self.factor.index:
            f = self.factor.loc[date].dropna()
            r = self.fwd_returns.loc[date].dropna()
            common = f.index.intersection(r.index)
            if len(common) < 30:
                ic_list.append(np.nan)
                continue
            corr, _ = stats.spearmanr(f.loc[common], r.loc[common])
            ic_list.append(corr)
        return pd.Series(ic_list, index=self.factor.index, name="IC")

    def calc_ic_stats(self) -> dict:
        """IC 统计摘要"""
        ic = self.calc_ic_series().dropna()
        return {
            "IC_mean": ic.mean(),
            "IC_std": ic.std(),
            "ICIR": ic.mean() / ic.std() if ic.std() > 0 else 0,
            "IC_positive_ratio": (ic > 0).mean(),
            "IC_abs_gt_002": (ic.abs() > 0.02).mean(),
            "t_stat": ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0,
        }

    def calc_group_returns(self) -> pd.DataFrame:
        """分层回测: 按因子值分成 n_groups 组，计算每组平均收益"""
        group_returns = {}

        for date in self.factor.index:
            f = self.factor.loc[date].dropna()
            r = self.fwd_returns.loc[date].dropna()
            common = f.index.intersection(r.index)
            if len(common) < self.n_groups * 5:
                continue

            f_sorted = f.loc[common].rank(pct=True)
            groups = pd.cut(f_sorted, bins=self.n_groups, labels=False) + 1

            row = {}
            for g in range(1, self.n_groups + 1):
                mask = groups == g
                if mask.sum() > 0:
                    row[f"G{g}"] = r.loc[common][mask].mean()
            row["L-S"] = row.get(f"G{self.n_groups}", 0) - row.get("G1", 0)
            group_returns[date] = row

        return pd.DataFrame(group_returns).T

    def calc_group_nav(self) -> pd.DataFrame:
        """分层净值曲线"""
        gr = self.calc_group_returns()
        return (1 + gr).cumprod()

    def summary(self) -> dict:
        """综合回测报告"""
        ic_stats = self.calc_ic_stats()
        group_ret = self.calc_group_returns()

        # 年化收益和夏普
        ann_factor = 52 / self.holding_period  # 周频
        group_ann = {}
        for col in group_ret.columns:
            mean_ret = group_ret[col].mean()
            std_ret = group_ret[col].std()
            group_ann[col] = {
                "ann_return": mean_ret * ann_factor,
                "ann_vol": std_ret * np.sqrt(ann_factor),
                "sharpe": (mean_ret * ann_factor) / (std_ret * np.sqrt(ann_factor))
                if std_ret > 0 else 0,
            }

        return {
            "ic_stats": ic_stats,
            "group_performance": group_ann,
            "n_periods": len(group_ret),
        }


def run_multi_factor_backtest(
    factors: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    n_groups: int = 5,
) -> dict[str, dict]:
    """并行回测多个因子"""
    results = {}
    for name, factor in factors.items():
        bt = FactorBacktest(factor, returns, n_groups=n_groups)
        results[name] = bt.summary()
    return results


def format_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """格式化多因子对比表"""
    rows = []
    for name, res in results.items():
        ic = res["ic_stats"]
        ls = res["group_performance"].get("L-S", {})
        rows.append({
            "Factor": name,
            "IC_mean": f"{ic['IC_mean']:.4f}",
            "ICIR": f"{ic['ICIR']:.4f}",
            "IC_pos%": f"{ic['IC_positive_ratio']:.1%}",
            "t_stat": f"{ic['t_stat']:.2f}",
            "LS_ann_ret": f"{ls.get('ann_return', 0):.2%}",
            "LS_sharpe": f"{ls.get('sharpe', 0):.2f}",
        })
    return pd.DataFrame(rows).set_index("Factor")
