"""
交易成本与换手率分析模块
- 线性冲击成本模型
- 换手率跟踪
- 净收益计算
"""
import numpy as np
import pandas as pd


class TransactionCostModel:
    """
    交易成本模型

    总成本 = 佣金 + 印花税 + 冲击成本

    A股特色:
    - 买入无印花税, 卖出 0.05% (2023年起)
    - 佣金双边各 ~0.01%-0.03%
    - 冲击成本与成交金额/日均成交额比值相关
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,      # 佣金率 (双边)
        stamp_tax_rate: float = 0.0005,        # 印花税率 (卖出单边)
        impact_coeff: float = 0.1,             # 冲击成本系数
        min_commission: float = 5.0,           # 最低佣金 (元)
    ):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.impact_coeff = impact_coeff
        self.min_commission = min_commission

    def estimate_single_trade_cost(
        self,
        trade_value: float,
        daily_volume: float = 1e8,
        is_buy: bool = True,
    ) -> float:
        """
        估算单笔交易成本 (占交易金额比例)

        Parameters:
            trade_value: 交易金额
            daily_volume: 日均成交金额
            is_buy: 是否为买入
        """
        # 佣金
        commission = self.commission_rate

        # 印花税 (仅卖出)
        stamp_tax = 0 if is_buy else self.stamp_tax_rate

        # 市场冲击成本: c_impact = coeff * sqrt(trade_value / daily_volume)
        participation_rate = abs(trade_value) / daily_volume if daily_volume > 0 else 0.01
        impact = self.impact_coeff * np.sqrt(min(participation_rate, 1.0))

        return commission + stamp_tax + impact

    def estimate_turnover_cost(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        portfolio_value: float = 1e8,
    ) -> dict:
        """
        估算调仓成本

        Returns:
            dict: turnover, buy_cost, sell_cost, total_cost
        """
        # 对齐
        all_stocks = old_weights.index.union(new_weights.index)
        w_old = old_weights.reindex(all_stocks, fill_value=0)
        w_new = new_weights.reindex(all_stocks, fill_value=0)

        delta = w_new - w_old
        turnover = delta.abs().sum() / 2  # 单边换手率

        # 买入部分
        buy_mask = delta > 0
        buy_value = (delta[buy_mask] * portfolio_value).sum()
        buy_cost = buy_value * self.estimate_single_trade_cost(buy_value, is_buy=True)

        # 卖出部分
        sell_mask = delta < 0
        sell_value = (-delta[sell_mask] * portfolio_value).sum()
        sell_cost = sell_value * self.estimate_single_trade_cost(sell_value, is_buy=False)

        total_cost = (buy_cost + sell_cost) / portfolio_value

        return {
            "turnover": turnover,
            "buy_cost_bps": buy_cost / portfolio_value * 1e4,
            "sell_cost_bps": sell_cost / portfolio_value * 1e4,
            "total_cost_bps": total_cost * 1e4,
            "total_cost_pct": total_cost,
        }


class TurnoverAnalyzer:
    """换手率分析"""

    def __init__(
        self,
        factor: pd.DataFrame,
        n_groups: int = 5,
        cost_model: TransactionCostModel = None,
    ):
        self.factor = factor
        self.n_groups = n_groups
        self.cost_model = cost_model or TransactionCostModel()

    def _build_group_weights(self, date) -> dict[str, pd.Series]:
        """按因子分层构建等权组合权重"""
        f = self.factor.loc[date].dropna()
        if len(f) < self.n_groups * 5:
            return {}

        ranks = f.rank(pct=True)
        groups = pd.cut(ranks, bins=self.n_groups, labels=False) + 1

        weights = {}
        for g in range(1, self.n_groups + 1):
            mask = groups == g
            w = pd.Series(0.0, index=f.index)
            if mask.sum() > 0:
                w[mask] = 1.0 / mask.sum()
            weights[f"G{g}"] = w

        # 多空组合
        w_ls = weights[f"G{self.n_groups}"] - weights["G1"]
        weights["L-S"] = w_ls

        return weights

    def calc_turnover_series(self) -> pd.DataFrame:
        """计算每期各层换手率"""
        dates = self.factor.index
        prev_weights = {}
        turnover_records = []

        for date in dates:
            curr_weights = self._build_group_weights(date)
            if not curr_weights:
                continue

            if prev_weights:
                row = {"date": date}
                for group_name in curr_weights:
                    if group_name in prev_weights:
                        cost_info = self.cost_model.estimate_turnover_cost(
                            prev_weights[group_name], curr_weights[group_name]
                        )
                        row[f"{group_name}_turnover"] = cost_info["turnover"]
                        row[f"{group_name}_cost_bps"] = cost_info["total_cost_bps"]
                    else:
                        row[f"{group_name}_turnover"] = np.nan
                        row[f"{group_name}_cost_bps"] = np.nan
                turnover_records.append(row)

            prev_weights = curr_weights

        return pd.DataFrame(turnover_records).set_index("date")

    def calc_net_group_returns(
        self,
        gross_group_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """扣除交易成本后的分层收益"""
        turnover_df = self.calc_turnover_series()

        net_returns = gross_group_returns.copy()
        for col in gross_group_returns.columns:
            cost_col = f"{col}_cost_bps"
            if cost_col in turnover_df.columns:
                common_idx = net_returns.index.intersection(turnover_df.index)
                net_returns.loc[common_idx, col] -= (
                    turnover_df.loc[common_idx, cost_col] / 1e4
                )

        return net_returns

    def summary(self) -> dict:
        """换手率摘要"""
        turnover_df = self.calc_turnover_series()
        if turnover_df.empty:
            return {}

        result = {}
        for g in range(1, self.n_groups + 1):
            col = f"G{g}_turnover"
            cost_col = f"G{g}_cost_bps"
            if col in turnover_df.columns:
                result[f"G{g}"] = {
                    "avg_turnover": turnover_df[col].mean(),
                    "avg_cost_bps": turnover_df[cost_col].mean(),
                    "ann_cost": turnover_df[cost_col].mean() * 52 / 1e4,
                }

        ls_col = "L-S_turnover"
        ls_cost_col = "L-S_cost_bps"
        if ls_col in turnover_df.columns:
            result["L-S"] = {
                "avg_turnover": turnover_df[ls_col].mean(),
                "avg_cost_bps": turnover_df[ls_cost_col].mean(),
                "ann_cost": turnover_df[ls_cost_col].mean() * 52 / 1e4,
            }

        return result
