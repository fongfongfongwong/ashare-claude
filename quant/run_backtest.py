"""
主运行脚本: 完整量化因子分析 Pipeline
1. 数据加载 (模拟数据 / AKShare 真实数据)
2. 5因子构建 + 行业中性化
3. 单因子回测 (IC/ICIR/分层)
4. 多因子合成 (等权/IC/ICIR/MaxICIR)
5. 换手率与交易成本分析
6. Barra 风险模型
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from data.generator import generate_stock_universe
from factors import build_all_factors
from backtest import FactorBacktest, run_multi_factor_backtest, format_comparison_table
from alpha_combine import AlphaCombiner
from cost_model import TransactionCostModel, TurnoverAnalyzer
from risk_model import BarraRiskModel

warnings.filterwarnings("ignore")


def run_full_pipeline(use_real_data: bool = False, n_stocks: int = 300):
    """运行完整回测流程"""
    total_start = time.time()

    print("=" * 72)
    print("  量化多因子分析系统 v2.0")
    print("  Momentum | Value | Quality | Low-Vol | Size")
    print("=" * 72)

    # =================================================================
    #  1. 数据
    # =================================================================
    print("\n" + "─" * 72)
    print("  [1/6] 数据加载")
    print("─" * 72)
    t0 = time.time()

    if use_real_data:
        try:
            from data.akshare_loader import load_stock_universe
            data = load_stock_universe(
                n_stocks=n_stocks,
                start_date="20150105",
                end_date="20241231",
                freq="W-FRI",
            )
        except Exception as e:
            print(f"  [!] 真实数据加载失败: {e}")
            print(f"  [!] 回退到模拟数据")
            data = generate_stock_universe(n_stocks=n_stocks)
    else:
        data = generate_stock_universe(n_stocks=n_stocks)

    n_stk = len(data["returns"].columns)
    n_periods = len(data["returns"])
    print(f"  股票数: {n_stk}")
    print(f"  时间范围: {data['returns'].index[0].date()} ~ {data['returns'].index[-1].date()}")
    print(f"  周期数: {n_periods}")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # =================================================================
    #  2. 因子构建
    # =================================================================
    print("\n" + "─" * 72)
    print("  [2/6] 因子构建 (行业中性化)")
    print("─" * 72)
    t0 = time.time()
    factors = build_all_factors(data, neutralize=True)
    for name, f in factors.items():
        valid_pct = f.notna().sum().sum() / (f.shape[0] * f.shape[1]) * 100
        print(f"  {name:12s}: 覆盖率 {valid_pct:.1f}%")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # =================================================================
    #  3. 单因子回测
    # =================================================================
    print("\n" + "─" * 72)
    print("  [3/6] 单因子回测 (5分层)")
    print("─" * 72)
    t0 = time.time()
    results = run_multi_factor_backtest(factors, data["returns"], n_groups=5)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # 对比表
    table = format_comparison_table(results)
    print("\n  ┌─ 因子有效性对比 ─────────────────────────────────────────────┐")
    print(table.to_string())
    print("  └──────────────────────────────────────────────────────────────┘")

    # 分层详情
    print("\n  分层年化收益:")
    for name in factors:
        gp = results[name]["group_performance"]
        line = f"    {name:12s} │"
        for g in ["G1", "G2", "G3", "G4", "G5", "L-S"]:
            if g in gp:
                ret = gp[g]["ann_return"]
                line += f" {g}:{ret:+.1%} │"
        print(line)

    # =================================================================
    #  4. 多因子合成
    # =================================================================
    print("\n" + "─" * 72)
    print("  [4/6] 多因子合成")
    print("─" * 72)
    t0 = time.time()

    combiner = AlphaCombiner(factors, data["returns"], ic_window=52, halflife=26)

    combine_results = {}
    for method in ["equal", "ic", "icir", "max_icir"]:
        res = combiner.summary(method)
        combine_results[method] = res

    # 对比表
    print("\n  ┌─ 合成方法对比 ──────────────────────────────────────┐")
    print(f"  │ {'Method':12s} │ {'IC_mean':>8s} │ {'ICIR':>8s} │ {'IC正比例':>8s} │")
    print(f"  ├{'─' * 14}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10}┤")
    for method, res in combine_results.items():
        print(f"  │ {method:12s} │ {res['IC_mean']:>8.4f} │ {res['ICIR']:>8.4f} │ {res['IC_positive_ratio']:>7.1%} │")
    print(f"  └{'─' * 14}┴{'─' * 10}┴{'─' * 10}┴{'─' * 10}┘")

    # 最佳方法的权重分布
    best_method = max(combine_results, key=lambda m: abs(combine_results[m]["ICIR"]))
    best_weights = combine_results[best_method]["avg_weights"]
    print(f"\n  最佳合成方法: {best_method}")
    print(f"  平均权重分配:")
    for name, w in sorted(best_weights.items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(w) * 30)
        sign = "+" if w > 0 else "-"
        print(f"    {name:12s}: {sign}{abs(w):.2%}  {bar}")

    print(f"  耗时: {time.time() - t0:.1f}s")

    # =================================================================
    #  5. 换手率 & 交易成本
    # =================================================================
    print("\n" + "─" * 72)
    print("  [5/6] 换手率与交易成本分析")
    print("─" * 72)
    t0 = time.time()

    cost_model = TransactionCostModel(
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        impact_coeff=0.1,
    )

    print("\n  各因子换手率 & 交易成本:")
    print(f"  {'Factor':12s} │ {'周换手率':>8s} │ {'成本(bps)':>9s} │ {'年化成本':>8s} │ {'净LS年化':>9s}")
    print(f"  {'─' * 12}─┼{'─' * 10}┼{'─' * 11}┼{'─' * 10}┼{'─' * 11}")

    for name, factor in factors.items():
        analyzer = TurnoverAnalyzer(factor, n_groups=5, cost_model=cost_model)
        turnover_summary = analyzer.summary()

        ls_info = turnover_summary.get("L-S", {})
        ls_turnover = ls_info.get("avg_turnover", 0)
        ls_cost_bps = ls_info.get("avg_cost_bps", 0)
        ls_ann_cost = ls_info.get("ann_cost", 0)

        # 净多空收益
        gross_ls = results[name]["group_performance"].get("L-S", {}).get("ann_return", 0)
        net_ls = gross_ls - ls_ann_cost

        print(f"  {name:12s} │ {ls_turnover:>7.1%} │ {ls_cost_bps:>8.1f} │ {ls_ann_cost:>7.2%} │ {net_ls:>+8.2%}")

    print(f"\n  耗时: {time.time() - t0:.1f}s")

    # =================================================================
    #  6. Barra 风险模型
    # =================================================================
    print("\n" + "─" * 72)
    print("  [6/6] Barra 风险模型 (增强版)")
    print("─" * 72)
    t0 = time.time()

    model = BarraRiskModel(
        factors=factors,
        returns=data["returns"],
        industry=data["industry"],
        market_cap=data["market_cap"],
        halflife=52,
        nw_lags=2,
        vol_halflife=26,
        eigenfactor_mc=100,
        eigenfactor_alpha=1.2,
    )

    # 因子收益率
    factor_ret = model.estimate_factor_returns(lookback=52)
    print(f"  因子收益率: {len(factor_ret)} 期 x {len(factor_ret.columns)} 因子")

    # 协方差矩阵
    factor_cov = model.estimate_factor_covariance(factor_ret)
    style_factors = [f for f in factor_cov.index
                     if not f.startswith("IND_") and f != "country"]

    print(f"\n  风格因子年化波动率:")
    for sf in style_factors:
        vol = np.sqrt(factor_cov.loc[sf, sf]) * np.sqrt(52) * 100
        print(f"    {sf:12s}: {vol:.2f}%")

    # 相关系数矩阵
    style_cov = factor_cov.loc[style_factors, style_factors]
    style_std = np.sqrt(np.diag(style_cov.values))
    style_corr = style_cov.values / np.outer(style_std, style_std)
    corr_df = pd.DataFrame(style_corr, index=style_factors, columns=style_factors)
    print(f"\n  风格因子相关系数:")
    print(corr_df.round(3).to_string())

    # 特异性风险
    spec_risk = model.estimate_specific_risk(factor_ret)
    print(f"\n  特异性风险 (年化):")
    print(f"    均值:   {spec_risk.mean():.2%}")
    print(f"    中位数: {spec_risk.median():.2%}")
    print(f"    [10%, 90%]: [{spec_risk.quantile(0.1):.2%}, {spec_risk.quantile(0.9):.2%}]")

    # 纯因子组合
    last_date = data["returns"].index[-2]
    print(f"\n  纯因子组合 (date={last_date.date()}):")
    for factor_name in style_factors:
        try:
            w = model.build_pure_factor_portfolio(factor_name, last_date)
            n_long = (w > 0).sum()
            n_short = (w < 0).sum()
            top_3 = w.nlargest(3)
            print(f"    {factor_name:12s}: L{n_long}/S{n_short}, "
                  f"top3权重: [{', '.join(f'{v:.2%}' for v in top_3.values)}]")
        except Exception as e:
            print(f"    {factor_name:12s}: 构建失败 - {e}")

    # 风险分解示例
    print(f"\n  风险分解示例 (momentum 纯因子组合):")
    try:
        w_mom = model.build_pure_factor_portfolio("momentum", last_date)
        risk_decomp = model.full_risk_decomposition(w_mom, last_date)
        print(f"    总风险:     {risk_decomp['total_risk']:.2%}")
        print(f"    因子风险:   {risk_decomp['factor_risk']:.2%} "
              f"({risk_decomp['factor_risk_pct']:.1%})")
        print(f"    特异性风险: {risk_decomp['specific_risk']:.2%}")

        if "style_risk_contrib" in risk_decomp:
            print(f"    风格因子风险贡献:")
            for sname, contrib in risk_decomp["style_risk_contrib"].items():
                pct = contrib["risk_contrib_pct"]
                if abs(pct) > 0.001:
                    print(f"      {sname:12s}: {pct:>+6.1%}")
    except Exception as e:
        print(f"    风险分解失败: {e}")

    print(f"\n  耗时: {time.time() - t0:.1f}s")

    # =================================================================
    #  总结
    # =================================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 72)
    print(f"  Pipeline 完成 | 总耗时: {total_time:.1f}s")
    print(f"  股票: {n_stk} | 周期: {n_periods} | 因子: {len(factors)}")
    print("=" * 72)


if __name__ == "__main__":
    use_real = "--real" in sys.argv
    n = 300
    for arg in sys.argv[1:]:
        if arg.startswith("--n="):
            n = int(arg.split("=")[1])

    run_full_pipeline(use_real_data=use_real, n_stocks=n)
