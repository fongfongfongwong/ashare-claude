"""
完整 Factor Mining Pipeline
─────────────────────────────
1. 数据加载
2. 因子库构建 (5000+)
3. IC 筛选 + 去冗余
4. 14 模型对比
5. 模型融合
6. 结果输出
"""
import sys
import os
import time
import warnings

import numpy as np
import pandas as pd

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")


def run_full_pipeline(
    use_real_data: bool = False,
    n_stocks: int = 300,
    enable_gp: bool = False,
    gp_population: int = 500,
    gp_generations: int = 100,
    max_model_features: int = 500,
    models_to_run: list = None,
):
    """运行完整 Factor Mining Pipeline"""
    total_start = time.time()

    print("=" * 76)
    print("  Factor Mining Pipeline v3.0")
    print("  5000+ Factors × 14 Models | Polars + MLX")
    print("=" * 76)

    # ═══════════════════════════════════════════════════════════════════
    #  1. 数据加载
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 76)
    print("  [1/5] 数据加载")
    print("─" * 76)
    t0 = time.time()

    if use_real_data:
        try:
            from data.akshare_loader import load_stock_universe
            data = load_stock_universe(
                n_stocks=n_stocks,
                start_date="20200105",
                end_date="20241231",
                freq="W-FRI",
            )
        except Exception as e:
            print(f"  [!] 真实数据失败: {e}, 回退模拟数据")
            from data.generator import generate_stock_universe
            data = generate_stock_universe(n_stocks=n_stocks)
    else:
        from data.generator import generate_stock_universe
        data = generate_stock_universe(n_stocks=n_stocks)

    n_stk = len(data["returns"].columns)
    n_periods = len(data["returns"])
    print(f"  股票数: {n_stk}")
    print(f"  时间: {data['returns'].index[0].date()} ~ {data['returns'].index[-1].date()}")
    print(f"  周期: {n_periods}")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    #  2. 因子库构建
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 76)
    print("  [2/5] 因子库构建 (目标 5000+)")
    print("─" * 76)
    t0 = time.time()

    from pipeline.factor_pipeline import build_factor_library

    factor_result = build_factor_library(
        data,
        enable_alpha101=True,
        enable_alpha158=True,
        enable_alpha360=True,
        enable_operator_expand=True,
        enable_gp=enable_gp,
        enable_cross=True,
        gp_population=gp_population,
        gp_generations=gp_generations,
        ic_threshold=0.02,
        corr_threshold=0.7,
        verbose=True,
    )

    stats = factor_result["stats"]
    filtered_factors = factor_result["filtered_factors"]

    print(f"\n  ┌─ 因子库统计 ─────────────────────────────────────────────┐")
    print(f"  │ 层级            │ 数量      │ 说明                      │")
    print(f"  ├─────────────────┼───────────┼───────────────────────────┤")
    print(f"  │ Alpha101        │ {stats['alpha101']:>6d}    │ WorldQuant 公式因子       │")
    print(f"  │ Alpha158        │ {stats['alpha158']:>6d}    │ Qlib 技术指标             │")
    print(f"  │ Alpha360        │ {stats['alpha360']:>6d}    │ Qlib 标准化价量序列       │")
    print(f"  │ 算子扩展        │ {stats['operator_expand']:>6d}    │ 时间窗口×算子组合         │")
    print(f"  │ GP 遗传编程     │ {stats['gp']:>6d}    │ Warm-Start GP             │")
    print(f"  │ 交叉特征        │ {stats['cross']:>6d}    │ 因子间交互                │")
    print(f"  ├─────────────────┼───────────┼───────────────────────────┤")
    print(f"  │ 原始总数        │ {stats['total_raw']:>6d}    │                           │")
    print(f"  │ 筛选后          │ {stats['total_filtered']:>6d}    │ IC>0.02, corr<0.7         │")
    print(f"  └─────────────────┴───────────┴───────────────────────────┘")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    #  3. 模型对比
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 76)
    print("  [3/5] 多模型对比 (14 Models)")
    print("─" * 76)
    t0 = time.time()

    from pipeline.model_pipeline import run_model_comparison, format_model_comparison

    model_results = run_model_comparison(
        factors=filtered_factors,
        returns=data["returns"],
        train_years=3,
        val_years=1,
        test_years=1,
        n_splits=3,
        models_to_run=models_to_run,
        verbose=True,
        max_features=max_model_features,
    )

    # 格式化输出
    comp_table = format_model_comparison(model_results)
    print(f"\n  ┌─ 模型 IC 有效性对比 ──────────────────────────────────────────────────┐")
    print(comp_table.to_string(index=False))
    print(f"  └────────────────────────────────────────────────────────────────────────┘")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    #  4. Top 因子排名
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 76)
    print("  [4/5] Top-20 因子 (按 ICIR)")
    print("─" * 76)
    t0 = time.time()

    # 计算每个因子的 IC/ICIR
    from scipy import stats as sp_stats

    fwd_ret = data["returns"].shift(-1)
    factor_ic_stats = []

    for fname, factor_df in filtered_factors.items():
        try:
            if isinstance(factor_df, pd.DataFrame):
                pdf = factor_df
            else:
                pdf = factor_df.to_pandas()
                if "date" in pdf.columns:
                    pdf = pdf.set_index("date")

            ic_list = []
            common_dates = pdf.index.intersection(fwd_ret.index)
            for date in common_dates:
                f = pdf.loc[date].dropna()
                r = fwd_ret.loc[date].dropna()
                common = f.index.intersection(r.index)
                if len(common) < 30:
                    continue
                corr, _ = sp_stats.spearmanr(f.loc[common], r.loc[common])
                ic_list.append(corr)

            if len(ic_list) > 10:
                ic_arr = np.array(ic_list)
                ic_mean = ic_arr.mean()
                ic_std = ic_arr.std()
                icir = ic_mean / ic_std if ic_std > 0 else 0
                factor_ic_stats.append({
                    "Factor": fname,
                    "IC_mean": ic_mean,
                    "IC_std": ic_std,
                    "ICIR": icir,
                    "IC>0%": (ic_arr > 0).mean(),
                })
        except Exception:
            pass

    if factor_ic_stats:
        ic_df = pd.DataFrame(factor_ic_stats)
        ic_df = ic_df.sort_values("ICIR", ascending=False).head(20).reset_index(drop=True)
        print(ic_df.to_string(index=True))
    else:
        print("  无因子 IC 统计")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    #  5. 总结
    # ═══════════════════════════════════════════════════════════════════
    total_time = time.time() - total_start
    print("\n" + "=" * 76)
    print(f"  Pipeline 完成")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  股票: {n_stk} | 周期: {n_periods}")
    print(f"  因子: {stats['total_raw']} (原始) → {stats['total_filtered']} (筛选后)")
    print(f"  模型: {len(model_results)} 个")

    # 最优模型
    if model_results:
        best = max(model_results.items(), key=lambda x: x[1].get("ICIR", 0) if np.isfinite(x[1].get("ICIR", 0)) else 0)
        print(f"  最优模型: {best[0]} (ICIR={best[1]['ICIR']:.4f})")

    print("=" * 76)

    return {
        "data": data,
        "factor_result": factor_result,
        "model_results": model_results,
        "factor_ic_stats": factor_ic_stats if factor_ic_stats else [],
    }


if __name__ == "__main__":
    use_real = "--real" in sys.argv
    enable_gp = "--gp" in sys.argv

    n = 300
    for arg in sys.argv[1:]:
        if arg.startswith("--n="):
            n = int(arg.split("=")[1])

    run_full_pipeline(
        use_real_data=use_real,
        n_stocks=n,
        enable_gp=enable_gp,
    )
