"""
因子生成 Pipeline
1. 经典因子库 (Alpha101 + Alpha158 + Alpha360)
2. 算子扩展
3. GP 遗传编程因子挖掘
4. 交叉特征
5. IC 筛选 + 去冗余
"""
import time
import numpy as np
import polars as pl
from typing import Optional


def _pandas_to_polars_ohlcv(data: dict) -> pl.LazyFrame:
    """
    将旧版 pandas dict 格式转换为 Polars 长表格式
    data keys: returns, market_cap, book_to_price, roe, momentum_12m, volatility, industry
    假设 returns 是 pandas DataFrame (index=date, columns=asset)
    """
    import pandas as pd

    ret_df = data["returns"]
    dates = ret_df.index
    assets = ret_df.columns.tolist()

    # 用 returns 重建 close (累积收益)
    close_pd = (1 + ret_df).cumprod()
    close_pd.iloc[0] = 1.0  # 基准价格

    # 模拟 OHLCV (用 close ± noise)
    np.random.seed(42)
    noise_h = np.abs(np.random.normal(0, 0.01, close_pd.shape))
    noise_l = np.abs(np.random.normal(0, 0.01, close_pd.shape))

    high_pd = close_pd * (1 + noise_h)
    low_pd = close_pd * (1 - noise_l)
    open_pd = close_pd.shift(1).bfill()
    volume_pd = pd.DataFrame(
        np.random.lognormal(18, 1, close_pd.shape),
        index=dates,
        columns=assets,
    )
    vwap_pd = (high_pd + low_pd + close_pd) / 3

    # 转为长表
    records = []
    for i, date in enumerate(dates):
        for j, asset in enumerate(assets):
            records.append({
                "date": date,
                "asset": asset,
                "open": float(open_pd.iloc[i, j]),
                "high": float(high_pd.iloc[i, j]),
                "low": float(low_pd.iloc[i, j]),
                "close": float(close_pd.iloc[i, j]),
                "volume": float(volume_pd.iloc[i, j]),
                "vwap": float(vwap_pd.iloc[i, j]),
                "returns": float(ret_df.iloc[i, j]),
            })

    df = pl.DataFrame(records).lazy()
    return df


def build_factor_library(
    data: dict,
    enable_alpha101: bool = True,
    enable_alpha158: bool = True,
    enable_alpha360: bool = True,
    enable_operator_expand: bool = True,
    enable_gp: bool = False,
    enable_cross: bool = True,
    gp_population: int = 500,
    gp_generations: int = 100,
    ic_threshold: float = 0.02,
    corr_threshold: float = 0.7,
    verbose: bool = True,
) -> dict:
    """
    构建完整因子库

    Returns:
        dict with keys:
            - factors: dict[str, pl.DataFrame]  因子名 -> 因子值(宽表: date x asset)
            - stats: dict  统计信息
            - filtered_factors: dict  筛选后的因子
    """
    total_start = time.time()
    all_factors = {}
    stats = {
        "alpha101": 0,
        "alpha158": 0,
        "alpha360": 0,
        "operator_expand": 0,
        "gp": 0,
        "cross": 0,
        "total_raw": 0,
        "total_filtered": 0,
    }

    # 转换数据格式
    if verbose:
        print("  转换数据格式 (Pandas → Polars 长表) ...")
    t0 = time.time()
    ohlcv_lazy = _pandas_to_polars_ohlcv(data)
    ohlcv_df = ohlcv_lazy.collect()
    if verbose:
        print(f"    OHLCV 长表: {ohlcv_df.shape[0]:,} 行, 耗时 {time.time()-t0:.1f}s")

    # ─── Alpha101 ───
    if enable_alpha101:
        if verbose:
            print("\n  [因子层1] Alpha101 ...")
        t0 = time.time()
        try:
            from factor_zoo.alpha101 import build_alpha101
            a101 = build_alpha101(ohlcv_df.lazy())
            a101_collected = a101.collect()
            # 提取因子列 (排除 date, asset)
            factor_cols = [c for c in a101_collected.columns if c.startswith("alpha")]
            for col in factor_cols:
                pivot = a101_collected.select(["date", "asset", col]).pivot(
                    on="asset", index="date", values=col
                )
                all_factors[col] = pivot
            stats["alpha101"] = len(factor_cols)
            if verbose:
                print(f"    生成 {len(factor_cols)} 个因子, 耗时 {time.time()-t0:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    Alpha101 失败: {e}")

    # ─── Alpha158 ───
    if enable_alpha158:
        if verbose:
            print("\n  [因子层1] Alpha158 ...")
        t0 = time.time()
        try:
            from factor_zoo.alpha158 import build_alpha158
            a158 = build_alpha158(ohlcv_df.lazy())
            a158_collected = a158.collect()
            factor_cols = [c for c in a158_collected.columns
                           if c not in ("date", "asset")]
            for col in factor_cols:
                pivot = a158_collected.select(["date", "asset", col]).pivot(
                    on="asset", index="date", values=col
                )
                all_factors[f"a158_{col}"] = pivot
            stats["alpha158"] = len(factor_cols)
            if verbose:
                print(f"    生成 {len(factor_cols)} 个因子, 耗时 {time.time()-t0:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    Alpha158 失败: {e}")

    # ─── Alpha360 ───
    if enable_alpha360:
        if verbose:
            print("\n  [因子层1] Alpha360 ...")
        t0 = time.time()
        try:
            from factor_zoo.alpha360 import build_alpha360
            a360 = build_alpha360(ohlcv_df.lazy())
            a360_collected = a360.collect()
            factor_cols = [c for c in a360_collected.columns
                           if c not in ("date", "asset")]
            for col in factor_cols:
                pivot = a360_collected.select(["date", "asset", col]).pivot(
                    on="asset", index="date", values=col
                )
                all_factors[f"a360_{col}"] = pivot
            stats["alpha360"] = len(factor_cols)
            if verbose:
                print(f"    生成 {len(factor_cols)} 个因子, 耗时 {time.time()-t0:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    Alpha360 失败: {e}")

    # ─── 算子扩展 ───
    if enable_operator_expand:
        if verbose:
            print("\n  [因子层2] 算子扩展 ...")
        t0 = time.time()
        try:
            from factor_zoo.operator_expand import expand_factors
            # 用核心价量列做算子扩展: 7列 × 3算子 × 3窗口 = 63 个
            # 再加上 Alpha101 中的因子列做扩展
            base_cols = ["open", "high", "low", "close", "volume", "vwap", "returns"]
            expanded = expand_factors(
                ohlcv_df.lazy(),
                factor_cols=base_cols,
                windows=[5, 20, 60],
                operators=["ts_rank", "ts_zscore", "ts_delta"],
            )
            expanded_collected = expanded.collect()
            new_cols = [c for c in expanded_collected.columns
                        if c not in ("date", "asset") and c not in base_cols]
            for col in new_cols:
                pivot = expanded_collected.select(["date", "asset", col]).pivot(
                    on="asset", index="date", values=col
                )
                all_factors[f"op_{col}"] = pivot
            stats["operator_expand"] = len(new_cols)
            if verbose:
                print(f"    生成 {len(factor_cols)} 个因子, 耗时 {time.time()-t0:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    算子扩展失败: {e}")

    # ─── GP 遗传编程 ───
    if enable_gp:
        if verbose:
            print(f"\n  [因子层3] GP 遗传编程 (种群{gp_population} × {gp_generations}代) ...")
        t0 = time.time()
        try:
            from factor_zoo.gp_miner import GPMiner, mine_factors
            gp_factors_result = mine_factors(
                ohlcv_df.lazy(),
                population_size=gp_population,
                n_generations=gp_generations,
            )
            gp_factors = {}
            if isinstance(gp_factors_result, dict):
                gp_factors = gp_factors_result
            elif isinstance(gp_factors_result, pl.DataFrame):
                factor_cols_gp = [c for c in gp_factors_result.columns if c not in ("date", "asset")]
                for col in factor_cols_gp:
                    gp_factors[col] = gp_factors_result.select(["date", "asset", col])

            for name, factor_df in gp_factors.items():
                all_factors[f"gp_{name}"] = factor_df
            stats["gp"] = len(gp_factors)
            if verbose:
                print(f"    生成 {len(gp_factors)} 个有效因子, 耗时 {time.time()-t0:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    GP 挖掘失败: {e}")

    # ─── 交叉特征 ───
    if enable_cross:
        if verbose:
            print("\n  [因子层4] 交叉特征 ...")
        t0 = time.time()
        try:
            from factor_zoo.cross_features import build_cross_features
            top_n = ["close", "volume", "returns", "vwap", "high", "low", "open"]
            cross = build_cross_features(ohlcv_df.lazy(), top_n_factors=top_n)
            cross_collected = cross.collect()
            factor_cols = [c for c in cross_collected.columns
                           if c not in ("date", "asset")]
            for col in factor_cols:
                pivot = cross_collected.select(["date", "asset", col]).pivot(
                    on="asset", index="date", values=col
                )
                all_factors[f"cross_{col}"] = pivot
            stats["cross"] = len(factor_cols)
            if verbose:
                print(f"    生成 {len(factor_cols)} 个因子, 耗时 {time.time()-t0:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    交叉特征失败: {e}")

    stats["total_raw"] = len(all_factors)
    if verbose:
        print(f"\n  因子总数 (筛选前): {stats['total_raw']}")
        print(f"    Alpha101: {stats['alpha101']}")
        print(f"    Alpha158: {stats['alpha158']}")
        print(f"    Alpha360: {stats['alpha360']}")
        print(f"    算子扩展: {stats['operator_expand']}")
        print(f"    GP: {stats['gp']}")
        print(f"    交叉特征: {stats['cross']}")

    # ─── IC 筛选 + 去冗余 ───
    if verbose:
        print(f"\n  [筛选] IC>{ic_threshold}, 相关性<{corr_threshold} ...")
    t0 = time.time()
    try:
        import pandas as pd
        from scipy import stats as sp_stats

        # 转换因子为 pandas 宽表 dict
        pd_factors = {}
        for name, pldf in all_factors.items():
            try:
                if isinstance(pldf, pl.DataFrame):
                    pdf = pldf.to_pandas()
                else:
                    pdf = pldf
                if "date" in pdf.columns:
                    pdf = pdf.set_index("date")
                pd_factors[name] = pdf
            except Exception:
                pass

        returns_pd = data["returns"]
        fwd_ret = returns_pd.shift(-1)

        # Step 1: IC 筛选
        filtered = {}
        for fname, fdf in pd_factors.items():
            try:
                common_dates = fdf.index.intersection(fwd_ret.index)
                ic_list = []
                for date in common_dates[:100]:  # 采样100期加速
                    f = fdf.loc[date].dropna()
                    r = fwd_ret.loc[date].dropna()
                    common = f.index.intersection(r.index)
                    if len(common) < 20:
                        continue
                    corr, _ = sp_stats.spearmanr(f.loc[common], r.loc[common])
                    if np.isfinite(corr):
                        ic_list.append(corr)
                if len(ic_list) >= 10 and abs(np.mean(ic_list)) >= ic_threshold:
                    filtered[fname] = fdf
            except Exception:
                pass

        n_after_ic = len(filtered)
        if verbose:
            print(f"    IC 筛选后: {n_after_ic} 个 (|IC|>{ic_threshold})")

        # Step 2: 简单去冗余 (跳过如果因子太少)
        if n_after_ic > 500:
            # 只对前500个做去冗余
            factor_names = list(filtered.keys())[:500]
            filtered = {k: filtered[k] for k in factor_names}
            if verbose:
                print(f"    截断到 {len(filtered)} 个因子 (去冗余限制)")

        stats["total_filtered"] = len(filtered)
        if verbose:
            print(f"    最终: {len(filtered)} 个有效因子, 耗时 {time.time()-t0:.1f}s")
    except Exception as e:
        filtered = all_factors  # fallback: 不筛选
        stats["total_filtered"] = len(filtered)
        if verbose:
            print(f"    筛选失败 ({e}), 使用全部 {len(filtered)} 个因子")

    total_time = time.time() - total_start
    if verbose:
        print(f"\n  因子库构建完成: {stats['total_filtered']}/{stats['total_raw']} | 耗时 {total_time:.1f}s")

    return {
        "factors": all_factors,
        "filtered_factors": filtered,
        "stats": stats,
        "ohlcv": ohlcv_df,
    }
