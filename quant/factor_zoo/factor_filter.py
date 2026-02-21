"""
因子筛选与去冗余模块 - Polars 实现
=====================================
提供以下功能:
- calc_ic: 计算截面 Rank IC
- filter_by_ic: 按 IC 阈值筛选因子
- remove_redundant: 因子间相关性去冗余
- factor_report: 因子综合评估报告 (IC/ICIR/覆盖率/分位数收益)
"""

import polars as pl
import numpy as np
from typing import Optional


# ============================================================
# IC 计算
# ============================================================

def calc_ic(
    df: pl.LazyFrame,
    factor_cols: list[str],
    fwd_return_col: str = "fwd_return",
) -> pl.DataFrame:
    """
    计算截面 Rank IC (信息系数)

    对每个交易日，计算因子截面排名与前向收益截面排名的 Spearman 相关系数

    参数:
        df: 包含因子列和前向收益列的 LazyFrame
        factor_cols: 因子列名列表
        fwd_return_col: 前向收益列名

    返回:
        DataFrame，每行一个日期，每列一个因子的 IC 值
        列名: date, {factor_col}_ic, ...
    """
    # 过滤缺失值
    df_filtered = df.filter(pl.col(fwd_return_col).is_not_null())

    # 对每个因子计算截面 rank IC
    ic_exprs = []
    for col in factor_cols:
        ic_expr = pl.corr(
            pl.col(col).rank(),
            pl.col(fwd_return_col).rank(),
        ).alias(f"{col}_ic")
        ic_exprs.append(ic_expr)

    ic_df = (
        df_filtered
        .group_by("date")
        .agg(ic_exprs)
        .sort("date")
        .collect()
    )

    return ic_df


def calc_ic_summary(
    ic_df: pl.DataFrame,
    factor_cols: list[str],
) -> pl.DataFrame:
    """
    计算 IC 摘要统计

    参数:
        ic_df: calc_ic 返回的 IC 时间序列 DataFrame
        factor_cols: 因子列名列表

    返回:
        DataFrame，包含每个因子的:
        - mean_ic: 平均 IC
        - std_ic: IC 标准差
        - icir: IC / IC_std (信息比率)
        - abs_ic: 平均 |IC|
        - pos_ratio: IC > 0 的比例
    """
    records = []
    for col in factor_cols:
        ic_col = f"{col}_ic"
        if ic_col not in ic_df.columns:
            continue

        ic_series = ic_df.get_column(ic_col).drop_nulls()
        if len(ic_series) == 0:
            continue

        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std())
        icir = mean_ic / (std_ic + 1e-12)
        abs_ic = float(ic_series.abs().mean())
        pos_ratio = float((ic_series > 0).sum()) / len(ic_series)

        records.append({
            "factor": col,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "abs_ic": abs_ic,
            "pos_ratio": pos_ratio,
        })

    return pl.DataFrame(records)


# ============================================================
# IC 筛选
# ============================================================

def filter_by_ic(
    df: pl.LazyFrame,
    factor_cols: list[str],
    fwd_return_col: str = "fwd_return",
    threshold: float = 0.02,
) -> tuple[pl.LazyFrame, list[str]]:
    """
    按平均 |IC| 阈值筛选因子

    参数:
        df: 包含因子列和前向收益列的 LazyFrame
        factor_cols: 因子列名列表
        fwd_return_col: 前向收益列名
        threshold: IC 阈值，默认 0.02

    返回:
        (筛选后的 LazyFrame (只保留通过的因子列), 通过筛选的因子列名列表)
    """
    # 计算 IC
    ic_df = calc_ic(df, factor_cols, fwd_return_col)

    # 计算每个因子的平均 |IC|
    kept_cols = []
    for col in factor_cols:
        ic_col = f"{col}_ic"
        if ic_col not in ic_df.columns:
            continue

        ic_series = ic_df.get_column(ic_col).drop_nulls()
        if len(ic_series) == 0:
            continue

        avg_abs_ic = float(ic_series.abs().mean())
        if avg_abs_ic >= threshold:
            kept_cols.append(col)

    # 保留 date, asset, fwd_return 和通过筛选的因子列
    base_cols = ["date", "asset"]
    if fwd_return_col in df.collect_schema().names():
        base_cols.append(fwd_return_col)

    select_cols = base_cols + kept_cols
    df_filtered = df.select(select_cols)

    return df_filtered, kept_cols


# ============================================================
# 相关性去冗余
# ============================================================

def remove_redundant(
    df: pl.LazyFrame,
    factor_cols: list[str],
    corr_threshold: float = 0.7,
    fwd_return_col: Optional[str] = "fwd_return",
    method: str = "ic_priority",
) -> tuple[pl.LazyFrame, list[str]]:
    """
    因子间相关性去冗余

    对因子两两计算截面相关系数均值，若超过阈值则剔除 IC 较低的那个

    参数:
        df: 包含因子列的 LazyFrame
        factor_cols: 因子列名列表
        corr_threshold: 相关性阈值，默认 0.7
        fwd_return_col: 前向收益列名 (用于确定保留优先级)
        method: 去冗余策略
            - "ic_priority": 保留 IC 更高的因子 (需要 fwd_return_col)
            - "first": 保留先出现的因子

    返回:
        (去冗余后的 LazyFrame, 保留的因子列名列表)
    """
    if len(factor_cols) <= 1:
        return df, factor_cols

    df_collected = df.collect()

    # 计算因子间相关系数矩阵 (用截面均值)
    n = len(factor_cols)
    corr_matrix = np.eye(n)

    # 按日期分组计算截面相关系数，取时间均值
    dates = df_collected.get_column("date").unique().sort()

    # 使用 numpy 批量计算
    pair_corrs = np.zeros((n, n))
    pair_counts = np.zeros((n, n))

    for date_val in dates:
        day_df = df_collected.filter(pl.col("date") == date_val)
        # 提取因子矩阵
        factor_matrix = day_df.select(factor_cols).to_numpy()

        # 跳过样本太少的日期
        if factor_matrix.shape[0] < 10:
            continue

        # 对每列排名
        ranked = np.zeros_like(factor_matrix)
        for j in range(n):
            col_data = factor_matrix[:, j]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.sum() < 5:
                ranked[:, j] = np.nan
            else:
                from scipy.stats import rankdata
                ranked[valid_mask, j] = rankdata(col_data[valid_mask])
                ranked[~valid_mask, j] = np.nan

        # 计算相关矩阵
        for i_idx in range(n):
            for j_idx in range(i_idx + 1, n):
                mask = ~np.isnan(ranked[:, i_idx]) & ~np.isnan(ranked[:, j_idx])
                if mask.sum() < 5:
                    continue
                c = np.corrcoef(ranked[mask, i_idx], ranked[mask, j_idx])[0, 1]
                if not np.isnan(c):
                    pair_corrs[i_idx, j_idx] += c
                    pair_corrs[j_idx, i_idx] += c
                    pair_counts[i_idx, j_idx] += 1
                    pair_counts[j_idx, i_idx] += 1

    # 平均相关系数
    pair_counts[pair_counts == 0] = 1
    corr_matrix = pair_corrs / pair_counts
    np.fill_diagonal(corr_matrix, 1.0)

    # 确定保留优先级
    if method == "ic_priority" and fwd_return_col and fwd_return_col in df_collected.columns:
        # 按 IC 大小排序
        ic_df = calc_ic(df.lazy() if isinstance(df, pl.DataFrame) else df, factor_cols, fwd_return_col)
        ic_summary = calc_ic_summary(ic_df, factor_cols)
        # 按 abs_ic 降序排列
        ic_dict = {}
        for row in ic_summary.iter_rows(named=True):
            ic_dict[row["factor"]] = row["abs_ic"]
        # 按 IC 降序排列索引
        priority = sorted(range(n), key=lambda i: ic_dict.get(factor_cols[i], 0), reverse=True)
    else:
        priority = list(range(n))

    # 贪心去冗余: 按优先级遍历，标记冗余因子
    removed = set()
    for idx in priority:
        if idx in removed:
            continue
        for other_idx in priority:
            if other_idx == idx or other_idx in removed:
                continue
            if abs(corr_matrix[idx, other_idx]) >= corr_threshold:
                removed.add(other_idx)

    kept_indices = [i for i in range(n) if i not in removed]
    kept_cols = [factor_cols[i] for i in kept_indices]

    # 保留 date, asset, fwd_return 和去冗余后的因子列
    base_cols = ["date", "asset"]
    if fwd_return_col and fwd_return_col in df_collected.columns:
        base_cols.append(fwd_return_col)

    select_cols = base_cols + kept_cols
    df_filtered = df.select(select_cols) if not isinstance(df, pl.DataFrame) else df.lazy().select(select_cols)

    return df_filtered, kept_cols


# ============================================================
# 因子评估报告
# ============================================================

def factor_report(
    df: pl.LazyFrame,
    factor_cols: list[str],
    fwd_return_col: str = "fwd_return",
    n_quantiles: int = 5,
) -> pl.DataFrame:
    """
    生成因子综合评估报告

    参数:
        df: 包含因子列和前向收益列的 LazyFrame
        factor_cols: 因子列名列表
        fwd_return_col: 前向收益列名
        n_quantiles: 分位数组数，默认 5

    返回:
        DataFrame，每行一个因子，包含:
        - factor: 因子名
        - mean_ic: 平均 IC
        - icir: IC 信息比率
        - abs_ic: 平均 |IC|
        - pos_ratio: IC > 0 的比例
        - coverage: 因子覆盖率 (非空比例)
        - q1_return: 第1分位数组的平均收益
        - q5_return: 第5分位数组的平均收益
        - long_short: Q5 - Q1 收益 (多空收益)
    """
    df_collected = df.collect()

    # 计算 IC 统计
    ic_df = calc_ic(df.lazy() if isinstance(df, pl.DataFrame) else df, factor_cols, fwd_return_col)
    ic_summary = calc_ic_summary(ic_df, factor_cols)

    # 转换为字典便于查找
    ic_dict = {}
    for row in ic_summary.iter_rows(named=True):
        ic_dict[row["factor"]] = row

    total_rows = len(df_collected)
    records = []

    for col in factor_cols:
        record = {"factor": col}

        # IC 统计
        ic_info = ic_dict.get(col, {})
        record["mean_ic"] = ic_info.get("mean_ic", None)
        record["icir"] = ic_info.get("icir", None)
        record["abs_ic"] = ic_info.get("abs_ic", None)
        record["pos_ratio"] = ic_info.get("pos_ratio", None)

        # 覆盖率
        non_null = df_collected.get_column(col).drop_nulls().len()
        record["coverage"] = non_null / total_rows if total_rows > 0 else 0.0

        # 分位数收益
        if fwd_return_col in df_collected.columns:
            valid_df = df_collected.filter(
                pl.col(col).is_not_null() & pl.col(fwd_return_col).is_not_null()
            )

            if len(valid_df) > 0:
                # 截面分位数标签
                quantile_df = (
                    valid_df.lazy()
                    .with_columns(
                        (pl.col(col).rank().over("date") /
                         pl.col(col).count().over("date") * n_quantiles)
                        .cast(pl.Int32)
                        .clip(0, n_quantiles - 1)
                        .alias("_quantile")
                    )
                    .group_by("_quantile")
                    .agg(pl.col(fwd_return_col).mean().alias("mean_ret"))
                    .sort("_quantile")
                    .collect()
                )

                q_returns = {}
                for row in quantile_df.iter_rows(named=True):
                    q_returns[row["_quantile"]] = row["mean_ret"]

                record["q1_return"] = q_returns.get(0, None)
                record[f"q{n_quantiles}_return"] = q_returns.get(n_quantiles - 1, None)

                # 多空收益
                q1 = q_returns.get(0, 0) or 0
                qn = q_returns.get(n_quantiles - 1, 0) or 0
                record["long_short"] = qn - q1
            else:
                record["q1_return"] = None
                record[f"q{n_quantiles}_return"] = None
                record["long_short"] = None
        else:
            record["q1_return"] = None
            record[f"q{n_quantiles}_return"] = None
            record["long_short"] = None

        records.append(record)

    report = pl.DataFrame(records)

    # 按 abs_ic 降序排列
    if "abs_ic" in report.columns:
        report = report.sort("abs_ic", descending=True, nulls_last=True)

    return report
