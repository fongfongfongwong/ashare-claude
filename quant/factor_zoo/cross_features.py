"""
因子交叉特征模块 - Polars 实现
================================
对 top-N 因子两两做交叉运算:
- multiply: 因子乘积
- divide: 因子比值
- subtract: 因子差值

保留 IC > 阈值的交叉因子

输入: 已有因子 DataFrame + top-N 因子名列表
输出: 扩展后的因子 DataFrame (~500 新列)
"""

import polars as pl
import numpy as np
from itertools import combinations
from typing import Optional


# ============================================================
# 交叉运算算子
# ============================================================

def _cross_multiply(col_a: str, col_b: str) -> tuple[pl.Expr, str]:
    """因子乘积"""
    name = f"{col_a}_X_{col_b}_mul"
    expr = (pl.col(col_a) * pl.col(col_b)).alias(name)
    return expr, name


def _cross_divide(col_a: str, col_b: str) -> tuple[pl.Expr, str]:
    """因子比值 (带保护)"""
    name = f"{col_a}_X_{col_b}_div"
    expr = (pl.col(col_a) / (pl.col(col_b).abs() + 1e-12) * pl.col(col_b).sign()).alias(name)
    return expr, name


def _cross_subtract(col_a: str, col_b: str) -> tuple[pl.Expr, str]:
    """因子差值"""
    name = f"{col_a}_X_{col_b}_sub"
    expr = (pl.col(col_a) - pl.col(col_b)).alias(name)
    return expr, name


# 交叉算子注册表
CROSS_OPERATORS = {
    "multiply": _cross_multiply,
    "divide": _cross_divide,
    "subtract": _cross_subtract,
}


# ============================================================
# IC 计算 (截面 Rank IC)
# ============================================================

def _calc_rank_ic_single(
    df: pl.DataFrame,
    factor_col: str,
    fwd_return_col: str,
) -> float:
    """
    计算单个因子的截面平均 Rank IC

    参数:
        df: 包含因子列和前向收益列的 DataFrame
        factor_col: 因子列名
        fwd_return_col: 前向收益列名

    返回:
        平均 Rank IC 绝对值
    """
    # 按日期分组，计算截面 rank 相关系数
    ic_series = (
        df.lazy()
        .filter(pl.col(factor_col).is_not_null() & pl.col(fwd_return_col).is_not_null())
        .group_by("date")
        .agg(
            pl.corr(
                pl.col(factor_col).rank(),
                pl.col(fwd_return_col).rank(),
            ).alias("ic")
        )
        .collect()
        .get_column("ic")
    )

    # 返回平均 IC 绝对值
    valid = ic_series.drop_nulls()
    if len(valid) == 0:
        return 0.0
    return abs(float(valid.mean()))


# ============================================================
# 主函数
# ============================================================

def build_cross_features(
    df: pl.LazyFrame,
    top_n_factors: list[str],
    fwd_return_col: str = "fwd_return",
    ic_threshold: float = 0.02,
    operators: Optional[list[str]] = None,
    batch_size: int = 100,
    max_output_cols: int = 500,
) -> pl.LazyFrame:
    """
    构建因子交叉特征

    参数:
        df: 包含因子列和前向收益列的 LazyFrame
        top_n_factors: top-N 因子名称列表
        fwd_return_col: 前向收益列名 (用于 IC 筛选)
        ic_threshold: IC 筛选阈值，默认 0.02
        operators: 交叉算子列表，默认 ["multiply", "divide", "subtract"]
        batch_size: 每批处理的交叉因子数量
        max_output_cols: 最大输出列数限制

    返回:
        附加了交叉因子列的 LazyFrame
    """
    if operators is None:
        operators = list(CROSS_OPERATORS.keys())

    # 验证算子名称
    invalid_ops = set(operators) - set(CROSS_OPERATORS.keys())
    if invalid_ops:
        raise ValueError(f"未知交叉算子: {invalid_ops}. 可用: {list(CROSS_OPERATORS.keys())}")

    # 生成所有因子对
    factor_pairs = list(combinations(top_n_factors, 2))

    # 第一步: 生成所有交叉因子候选
    all_candidates: list[tuple[pl.Expr, str]] = []
    for col_a, col_b in factor_pairs:
        for op_name in operators:
            op_func = CROSS_OPERATORS[op_name]
            expr, name = op_func(col_a, col_b)
            all_candidates.append((expr, name))

    if not all_candidates:
        return df

    # 第二步: 分批计算交叉因子，然后用 IC 筛选
    # 先 collect 一份用于 IC 计算
    df_collected = df.collect()

    kept_names: list[str] = []

    for i in range(0, len(all_candidates), batch_size):
        batch_candidates = all_candidates[i : i + batch_size]
        batch_exprs = [expr for expr, _ in batch_candidates]
        batch_names = [name for _, name in batch_candidates]

        # 计算当前批次的交叉因子
        df_with_cross = df_collected.lazy().with_columns(batch_exprs).collect()

        # 计算每个交叉因子的 IC，筛选
        for name in batch_names:
            if fwd_return_col in df_with_cross.columns:
                ic = _calc_rank_ic_single(df_with_cross, name, fwd_return_col)
                if ic > ic_threshold:
                    kept_names.append(name)
            else:
                # 没有前向收益列时保留全部
                kept_names.append(name)

            # 达到上限则停止
            if len(kept_names) >= max_output_cols:
                break

        if len(kept_names) >= max_output_cols:
            break

    # 第三步: 只添加通过筛选的交叉因子
    if not kept_names:
        return df

    # 重新构建筛选后的表达式
    name_to_expr = {name: expr for expr, name in all_candidates}
    kept_exprs = [name_to_expr[name] for name in kept_names]

    # 分批添加到 LazyFrame
    for i in range(0, len(kept_exprs), batch_size):
        batch = kept_exprs[i : i + batch_size]
        df = df.with_columns(batch)

    return df


def build_cross_features_no_filter(
    df: pl.LazyFrame,
    top_n_factors: list[str],
    operators: Optional[list[str]] = None,
    batch_size: int = 100,
) -> pl.LazyFrame:
    """
    构建因子交叉特征 (不做 IC 筛选，保留全部)

    适用于后续统一做因子筛选的场景

    参数:
        df: 包含因子列的 LazyFrame
        top_n_factors: top-N 因子名称列表
        operators: 交叉算子列表
        batch_size: 每批处理的交叉因子数量

    返回:
        附加了所有交叉因子列的 LazyFrame
    """
    if operators is None:
        operators = list(CROSS_OPERATORS.keys())

    factor_pairs = list(combinations(top_n_factors, 2))

    all_exprs = []
    for col_a, col_b in factor_pairs:
        for op_name in operators:
            op_func = CROSS_OPERATORS[op_name]
            expr, _ = op_func(col_a, col_b)
            all_exprs.append(expr)

    # 分批添加
    for i in range(0, len(all_exprs), batch_size):
        batch = all_exprs[i : i + batch_size]
        df = df.with_columns(batch)

    return df


def get_cross_feature_names(
    top_n_factors: list[str],
    operators: Optional[list[str]] = None,
) -> list[str]:
    """
    获取所有可能的交叉因子列名 (不执行计算)

    参数:
        top_n_factors: top-N 因子名称列表
        operators: 交叉算子列表

    返回:
        交叉因子列名列表
    """
    if operators is None:
        operators = list(CROSS_OPERATORS.keys())

    op_suffixes = {"multiply": "mul", "divide": "div", "subtract": "sub"}
    factor_pairs = list(combinations(top_n_factors, 2))

    names = []
    for col_a, col_b in factor_pairs:
        for op_name in operators:
            suffix = op_suffixes.get(op_name, op_name)
            names.append(f"{col_a}_X_{col_b}_{suffix}")
    return names
