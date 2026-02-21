"""
算子扩展模块 - Polars 实现
============================
对任意因子列表做时序算子扩展:
- 时间窗口: [5, 10, 20, 40, 60]
- 算子: ts_rank, ts_zscore, ts_delta, ts_decay_linear, ts_max, ts_min, ts_std

输入: 已有因子 DataFrame + 要扩展的列名列表
输出: 扩展后的因子 DataFrame (~因子数 x 7算子 x 5窗口 = 新列)

全部用 Polars 的 rolling/group_by 高性能 API
"""

import polars as pl
import numpy as np
from typing import Optional


# 默认扩展窗口
DEFAULT_WINDOWS = [5, 10, 20, 40, 60]


# ============================================================
# 时序算子定义
# ============================================================

def _ts_rank_expr(col_name: str, window: int) -> pl.Expr:
    """时间序列排名 (归一化到 0~1)"""
    return (
        pl.col(col_name)
        .rolling_rank(window_size=window, min_samples=window)
        .over("asset")
        / window
    ).alias(f"{col_name}_tsrank_{window}")


def _ts_zscore_expr(col_name: str, window: int) -> pl.Expr:
    """时间序列 Z-score 标准化"""
    mean = pl.col(col_name).rolling_mean(window_size=window, min_samples=window).over("asset")
    std = pl.col(col_name).rolling_std(window_size=window, min_samples=window).over("asset")
    return ((pl.col(col_name) - mean) / (std + 1e-12)).alias(f"{col_name}_tszscore_{window}")


def _ts_delta_expr(col_name: str, window: int) -> pl.Expr:
    """时间序列差分"""
    return (
        pl.col(col_name) - pl.col(col_name).shift(window).over("asset")
    ).alias(f"{col_name}_tsdelta_{window}")


def _ts_decay_linear_expr(col_name: str, window: int) -> pl.Expr:
    """线性衰减加权均值 (近期权重更大)"""
    weights = list(range(1, window + 1))
    w_sum = sum(weights)
    weights_norm = [w / w_sum for w in weights]

    return (
        pl.col(col_name)
        .rolling_map(
            lambda s: float(np.dot(s, weights_norm)) if len(s) == window else float("nan"),
            window_size=window,
            min_samples=window,
        )
        .over("asset")
    ).alias(f"{col_name}_tsdecay_{window}")


def _ts_max_expr(col_name: str, window: int) -> pl.Expr:
    """滚动最大值"""
    return (
        pl.col(col_name)
        .rolling_max(window_size=window, min_samples=window)
        .over("asset")
    ).alias(f"{col_name}_tsmax_{window}")


def _ts_min_expr(col_name: str, window: int) -> pl.Expr:
    """滚动最小值"""
    return (
        pl.col(col_name)
        .rolling_min(window_size=window, min_samples=window)
        .over("asset")
    ).alias(f"{col_name}_tsmin_{window}")


def _ts_std_expr(col_name: str, window: int) -> pl.Expr:
    """滚动标准差"""
    return (
        pl.col(col_name)
        .rolling_std(window_size=window, min_samples=window)
        .over("asset")
    ).alias(f"{col_name}_tsstd_{window}")


# 算子注册表
OPERATORS = {
    "ts_rank": _ts_rank_expr,
    "ts_zscore": _ts_zscore_expr,
    "ts_delta": _ts_delta_expr,
    "ts_decay_linear": _ts_decay_linear_expr,
    "ts_max": _ts_max_expr,
    "ts_min": _ts_min_expr,
    "ts_std": _ts_std_expr,
}


# ============================================================
# 主扩展函数
# ============================================================

def expand_factors(
    df: pl.LazyFrame,
    factor_cols: list[str],
    windows: Optional[list[int]] = None,
    operators: Optional[list[str]] = None,
    batch_size: int = 50,
) -> pl.LazyFrame:
    """
    对已有因子做算子扩展

    参数:
        df: 包含 date, asset 列和因子列的 LazyFrame
        factor_cols: 要扩展的因子列名列表
        windows: 时间窗口列表，默认 [5, 10, 20, 40, 60]
        operators: 算子名称列表，默认全部 7 个算子
        batch_size: 每批处理的表达式数量 (避免内存溢出)

    返回:
        附加了扩展因子列的 LazyFrame
        新列命名: {原因子名}_{算子名}_{窗口}
        例如: alpha001_tsrank_5, MA_20_tszscore_10
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    if operators is None:
        operators = list(OPERATORS.keys())

    # 确保排序
    df = df.sort(["asset", "date"])

    # 验证算子名称
    invalid_ops = set(operators) - set(OPERATORS.keys())
    if invalid_ops:
        raise ValueError(f"未知算子: {invalid_ops}. 可用算子: {list(OPERATORS.keys())}")

    # 收集所有表达式
    all_exprs = []
    for col_name in factor_cols:
        for op_name in operators:
            op_func = OPERATORS[op_name]
            for w in windows:
                all_exprs.append(op_func(col_name, w))

    # 分批应用表达式，避免单次 with_columns 过多导致内存问题
    for i in range(0, len(all_exprs), batch_size):
        batch = all_exprs[i : i + batch_size]
        df = df.with_columns(batch)

    return df


def get_expanded_column_names(
    factor_cols: list[str],
    windows: Optional[list[int]] = None,
    operators: Optional[list[str]] = None,
) -> list[str]:
    """
    获取扩展后的所有因子列名 (不执行计算)

    参数:
        factor_cols: 原始因子列名列表
        windows: 时间窗口列表
        operators: 算子名称列表

    返回:
        扩展后的因子列名列表
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    if operators is None:
        operators = list(OPERATORS.keys())

    # 算子名到后缀的映射
    op_suffixes = {
        "ts_rank": "tsrank",
        "ts_zscore": "tszscore",
        "ts_delta": "tsdelta",
        "ts_decay_linear": "tsdecay",
        "ts_max": "tsmax",
        "ts_min": "tsmin",
        "ts_std": "tsstd",
    }

    cols = []
    for col_name in factor_cols:
        for op_name in operators:
            suffix = op_suffixes.get(op_name, op_name)
            for w in windows:
                cols.append(f"{col_name}_{suffix}_{w}")
    return cols
