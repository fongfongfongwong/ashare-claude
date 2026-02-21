"""
WorldQuant 101 公式化因子 - Polars 实现
========================================
精选 40 个最有效的因子，覆盖以下类别:
- 价量相关 (correlation of price/volume)
- 动量类 (momentum, returns)
- 均值回复 (mean reversion)
- 波动率 (volatility)
- 换手率 (turnover)

参考: Kakushadze, Z. (2016). "101 Formulaic Alphas"

输入 DataFrame 列要求: date, asset, open, high, low, close, volume, vwap, returns
"""

import polars as pl
import numpy as np


# ============================================================
# 辅助算子 (Helper Operators)
# ============================================================

def _sort_by_date(df: pl.LazyFrame) -> pl.LazyFrame:
    """确保按 asset, date 排序"""
    return df.sort(["asset", "date"])


def _ts_rank(expr: pl.Expr, window: int) -> pl.Expr:
    """时间序列排名 (归一化到 0~1)"""
    return expr.rolling_rank(window_size=window, min_samples=window).over("asset") / window


def _ts_argmax(expr: pl.Expr, window: int) -> pl.Expr:
    """滚动窗口内最大值位置 (距当前的距离)"""
    return expr.rolling_map(
        lambda s: float(np.argmax(s)),
        window_size=window,
        min_samples=window,
    ).over("asset")


def _ts_argmin(expr: pl.Expr, window: int) -> pl.Expr:
    """滚动窗口内最小值位置 (距当前的距离)"""
    return expr.rolling_map(
        lambda s: float(np.argmin(s)),
        window_size=window,
        min_samples=window,
    ).over("asset")


def _ts_corr(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """滚动相关系数"""
    return pl.rolling_corr(x, y, window_size=window)


def _ts_cov(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """滚动协方差"""
    return pl.rolling_corr(x, y, window_size=window) * x.rolling_std(window).over("asset") * y.rolling_std(window).over("asset")


def _ts_stddev(expr: pl.Expr, window: int) -> pl.Expr:
    """滚动标准差"""
    return expr.rolling_std(window_size=window, min_samples=window).over("asset")


def _ts_mean(expr: pl.Expr, window: int) -> pl.Expr:
    """滚动均值"""
    return expr.rolling_mean(window_size=window, min_samples=window).over("asset")


def _ts_sum(expr: pl.Expr, window: int) -> pl.Expr:
    """滚动求和"""
    return expr.rolling_sum(window_size=window, min_samples=window).over("asset")


def _ts_min(expr: pl.Expr, window: int) -> pl.Expr:
    """滚动最小值"""
    return expr.rolling_min(window_size=window, min_samples=window).over("asset")


def _ts_max(expr: pl.Expr, window: int) -> pl.Expr:
    """滚动最大值"""
    return expr.rolling_max(window_size=window, min_samples=window).over("asset")


def _ts_delta(expr: pl.Expr, period: int) -> pl.Expr:
    """差分"""
    return (expr - expr.shift(period)).over("asset")


def _ts_delay(expr: pl.Expr, period: int) -> pl.Expr:
    """延迟"""
    return expr.shift(period).over("asset")


def _ts_product(expr: pl.Expr, window: int) -> pl.Expr:
    """滚动乘积"""
    return expr.log().rolling_sum(window_size=window, min_samples=window).over("asset").exp()


def _rank(expr: pl.Expr) -> pl.Expr:
    """截面排名 (归一化到 0~1)"""
    return expr.rank().over("date") / expr.count().over("date")


def _decay_linear(expr: pl.Expr, window: int) -> pl.Expr:
    """线性衰减加权均值"""
    weights = list(range(1, window + 1))
    w_sum = sum(weights)
    weights_norm = [w / w_sum for w in weights]
    return expr.rolling_map(
        lambda s: float(np.dot(s, weights_norm)) if len(s) == window else float("nan"),
        window_size=window,
        min_samples=window,
    ).over("asset")


def _signed_power(expr: pl.Expr, exp: float) -> pl.Expr:
    """有符号幂次"""
    return expr.sign() * expr.abs().pow(exp)


def _scale(expr: pl.Expr, a: float = 1.0) -> pl.Expr:
    """截面缩放，使绝对值之和 = a"""
    return a * expr / expr.abs().sum().over("date")


# ============================================================
# Alpha 因子实现 (40 个精选因子)
# ============================================================

# --- 类别一: 价量相关因子 (8个) ---

def alpha001(df: pl.LazyFrame) -> pl.LazyFrame:
    """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    类别: 均值回复 / 波动率
    """
    inner = pl.when(pl.col("returns") < 0).then(
        _ts_stddev(pl.col("returns"), 20)
    ).otherwise(pl.col("close"))
    powered = _signed_power(inner, 2.0)
    argmax = _ts_argmax(powered, 5)
    return df.with_columns(
        (_rank(argmax) - 0.5).alias("alpha001")
    )


def alpha002(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * corr(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)
    类别: 价量相关
    """
    delta_log_vol = _ts_delta(pl.col("volume").log(), 2)
    price_change = (pl.col("close") - pl.col("open")) / pl.col("open")
    corr = _ts_corr(_rank(delta_log_vol), _rank(price_change), 6)
    return df.with_columns(
        (-1 * corr).alias("alpha002")
    )


def alpha003(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * corr(rank(open), rank(volume), 10)
    类别: 价量相关
    """
    corr = _ts_corr(_rank(pl.col("open")), _rank(pl.col("volume")), 10)
    return df.with_columns(
        (-1 * corr).alias("alpha003")
    )


def alpha004(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * Ts_Rank(rank(low), 9)
    类别: 均值回复
    """
    ts_r = _ts_rank(_rank(pl.col("low")), 9)
    return df.with_columns(
        (-1 * ts_r).alias("alpha004")
    )


def alpha006(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * corr(open, volume, 10)
    类别: 价量相关
    """
    corr = _ts_corr(pl.col("open"), pl.col("volume"), 10)
    return df.with_columns(
        (-1 * corr).alias("alpha006")
    )


def alpha007(df: pl.LazyFrame) -> pl.LazyFrame:
    """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))
    类别: 价量相关 / 动量
    """
    adv20 = _ts_mean(pl.col("volume"), 20)
    delta_c7 = _ts_delta(pl.col("close"), 7)
    ts_r = _ts_rank(delta_c7.abs(), 60)
    factor = pl.when(adv20 < pl.col("volume")).then(
        -1 * ts_r * delta_c7.sign()
    ).otherwise(pl.lit(-1.0))
    return df.with_columns(factor.alias("alpha007"))


def alpha008(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))
    类别: 动量 / 价量
    """
    inner = _ts_sum(pl.col("open"), 5) * _ts_sum(pl.col("returns"), 5)
    factor = -1 * _rank(inner - _ts_delay(inner, 10))
    return df.with_columns(factor.alias("alpha008"))


def alpha009(df: pl.LazyFrame) -> pl.LazyFrame:
    """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    类别: 动量 / 均值回复
    """
    d = _ts_delta(pl.col("close"), 1)
    ts_min_d = _ts_min(d, 5)
    ts_max_d = _ts_max(d, 5)
    factor = pl.when(ts_min_d > 0).then(d).when(ts_max_d < 0).then(d).otherwise(-1 * d)
    return df.with_columns(factor.alias("alpha009"))


# --- 类别二: 动量因子 (8个) ---

def alpha012(df: pl.LazyFrame) -> pl.LazyFrame:
    """sign(delta(volume, 1)) * (-1 * delta(close, 1))
    类别: 动量 / 价量
    """
    factor = _ts_delta(pl.col("volume"), 1).sign() * (-1 * _ts_delta(pl.col("close"), 1))
    return df.with_columns(factor.alias("alpha012"))


def alpha013(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * rank(covariance(rank(close), rank(volume), 5))
    类别: 价量相关
    """
    cov = _ts_cov(_rank(pl.col("close")), _rank(pl.col("volume")), 5)
    return df.with_columns((-1 * _rank(cov)).alias("alpha013"))


def alpha014(df: pl.LazyFrame) -> pl.LazyFrame:
    """((-1 * rank(delta(returns, 3))) * corr(open, volume, 10))
    类别: 动量 / 价量
    """
    factor = -1 * _rank(_ts_delta(pl.col("returns"), 3)) * _ts_corr(pl.col("open"), pl.col("volume"), 10)
    return df.with_columns(factor.alias("alpha014"))


def alpha016(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * rank(covariance(rank(high), rank(volume), 5))
    类别: 价量相关
    """
    cov = _ts_cov(_rank(pl.col("high")), _rank(pl.col("volume")), 5)
    return df.with_columns((-1 * _rank(cov)).alias("alpha016"))


def alpha017(df: pl.LazyFrame) -> pl.LazyFrame:
    """(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    类别: 动量 / 换手率
    """
    adv20 = _ts_mean(pl.col("volume"), 20)
    p1 = -1 * _rank(_ts_rank(pl.col("close"), 10))
    p2 = _rank(_ts_delta(_ts_delta(pl.col("close"), 1), 1))
    p3 = _rank(_ts_rank(pl.col("volume") / adv20, 5))
    return df.with_columns((p1 * p2 * p3).alias("alpha017"))


def alpha018(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + corr(close, open, 10)))
    类别: 波动率 / 均值回复
    """
    diff = pl.col("close") - pl.col("open")
    inner = _ts_stddev(diff.abs(), 5) + diff + _ts_corr(pl.col("close"), pl.col("open"), 10)
    return df.with_columns((-1 * _rank(inner)).alias("alpha018"))


def alpha020(df: pl.LazyFrame) -> pl.LazyFrame:
    """(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    类别: 动量 / 缺口
    """
    p1 = -1 * _rank(pl.col("open") - _ts_delay(pl.col("high"), 1))
    p2 = _rank(pl.col("open") - _ts_delay(pl.col("close"), 1))
    p3 = _rank(pl.col("open") - _ts_delay(pl.col("low"), 1))
    return df.with_columns((p1 * p2 * p3).alias("alpha020"))


def alpha021(df: pl.LazyFrame) -> pl.LazyFrame:
    """((mean(close, 8) + stddev(close, 8)) < mean(close, 2)) ? -1 : (mean(close,2) < (mean(close,8) - stddev(close,8)) ? 1 : ((volume/adv20) < 1 ? -1 : 1))
    类别: 均值回复 / 换手率
    """
    mean8 = _ts_mean(pl.col("close"), 8)
    std8 = _ts_stddev(pl.col("close"), 8)
    mean2 = _ts_mean(pl.col("close"), 2)
    adv20 = _ts_mean(pl.col("volume"), 20)
    factor = pl.when((mean8 + std8) < mean2).then(pl.lit(-1.0)).when(
        mean2 < (mean8 - std8)
    ).then(pl.lit(1.0)).when(
        (pl.col("volume") / adv20) < 1
    ).then(pl.lit(-1.0)).otherwise(pl.lit(1.0))
    return df.with_columns(factor.alias("alpha021"))


# --- 类别三: 均值回复因子 (8个) ---

def alpha022(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * (delta(corr(high, volume, 5), 5) * rank(stddev(close, 20)))
    类别: 均值回复 / 波动率
    """
    corr_hv = _ts_corr(pl.col("high"), pl.col("volume"), 5)
    factor = -1 * _ts_delta(corr_hv, 5) * _rank(_ts_stddev(pl.col("close"), 20))
    return df.with_columns(factor.alias("alpha022"))


def alpha023(df: pl.LazyFrame) -> pl.LazyFrame:
    """(mean(high, 20) < high) ? (-1 * delta(high, 2)) : 0
    类别: 均值回复
    """
    factor = pl.when(_ts_mean(pl.col("high"), 20) < pl.col("high")).then(
        -1 * _ts_delta(pl.col("high"), 2)
    ).otherwise(pl.lit(0.0))
    return df.with_columns(factor.alias("alpha023"))


def alpha024(df: pl.LazyFrame) -> pl.LazyFrame:
    """((((delta(mean(close, 100), 100) / delay(close, 100)) < 0.05) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
    类别: 均值回复 / 动量
    """
    cond = _ts_delta(_ts_mean(pl.col("close"), 100), 100) / _ts_delay(pl.col("close"), 100)
    factor = pl.when(cond < 0.05).then(
        -1 * (pl.col("close") - _ts_min(pl.col("close"), 100))
    ).otherwise(-1 * _ts_delta(pl.col("close"), 3))
    return df.with_columns(factor.alias("alpha024"))


def alpha026(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * ts_max(corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    类别: 价量相关 / 均值回复
    """
    corr = _ts_corr(_ts_rank(pl.col("volume"), 5), _ts_rank(pl.col("high"), 5), 5)
    return df.with_columns((-1 * _ts_max(corr, 3)).alias("alpha026"))


def alpha028(df: pl.LazyFrame) -> pl.LazyFrame:
    """scale(((corr(adv20, low, 5) + ((high + low) / 2)) - close))
    类别: 均值回复
    """
    adv20 = _ts_mean(pl.col("volume"), 20)
    inner = _ts_corr(adv20, pl.col("low"), 5) + (pl.col("high") + pl.col("low")) / 2 - pl.col("close")
    return df.with_columns(_scale(inner).alias("alpha028"))


def alpha029(df: pl.LazyFrame) -> pl.LazyFrame:
    """min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close-1),5))))),2),1))))),1),5)
    + ts_rank(delay((-1*returns),6),5)
    简化: 对 returns 的延迟做时序排名
    类别: 均值回复 / 动量
    """
    p1 = _ts_rank(_ts_delay(-1 * pl.col("returns"), 6), 5)
    delta_c = _rank(-1 * _rank(_ts_delta(pl.col("close") - 1, 5)))
    inner = _scale((_ts_sum(_ts_min(_rank(_rank(delta_c)), 2), 1)).log())
    p2 = _ts_min(_rank(_rank(inner)), 5)
    return df.with_columns((p1 + p2).alias("alpha029"))


def alpha031(df: pl.LazyFrame) -> pl.LazyFrame:
    """((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) +
    rank((-1 * delta(close, 3)))) + sign(scale(corr(adv20, low, 12))))
    类别: 均值回复 / 动量
    """
    adv20 = _ts_mean(pl.col("volume"), 20)
    p1 = _rank(_rank(_rank(_decay_linear(-1 * _rank(_rank(_ts_delta(pl.col("close"), 10))), 10))))
    p2 = _rank(-1 * _ts_delta(pl.col("close"), 3))
    p3 = _scale(_ts_corr(adv20, pl.col("low"), 12)).sign()
    return df.with_columns((p1 + p2 + p3).alias("alpha031"))


def alpha032(df: pl.LazyFrame) -> pl.LazyFrame:
    """scale(mean(close, 7) - close) + (20 * scale(corr(vwap, delay(close, 5), 230)))
    类别: 均值回复
    """
    p1 = _scale(_ts_mean(pl.col("close"), 7) - pl.col("close"))
    p2 = 20 * _scale(_ts_corr(pl.col("vwap"), _ts_delay(pl.col("close"), 5), 230))
    return df.with_columns((p1 + p2).alias("alpha032"))


# --- 类别四: 波动率因子 (8个) ---

def alpha033(df: pl.LazyFrame) -> pl.LazyFrame:
    """rank((-1 * ((1 - (open / close))^1)))
    类别: 价格形态
    """
    return df.with_columns(
        _rank(-1 * (1 - pl.col("open") / pl.col("close"))).alias("alpha033")
    )


def alpha034(df: pl.LazyFrame) -> pl.LazyFrame:
    """rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    类别: 波动率
    """
    ratio = _ts_stddev(pl.col("returns"), 2) / _ts_stddev(pl.col("returns"), 5)
    factor = _rank((1 - _rank(ratio)) + (1 - _rank(_ts_delta(pl.col("close"), 1))))
    return df.with_columns(factor.alias("alpha034"))


def alpha035(df: pl.LazyFrame) -> pl.LazyFrame:
    """((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    类别: 波动率 / 换手率
    """
    p1 = _ts_rank(pl.col("volume"), 32)
    p2 = 1 - _ts_rank(pl.col("close") + pl.col("high") - pl.col("low"), 16)
    p3 = 1 - _ts_rank(pl.col("returns"), 32)
    return df.with_columns((p1 * p2 * p3).alias("alpha035"))


def alpha037(df: pl.LazyFrame) -> pl.LazyFrame:
    """(rank(corr(delay((open - close), 1), close, 200)) + rank((open - close)))
    类别: 价格形态 / 均值回复
    """
    corr = _ts_corr(_ts_delay(pl.col("open") - pl.col("close"), 1), pl.col("close"), 200)
    return df.with_columns((_rank(corr) + _rank(pl.col("open") - pl.col("close"))).alias("alpha037"))


def alpha038(df: pl.LazyFrame) -> pl.LazyFrame:
    """((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    类别: 均值回复
    """
    factor = -1 * _rank(_ts_rank(pl.col("close"), 10)) * _rank(pl.col("close") / pl.col("open"))
    return df.with_columns(factor.alias("alpha038"))


def alpha040(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * rank(stddev(high, 10)) * corr(high, volume, 10)
    类别: 波动率 / 价量
    """
    factor = -1 * _rank(_ts_stddev(pl.col("high"), 10)) * _ts_corr(pl.col("high"), pl.col("volume"), 10)
    return df.with_columns(factor.alias("alpha040"))


def alpha041(df: pl.LazyFrame) -> pl.LazyFrame:
    """(((high * low)^0.5) - vwap)
    类别: 价格偏离
    """
    return df.with_columns(
        ((pl.col("high") * pl.col("low")).pow(0.5) - pl.col("vwap")).alias("alpha041")
    )


def alpha042(df: pl.LazyFrame) -> pl.LazyFrame:
    """(rank((vwap - close)) / rank((vwap + close)))
    类别: 价格偏离
    """
    return df.with_columns(
        (_rank(pl.col("vwap") - pl.col("close")) / _rank(pl.col("vwap") + pl.col("close"))).alias("alpha042")
    )


# --- 类别五: 换手率 / 综合因子 (8个) ---

def alpha043(df: pl.LazyFrame) -> pl.LazyFrame:
    """ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8)
    类别: 换手率 / 动量
    """
    adv20 = _ts_mean(pl.col("volume"), 20)
    p1 = _ts_rank(pl.col("volume") / adv20, 20)
    p2 = _ts_rank(-1 * _ts_delta(pl.col("close"), 7), 8)
    return df.with_columns((p1 * p2).alias("alpha043"))


def alpha044(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * corr(high, rank(volume), 5)
    类别: 价量相关
    """
    corr = _ts_corr(pl.col("high"), _rank(pl.col("volume")), 5)
    return df.with_columns((-1 * corr).alias("alpha044"))


def alpha045(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * ((rank((sum(delay(close, 5), 20) / 20)) * corr(close, volume, 2)) * rank(corr(sum(close, 5), sum(close, 20), 2)))
    类别: 价量相关 / 动量
    """
    p1 = _rank(_ts_sum(_ts_delay(pl.col("close"), 5), 20) / 20)
    p2 = _ts_corr(pl.col("close"), pl.col("volume"), 2)
    p3 = _rank(_ts_corr(_ts_sum(pl.col("close"), 5), _ts_sum(pl.col("close"), 20), 2))
    return df.with_columns((-1 * p1 * p2 * p3).alias("alpha045"))


def alpha046(df: pl.LazyFrame) -> pl.LazyFrame:
    """((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
    (-1 * (close - delay(close, 1))) : ...)
    类别: 动量
    """
    cond = (((_ts_delay(pl.col("close"), 20) - _ts_delay(pl.col("close"), 10)) / 10)
            - ((_ts_delay(pl.col("close"), 10) - pl.col("close")) / 10))
    delta1 = _ts_delta(pl.col("close"), 1)
    factor = pl.when(cond > 0.25).then(-1 * delta1).when(cond < 0).then(delta1).otherwise(-1 * delta1)
    return df.with_columns(factor.alias("alpha046"))


def alpha049(df: pl.LazyFrame) -> pl.LazyFrame:
    """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-0.1)) ?
    (1) : ((-1) * (close - delay(close, 1))))
    类别: 动量
    """
    cond = (((_ts_delay(pl.col("close"), 20) - _ts_delay(pl.col("close"), 10)) / 10)
            - ((_ts_delay(pl.col("close"), 10) - pl.col("close")) / 10))
    factor = pl.when(cond < -0.1).then(pl.lit(1.0)).otherwise(-1 * _ts_delta(pl.col("close"), 1))
    return df.with_columns(factor.alias("alpha049"))


def alpha051(df: pl.LazyFrame) -> pl.LazyFrame:
    """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-0.05)) ?
    (1) : ((-1) * (close - delay(close, 1))))
    类别: 动量
    """
    cond = (((_ts_delay(pl.col("close"), 20) - _ts_delay(pl.col("close"), 10)) / 10)
            - ((_ts_delay(pl.col("close"), 10) - pl.col("close")) / 10))
    factor = pl.when(cond < -0.05).then(pl.lit(1.0)).otherwise(-1 * _ts_delta(pl.col("close"), 1))
    return df.with_columns(factor.alias("alpha051"))


def alpha052(df: pl.LazyFrame) -> pl.LazyFrame:
    """((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    类别: 换手率 / 动量
    """
    ts_min_low5 = _ts_min(pl.col("low"), 5)
    p1 = (-1 * ts_min_low5 + _ts_delay(ts_min_low5, 5))
    p2 = _rank((_ts_sum(pl.col("returns"), 240) - _ts_sum(pl.col("returns"), 20)) / 220)
    p3 = _ts_rank(pl.col("volume"), 5)
    return df.with_columns((p1 * p2 * p3).alias("alpha052"))


def alpha053(df: pl.LazyFrame) -> pl.LazyFrame:
    """-1 * delta(((close - low) - (high - close)) / (close - low), 9)
    类别: 价格形态
    """
    inner = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (
        pl.col("close") - pl.col("low") + 1e-8
    )
    return df.with_columns((-1 * _ts_delta(inner, 9)).alias("alpha053"))


# ============================================================
# 汇总入口
# ============================================================

# 所有因子函数列表
_ALL_ALPHA_FUNCS = [
    alpha001, alpha002, alpha003, alpha004, alpha006, alpha007, alpha008, alpha009,
    alpha012, alpha013, alpha014, alpha016, alpha017, alpha018, alpha020, alpha021,
    alpha022, alpha023, alpha024, alpha026, alpha028, alpha029, alpha031, alpha032,
    alpha033, alpha034, alpha035, alpha037, alpha038, alpha040, alpha041, alpha042,
    alpha043, alpha044, alpha045, alpha046, alpha049, alpha051, alpha052, alpha053,
]


def build_alpha101(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    构建所有 WorldQuant 101 精选因子

    参数:
        df: 包含 date, asset, open, high, low, close, volume, vwap, returns 列的 LazyFrame

    返回:
        附加了 40 个 alpha 因子列的 LazyFrame
    """
    df = _sort_by_date(df)
    for func in _ALL_ALPHA_FUNCS:
        df = func(df)
    return df
