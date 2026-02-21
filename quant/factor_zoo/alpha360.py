"""
Qlib Alpha360 标准化价量序列因子 - Polars 实现
===============================================
360 个因子 = 6 类 x 60 天:
- 过去60天的 close / close[-1] 标准化 (60个)
- 过去60天的 open / close[-1] 标准化 (60个)
- 过去60天的 high / close[-1] 标准化 (60个)
- 过去60天的 low / close[-1] 标准化 (60个)
- 过去60天的 vwap / close[-1] 标准化 (60个)
- 过去60天的 volume / volume_mean 标准化 (60个)

合计: 6 x 60 = 360 个因子

输入 DataFrame 列要求: date, asset, open, high, low, close, volume, vwap
"""

import polars as pl


# 回望天数
LOOKBACK = 60

# 价格类因子前缀和对应列
PRICE_FEATURES = {
    "CLOSE": "close",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "VWAP": "vwap",
}


def _price_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    价格类因子: price_t-d / close_t-1
    对每种价格 (close, open, high, low, vwap)，
    计算过去 d 天该价格相对于前一日收盘价的比值

    命名: {PREFIX}_D{d}，如 CLOSE_D0, OPEN_D5, HIGH_D59
    """
    exprs = []
    for prefix, col_name in PRICE_FEATURES.items():
        for d in range(LOOKBACK):
            if d == 0:
                # 当天: price_t / close_t-1
                ratio = pl.col(col_name) / pl.col("close").shift(1).over("asset")
            else:
                # d天前: price_t-d / close_t-d-1
                ratio = pl.col(col_name).shift(d).over("asset") / pl.col("close").shift(d + 1).over("asset")
            exprs.append(ratio.alias(f"{prefix}_D{d}"))

    return df.with_columns(exprs)


def _volume_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    成交量因子: volume_t-d / mean(volume, 20)
    用20日均量做标准化

    命名: VOL_D{d}
    """
    # 计算20日均量
    df = df.with_columns(
        pl.col("volume").rolling_mean(20).over("asset").alias("_vol_mean20")
    )

    exprs = []
    for d in range(LOOKBACK):
        if d == 0:
            vol_ratio = pl.col("volume") / (pl.col("_vol_mean20") + 1e-12)
        else:
            vol_ratio = pl.col("volume").shift(d).over("asset") / (
                pl.col("_vol_mean20").shift(d).over("asset") + 1e-12
            )
        exprs.append(vol_ratio.alias(f"VOL_D{d}"))

    df = df.with_columns(exprs)
    df = df.drop("_vol_mean20")
    return df


def build_alpha360(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    构建所有 Alpha360 标准化价量序列因子

    参数:
        df: 包含 date, asset, open, high, low, close, volume, vwap 列的 LazyFrame

    返回:
        附加了 360 个标准化价量因子列的 LazyFrame
        列名格式: {CLOSE,OPEN,HIGH,LOW,VWAP,VOL}_D{0-59}
    """
    # 确保排序
    df = df.sort(["asset", "date"])

    # 分批构建因子，避免单次 with_columns 表达式过多
    df = _price_features(df)
    df = _volume_features(df)

    return df


def get_alpha360_columns() -> list[str]:
    """获取所有 Alpha360 因子列名列表"""
    cols = []
    for prefix in ["CLOSE", "OPEN", "HIGH", "LOW", "VWAP"]:
        for d in range(LOOKBACK):
            cols.append(f"{prefix}_D{d}")
    for d in range(LOOKBACK):
        cols.append(f"VOL_D{d}")
    return cols
