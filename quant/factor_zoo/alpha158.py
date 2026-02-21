"""
Qlib Alpha158 技术指标因子 - Polars 实现
==========================================
包含 158 个技术指标因子，分为以下类别:
- K线形态: KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2 (各5个窗口)
- 收益率: ROC (5个窗口)
- 均线偏离: MA (5个窗口)
- 波动率: STD (5个窗口)
- Beta: BETA (5个窗口)
- R-squared: RSQR (5个窗口)
- 残差: RESI (5个窗口)
- 极值: MAX, MIN (各5个窗口)
- 分位数: QTLU, QTLD (各5个窗口)
- 排名: RANK (5个窗口)
- RSV: RSV (5个窗口)
- 极值位置: IMAX, IMIN (各5个窗口)
- 极值距离差: IMXD (5个窗口)
- 相关: CORR, CORD (各5个窗口)
- 计数: CNTP, CNTN, CNTD (各5个窗口)
- 求和: SUMP, SUMN, SUMD (各5个窗口)
- 成交量: VMA, VSTD (各5个窗口)
- 加权成交量: WVMA (5个窗口)
- 成交量方向: VSUMP, VSUMN, VSUMD (各5个窗口)

窗口: [5, 10, 20, 30, 60]

输入 DataFrame 列要求: date, asset, open, high, low, close, volume, vwap, returns
"""

import polars as pl
import numpy as np

# 标准窗口列表
WINDOWS = [5, 10, 20, 30, 60]


# ============================================================
# K线形态因子 (7组 x 5窗口 = 35个)
# ============================================================

def _kline_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    K线形态因子:
    - KMID: (close - open) / open
    - KLEN: (high - low) / open
    - KMID2: (close - open) / (high - low + 1e-12)
    - KUP: (high - max(open, close)) / open
    - KUP2: (high - max(open, close)) / (high - low + 1e-12)
    - KLOW: (min(open, close) - low) / open
    - KLOW2: (min(open, close) - low) / (high - low + 1e-12)
    对每种形态在各窗口内取均值
    """
    exprs = []
    for w in WINDOWS:
        # KMID: K线中心位置
        kmid = ((pl.col("close") - pl.col("open")) / pl.col("open")).rolling_mean(w).over("asset")
        exprs.append(kmid.alias(f"KMID_{w}"))

        # KLEN: K线长度
        klen = ((pl.col("high") - pl.col("low")) / pl.col("open")).rolling_mean(w).over("asset")
        exprs.append(klen.alias(f"KLEN_{w}"))

        # KMID2: K线中心位置 (归一化)
        kmid2 = ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-12)).rolling_mean(w).over("asset")
        exprs.append(kmid2.alias(f"KMID2_{w}"))

        # KUP: 上影线
        kup = ((pl.col("high") - pl.max_horizontal("open", "close")) / pl.col("open")).rolling_mean(w).over("asset")
        exprs.append(kup.alias(f"KUP_{w}"))

        # KUP2: 上影线 (归一化)
        kup2 = ((pl.col("high") - pl.max_horizontal("open", "close")) / (pl.col("high") - pl.col("low") + 1e-12)).rolling_mean(w).over("asset")
        exprs.append(kup2.alias(f"KUP2_{w}"))

        # KLOW: 下影线
        klow = ((pl.min_horizontal("open", "close") - pl.col("low")) / pl.col("open")).rolling_mean(w).over("asset")
        exprs.append(klow.alias(f"KLOW_{w}"))

        # KLOW2: 下影线 (归一化)
        klow2 = ((pl.min_horizontal("open", "close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-12)).rolling_mean(w).over("asset")
        exprs.append(klow2.alias(f"KLOW2_{w}"))

    return df.with_columns(exprs)


# ============================================================
# 收益率变化因子 ROC (1组 x 5窗口 = 5个)
# ============================================================

def _roc_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    ROC: 收益率变化 = close / delay(close, w) - 1
    """
    exprs = []
    for w in WINDOWS:
        roc = (pl.col("close") / pl.col("close").shift(w).over("asset") - 1)
        exprs.append(roc.alias(f"ROC_{w}"))
    return df.with_columns(exprs)


# ============================================================
# 均线偏离因子 MA (1组 x 5窗口 = 5个)
# ============================================================

def _ma_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    MA: 均线偏离 = close / mean(close, w) - 1
    """
    exprs = []
    for w in WINDOWS:
        ma = (pl.col("close") / pl.col("close").rolling_mean(w).over("asset") - 1)
        exprs.append(ma.alias(f"MA_{w}"))
    return df.with_columns(exprs)


# ============================================================
# 波动率因子 STD (1组 x 5窗口 = 5个)
# ============================================================

def _std_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    STD: 收益率的滚动标准差
    """
    exprs = []
    for w in WINDOWS:
        std = pl.col("returns").rolling_std(w, min_samples=w).over("asset")
        exprs.append(std.alias(f"STD_{w}"))
    return df.with_columns(exprs)


# ============================================================
# Beta, R-squared, 残差因子 (3组 x 5窗口 = 15个)
# ============================================================

def _beta_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    BETA: 个股收益率对市场收益率的回归 beta
    RSQR: 回归的 R-squared
    RESI: 回归残差的标准差

    市场收益率用截面均值近似
    """
    # 先计算市场收益率 (截面均值)
    df = df.with_columns(
        pl.col("returns").mean().over("date").alias("_mkt_ret")
    )

    exprs = []
    for w in WINDOWS:
        # Beta = Cov(r, rm) / Var(rm)
        cov = pl.rolling_corr(
            pl.col("returns"), pl.col("_mkt_ret"), window_size=w
        ) * pl.col("returns").rolling_std(w).over("asset") * pl.col("_mkt_ret").rolling_std(w).over("asset")

        var_mkt = pl.col("_mkt_ret").rolling_var(w).over("asset")
        beta = cov / (var_mkt + 1e-12)
        exprs.append(beta.alias(f"BETA_{w}"))

        # R-squared = corr^2
        corr = pl.rolling_corr(pl.col("returns"), pl.col("_mkt_ret"), window_size=w)
        rsqr = corr.pow(2)
        exprs.append(rsqr.alias(f"RSQR_{w}"))

        # 残差 std: std(r - beta * rm)
        # 用近似: std(r) * sqrt(1 - R^2)
        resi = pl.col("returns").rolling_std(w).over("asset") * (1 - rsqr).abs().sqrt()
        exprs.append(resi.alias(f"RESI_{w}"))

    df = df.with_columns(exprs)
    # 删除临时列
    df = df.drop("_mkt_ret")
    return df


# ============================================================
# 最高最低因子 MAX, MIN (2组 x 5窗口 = 10个)
# ============================================================

def _max_min_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    MAX: (high的滚动最大值 / close - 1)
    MIN: (low的滚动最小值 / close - 1)
    """
    exprs = []
    for w in WINDOWS:
        max_f = (pl.col("high").rolling_max(w).over("asset") / pl.col("close") - 1)
        exprs.append(max_f.alias(f"MAX_{w}"))

        min_f = (pl.col("low").rolling_min(w).over("asset") / pl.col("close") - 1)
        exprs.append(min_f.alias(f"MIN_{w}"))
    return df.with_columns(exprs)


# ============================================================
# 分位数因子 QTLU, QTLD (2组 x 5窗口 = 10个)
# ============================================================

def _quantile_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    QTLU: close在窗口内的上分位数位置 (0.8分位数 / close - 1)
    QTLD: close在窗口内的下分位数位置 (0.2分位数 / close - 1)
    """
    exprs = []
    for w in WINDOWS:
        qtlu = (pl.col("close").rolling_quantile(0.8, window_size=w).over("asset") / pl.col("close") - 1)
        exprs.append(qtlu.alias(f"QTLU_{w}"))

        qtld = (pl.col("close").rolling_quantile(0.2, window_size=w).over("asset") / pl.col("close") - 1)
        exprs.append(qtld.alias(f"QTLD_{w}"))
    return df.with_columns(exprs)


# ============================================================
# 排名因子 RANK (1组 x 5窗口 = 5个)
# ============================================================

def _rank_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    RANK: close在窗口内的排名百分位
    """
    exprs = []
    for w in WINDOWS:
        rank_f = pl.col("close").rolling_rank(window_size=w).over("asset") / w
        exprs.append(rank_f.alias(f"RANK_{w}"))
    return df.with_columns(exprs)


# ============================================================
# RSV 相对强弱因子 (1组 x 5窗口 = 5个)
# ============================================================

def _rsv_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    RSV: (close - low_min) / (high_max - low_min + 1e-12)
    衡量收盘价在窗口内高低范围中的相对位置
    """
    exprs = []
    for w in WINDOWS:
        high_max = pl.col("high").rolling_max(w).over("asset")
        low_min = pl.col("low").rolling_min(w).over("asset")
        rsv = (pl.col("close") - low_min) / (high_max - low_min + 1e-12)
        exprs.append(rsv.alias(f"RSV_{w}"))
    return df.with_columns(exprs)


# ============================================================
# 极值位置因子 IMAX, IMIN, IMXD (3组 x 5窗口 = 15个)
# ============================================================

def _imax_imin_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    IMAX: high在窗口内最大值的位置 / 窗口大小
    IMIN: low在窗口内最小值的位置 / 窗口大小
    IMXD: (IMAX - IMIN) / 窗口大小
    """
    exprs = []
    for w in WINDOWS:
        # IMAX: argmax(high) / w
        imax = pl.col("high").rolling_map(
            lambda s: float(np.argmax(s)) / len(s),
            window_size=w, min_samples=w
        ).over("asset")
        exprs.append(imax.alias(f"IMAX_{w}"))

        # IMIN: argmin(low) / w
        imin = pl.col("low").rolling_map(
            lambda s: float(np.argmin(s)) / len(s),
            window_size=w, min_samples=w
        ).over("asset")
        exprs.append(imin.alias(f"IMIN_{w}"))

    df = df.with_columns(exprs)

    # IMXD = IMAX - IMIN
    imxd_exprs = []
    for w in WINDOWS:
        imxd = pl.col(f"IMAX_{w}") - pl.col(f"IMIN_{w}")
        imxd_exprs.append(imxd.alias(f"IMXD_{w}"))
    return df.with_columns(imxd_exprs)


# ============================================================
# 相关因子 CORR, CORD (2组 x 5窗口 = 10个)
# ============================================================

def _corr_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    CORR: close和log(volume+1)的滚动相关系数
    CORD: close和open的滚动相关系数
    """
    df = df.with_columns(
        (pl.col("volume") + 1).log().alias("_log_vol")
    )

    exprs = []
    for w in WINDOWS:
        corr = pl.rolling_corr(pl.col("close"), pl.col("_log_vol"), window_size=w)
        exprs.append(corr.alias(f"CORR_{w}"))

        cord = pl.rolling_corr(pl.col("close"), pl.col("open"), window_size=w)
        exprs.append(cord.alias(f"CORD_{w}"))

    df = df.with_columns(exprs)
    df = df.drop("_log_vol")
    return df


# ============================================================
# 计数因子 CNTP, CNTN, CNTD (3组 x 5窗口 = 15个)
# ============================================================

def _count_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    CNTP: 窗口内 returns > 0 的天数占比
    CNTN: 窗口内 returns < 0 的天数占比
    CNTD: CNTP - CNTN
    """
    exprs = []
    for w in WINDOWS:
        # 正收益占比
        cntp = (pl.col("returns") > 0).cast(pl.Float64).rolling_mean(w).over("asset")
        exprs.append(cntp.alias(f"CNTP_{w}"))

        # 负收益占比
        cntn = (pl.col("returns") < 0).cast(pl.Float64).rolling_mean(w).over("asset")
        exprs.append(cntn.alias(f"CNTN_{w}"))

    df = df.with_columns(exprs)

    # CNTD = CNTP - CNTN
    cntd_exprs = []
    for w in WINDOWS:
        cntd_exprs.append((pl.col(f"CNTP_{w}") - pl.col(f"CNTN_{w}")).alias(f"CNTD_{w}"))
    return df.with_columns(cntd_exprs)


# ============================================================
# 求和因子 SUMP, SUMN, SUMD (3组 x 5窗口 = 15个)
# ============================================================

def _sum_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    SUMP: 窗口内正收益之和 / 总绝对收益之和
    SUMN: 窗口内负收益绝对值之和 / 总绝对收益之和
    SUMD: SUMP - SUMN
    """
    # 正收益和负收益绝对值
    df = df.with_columns([
        pl.when(pl.col("returns") > 0).then(pl.col("returns")).otherwise(0.0).alias("_ret_pos"),
        pl.when(pl.col("returns") < 0).then(-pl.col("returns")).otherwise(0.0).alias("_ret_neg"),
    ])

    exprs = []
    for w in WINDOWS:
        sum_pos = pl.col("_ret_pos").rolling_sum(w).over("asset")
        sum_neg = pl.col("_ret_neg").rolling_sum(w).over("asset")
        sum_abs = sum_pos + sum_neg + 1e-12

        exprs.append((sum_pos / sum_abs).alias(f"SUMP_{w}"))
        exprs.append((sum_neg / sum_abs).alias(f"SUMN_{w}"))

    df = df.with_columns(exprs)

    # SUMD = SUMP - SUMN
    sumd_exprs = []
    for w in WINDOWS:
        sumd_exprs.append((pl.col(f"SUMP_{w}") - pl.col(f"SUMN_{w}")).alias(f"SUMD_{w}"))

    df = df.with_columns(sumd_exprs)
    df = df.drop(["_ret_pos", "_ret_neg"])
    return df


# ============================================================
# 成交量因子 VMA, VSTD (2组 x 5窗口 = 10个)
# ============================================================

def _volume_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    VMA: volume / mean(volume, w) - 1
    VSTD: 成交量的滚动标准差 / (mean(volume, w) + 1e-12)
    """
    exprs = []
    for w in WINDOWS:
        vol_mean = pl.col("volume").rolling_mean(w).over("asset")
        vma = pl.col("volume") / vol_mean - 1
        exprs.append(vma.alias(f"VMA_{w}"))

        vstd = pl.col("volume").rolling_std(w).over("asset") / (vol_mean + 1e-12)
        exprs.append(vstd.alias(f"VSTD_{w}"))
    return df.with_columns(exprs)


# ============================================================
# 加权成交量因子 WVMA (1组 x 5窗口 = 5个)
# ============================================================

def _wvma_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    WVMA: 用收益率绝对值加权的成交量标准差
    std(abs(returns) * volume, w) / (mean(abs(returns) * volume, w) + 1e-12)
    """
    df = df.with_columns(
        (pl.col("returns").abs() * pl.col("volume")).alias("_wvol")
    )

    exprs = []
    for w in WINDOWS:
        wvma = pl.col("_wvol").rolling_std(w).over("asset") / (
            pl.col("_wvol").rolling_mean(w).over("asset") + 1e-12
        )
        exprs.append(wvma.alias(f"WVMA_{w}"))

    df = df.with_columns(exprs)
    df = df.drop("_wvol")
    return df


# ============================================================
# 成交量方向因子 VSUMP, VSUMN, VSUMD (3组 x 5窗口 = 15个)
# ============================================================

def _volume_direction_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    VSUMP: 上涨日成交量之和 / 总成交量之和
    VSUMN: 下跌日成交量之和 / 总成交量之和
    VSUMD: VSUMP - VSUMN
    """
    df = df.with_columns([
        pl.when(pl.col("returns") > 0).then(pl.col("volume")).otherwise(0.0).alias("_vol_up"),
        pl.when(pl.col("returns") < 0).then(pl.col("volume")).otherwise(0.0).alias("_vol_dn"),
    ])

    exprs = []
    for w in WINDOWS:
        vol_up_sum = pl.col("_vol_up").rolling_sum(w).over("asset")
        vol_dn_sum = pl.col("_vol_dn").rolling_sum(w).over("asset")
        vol_total = pl.col("volume").rolling_sum(w).over("asset") + 1e-12

        exprs.append((vol_up_sum / vol_total).alias(f"VSUMP_{w}"))
        exprs.append((vol_dn_sum / vol_total).alias(f"VSUMN_{w}"))

    df = df.with_columns(exprs)

    # VSUMD = VSUMP - VSUMN
    vsumd_exprs = []
    for w in WINDOWS:
        vsumd_exprs.append((pl.col(f"VSUMP_{w}") - pl.col(f"VSUMN_{w}")).alias(f"VSUMD_{w}"))

    df = df.with_columns(vsumd_exprs)
    df = df.drop(["_vol_up", "_vol_dn"])
    return df


# ============================================================
# 汇总入口
# ============================================================

# 各组因子数量:
# K线形态: 7 x 5 = 35
# ROC: 1 x 5 = 5
# MA: 1 x 5 = 5
# STD: 1 x 5 = 5
# BETA/RSQR/RESI: 3 x 5 = 15
# MAX/MIN: 2 x 5 = 10
# QTLU/QTLD: 2 x 5 = 10
# RANK: 1 x 5 = 5
# RSV: 1 x 5 = 5
# IMAX/IMIN/IMXD: 3 x 5 = 15
# CORR/CORD: 2 x 5 = 10
# CNTP/CNTN/CNTD: 3 x 5 = 15
# SUMP/SUMN/SUMD: 3 x 5 = 15
# VMA/VSTD: 2 x 5 = 10
# WVMA: 1 x 5 = 5
# VSUMP/VSUMN/VSUMD: 3 x 5 = 15
# 合计: 35+5+5+5+15+10+10+5+5+15+10+15+15+10+5+15 = 180
# 注: 原始158个因子 + 衍生的 IMXD/CNTD/SUMD/VSUMD 等
# 实际输出约 180 列，覆盖所有 Alpha158 类别

_ALL_FEATURE_FUNCS = [
    _kline_features,
    _roc_features,
    _ma_features,
    _std_features,
    _beta_features,
    _max_min_features,
    _quantile_features,
    _rank_features,
    _rsv_features,
    _imax_imin_features,
    _corr_features,
    _count_features,
    _sum_features,
    _volume_features,
    _wvma_features,
    _volume_direction_features,
]


def build_alpha158(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    构建所有 Alpha158 技术指标因子

    参数:
        df: 包含 date, asset, open, high, low, close, volume, vwap, returns 列的 LazyFrame

    返回:
        附加了 ~158 个技术指标因子列的 LazyFrame
    """
    # 确保排序
    df = df.sort(["asset", "date"])

    for func in _ALL_FEATURE_FUNCS:
        df = func(df)

    return df


def get_alpha158_columns() -> list[str]:
    """获取所有 Alpha158 因子列名列表"""
    cols = []
    # K线形态
    for prefix in ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # ROC, MA, STD
    for prefix in ["ROC", "MA", "STD"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # BETA, RSQR, RESI
    for prefix in ["BETA", "RSQR", "RESI"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # MAX, MIN
    for prefix in ["MAX", "MIN"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # QTLU, QTLD
    for prefix in ["QTLU", "QTLD"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # RANK, RSV
    for prefix in ["RANK", "RSV"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # IMAX, IMIN, IMXD
    for prefix in ["IMAX", "IMIN", "IMXD"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # CORR, CORD
    for prefix in ["CORR", "CORD"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # CNTP, CNTN, CNTD
    for prefix in ["CNTP", "CNTN", "CNTD"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # SUMP, SUMN, SUMD
    for prefix in ["SUMP", "SUMN", "SUMD"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # VMA, VSTD
    for prefix in ["VMA", "VSTD"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    # WVMA
    for w in WINDOWS:
        cols.append(f"WVMA_{w}")
    # VSUMP, VSUMN, VSUMD
    for prefix in ["VSUMP", "VSUMN", "VSUMD"]:
        for w in WINDOWS:
            cols.append(f"{prefix}_{w}")
    return cols
