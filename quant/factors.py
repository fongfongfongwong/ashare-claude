"""
因子构建模块
支持 5 个经典因子: momentum, value, quality, low_vol, size
"""
import numpy as np
import pandas as pd


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面标准化 (每期去均值、除以标准差)"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def _winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """截面缩尾"""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lo, hi, axis=0)


def _neutralize_industry(
    factor: pd.DataFrame, industry: pd.Series
) -> pd.DataFrame:
    """行业中性化: 对每期截面做行业哑变量回归取残差"""
    result = factor.copy()
    dummies = pd.get_dummies(industry, dtype=float)
    for date in factor.index:
        row = factor.loc[date].dropna()
        common = row.index.intersection(dummies.index)
        if len(common) < 10:
            continue
        X = dummies.loc[common].values
        y = row.loc[common].values
        # OLS 残差
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        result.loc[date, common] = resid
    return result


def build_momentum(data: dict) -> pd.DataFrame:
    """动量因子: 过去12个月累计收益 (跳过最近1个月)"""
    ret = data["returns"]
    # 12个月累计 - 最近1个月
    mom_12 = ret.rolling(52, min_periods=40).sum()
    mom_1 = ret.rolling(4, min_periods=3).sum()
    raw = mom_12 - mom_1
    return _cross_sectional_zscore(_winsorize(raw))


def build_value(data: dict) -> pd.DataFrame:
    """价值因子: Book-to-Price"""
    raw = data["book_to_price"]
    return _cross_sectional_zscore(_winsorize(raw))


def build_quality(data: dict) -> pd.DataFrame:
    """质量因子: ROE"""
    raw = data["roe"]
    return _cross_sectional_zscore(_winsorize(raw))


def build_low_vol(data: dict) -> pd.DataFrame:
    """低波因子: 历史波动率取反 (低波 = 高因子值)"""
    raw = -data["volatility"]
    return _cross_sectional_zscore(_winsorize(raw))


def build_size(data: dict) -> pd.DataFrame:
    """市值因子: 对数市值取反 (小市值 = 高因子值)"""
    raw = -np.log(data["market_cap"])
    return _cross_sectional_zscore(_winsorize(raw))


FACTOR_BUILDERS = {
    "momentum": build_momentum,
    "value": build_value,
    "quality": build_quality,
    "low_vol": build_low_vol,
    "size": build_size,
}


def build_all_factors(
    data: dict, neutralize: bool = True
) -> dict[str, pd.DataFrame]:
    """构建全部因子，可选行业中性化"""
    factors = {}
    for name, builder in FACTOR_BUILDERS.items():
        f = builder(data)
        if neutralize and "industry" in data:
            f = _neutralize_industry(f, data["industry"])
            f = _cross_sectional_zscore(f)  # 中性化后再标准化
        factors[name] = f
    return factors
