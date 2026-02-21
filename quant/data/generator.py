"""
模拟股票数据生成器
用于因子回测框架的演示和测试
"""
import numpy as np
import pandas as pd
from datetime import datetime


def generate_stock_universe(
    n_stocks: int = 300,
    start_date: str = "2015-01-05",
    end_date: str = "2024-12-31",
    freq: str = "W-FRI",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    生成模拟的A股数据集，包含价格、市值、财务指标等

    Returns:
        dict with keys: 'returns', 'market_cap', 'book_to_price',
                        'roe', 'volatility', 'momentum_12m'
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq=freq)
    tickers = [f"SZ{str(i).zfill(6)}" for i in range(1, n_stocks + 1)]

    n_dates = len(dates)

    # --- 收益率: 带因子结构的模拟 ---
    # 市场因子
    market_ret = rng.normal(0.0015, 0.025, n_dates)
    # 个股 beta
    betas = rng.uniform(0.5, 1.5, n_stocks)
    # 个股特异收益
    idio = rng.normal(0, 0.04, (n_dates, n_stocks))
    # 总收益
    returns_arr = market_ret[:, None] * betas[None, :] + idio
    returns = pd.DataFrame(returns_arr, index=dates, columns=tickers)

    # --- 市值 (对数正态) ---
    log_cap_base = rng.normal(23, 1.5, n_stocks)  # ln(市值), 约 100亿中位数
    market_cap = pd.DataFrame(
        np.exp(log_cap_base[None, :] + np.cumsum(returns_arr * 0.3, axis=0)),
        index=dates,
        columns=tickers,
    )

    # --- Book-to-Price (价值因子原始指标) ---
    bp_base = rng.uniform(0.3, 2.0, n_stocks)
    bp_noise = rng.normal(0, 0.05, (n_dates, n_stocks)).cumsum(axis=0)
    book_to_price = pd.DataFrame(
        np.clip(bp_base[None, :] + bp_noise, 0.1, 5.0),
        index=dates,
        columns=tickers,
    )

    # --- ROE (质量因子原始指标) ---
    roe_base = rng.uniform(0.02, 0.30, n_stocks)
    roe_noise = rng.normal(0, 0.01, (n_dates, n_stocks))
    roe = pd.DataFrame(
        np.clip(roe_base[None, :] + roe_noise, -0.10, 0.60),
        index=dates,
        columns=tickers,
    )

    # --- 12个月动量 ---
    momentum_12m = returns.rolling(52, min_periods=40).sum()

    # --- 波动率 (过去60日) ---
    volatility = returns.rolling(26, min_periods=20).std() * np.sqrt(52)

    # --- 行业分类 (模拟30个行业) ---
    industry = pd.Series(
        rng.choice([f"IND_{str(i).zfill(2)}" for i in range(1, 31)], n_stocks),
        index=tickers,
        name="industry",
    )

    return {
        "returns": returns,
        "market_cap": market_cap,
        "book_to_price": book_to_price,
        "roe": roe,
        "momentum_12m": momentum_12m,
        "volatility": volatility,
        "industry": industry,
    }
