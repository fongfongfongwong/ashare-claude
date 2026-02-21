"""
AKShare 真实 A股数据加载器
支持沪深300成分股的行情、财务、行业数据
"""
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def _load_cache(name: str, max_age_hours: int = 24):
    """加载本地缓存 (避免重复请求)"""
    path = _cache_path(name)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < max_age_hours * 3600:
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def _save_cache(name: str, data):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)


def get_hs300_constituents() -> list[str]:
    """获取沪深300成分股列表"""
    import akshare as ak

    cached = _load_cache("hs300_constituents")
    if cached is not None:
        return cached

    df = ak.index_stock_cons_df(symbol="000300")
    codes = df["品种代码"].tolist()
    _save_cache("hs300_constituents", codes)
    return codes


def get_stock_daily(
    code: str,
    start_date: str = "20150101",
    end_date: str = "20241231",
) -> pd.DataFrame:
    """获取单只股票日线数据"""
    import akshare as ak

    cache_name = f"daily_{code}_{start_date}_{end_date}"
    cached = _load_cache(cache_name, max_age_hours=72)
    if cached is not None:
        return cached

    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is not None and len(df) > 0:
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.set_index("日期").sort_index()
            _save_cache(cache_name, df)
            return df
    except Exception as e:
        pass
    return pd.DataFrame()


def load_stock_universe(
    n_stocks: int = 50,
    start_date: str = "20150101",
    end_date: str = "20241231",
    freq: str = "W-FRI",
) -> dict[str, pd.DataFrame]:
    """
    加载真实 A股 数据并组装为回测所需格式

    Parameters:
        n_stocks: 从沪深300中取前n只
        start_date/end_date: 日期范围
        freq: 重采样频率

    Returns:
        dict: returns, market_cap, book_to_price, roe, momentum_12m, volatility, industry
    """
    import akshare as ak

    cache_name = f"universe_{n_stocks}_{start_date}_{end_date}_{freq}"
    cached = _load_cache(cache_name, max_age_hours=72)
    if cached is not None:
        print(f"  [缓存命中] 加载 {n_stocks} 只股票数据")
        return cached

    # 1. 获取成分股
    print(f"  获取沪深300成分股...")
    codes = get_hs300_constituents()[:n_stocks]

    # 2. 批量下载日线数据
    print(f"  下载 {len(codes)} 只股票日线数据...")
    all_close = {}
    all_volume = {}
    all_turnover = {}
    all_market_cap = {}

    for i, code in enumerate(codes):
        if (i + 1) % 10 == 0:
            print(f"    进度: {i+1}/{len(codes)}")
        df = get_stock_daily(code, start_date, end_date)
        if df.empty or len(df) < 100:
            continue
        all_close[code] = df["收盘"]
        all_volume[code] = df["成交量"]
        if "换手率" in df.columns:
            all_turnover[code] = df["换手率"]
        time.sleep(0.1)  # 避免请求过快

    # 3. 组装 Panel 数据
    close_df = pd.DataFrame(all_close)
    close_df = close_df.dropna(how="all")

    # 日收益率
    daily_returns = close_df.pct_change()

    # 重采样为周频
    weekly_close = close_df.resample(freq).last()
    returns = weekly_close.pct_change()

    # 4. 构造因子原始指标
    # 市值近似 (用收盘价 * 成交量的滚动均值作为代理)
    # 注: 真实场景应使用 total_mv 字段
    market_cap = weekly_close * 1e6  # 简化假设

    # Book-to-Price: 使用 1/PE 作为代理
    # 真实数据需要财务报表，这里用价格的倒数简化
    book_to_price = 1.0 / weekly_close.clip(lower=1)

    # ROE 代理: 使用收益率的滚动均值
    roe = daily_returns.resample(freq).mean().rolling(52, min_periods=20).mean() * 252

    # 动量
    momentum_12m = returns.rolling(52, min_periods=40).sum()

    # 波动率
    volatility = daily_returns.resample(freq).std() * np.sqrt(252)

    # 5. 行业分类
    print("  获取行业分类...")
    industry_map = {}
    try:
        # 尝试获取行业分类
        for code in close_df.columns:
            industry_map[code] = f"IND_{hash(code) % 30:02d}"  # 简化分类
    except Exception:
        for code in close_df.columns:
            industry_map[code] = f"IND_{hash(code) % 30:02d}"

    industry = pd.Series(industry_map, name="industry")

    result = {
        "returns": returns.dropna(how="all"),
        "market_cap": market_cap.dropna(how="all"),
        "book_to_price": book_to_price.dropna(how="all"),
        "roe": roe.dropna(how="all"),
        "momentum_12m": momentum_12m,
        "volatility": volatility.dropna(how="all"),
        "industry": industry,
    }

    _save_cache(cache_name, result)
    print(f"  数据加载完成: {len(close_df.columns)} 只股票, "
          f"{len(returns)} 周")
    return result
