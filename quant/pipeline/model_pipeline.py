"""
多模型训练与评估 Pipeline
- 滚动训练: 3年训练 + 1年验证 + 样本外测试
- 14 个模型对比
- 模型融合 (stacking/blending)
"""
import time
import numpy as np
import pandas as pd
import polars as pl
from typing import Optional
from scipy import stats as sp_stats


def _prepare_Xy(
    factors: dict,
    returns: pd.DataFrame,
    train_start: str,
    train_end: str,
    label_shift: int = 1,
) -> tuple:
    """
    准备特征矩阵 X 和标签 y (向量化实现)

    Parameters:
        factors: dict[str, pd.DataFrame] 因子宽表
        returns: pd.DataFrame 收益率宽表
        train_start, train_end: 训练区间
        label_shift: 标签前移期数 (T+1)

    Returns:
        X: np.ndarray (n_samples, n_features)
        y: np.ndarray (n_samples,)
        feature_names: list[str]
    """
    # 构建标签: T+label_shift 收益
    fwd_ret = returns.shift(-label_shift)

    # 时间筛选
    mask = (returns.index >= train_start) & (returns.index <= train_end)
    valid_dates = returns.index[mask]

    feature_names = list(factors.keys())

    # 预先转换所有 Polars DataFrame 为 Pandas
    pd_factors = {}
    for fname, f in factors.items():
        if isinstance(f, pl.DataFrame):
            pdf = f.to_pandas()
            if "date" in pdf.columns:
                pdf = pdf.set_index("date")
            pd_factors[fname] = pdf
        elif isinstance(f, pd.DataFrame):
            pd_factors[fname] = f
        else:
            pd_factors[fname] = f

    # 向量化: 将所有因子在时间区间内 stack 成 3D 数组
    # 先找公共资产
    all_assets = returns.columns
    for fname in feature_names:
        f = pd_factors[fname]
        all_assets = all_assets.intersection(f.columns)

    if len(all_assets) < 10:
        return np.array([]).reshape(0, len(feature_names)), np.array([]), feature_names

    # 构建特征矩阵: 按日期 stack
    X_all = []
    y_all = []

    for date in valid_dates:
        if date not in fwd_ret.index:
            continue

        # 标签
        y_row = fwd_ret.loc[date, all_assets].values.astype(np.float32)

        # 特征
        x_row = np.zeros((len(all_assets), len(feature_names)), dtype=np.float32)
        skip = False
        for j, fname in enumerate(feature_names):
            f = pd_factors[fname]
            if date not in f.index:
                skip = True
                break
            x_row[:, j] = f.loc[date, all_assets].values.astype(np.float32)

        if skip:
            continue

        # 仅保留无 NaN 的行
        valid_mask = np.isfinite(x_row).all(axis=1) & np.isfinite(y_row)
        if valid_mask.sum() < 20:
            continue

        X_all.append(x_row[valid_mask])
        y_all.append(y_row[valid_mask])

    if not X_all:
        return np.array([]).reshape(0, len(feature_names)), np.array([]), feature_names

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    return X, y, feature_names


def _calc_ic(predictions: np.ndarray, actual: np.ndarray) -> float:
    """计算 Rank IC (Spearman)"""
    if len(predictions) < 10:
        return np.nan
    corr, _ = sp_stats.spearmanr(predictions, actual)
    return corr


def run_model_comparison(
    factors: dict,
    returns: pd.DataFrame,
    train_years: int = 3,
    val_years: int = 1,
    test_years: int = 1,
    n_splits: int = 3,
    models_to_run: Optional[list] = None,
    verbose: bool = True,
    max_features: int = 500,
) -> dict:
    """
    多模型滚动训练与评估

    Parameters:
        factors: dict[str, pd.DataFrame/pl.DataFrame] 因子
        returns: pd.DataFrame 收益率
        train_years: 训练年数
        val_years: 验证年数
        test_years: 测试年数
        n_splits: 滚动窗口数
        models_to_run: 要运行的模型列表 (None=全部)
        max_features: 最大特征数 (超过时随机采样)

    Returns:
        dict: 模型名 -> {ic_mean, ic_std, icir, rank_ic, ...}
    """
    total_start = time.time()

    # 限制因子数量
    if len(factors) > max_features:
        if verbose:
            print(f"  因子数 {len(factors)} > {max_features}, 随机采样 {max_features} 个")
        import random
        random.seed(42)
        selected_keys = random.sample(list(factors.keys()), max_features)
        factors = {k: factors[k] for k in selected_keys}

    # 加载所有模型
    available_models = {}

    # Tree models
    try:
        from models.tree_models import LGBMModel, XGBModel, CatModel, DoubleEnsembleModel
        available_models["LightGBM"] = (LGBMModel, {})
        available_models["XGBoost"] = (XGBModel, {})
        available_models["CatBoost"] = (CatModel, {})
        available_models["DoubleEnsemble"] = (DoubleEnsembleModel, {})
    except ImportError as e:
        if verbose:
            print(f"  树模型加载失败: {e}")

    # Linear models
    try:
        from models.linear_models import RidgeModel, ElasticNetModel
        available_models["Ridge"] = (RidgeModel, {})
        available_models["ElasticNet"] = (ElasticNetModel, {})
    except ImportError as e:
        if verbose:
            print(f"  线性模型加载失败: {e}")

    # Deep learning models (MLX) - use seq_len=1 for cross-sectional data
    try:
        from models.mlx_models import (
            MLPModel, GRUModel, LSTMModel, ALSTMModel,
            TransformerModel, TCNModel, LocalformerModel, GATModel,
        )
        available_models["MLP"] = (MLPModel, {})
        available_models["GRU"] = (GRUModel, {"seq_len": 1, "n_epochs": 30})
        available_models["LSTM"] = (LSTMModel, {"seq_len": 1, "n_epochs": 30})
        available_models["ALSTM"] = (ALSTMModel, {"seq_len": 1, "n_epochs": 30})
        available_models["Transformer"] = (TransformerModel, {"seq_len": 1, "n_epochs": 30})
        available_models["TCN"] = (TCNModel, {"seq_len": 3, "n_epochs": 30})  # TCN needs seq_len > 1
        available_models["Localformer"] = (LocalformerModel, {"seq_len": 1, "n_epochs": 30})
        available_models["GAT"] = (GATModel, {"n_epochs": 30})  # GAT already sets seq_len=1
    except ImportError as e:
        if verbose:
            print(f"  MLX 深度模型加载失败: {e}")

    if models_to_run:
        available_models = {k: v for k, v in available_models.items() if k in models_to_run}

    if verbose:
        print(f"  可用模型: {len(available_models)} 个")
        for name in available_models:
            print(f"    - {name}")

    # 时间分割
    all_dates = sorted(returns.index)
    total_periods = len(all_dates)
    periods_per_year = 52  # 周频
    train_periods = train_years * periods_per_year
    val_periods = val_years * periods_per_year
    test_periods = test_years * periods_per_year
    window_size = train_periods + val_periods + test_periods

    if total_periods < window_size:
        if verbose:
            print(f"  数据不足: 需要 {window_size} 期, 仅有 {total_periods} 期")
            print(f"  使用简单 train/test split")
        n_splits = 1
        split_points = [(0, total_periods)]
    else:
        step = max(1, (total_periods - window_size) // max(1, n_splits - 1))
        split_points = []
        for i in range(n_splits):
            start = i * step
            if start + window_size > total_periods:
                break
            split_points.append((start, start + window_size))

    if verbose:
        print(f"  滚动窗口: {len(split_points)} 个")

    # 模型评估
    results = {}

    for model_name, model_info in available_models.items():
        # Support both plain class and (class, kwargs) tuple
        if isinstance(model_info, tuple):
            model_cls, model_kwargs = model_info
        else:
            model_cls, model_kwargs = model_info, {}

        if verbose:
            print(f"\n  ── {model_name} ──")
        t0 = time.time()

        ic_list = []

        for split_idx, (start, end) in enumerate(split_points):
            train_end_idx = start + train_periods
            val_end_idx = train_end_idx + val_periods

            train_dates = all_dates[start:train_end_idx]
            test_dates = all_dates[val_end_idx:end]

            if not train_dates or not test_dates:
                continue

            train_start_str = str(train_dates[0].date()) if hasattr(train_dates[0], 'date') else str(train_dates[0])
            train_end_str = str(train_dates[-1].date()) if hasattr(train_dates[-1], 'date') else str(train_dates[-1])
            test_start_str = str(test_dates[0].date()) if hasattr(test_dates[0], 'date') else str(test_dates[0])
            test_end_str = str(test_dates[-1].date()) if hasattr(test_dates[-1], 'date') else str(test_dates[-1])

            # 准备数据
            X_train, y_train, feat_names = _prepare_Xy(
                factors, returns, train_start_str, train_end_str
            )
            X_test, y_test, _ = _prepare_Xy(
                factors, returns, test_start_str, test_end_str
            )

            if len(X_train) < 100 or len(X_test) < 50:
                if verbose:
                    print(f"    split {split_idx}: 数据不足 (train={len(X_train)}, test={len(X_test)})")
                continue

            try:
                # Convert numpy to Polars (models expect Polars input)
                X_train_pl = pl.DataFrame(
                    {f"f{i}": X_train[:, i] for i in range(X_train.shape[1])}
                )
                y_train_pl = pl.Series("y", y_train)
                X_test_pl = pl.DataFrame(
                    {f"f{i}": X_test[:, i] for i in range(X_test.shape[1])}
                )

                # 训练
                model = model_cls(**model_kwargs)
                model.fit(X_train_pl, y_train_pl)

                # 预测
                preds_pl = model.predict(X_test_pl)
                preds = preds_pl.to_numpy() if isinstance(preds_pl, pl.Series) else np.array(preds_pl)

                # 计算 IC
                ic = _calc_ic(preds, y_test)
                ic_list.append(ic)

                if verbose:
                    print(f"    split {split_idx}: IC={ic:.4f} "
                          f"(train={len(X_train)}, test={len(X_test)})")

            except Exception as e:
                if verbose:
                    print(f"    split {split_idx}: 失败 - {e}")

        if ic_list:
            ic_arr = np.array([x for x in ic_list if np.isfinite(x)])
            results[model_name] = {
                "IC_mean": ic_arr.mean() if len(ic_arr) > 0 else np.nan,
                "IC_std": ic_arr.std() if len(ic_arr) > 0 else np.nan,
                "ICIR": (ic_arr.mean() / ic_arr.std()) if len(ic_arr) > 1 and ic_arr.std() > 0 else np.nan,
                "IC_positive_ratio": (ic_arr > 0).mean() if len(ic_arr) > 0 else np.nan,
                "n_splits": len(ic_arr),
                "train_time": time.time() - t0,
            }
            if verbose:
                r = results[model_name]
                print(f"    汇总: IC={r['IC_mean']:.4f}, ICIR={r['ICIR']:.4f}, "
                      f"IC>0={r['IC_positive_ratio']:.1%}, 耗时={r['train_time']:.1f}s")
        else:
            results[model_name] = {
                "IC_mean": np.nan,
                "IC_std": np.nan,
                "ICIR": np.nan,
                "IC_positive_ratio": np.nan,
                "n_splits": 0,
                "train_time": time.time() - t0,
            }
            if verbose:
                print(f"    无有效结果")

    total_time = time.time() - total_start
    if verbose:
        print(f"\n  模型对比完成 | 耗时 {total_time:.1f}s")

    return results


def format_model_comparison(results: dict) -> pd.DataFrame:
    """格式化模型对比表"""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "IC_mean": r.get("IC_mean", np.nan),
            "IC_std": r.get("IC_std", np.nan),
            "ICIR": r.get("ICIR", np.nan),
            "IC>0%": r.get("IC_positive_ratio", np.nan),
            "Splits": r.get("n_splits", 0),
            "Time(s)": r.get("train_time", 0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("ICIR", ascending=False).reset_index(drop=True)
    return df
