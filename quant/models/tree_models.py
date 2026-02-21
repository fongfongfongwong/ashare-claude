"""
树模型模块
包含 LightGBM, XGBoost, CatBoost 以及 DoubleEnsemble 算法
数据处理使用 Polars
"""
import logging
import numpy as np
import polars as pl

from .base import BaseModel

logger = logging.getLogger(__name__)

# ============================================================
# 安全导入: LightGBM / XGBoost / CatBoost
# ============================================================
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


# ============================================================
# LightGBM
# ============================================================
class LGBMModel(BaseModel):
    """
    LightGBM 回归模型

    默认参数针对 A 股多因子场景调优:
    - n_estimators=1000 配合 early_stopping=50 防止过拟合
    - num_leaves=128 提供足够的模型复杂度
    - feature_fraction=0.8 增加随机性
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 128,
        feature_fraction: float = 0.8,
        early_stopping: int = 50,
        **kwargs,
    ):
        if not HAS_LGBM:
            raise ImportError(
                "LightGBM 未安装，请执行: pip install lightgbm"
            )
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            early_stopping=early_stopping,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.early_stopping = early_stopping
        self.extra_params = kwargs
        self.model = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """训练 LightGBM 模型"""
        self._feature_names = X_train.columns

        # Polars -> numpy
        X_tr = X_train.to_numpy()
        y_tr = y_train.to_numpy().ravel()
        train_set = lgb.Dataset(X_tr, label=y_tr)

        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            X_v = X_val.to_numpy()
            y_v = y_val.to_numpy().ravel()
            val_set = lgb.Dataset(X_v, label=y_v, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("valid")

        params = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "feature_fraction": self.feature_fraction,
            "verbosity": -1,
            **self.extra_params,
        }

        callbacks = [lgb.log_evaluation(period=0)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping, verbose=False))

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self._fitted = True
        logger.info(
            f"LGBMModel 训练完成, best_iteration={self.model.best_iteration}"
        )
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """预测"""
        self._check_fitted()
        X_np = X.to_numpy()
        preds = self.model.predict(X_np, num_iteration=self.model.best_iteration)
        return pl.Series("prediction", preds)

    def get_feature_importance(self) -> dict[str, float]:
        """返回特征重要性 (gain)"""
        self._check_fitted()
        importance = self.model.feature_importance(importance_type="gain")
        total = importance.sum()
        if total > 0:
            importance = importance / total
        return dict(zip(self._feature_names, importance))


# ============================================================
# XGBoost
# ============================================================
class XGBModel(BaseModel):
    """
    XGBoost 回归模型

    默认参数:
    - max_depth=8 控制树深度
    - colsample_bytree=0.8 列采样
    - tree_method='hist' 加速训练
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        colsample_bytree: float = 0.8,
        early_stopping: int = 50,
        **kwargs,
    ):
        if not HAS_XGB:
            raise ImportError(
                "XGBoost 未安装，请执行: pip install xgboost"
            )
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            early_stopping=early_stopping,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.early_stopping = early_stopping
        self.extra_params = kwargs
        self.model = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """训练 XGBoost 模型"""
        self._feature_names = X_train.columns

        X_tr = X_train.to_numpy()
        y_tr = y_train.to_numpy().ravel()
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=self._feature_names)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            X_v = X_val.to_numpy()
            y_v = y_val.to_numpy().ravel()
            dval = xgb.DMatrix(X_v, label=y_v, feature_names=self._feature_names)
            evals.append((dval, "valid"))

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": "hist",
            "verbosity": 0,
            **self.extra_params,
        }

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping if X_val is not None else None,
            verbose_eval=False,
        )
        self._fitted = True
        try:
            logger.info(f"XGBModel 训练完成, best_iteration={self.model.best_iteration}")
        except AttributeError:
            logger.info(f"XGBModel 训练完成, n_boost_round={self.n_estimators}")
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """预测"""
        self._check_fitted()
        X_np = X.to_numpy()
        dtest = xgb.DMatrix(X_np, feature_names=self._feature_names)
        try:
            best_iter = self.model.best_iteration
            preds = self.model.predict(dtest, iteration_range=(0, best_iter))
        except Exception:
            preds = self.model.predict(dtest)
        return pl.Series("prediction", preds)

    def get_feature_importance(self) -> dict[str, float]:
        """返回特征重要性 (gain)"""
        self._check_fitted()
        score = self.model.get_score(importance_type="gain")
        total = sum(score.values())
        if total > 0:
            return {k: v / total for k, v in score.items()}
        return score


# ============================================================
# CatBoost
# ============================================================
class CatModel(BaseModel):
    """
    CatBoost 回归模型

    默认参数:
    - iterations=1000, depth=8
    - verbose=0 关闭输出
    """

    def __init__(
        self,
        iterations: int = 1000,
        depth: int = 8,
        learning_rate: float = 0.05,
        early_stopping: int = 50,
        verbose: int = 0,
        **kwargs,
    ):
        if not HAS_CAT:
            raise ImportError(
                "CatBoost 未安装，请执行: pip install catboost"
            )
        super().__init__(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            verbose=verbose,
            **kwargs,
        )
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.extra_params = kwargs
        self.model = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """训练 CatBoost 模型"""
        self._feature_names = X_train.columns

        X_tr = X_train.to_numpy()
        y_tr = y_train.to_numpy().ravel()
        train_pool = cb.Pool(X_tr, label=y_tr)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.to_numpy()
            y_v = y_val.to_numpy().ravel()
            eval_set = cb.Pool(X_v, label=y_v)

        self.model = cb.CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping if eval_set is not None else None,
            **self.extra_params,
        )
        self.model.fit(train_pool, eval_set=eval_set)
        self._fitted = True
        logger.info(
            f"CatModel 训练完成, best_iteration={self.model.get_best_iteration()}"
        )
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """预测"""
        self._check_fitted()
        X_np = X.to_numpy()
        preds = self.model.predict(X_np)
        return pl.Series("prediction", preds)

    def get_feature_importance(self) -> dict[str, float]:
        """返回特征重要性"""
        self._check_fitted()
        importance = self.model.get_feature_importance()
        total = importance.sum()
        if total > 0:
            importance = importance / total
        return dict(zip(self._feature_names, importance))


# ============================================================
# DoubleEnsemble (Zhang et al., ICDM 2020)
# ============================================================
class DoubleEnsembleModel(BaseModel):
    """
    DoubleEnsemble 算法 (基于 LightGBM)

    核心思想 (Zhang et al., ICDM 2020):
    1. 多轮迭代训练多个 base model
    2. 每轮迭代:
       a) 训练当前 base model
       b) 根据样本 loss 重新加权样本 (难样本获得更高权重)
       c) 通过 feature shuffling 评估特征重要性
       d) 选择 top-k 特征子集
       e) 在子集上重新训练
    3. 最终预测为所有 base model 的平均

    参数:
        n_rounds: 迭代轮数 (默认 6)
        n_estimators: 每个 base model 的树数量
        feature_ratio: 每轮保留的特征比例
        sample_alpha: 样本重加权的衰减系数
    """

    def __init__(
        self,
        n_rounds: int = 6,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 64,
        feature_ratio: float = 0.7,
        sample_alpha: float = 1.0,
        early_stopping: int = 30,
        **kwargs,
    ):
        if not HAS_LGBM:
            raise ImportError(
                "DoubleEnsemble 依赖 LightGBM，请执行: pip install lightgbm"
            )
        super().__init__(
            n_rounds=n_rounds,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            feature_ratio=feature_ratio,
            sample_alpha=sample_alpha,
            early_stopping=early_stopping,
            **kwargs,
        )
        self.n_rounds = n_rounds
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_ratio = feature_ratio
        self.sample_alpha = sample_alpha
        self.early_stopping = early_stopping
        self.extra_params = kwargs

        # 存储每轮训练的 base model 和对应使用的特征
        self._base_models: list[lgb.Booster] = []
        self._feature_subsets: list[list[int]] = []
        self._feature_names: list[str] = []

    def _compute_sample_weights(
        self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        根据预测残差计算样本权重
        loss 大的样本 (难样本) 获得更高权重
        """
        residuals = np.abs(y_true - y_pred)
        # 标准化残差到 [0, 1]
        r_min, r_max = residuals.min(), residuals.max()
        if r_max - r_min > 1e-10:
            normalized = (residuals - r_min) / (r_max - r_min)
        else:
            normalized = np.zeros_like(residuals)
        # 指数加权: 难样本权重更高
        weights = np.exp(alpha * normalized)
        weights = weights / weights.mean()  # 归一化使均值为 1
        return weights

    def _evaluate_feature_importance_by_shuffling(
        self,
        model: lgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        """
        通过特征 shuffling 评估特征重要性
        对每个特征: 随机打乱 → 计算 loss 增量 → loss 增加越多说明特征越重要
        """
        baseline_pred = model.predict(X)
        baseline_loss = np.mean((y - baseline_pred) ** 2)

        importance = np.zeros(n_features)
        for j in range(n_features):
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, j])
            shuffled_pred = model.predict(X_shuffled)
            shuffled_loss = np.mean((y - shuffled_pred) ** 2)
            importance[j] = shuffled_loss - baseline_loss

        # 确保非负
        importance = np.maximum(importance, 0)
        return importance

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """训练 DoubleEnsemble 模型"""
        self._feature_names = X_train.columns
        n_features = len(self._feature_names)
        n_keep = max(1, int(n_features * self.feature_ratio))

        X_tr = X_train.to_numpy().astype(np.float32)
        y_tr = y_train.to_numpy().ravel().astype(np.float32)

        # 验证集 (用于 early stopping)
        X_v, y_v = None, None
        if X_val is not None and y_val is not None:
            X_v = X_val.to_numpy().astype(np.float32)
            y_v = y_val.to_numpy().ravel().astype(np.float32)

        # 初始化: 样本权重均匀, 使用全部特征
        sample_weights = np.ones(len(y_tr), dtype=np.float32)
        feature_indices = list(range(n_features))

        self._base_models = []
        self._feature_subsets = []

        # 当前集成预测 (用于计算残差)
        ensemble_pred = np.zeros_like(y_tr)

        for round_idx in range(self.n_rounds):
            logger.info(
                f"DoubleEnsemble 第 {round_idx + 1}/{self.n_rounds} 轮, "
                f"使用 {len(feature_indices)} 个特征"
            )

            # --- 步骤 1: 用当前特征子集和样本权重训练 base model ---
            X_sub = X_tr[:, feature_indices]
            train_set = lgb.Dataset(X_sub, label=y_tr, weight=sample_weights)

            valid_sets = [train_set]
            valid_names = ["train"]
            if X_v is not None:
                X_v_sub = X_v[:, feature_indices]
                val_set = lgb.Dataset(X_v_sub, label=y_v, reference=train_set)
                valid_sets.append(val_set)
                valid_names.append("valid")

            params = {
                "objective": "regression",
                "metric": "mse",
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "feature_fraction": 0.8,
                "verbosity": -1,
                **self.extra_params,
            }

            callbacks = [lgb.log_evaluation(period=0)]
            if X_v is not None:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping, verbose=False)
                )

            model = lgb.train(
                params,
                train_set,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )

            self._base_models.append(model)
            self._feature_subsets.append(list(feature_indices))

            # --- 步骤 2: 更新集成预测 ---
            round_pred = model.predict(X_sub, num_iteration=model.best_iteration)
            ensemble_pred = (
                ensemble_pred * round_idx + round_pred
            ) / (round_idx + 1)

            # --- 步骤 3: 根据 loss 重加权样本 ---
            sample_weights = self._compute_sample_weights(
                y_tr, ensemble_pred, self.sample_alpha
            )

            # --- 步骤 4: 通过 shuffling 评估特征重要性, 选择 top 特征 ---
            if round_idx < self.n_rounds - 1:
                importance = self._evaluate_feature_importance_by_shuffling(
                    model, X_sub, y_tr, len(feature_indices)
                )
                # 将子集重要性映射回全局特征索引
                global_importance = np.zeros(n_features)
                for local_idx, global_idx in enumerate(feature_indices):
                    global_importance[global_idx] = importance[local_idx]

                # 选择 top-k 特征
                top_indices = np.argsort(global_importance)[::-1][:n_keep]
                feature_indices = sorted(top_indices.tolist())

                # 确保至少有 1 个特征
                if len(feature_indices) == 0:
                    feature_indices = list(range(n_features))

        self._fitted = True
        logger.info(
            f"DoubleEnsemble 训练完成, 共 {len(self._base_models)} 个 base model"
        )
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """预测: 对所有 base model 预测取平均"""
        self._check_fitted()
        X_np = X.to_numpy().astype(np.float32)
        preds = np.zeros(len(X_np), dtype=np.float64)

        for model, feat_indices in zip(self._base_models, self._feature_subsets):
            X_sub = X_np[:, feat_indices]
            preds += model.predict(X_sub, num_iteration=model.best_iteration)

        preds /= len(self._base_models)
        return pl.Series("prediction", preds)

    def get_feature_importance(self) -> dict[str, float]:
        """返回所有 base model 的平均特征重要性"""
        self._check_fitted()
        n_features = len(self._feature_names)
        avg_importance = np.zeros(n_features)

        for model, feat_indices in zip(self._base_models, self._feature_subsets):
            imp = model.feature_importance(importance_type="gain")
            total = imp.sum()
            if total > 0:
                imp = imp / total
            for local_idx, global_idx in enumerate(feat_indices):
                avg_importance[global_idx] += imp[local_idx]

        avg_importance /= len(self._base_models)
        total = avg_importance.sum()
        if total > 0:
            avg_importance /= total

        return dict(zip(self._feature_names, avg_importance))
