"""
模型融合模块
- StackingEnsemble: 用 Ridge 做 meta-learner，输入各模型的 OOF 预测
- BlendingEnsemble: 简单加权平均，权重按验证集 IC 分配
"""
import logging
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .base import BaseModel

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseModel):
    """
    Stacking 集成模型

    第一层: 多个 base model 通过 K-Fold 交叉验证生成 OOF (Out-Of-Fold) 预测
    第二层: 用 Ridge 回归作为 meta-learner，输入 OOF 预测，输出最终预测

    参数:
        base_models: dict[str, BaseModel] — 基模型字典 {名称: 模型实例}
        meta_alpha: Ridge 正则化系数 (默认 1.0)
        n_folds: 交叉验证折数 (默认 5)
    """

    def __init__(
        self,
        base_models: dict[str, BaseModel],
        meta_alpha: float = 1.0,
        n_folds: int = 5,
    ):
        super().__init__(
            meta_alpha=meta_alpha,
            n_folds=n_folds,
        )
        self.base_models = base_models
        self.model_names = list(base_models.keys())
        self.meta_alpha = meta_alpha
        self.n_folds = n_folds

        self.meta_model = None
        # 存储每个 base model 在全量训练集上训练的版本 (用于 predict)
        self._trained_base_models: dict[str, BaseModel] = {}

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """
        Stacking 训练流程:
        1. 对每个 base model 做 K-Fold 生成 OOF 预测
        2. 用 OOF 预测作为特征训练 meta-learner
        3. 用全量数据重新训练所有 base model (用于 predict 阶段)
        """
        n_samples = len(y_train)
        n_models = len(self.base_models)

        # OOF 预测矩阵 (n_samples, n_models)
        oof_preds = np.zeros((n_samples, n_models))
        y_np = y_train.to_numpy().ravel()

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for model_idx, (name, model_template) in enumerate(self.base_models.items()):
            logger.info(f"Stacking: 训练 base model '{name}' ({self.n_folds}-Fold)")

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
                # 用索引切分 Polars DataFrame
                X_tr_fold = X_train[train_idx.tolist()]
                y_tr_fold = y_train[train_idx.tolist()]
                X_val_fold = X_train[val_idx.tolist()]
                y_val_fold = y_train[val_idx.tolist()]

                # 创建新模型实例 (避免污染模板)
                model = model_template.__class__(**model_template.params)
                model.fit(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold)

                # OOF 预测
                fold_preds = model.predict(X_val_fold).to_numpy()
                oof_preds[val_idx, model_idx] = fold_preds

                logger.info(
                    f"  Fold {fold_idx + 1}/{self.n_folds} 完成"
                )

        # 训练 meta-learner
        logger.info("Stacking: 训练 meta-learner (Ridge)")
        self.meta_model = Ridge(alpha=self.meta_alpha)
        self.meta_model.fit(oof_preds, y_np)

        # 用全量数据重新训练所有 base model
        logger.info("Stacking: 用全量数据重训所有 base model")
        for name, model_template in self.base_models.items():
            model = model_template.__class__(**model_template.params)
            model.fit(X_train, y_train, X_val, y_val)
            self._trained_base_models[name] = model

        self._fitted = True
        logger.info("StackingEnsemble 训练完成")
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """
        Stacking 预测:
        1. 每个 base model 产生预测
        2. meta-learner 融合为最终预测
        """
        self._check_fitted()
        n_samples = len(X)
        n_models = len(self._trained_base_models)
        test_preds = np.zeros((n_samples, n_models))

        for model_idx, (name, model) in enumerate(self._trained_base_models.items()):
            test_preds[:, model_idx] = model.predict(X).to_numpy()

        final_preds = self.meta_model.predict(test_preds)
        return pl.Series("prediction", final_preds)

    def get_feature_importance(self) -> dict[str, float]:
        """返回 meta-learner 对各 base model 的权重"""
        self._check_fitted()
        coef = np.abs(self.meta_model.coef_)
        total = coef.sum()
        if total > 0:
            coef = coef / total
        return dict(zip(self.model_names, coef))


class BlendingEnsemble(BaseModel):
    """
    Blending 集成模型

    简单加权平均融合，权重按验证集上的 Rank IC 自动分配:
    - IC 越高的模型获得越大权重
    - 也可以手动指定权重

    参数:
        base_models: dict[str, BaseModel] — 基模型字典
        weights: dict[str, float] — 手动指定权重 (可选)
    """

    def __init__(
        self,
        base_models: dict[str, BaseModel],
        weights: dict[str, float] = None,
    ):
        super().__init__()
        self.base_models = base_models
        self.model_names = list(base_models.keys())
        self._weights: dict[str, float] = weights or {}
        self._trained_models: dict[str, BaseModel] = {}

    def _compute_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算 Rank IC (Spearman 相关系数)"""
        if len(y_true) < 10:
            return 0.0
        corr, _ = stats.spearmanr(y_true, y_pred)
        return corr if not np.isnan(corr) else 0.0

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """
        Blending 训练:
        1. 训练所有 base model
        2. 如果未手动指定权重，在验证集上计算 IC 自动分配权重
        """
        for name, model_template in self.base_models.items():
            logger.info(f"Blending: 训练 base model '{name}'")
            model = model_template.__class__(**model_template.params)
            model.fit(X_train, y_train, X_val, y_val)
            self._trained_models[name] = model

        # 自动计算权重 (基于验证集 IC)
        if not self._weights and X_val is not None and y_val is not None:
            y_val_np = y_val.to_numpy().ravel()
            ic_scores = {}

            for name, model in self._trained_models.items():
                preds = model.predict(X_val).to_numpy()
                ic = self._compute_ic(y_val_np, preds)
                ic_scores[name] = max(ic, 0.0)  # 负 IC 的模型权重为 0
                logger.info(f"  {name}: val IC = {ic:.4f}")

            total_ic = sum(ic_scores.values())
            if total_ic > 0:
                self._weights = {k: v / total_ic for k, v in ic_scores.items()}
            else:
                # fallback: 等权
                n = len(self.model_names)
                self._weights = {name: 1.0 / n for name in self.model_names}

            logger.info(f"Blending 权重: {self._weights}")

        elif not self._weights:
            # 没有验证集也没有手动权重: 等权
            n = len(self.model_names)
            self._weights = {name: 1.0 / n for name in self.model_names}

        self._fitted = True
        logger.info("BlendingEnsemble 训练完成")
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """加权平均预测"""
        self._check_fitted()
        n_samples = len(X)
        final_preds = np.zeros(n_samples)

        for name, model in self._trained_models.items():
            weight = self._weights.get(name, 0.0)
            if weight > 0:
                preds = model.predict(X).to_numpy()
                final_preds += weight * preds

        return pl.Series("prediction", final_preds)

    def get_feature_importance(self) -> dict[str, float]:
        """返回各 base model 的融合权重"""
        self._check_fitted()
        return dict(self._weights)

    @property
    def weights(self) -> dict[str, float]:
        """获取当前融合权重"""
        return dict(self._weights)
