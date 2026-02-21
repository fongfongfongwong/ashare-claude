"""
线性模型模块
包含 Ridge Regression 和 ElasticNet
数据处理使用 Polars
"""
import logging
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class RidgeModel(BaseModel):
    """
    Ridge 回归模型

    内置标准化: 训练前自动对特征做 z-score 标准化
    适合作为线性基准模型或 stacking 的 meta-learner

    参数:
        alpha: L2 正则化系数 (默认 1.0)
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = alpha
        self.extra_params = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """训练 Ridge 回归"""
        self._feature_names = X_train.columns

        X_tr = X_train.to_numpy()
        y_tr = y_train.to_numpy().ravel()

        # 标准化
        X_scaled = self.scaler.fit_transform(X_tr)

        self.model = Ridge(alpha=self.alpha, **self.extra_params)
        self.model.fit(X_scaled, y_tr)
        self._fitted = True

        logger.info(f"RidgeModel 训练完成, alpha={self.alpha}")
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """预测"""
        self._check_fitted()
        X_np = X.to_numpy()
        X_scaled = self.scaler.transform(X_np)
        preds = self.model.predict(X_scaled)
        return pl.Series("prediction", preds)

    def get_feature_importance(self) -> dict[str, float]:
        """返回系数绝对值作为特征重要性"""
        self._check_fitted()
        coef = np.abs(self.model.coef_)
        total = coef.sum()
        if total > 0:
            coef = coef / total
        return dict(zip(self._feature_names, coef))


class ElasticNetModel(BaseModel):
    """
    ElasticNet 回归模型

    结合 L1 和 L2 正则化:
    - alpha 控制总正则化强度
    - l1_ratio 控制 L1 和 L2 的比例 (0=纯 Ridge, 1=纯 Lasso)

    参数:
        alpha: 总正则化系数 (默认 0.01)
        l1_ratio: L1 比例 (默认 0.5)
        max_iter: 最大迭代次数 (默认 5000)
    """

    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
        **kwargs,
    ):
        super().__init__(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, **kwargs
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.extra_params = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """训练 ElasticNet"""
        self._feature_names = X_train.columns

        X_tr = X_train.to_numpy()
        y_tr = y_train.to_numpy().ravel()

        # 标准化
        X_scaled = self.scaler.fit_transform(X_tr)

        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            **self.extra_params,
        )
        self.model.fit(X_scaled, y_tr)
        self._fitted = True

        logger.info(
            f"ElasticNetModel 训练完成, alpha={self.alpha}, l1_ratio={self.l1_ratio}"
        )
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """预测"""
        self._check_fitted()
        X_np = X.to_numpy()
        X_scaled = self.scaler.transform(X_np)
        preds = self.model.predict(X_scaled)
        return pl.Series("prediction", preds)

    def get_feature_importance(self) -> dict[str, float]:
        """返回系数绝对值作为特征重要性"""
        self._check_fitted()
        coef = np.abs(self.model.coef_)
        total = coef.sum()
        if total > 0:
            coef = coef / total
        return dict(zip(self._feature_names, coef))
