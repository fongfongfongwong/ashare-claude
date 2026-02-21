"""
模型基类模块
定义所有模型的统一接口
"""
import polars as pl
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """所有模型的基类，定义统一的 fit/predict 接口"""

    def __init__(self, **kwargs):
        self.params = kwargs
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """
        训练模型

        参数:
            X_train: 训练特征 (Polars DataFrame)
            y_train: 训练标签 (Polars Series)
            X_val: 验证特征 (可选，用于 early stopping)
            y_val: 验证标签
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pl.DataFrame) -> pl.Series:
        """
        模型预测

        参数:
            X: 特征 (Polars DataFrame)

        返回:
            预测值 (Polars Series)
        """
        raise NotImplementedError

    def get_feature_importance(self) -> dict[str, float]:
        """返回特征重要性 (如模型支持)"""
        return {}

    def _check_fitted(self):
        """检查模型是否已训练"""
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__} 尚未训练，请先调用 fit()")

    def __repr__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
