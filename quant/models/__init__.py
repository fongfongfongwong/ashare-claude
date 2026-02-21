"""
模型模块
提供 14 个模型的统一接口

树模型 (4):
    - LGBMModel: LightGBM
    - XGBModel: XGBoost
    - CatModel: CatBoost
    - DoubleEnsembleModel: DoubleEnsemble (Zhang et al., ICDM 2020)

线性模型 (2):
    - RidgeModel: Ridge Regression
    - ElasticNetModel: ElasticNet

深度学习模型 (6 + MLX):
    - MLPModel: 多层感知机
    - GRUModel: GRU
    - LSTMModel: LSTM
    - ALSTMModel: LSTM + Temporal Attention
    - TransformerModel: Transformer Encoder
    - TCNModel: 时间卷积网络
    - LocalformerModel: 局部注意力 + 全局注意力
    - GATModel: 图注意力网络

集成模型 (2):
    - StackingEnsemble: Stacking (Ridge meta-learner)
    - BlendingEnsemble: 加权平均 (IC 权重)
"""
from .base import BaseModel

# 树模型
from .tree_models import LGBMModel, XGBModel, CatModel, DoubleEnsembleModel

# 线性模型
from .linear_models import RidgeModel, ElasticNetModel

# 深度学习模型 (MLX)
from .mlx_models import (
    MLPModel,
    GRUModel,
    LSTMModel,
    ALSTMModel,
    TransformerModel,
    TCNModel,
    LocalformerModel,
    GATModel,
)

# 集成模型
from .ensemble import StackingEnsemble, BlendingEnsemble


# 所有可用模型的注册表 (名称 -> 类)
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    # 树模型
    "lgbm": LGBMModel,
    "xgb": XGBModel,
    "catboost": CatModel,
    "double_ensemble": DoubleEnsembleModel,
    # 线性模型
    "ridge": RidgeModel,
    "elastic_net": ElasticNetModel,
    # 深度学习模型
    "mlp": MLPModel,
    "gru": GRUModel,
    "lstm": LSTMModel,
    "alstm": ALSTMModel,
    "transformer": TransformerModel,
    "tcn": TCNModel,
    "localformer": LocalformerModel,
    "gat": GATModel,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """
    通过名称获取模型实例

    参数:
        name: 模型名称 (见 MODEL_REGISTRY)
        **kwargs: 传递给模型构造函数的参数

    示例:
        model = get_model("lgbm", n_estimators=500)
        model = get_model("transformer", seq_len=20, d_model=64)
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"未知模型 '{name}'，可用模型: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """返回所有可用模型名称"""
    return sorted(MODEL_REGISTRY.keys())


__all__ = [
    # 基类
    "BaseModel",
    # 树模型
    "LGBMModel",
    "XGBModel",
    "CatModel",
    "DoubleEnsembleModel",
    # 线性模型
    "RidgeModel",
    "ElasticNetModel",
    # 深度学习模型
    "MLPModel",
    "GRUModel",
    "LSTMModel",
    "ALSTMModel",
    "TransformerModel",
    "TCNModel",
    "LocalformerModel",
    "GATModel",
    # 集成模型
    "StackingEnsemble",
    "BlendingEnsemble",
    # 工具函数
    "get_model",
    "list_models",
    "MODEL_REGISTRY",
]
