"""
因子库 (Factor Zoo)
==================
基于 Polars 的高性能因子计算库，包含:
- alpha101: WorldQuant 101 公式化因子 (精选40个)
- alpha158: Qlib Alpha158 技术指标因子
- alpha360: Qlib Alpha360 标准化价量序列因子
- operator_expand: 算子扩展模块
- cross_features: 因子交叉特征
- factor_filter: 因子筛选与去冗余
- gp_miner: 基于遗传编程的因子挖掘器 (WS-GP + AlphaGen)
"""

from .alpha101 import build_alpha101
from .alpha158 import build_alpha158
from .alpha360 import build_alpha360
from .operator_expand import expand_factors
from .cross_features import build_cross_features
from .factor_filter import calc_ic, filter_by_ic, remove_redundant, factor_report
from .gp_miner import GPMiner, ExprNode, mine_factors, parse_expression

__all__ = [
    "build_alpha101",
    "build_alpha158",
    "build_alpha360",
    "expand_factors",
    "build_cross_features",
    "calc_ic",
    "filter_by_ic",
    "remove_redundant",
    "factor_report",
    "GPMiner",
    "ExprNode",
    "mine_factors",
    "parse_expression",
]
