"""
基于遗传编程的因子挖掘器 (GP Factor Miner)
============================================
参考论文:
  - Warm-Start GP (WS-GP, 2024): 利用已知优质因子模板初始化种群
  - AlphaGen (KDD 2023): 基于表达式树的 Alpha 因子自动生成

全部使用 Polars 进行高性能向量化计算。

核心流程:
  1. 用 Alpha101 模板 + 随机变异构建初始种群 (Warm-Start)
  2. 用 Rank IC (Spearman) 作为适应度函数评估因子质量
  3. 锦标赛选择 + 子树交叉 + 点变异 / 子树变异 驱动进化
  4. 自动去冗余 (相关性阈值)，保留独立有效因子
"""

from __future__ import annotations

import copy
import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ============================================================
# 常量定义
# ============================================================

# 叶节点: 数据列名
FEATURE_NAMES = ["open", "high", "low", "close", "volume", "vwap", "returns"]

# 叶节点: 常数值
CONSTANTS = [-1.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# 时序算子的候选窗口期
TS_WINDOWS = [5, 10, 20, 40, 60]

# 算子分类
BINARY_OPS = ["add", "sub", "mul", "safe_div"]
UNARY_OPS = ["abs", "neg", "sign", "log1p", "sqrt_abs"]
TS_OPS_1ARG = [
    "ts_mean", "ts_std", "ts_rank", "ts_delta",
    "ts_max", "ts_min", "ts_decay_linear",
]
TS_OPS_2ARG = ["ts_corr"]
CS_OPS = ["cs_rank", "cs_zscore"]

# 所有算子及其元数 (children 数量)
OP_ARITY: dict[str, int] = {}
for op in BINARY_OPS:
    OP_ARITY[op] = 2
for op in UNARY_OPS:
    OP_ARITY[op] = 1
for op in TS_OPS_1ARG:
    OP_ARITY[op] = 1  # 数据子节点 (窗口期存储在节点参数中)
for op in TS_OPS_2ARG:
    OP_ARITY[op] = 2  # 两个数据子节点
for op in CS_OPS:
    OP_ARITY[op] = 1

ALL_OPS = list(OP_ARITY.keys())


# ============================================================
# 表达式树节点
# ============================================================

@dataclass
class ExprNode:
    """
    表达式树节点。

    - 内部节点: op 为算子名, children 为子节点列表
    - 叶节点: op = "feature" 或 "const", value 存储具体的列名或数值
    - 时序算子: window 存储窗口期参数
    """

    op: str = ""
    children: list["ExprNode"] = field(default_factory=list)
    value: Optional[str | float] = None
    window: Optional[int] = None

    # ------ 序列化 / 可读表示 ------

    def to_string(self) -> str:
        """将表达式树转为人类可读的字符串表达式"""
        if self.op == "feature":
            return str(self.value)
        if self.op == "const":
            v = self.value
            # 整数显示不带小数点
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)

        # 时序算子: ts_xxx(child, window)
        if self.op in TS_OPS_1ARG:
            child_str = self.children[0].to_string()
            return f"{self.op}({child_str}, {self.window})"
        if self.op in TS_OPS_2ARG:
            c0 = self.children[0].to_string()
            c1 = self.children[1].to_string()
            return f"{self.op}({c0}, {c1}, {self.window})"

        # 截面算子
        if self.op in CS_OPS:
            child_str = self.children[0].to_string()
            return f"{self.op}({child_str})"

        # 一元算子
        if self.op in UNARY_OPS:
            child_str = self.children[0].to_string()
            return f"{self.op}({child_str})"

        # 二元算子
        if self.op in BINARY_OPS:
            left_str = self.children[0].to_string()
            right_str = self.children[1].to_string()
            return f"{self.op}({left_str}, {right_str})"

        return f"UNKNOWN({self.op})"

    def depth(self) -> int:
        """计算树的深度"""
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        """计算树的节点总数"""
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> "ExprNode":
        """深拷贝整棵子树"""
        return copy.deepcopy(self)

    def all_nodes(self) -> list["ExprNode"]:
        """返回所有节点的列表 (先序遍历)"""
        result = [self]
        for c in self.children:
            result.extend(c.all_nodes())
        return result

    def _fingerprint(self) -> str:
        """表达式的哈希指纹，用于快速去重"""
        s = self.to_string()
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        return f"ExprNode({self.to_string()})"


# ============================================================
# 表达式解析器: 字符串 -> ExprNode
# ============================================================

def _tokenize(expr: str) -> list[str]:
    """将表达式字符串分词"""
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in " \t\n":
            i += 1
            continue
        if c in "(),":
            tokens.append(c)
            i += 1
            continue
        # 数字 (含负号和小数)
        if c.isdigit() or (c == "-" and i + 1 < len(expr) and expr[i + 1].isdigit()):
            j = i + 1
            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        # 标识符 (算子名 / 特征名)
        if c.isalpha() or c == "_":
            j = i + 1
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        # 运算符 *
        if c == "*":
            tokens.append("*")
            i += 1
            continue
        if c == "/":
            tokens.append("/")
            i += 1
            continue
        if c == "+":
            tokens.append("+")
            i += 1
            continue
        if c == "-":
            tokens.append("-")
            i += 1
            continue
        i += 1
    return tokens


def parse_expression(expr: str) -> ExprNode:
    """
    将因子表达式字符串解析为 ExprNode 树。

    支持的格式:
      - ts_corr(cs_rank(close), cs_rank(volume), 10)
      - ts_delta(close, 5) * -1
      - cs_rank(ts_mean(close, 10) / close - 1)
    """
    # 预处理: 把中缀 *, /, +, - 转为函数调用
    expr = expr.strip()
    tokens = _tokenize(expr)
    pos = [0]  # 可变引用

    def _peek() -> str | None:
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def _consume() -> str:
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def _parse_expr() -> ExprNode:
        """解析表达式，处理中缀 + - """
        node = _parse_term()
        while _peek() in ("+", "-"):
            op_tok = _consume()
            right = _parse_term()
            op_name = "add" if op_tok == "+" else "sub"
            node = ExprNode(op=op_name, children=[node, right])
        return node

    def _parse_term() -> ExprNode:
        """解析项，处理中缀 * / """
        node = _parse_unary()
        while _peek() in ("*", "/"):
            op_tok = _consume()
            right = _parse_unary()
            op_name = "mul" if op_tok == "*" else "safe_div"
            node = ExprNode(op=op_name, children=[node, right])
        return node

    def _parse_unary() -> ExprNode:
        """解析一元前缀 (负号)"""
        if _peek() == "-":
            _consume()
            child = _parse_atom()
            return ExprNode(op="neg", children=[child])
        return _parse_atom()

    def _parse_atom() -> ExprNode:
        """解析原子: 函数调用 / 括号 / 标识符 / 数字"""
        tok = _peek()
        if tok is None:
            raise ValueError("表达式意外结束")

        # 括号
        if tok == "(":
            _consume()  # (
            node = _parse_expr()
            if _peek() == ")":
                _consume()  # )
            return node

        # 数字常量
        if tok[0].isdigit() or (tok[0] == "-" and len(tok) > 1):
            _consume()
            val = float(tok)
            return ExprNode(op="const", value=val)

        # 标识符
        if tok[0].isalpha() or tok[0] == "_":
            name = _consume()

            # 如果下一个是 '('，说明是函数调用
            if _peek() == "(":
                _consume()  # (
                args: list[ExprNode | int | float] = []
                while _peek() != ")":
                    if _peek() == ",":
                        _consume()
                        continue
                    # 尝试解析为数字 (窗口期参数)
                    next_tok = _peek()
                    if next_tok and (next_tok[0].isdigit() or
                                     (next_tok[0] == "-" and len(next_tok) > 1)):
                        num_tok = _consume()
                        val = float(num_tok)
                        if val == int(val):
                            val = int(val)
                        args.append(val)
                    else:
                        args.append(_parse_expr())
                if _peek() == ")":
                    _consume()  # )

                return _build_func_node(name, args)
            else:
                # 纯标识符 -> 特征叶节点
                if name in FEATURE_NAMES:
                    return ExprNode(op="feature", value=name)
                # 可能是 Python 级别的常量名，按特征处理
                return ExprNode(op="feature", value=name)

        raise ValueError(f"无法解析的 token: {tok}")

    def _build_func_node(name: str, args) -> ExprNode:
        """根据函数名和参数列表构建 ExprNode"""
        if name in TS_OPS_1ARG:
            # ts_xxx(child, window)
            child_args = [a for a in args if isinstance(a, ExprNode)]
            num_args = [a for a in args if isinstance(a, (int, float))]
            child = child_args[0] if child_args else ExprNode(op="feature", value="close")
            window = int(num_args[0]) if num_args else 10
            return ExprNode(op=name, children=[child], window=window)

        if name in TS_OPS_2ARG:
            # ts_corr(child1, child2, window)
            child_args = [a for a in args if isinstance(a, ExprNode)]
            num_args = [a for a in args if isinstance(a, (int, float))]
            c0 = child_args[0] if len(child_args) > 0 else ExprNode(op="feature", value="close")
            c1 = child_args[1] if len(child_args) > 1 else ExprNode(op="feature", value="volume")
            window = int(num_args[0]) if num_args else 10
            return ExprNode(op=name, children=[c0, c1], window=window)

        if name in CS_OPS:
            child_args = [a for a in args if isinstance(a, ExprNode)]
            child = child_args[0] if child_args else ExprNode(op="feature", value="close")
            return ExprNode(op=name, children=[child])

        if name in UNARY_OPS:
            child_args = [a for a in args if isinstance(a, ExprNode)]
            child = child_args[0] if child_args else ExprNode(op="feature", value="close")
            return ExprNode(op=name, children=[child])

        if name in BINARY_OPS:
            child_args = [a for a in args if isinstance(a, ExprNode)]
            c0 = child_args[0] if len(child_args) > 0 else ExprNode(op="feature", value="close")
            c1 = child_args[1] if len(child_args) > 1 else ExprNode(op="const", value=1.0)
            return ExprNode(op=name, children=[c0, c1])

        # 未知函数名，尝试当做特征
        logger.warning(f"未知函数: {name}, 回退为特征节点")
        return ExprNode(op="feature", value=name)

    node = _parse_expr()
    return node


# ============================================================
# Warm-Start 模板 (来自 Alpha101 / 经典量价因子)
# ============================================================

WARM_START_TEMPLATES: list[str] = [
    # --- 量价相关 ---
    "ts_corr(cs_rank(close), cs_rank(volume), 10)",
    "ts_corr(open, volume, 10)",
    "ts_corr(high, volume, 10)",
    "ts_corr(low, volume, 5)",
    "ts_corr(vwap, volume, 20)",
    "ts_corr(close, volume, 5)",

    # --- 动量 / 反转 ---
    "ts_delta(close, 5) * -1",
    "ts_delta(close, 10) * -1",
    "ts_delta(close, 20)",
    "ts_rank(returns, 5)",
    "ts_rank(returns, 20)",
    "ts_rank(volume, 20)",
    "cs_rank(ts_std(returns, 20)) * -1",

    # --- 衰减加权 ---
    "ts_decay_linear(ts_delta(close, 5), 10)",
    "ts_decay_linear(ts_delta(volume, 5), 10)",
    "ts_decay_linear(returns, 20)",

    # --- 均值回复 ---
    "cs_rank(ts_mean(close, 10) / close - 1)",
    "cs_rank(ts_mean(close, 5) / close - 1)",
    "cs_rank(close / ts_mean(close, 20) - 1)",

    # --- 波动率 ---
    "ts_std(returns, 10)",
    "ts_std(returns, 20) * -1",
    "ts_std(close / ts_mean(close, 5) - 1, 10)",

    # --- 最高最低价 ---
    "ts_max(high, 10) / close - 1",
    "close / ts_min(low, 10) - 1",
    "ts_max(high, 20) / ts_min(low, 20) - 1",

    # --- VWAP 相关 ---
    "cs_rank(vwap / close - 1)",
    "ts_delta(vwap, 5) * -1",
    "ts_corr(vwap, ts_mean(volume, 10), 10)",

    # --- 成交量异动 ---
    "cs_zscore(volume / ts_mean(volume, 20))",
    "ts_rank(volume / ts_mean(volume, 10), 5)",
]


# ============================================================
# Polars 表达式编译器: ExprNode -> pl.Expr
# ============================================================

def compile_to_polars(node: ExprNode) -> pl.Expr:
    """
    将表达式树编译为 Polars 表达式。

    假设 DataFrame 包含以下列:
      - date: 日期列
      - symbol: 股票代码列
      - open, high, low, close, volume, vwap, returns: 价量数据列

    所有时序算子在 group_by("symbol") 的上下文中执行。
    所有截面算子在 group_by("date") 的上下文中执行。

    注意: 由于 Polars 表达式的上下文限制，截面/时序算子
    需要在 GPMiner._evaluate 中通过 with_columns + over() 分步计算。
    这里返回的 pl.Expr 支持 .over("symbol") / .over("date") 修饰。
    """
    # --- 叶节点 ---
    if node.op == "feature":
        return pl.col(str(node.value))

    if node.op == "const":
        return pl.lit(float(node.value))

    # --- 一元算子 ---
    if node.op == "abs":
        child = compile_to_polars(node.children[0])
        return child.abs()

    if node.op == "neg":
        child = compile_to_polars(node.children[0])
        return child * -1.0

    if node.op == "sign":
        child = compile_to_polars(node.children[0])
        return child.sign()

    if node.op == "log1p":
        child = compile_to_polars(node.children[0])
        # log1p(|x|) * sign(x)，避免对负值取对数
        return child.abs().log1p() * child.sign()

    if node.op == "sqrt_abs":
        child = compile_to_polars(node.children[0])
        return child.abs().sqrt() * child.sign()

    # --- 二元算子 ---
    if node.op == "add":
        left = compile_to_polars(node.children[0])
        right = compile_to_polars(node.children[1])
        return left + right

    if node.op == "sub":
        left = compile_to_polars(node.children[0])
        right = compile_to_polars(node.children[1])
        return left - right

    if node.op == "mul":
        left = compile_to_polars(node.children[0])
        right = compile_to_polars(node.children[1])
        return left * right

    if node.op == "safe_div":
        left = compile_to_polars(node.children[0])
        right = compile_to_polars(node.children[1])
        # 安全除法: 分母为 0 时返回 0
        return pl.when(right.abs() > 1e-10).then(left / right).otherwise(0.0)

    # --- 时序算子 (需配合 .over("symbol") 使用) ---
    w = node.window or 10

    if node.op == "ts_mean":
        child = compile_to_polars(node.children[0])
        return child.rolling_mean(window_size=w, min_samples=max(1, w // 2))

    if node.op == "ts_std":
        child = compile_to_polars(node.children[0])
        return child.rolling_std(window_size=w, min_samples=max(2, w // 2))

    if node.op == "ts_rank":
        # 滚动窗口内的百分位排名
        child = compile_to_polars(node.children[0])
        # Polars 无直接 rolling_rank; 用 rolling_map 近似
        # 简化实现: 使用 rolling_mean(rank_within_window)
        # 准确实现需要自定义函数，此处用 rolling_quantile 的逆思路
        # 退而用 rolling_min / rolling_max 归一化
        c_min = child.rolling_min(window_size=w, min_samples=max(1, w // 2))
        c_max = child.rolling_max(window_size=w, min_samples=max(1, w // 2))
        return pl.when((c_max - c_min).abs() > 1e-10).then(
            (child - c_min) / (c_max - c_min)
        ).otherwise(0.5)

    if node.op == "ts_delta":
        child = compile_to_polars(node.children[0])
        return child - child.shift(w)

    if node.op == "ts_max":
        child = compile_to_polars(node.children[0])
        return child.rolling_max(window_size=w, min_samples=max(1, w // 2))

    if node.op == "ts_min":
        child = compile_to_polars(node.children[0])
        return child.rolling_min(window_size=w, min_samples=max(1, w // 2))

    if node.op == "ts_decay_linear":
        # 线性衰减加权移动平均: 权重 [1, 2, ..., w] / sum
        child = compile_to_polars(node.children[0])
        # Polars 原生不直接支持 decay_linear
        # 近似实现: ewm 指数加权 (半衰期 = w/2)
        return child.ewm_mean(half_life=max(1, w // 2), min_samples=max(1, w // 2))

    if node.op == "ts_corr":
        # 滚动相关系数
        left = compile_to_polars(node.children[0])
        right = compile_to_polars(node.children[1])
        # Polars 不直接支持 rolling_corr 两列
        # 用 Pearson 公式: corr = (E[XY] - E[X]E[Y]) / (std[X] * std[Y])
        # 此处返回近似 — 在 _evaluate 中用特殊路径处理
        # 占位: 返回两者乘积的滚动均值归一化
        xy = (left * right).rolling_mean(window_size=w, min_samples=max(2, w // 2))
        ex = left.rolling_mean(window_size=w, min_samples=max(2, w // 2))
        ey = right.rolling_mean(window_size=w, min_samples=max(2, w // 2))
        sx = left.rolling_std(window_size=w, min_samples=max(2, w // 2))
        sy = right.rolling_std(window_size=w, min_samples=max(2, w // 2))
        cov = xy - ex * ey
        denom = sx * sy
        return pl.when(denom.abs() > 1e-10).then(cov / denom).otherwise(0.0)

    # --- 截面算子 (需配合 .over("date") 使用) ---
    if node.op == "cs_rank":
        child = compile_to_polars(node.children[0])
        # 截面排名百分位 — 需要 .rank().over("date") 外部包装
        return child.rank(method="average")

    if node.op == "cs_zscore":
        child = compile_to_polars(node.children[0])
        # 截面 z-score — 需要 .over("date") 外部包装
        return child  # GPMiner._evaluate 中做 (x - mean) / std

    raise ValueError(f"未知算子: {node.op}")


def _has_ts_ops(node: ExprNode) -> bool:
    """检查表达式树中是否包含时序算子"""
    if node.op in TS_OPS_1ARG or node.op in TS_OPS_2ARG:
        return True
    return any(_has_ts_ops(c) for c in node.children)


def _has_cs_ops(node: ExprNode) -> bool:
    """检查表达式树中是否包含截面算子"""
    if node.op in CS_OPS:
        return True
    return any(_has_cs_ops(c) for c in node.children)


def _collect_cs_ops(node: ExprNode) -> bool:
    """检查树的 *根* 是否为截面算子"""
    return node.op in CS_OPS


# ============================================================
# 辅助: 安全的 Polars 因子计算
# ============================================================

def _safe_eval_factor(
    df: pl.DataFrame,
    tree: ExprNode,
    factor_col_name: str = "__factor__",
) -> pl.DataFrame | None:
    """
    安全地在 DataFrame 上计算表达式树的因子值。

    采用两阶段策略:
      1. 时序算子: 在 group_by("symbol").sort("date") 上下文中计算
      2. 截面算子: 在 group_by("date") 上下文中计算

    为简化实现, 将整棵树编译为单个 pl.Expr,
    然后根据树中包含的算子类型决定 .over() 方式。

    对于混合了时序和截面算子的复杂表达式, 采用分层递归计算。

    Returns:
        带有 factor_col_name 列的 DataFrame, 或 None (计算失败)
    """
    try:
        result_df = _recursive_eval(df, tree, factor_col_name)
        return result_df
    except Exception as e:
        logger.debug(f"因子计算失败: {tree.to_string()}: {e}")
        return None


def _recursive_eval(
    df: pl.DataFrame,
    node: ExprNode,
    col_name: str,
) -> pl.DataFrame:
    """
    递归计算表达式树。

    对于截面算子, 先计算子树, 存为临时列, 再在截面维度做 rank/zscore。
    对于时序算子, 先计算子树, 存为临时列, 再在时序维度做滚动计算。
    """
    # --- 叶节点 ---
    if node.op == "feature":
        col = str(node.value)
        if col not in df.columns:
            raise ValueError(f"缺失列: {col}")
        return df.with_columns(pl.col(col).alias(col_name))

    if node.op == "const":
        return df.with_columns(pl.lit(float(node.value)).alias(col_name))

    # --- 截面算子: 先递归计算子树, 再做截面操作 ---
    if node.op in CS_OPS:
        tmp_name = f"_tmp_{id(node)}"
        df = _recursive_eval(df, node.children[0], tmp_name)

        if node.op == "cs_rank":
            df = df.with_columns(
                pl.col(tmp_name)
                .rank(method="average")
                .over("date")
                .alias(col_name)
            )
        elif node.op == "cs_zscore":
            df = df.with_columns(
                (
                    (pl.col(tmp_name) - pl.col(tmp_name).mean().over("date"))
                    / (pl.col(tmp_name).std().over("date") + 1e-10)
                ).alias(col_name)
            )

        # 清理临时列
        df = df.drop(tmp_name)
        return df

    # --- 时序算子: 先递归计算子树, 再做时序滚动 ---
    if node.op in TS_OPS_1ARG:
        tmp_name = f"_tmp_{id(node)}"
        df = _recursive_eval(df, node.children[0], tmp_name)
        w = node.window or 10
        min_p = max(1, w // 2)

        if node.op == "ts_mean":
            expr = pl.col(tmp_name).rolling_mean(window_size=w, min_samples=min_p)
        elif node.op == "ts_std":
            expr = pl.col(tmp_name).rolling_std(window_size=w, min_samples=max(2, min_p))
        elif node.op == "ts_rank":
            c_min = pl.col(tmp_name).rolling_min(window_size=w, min_samples=min_p)
            c_max = pl.col(tmp_name).rolling_max(window_size=w, min_samples=min_p)
            expr = pl.when((c_max - c_min).abs() > 1e-10).then(
                (pl.col(tmp_name) - c_min) / (c_max - c_min)
            ).otherwise(0.5)
        elif node.op == "ts_delta":
            expr = pl.col(tmp_name) - pl.col(tmp_name).shift(w)
        elif node.op == "ts_max":
            expr = pl.col(tmp_name).rolling_max(window_size=w, min_samples=min_p)
        elif node.op == "ts_min":
            expr = pl.col(tmp_name).rolling_min(window_size=w, min_samples=min_p)
        elif node.op == "ts_decay_linear":
            expr = pl.col(tmp_name).ewm_mean(
                half_life=max(1, w // 2), min_samples=min_p
            )
        else:
            raise ValueError(f"未知时序算子: {node.op}")

        df = df.with_columns(
            expr.over("symbol").alias(col_name)
        )
        df = df.drop(tmp_name)
        return df

    if node.op in TS_OPS_2ARG:
        # ts_corr(child0, child1, window)
        tmp0 = f"_tmp0_{id(node)}"
        tmp1 = f"_tmp1_{id(node)}"
        df = _recursive_eval(df, node.children[0], tmp0)
        df = _recursive_eval(df, node.children[1], tmp1)

        w = node.window or 10
        min_p = max(2, w // 2)

        # Pearson 相关系数: cov(X,Y) / (std(X) * std(Y))
        xy = (pl.col(tmp0) * pl.col(tmp1)).rolling_mean(window_size=w, min_samples=min_p)
        ex = pl.col(tmp0).rolling_mean(window_size=w, min_samples=min_p)
        ey = pl.col(tmp1).rolling_mean(window_size=w, min_samples=min_p)
        sx = pl.col(tmp0).rolling_std(window_size=w, min_samples=min_p)
        sy = pl.col(tmp1).rolling_std(window_size=w, min_samples=min_p)
        cov_expr = xy - ex * ey
        denom_expr = sx * sy

        corr_expr = pl.when(denom_expr.abs() > 1e-10).then(
            (cov_expr / denom_expr).clip(-1.0, 1.0)
        ).otherwise(0.0)

        df = df.with_columns(
            corr_expr.over("symbol").alias(col_name)
        )
        df = df.drop([tmp0, tmp1])
        return df

    # --- 一元算子 ---
    if node.op in UNARY_OPS:
        tmp_name = f"_tmp_{id(node)}"
        df = _recursive_eval(df, node.children[0], tmp_name)

        if node.op == "abs":
            expr = pl.col(tmp_name).abs()
        elif node.op == "neg":
            expr = pl.col(tmp_name) * -1.0
        elif node.op == "sign":
            expr = pl.col(tmp_name).sign()
        elif node.op == "log1p":
            expr = pl.col(tmp_name).abs().log1p() * pl.col(tmp_name).sign()
        elif node.op == "sqrt_abs":
            expr = pl.col(tmp_name).abs().sqrt() * pl.col(tmp_name).sign()
        else:
            raise ValueError(f"未知一元算子: {node.op}")

        df = df.with_columns(expr.alias(col_name))
        df = df.drop(tmp_name)
        return df

    # --- 二元算子 ---
    if node.op in BINARY_OPS:
        tmp0 = f"_tmp0_{id(node)}"
        tmp1 = f"_tmp1_{id(node)}"
        df = _recursive_eval(df, node.children[0], tmp0)
        df = _recursive_eval(df, node.children[1], tmp1)

        if node.op == "add":
            expr = pl.col(tmp0) + pl.col(tmp1)
        elif node.op == "sub":
            expr = pl.col(tmp0) - pl.col(tmp1)
        elif node.op == "mul":
            expr = pl.col(tmp0) * pl.col(tmp1)
        elif node.op == "safe_div":
            expr = pl.when(pl.col(tmp1).abs() > 1e-10).then(
                pl.col(tmp0) / pl.col(tmp1)
            ).otherwise(0.0)
        else:
            raise ValueError(f"未知二元算子: {node.op}")

        df = df.with_columns(expr.alias(col_name))
        df = df.drop([tmp0, tmp1])
        return df

    raise ValueError(f"无法计算的节点: {node.op}")


# ============================================================
# GPMiner: 遗传编程因子挖掘器
# ============================================================

class GPMiner:
    """
    基于遗传编程 (Genetic Programming) 的因子挖掘器。

    参考:
      - WS-GP (2024): Warm-Start 策略, 用已知因子模板初始化种群
      - AlphaGen (KDD 2023): 表达式树 + GP 搜索 Alpha 因子

    Parameters:
        data: 原始 OHLCV 面板数据 (长格式 Polars DataFrame)
              必须包含列: date, symbol, open, high, low, close, volume
              可选列: vwap, returns
        fwd_returns: 前瞻收益 DataFrame (长格式, 列: date, symbol, fwd_ret)
        population_size: 种群大小
        n_generations: 最大进化代数
        max_depth: 表达式树最大深度
        tournament_k: 锦标赛选择的候选个体数
        crossover_prob: 交叉概率
        mutation_prob: 变异概率
        ic_threshold: 有效因子的最低 IC 阈值
        corr_threshold: 因子去重的相关性阈值
        seed: 随机种子
    """

    def __init__(
        self,
        data: pl.DataFrame,
        fwd_returns: pl.DataFrame,
        population_size: int = 500,
        n_generations: int = 100,
        max_depth: int = 5,
        tournament_k: int = 5,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
        ic_threshold: float = 0.02,
        corr_threshold: float = 0.7,
        seed: int = 42,
    ):
        self.data = data
        self.fwd_returns = fwd_returns
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.tournament_k = tournament_k
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.ic_threshold = ic_threshold
        self.corr_threshold = corr_threshold
        self.seed = seed

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # 预处理数据: 确保有 vwap 和 returns 列
        self.data = self._preprocess_data(self.data)

        # 用于粗筛的 10% 随机采样日期
        all_dates = self.data.get_column("date").unique().sort()
        n_sample = max(10, len(all_dates) // 10)
        sample_idx = self._np_rng.choice(len(all_dates), size=n_sample, replace=False)
        self._sample_dates = all_dates.gather(sample_idx.tolist())
        self._sample_data = self.data.filter(
            pl.col("date").is_in(self._sample_dates)
        )
        self._sample_fwd = self.fwd_returns.filter(
            pl.col("date").is_in(self._sample_dates)
        )

        # 因子库: 存储已发现的有效因子及其值
        self._factor_pool: list[dict] = []
        # 因子值缓存: fingerprint -> (date, symbol) -> value 的 DataFrame
        self._factor_values_cache: dict[str, pl.DataFrame] = {}

        # 表达式指纹去重集合
        self._seen_fingerprints: set[str] = set()

        logger.info(
            f"GPMiner 初始化完成: "
            f"数据 {self.data.shape}, "
            f"种群={population_size}, "
            f"代数={n_generations}, "
            f"最大深度={max_depth}"
        )

    # ------ 数据预处理 ------

    def _preprocess_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """预处理数据: 补齐 vwap / returns 列"""
        if "vwap" not in df.columns:
            # VWAP 近似 = (high + low + close) / 3
            df = df.with_columns(
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0)
                .alias("vwap")
            )
        if "returns" not in df.columns:
            # 日收益率 = close / close.shift(1) - 1 (按 symbol 分组)
            df = df.sort(["symbol", "date"]).with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0)
                .alias("returns")
            )
        return df.sort(["symbol", "date"])

    # ------ 随机树生成 ------

    def _random_leaf(self) -> ExprNode:
        """随机生成叶节点"""
        if self._rng.random() < 0.7:
            # 70% 概率选择特征
            feat = self._rng.choice(FEATURE_NAMES)
            return ExprNode(op="feature", value=feat)
        else:
            # 30% 概率选择常数
            const = self._rng.choice(CONSTANTS)
            return ExprNode(op="const", value=const)

    def _random_tree(self, max_depth: int, method: str = "grow") -> ExprNode:
        """
        随机生成表达式树。

        method:
          - "grow": 每个节点随机选择是终端还是非终端 (深度 < max_depth 时)
          - "full": 所有分支都长到 max_depth
        """
        if max_depth <= 0:
            return self._random_leaf()

        if method == "grow" and self._rng.random() < 0.3:
            return self._random_leaf()

        # 随机选择算子
        op = self._rng.choice(ALL_OPS)
        arity = OP_ARITY[op]

        children = [self._random_tree(max_depth - 1, method) for _ in range(arity)]

        window = None
        if op in TS_OPS_1ARG or op in TS_OPS_2ARG:
            window = self._rng.choice(TS_WINDOWS)

        return ExprNode(op=op, children=children, window=window)

    def _ramped_half_and_half(self, n: int) -> list[ExprNode]:
        """
        Ramped half-and-half 初始化。
        一半用 grow 方法, 一半用 full 方法, 深度从 2 到 max_depth 均匀分布。
        """
        trees = []
        for i in range(n):
            depth = 2 + i % (self.max_depth - 1)
            depth = min(depth, self.max_depth)
            method = "full" if i % 2 == 0 else "grow"
            trees.append(self._random_tree(depth, method))
        return trees

    # ------ Warm-Start 初始化 ------

    def _warm_start_population(
        self,
        templates: list[str] | None = None,
    ) -> list[ExprNode]:
        """
        Warm-Start 种群初始化 (WS-GP)。

        策略:
          1. 解析预定义模板为表达式树
          2. 对每个模板生成多个变异体 (修改窗口期 / 替换叶节点 / 子树变异)
          3. 剩余位置用 ramped half-and-half 随机生成

        这样既保留了已知有效因子的结构知识, 又保持了种群多样性。
        """
        if templates is None:
            templates = WARM_START_TEMPLATES

        population: list[ExprNode] = []

        # 1. 解析模板并添加原始版本
        parsed_templates: list[ExprNode] = []
        for tmpl_str in templates:
            try:
                tree = parse_expression(tmpl_str)
                parsed_templates.append(tree)
                population.append(tree.copy())
            except Exception as e:
                logger.warning(f"模板解析失败: {tmpl_str}: {e}")

        # 2. 对每个模板生成 ~5 个变异体
        variants_per_template = max(
            1,
            (self.population_size // 2 - len(parsed_templates)) // max(1, len(parsed_templates))
        )
        for template in parsed_templates:
            for _ in range(variants_per_template):
                variant = template.copy()
                # 随机应用 1-3 次变异
                n_mutations = self._rng.randint(1, 3)
                for _ in range(n_mutations):
                    variant = self._mutate(variant)
                # 控制深度
                if variant.depth() <= self.max_depth:
                    population.append(variant)

        # 3. 剩余位置用随机树填充
        n_remaining = self.population_size - len(population)
        if n_remaining > 0:
            population.extend(self._ramped_half_and_half(n_remaining))

        # 截断到种群大小
        population = population[: self.population_size]

        logger.info(
            f"Warm-Start 种群初始化: "
            f"{len(parsed_templates)} 个模板, "
            f"{len(population)} 个个体"
        )
        return population

    # ------ 适应度评估 ------

    def _evaluate(self, tree: ExprNode) -> float:
        """
        计算因子的适应度 = Rank IC (Spearman 相关系数)。

        采用两阶段策略:
          1. 先在 10% 随机采样数据上粗筛 (快速排除差因子)
          2. 通过粗筛后, 在全量数据上精确计算

        Returns:
            平均 Rank IC 绝对值 (越大越好); 计算失败返回 -1.0
        """
        # 表达式去重: 跳过已见过的完全相同的表达式
        fp = tree._fingerprint()

        # 阶段 1: 粗筛 (10% 样本)
        coarse_ic = self._compute_ic(tree, self._sample_data, self._sample_fwd)
        if coarse_ic is None or abs(coarse_ic) < self.ic_threshold * 0.5:
            return abs(coarse_ic) if coarse_ic is not None else -1.0

        # 阶段 2: 全量评估
        full_ic = self._compute_ic(tree, self.data, self.fwd_returns)
        if full_ic is None:
            return -1.0

        return abs(full_ic)

    def _compute_ic(
        self,
        tree: ExprNode,
        data: pl.DataFrame,
        fwd_returns: pl.DataFrame,
    ) -> float | None:
        """
        在给定数据上计算因子的 Rank IC。

        Returns:
            平均 Rank IC (带符号), 或 None (计算失败)
        """
        # 计算因子值
        factor_df = _safe_eval_factor(data, tree, "__factor__")
        if factor_df is None:
            return None

        # 合并因子值和前瞻收益
        try:
            merged = factor_df.select(["date", "symbol", "__factor__"]).join(
                fwd_returns.select(["date", "symbol", "fwd_ret"]),
                on=["date", "symbol"],
                how="inner",
            )
        except Exception:
            return None

        # 去掉 NaN / Inf
        merged = merged.filter(
            pl.col("__factor__").is_finite() & pl.col("fwd_ret").is_finite()
        )

        if merged.height < 100:
            return None

        # 按日期分组计算 Rank IC (Spearman)
        try:
            ic_per_date = (
                merged
                .with_columns([
                    pl.col("__factor__").rank(method="average").over("date").alias("f_rank"),
                    pl.col("fwd_ret").rank(method="average").over("date").alias("r_rank"),
                ])
                .group_by("date")
                .agg([
                    pl.count().alias("n"),
                    pl.pearson_corr("f_rank", "r_rank").alias("ic"),
                ])
                .filter(pl.col("n") >= 20)
                .filter(pl.col("ic").is_finite())
            )
        except Exception:
            return None

        if ic_per_date.height < 5:
            return None

        ic_values = ic_per_date.get_column("ic")
        mean_ic = ic_values.mean()

        if mean_ic is None or np.isnan(mean_ic):
            return None

        return float(mean_ic)

    def _compute_ic_series(
        self,
        tree: ExprNode,
    ) -> tuple[float, float] | None:
        """
        在全量数据上计算因子的 IC 和 ICIR。

        Returns:
            (mean_ic, icir) 或 None
        """
        factor_df = _safe_eval_factor(self.data, tree, "__factor__")
        if factor_df is None:
            return None

        try:
            merged = factor_df.select(["date", "symbol", "__factor__"]).join(
                self.fwd_returns.select(["date", "symbol", "fwd_ret"]),
                on=["date", "symbol"],
                how="inner",
            )
        except Exception:
            return None

        merged = merged.filter(
            pl.col("__factor__").is_finite() & pl.col("fwd_ret").is_finite()
        )

        if merged.height < 100:
            return None

        try:
            ic_per_date = (
                merged
                .with_columns([
                    pl.col("__factor__").rank(method="average").over("date").alias("f_rank"),
                    pl.col("fwd_ret").rank(method="average").over("date").alias("r_rank"),
                ])
                .group_by("date")
                .agg([
                    pl.count().alias("n"),
                    pl.pearson_corr("f_rank", "r_rank").alias("ic"),
                ])
                .filter(pl.col("n") >= 20)
                .filter(pl.col("ic").is_finite())
            )
        except Exception:
            return None

        if ic_per_date.height < 5:
            return None

        ic_values = ic_per_date.get_column("ic")
        mean_ic = float(ic_values.mean())
        std_ic = float(ic_values.std())

        if np.isnan(mean_ic) or np.isnan(std_ic) or std_ic < 1e-10:
            return None

        icir = mean_ic / std_ic
        return (mean_ic, icir)

    # ------ 因子去重 ------

    def _is_redundant(self, tree: ExprNode) -> bool:
        """
        检查新因子与因子库中已有因子是否冗余。

        计算新因子值与已有因子值的截面相关性, 如果最大相关性 > corr_threshold 则冗余。
        """
        if not self._factor_pool:
            return False

        # 计算新因子值
        new_factor_df = _safe_eval_factor(self.data, tree, "__new_factor__")
        if new_factor_df is None:
            return True  # 计算失败视为冗余

        new_vals = new_factor_df.select(["date", "symbol", "__new_factor__"])

        for existing in self._factor_pool:
            existing_tree = existing["tree"]
            fp = existing_tree._fingerprint()

            # 使用缓存的因子值
            if fp in self._factor_values_cache:
                existing_vals = self._factor_values_cache[fp]
            else:
                existing_df = _safe_eval_factor(
                    self.data, existing_tree, "__existing_factor__"
                )
                if existing_df is None:
                    continue
                existing_vals = existing_df.select(
                    ["date", "symbol", "__existing_factor__"]
                )
                self._factor_values_cache[fp] = existing_vals

            # 计算截面相关性
            try:
                merged = new_vals.join(existing_vals, on=["date", "symbol"], how="inner")
                merged = merged.filter(
                    pl.col("__new_factor__").is_finite()
                    & pl.col("__existing_factor__").is_finite()
                )
                if merged.height < 100:
                    continue

                corr = merged.select(
                    pl.pearson_corr("__new_factor__", "__existing_factor__").alias("corr")
                ).item()

                if corr is not None and abs(corr) > self.corr_threshold:
                    return True
            except Exception:
                continue

        return False

    # ------ 选择算子 ------

    def _tournament_select(
        self,
        population: list[ExprNode],
        fitnesses: list[float],
    ) -> ExprNode:
        """
        锦标赛选择: 随机抽 tournament_k 个个体, 取适应度最高者。
        """
        indices = self._rng.sample(range(len(population)), k=self.tournament_k)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    # ------ 交叉算子 ------

    def _crossover(
        self, parent1: ExprNode, parent2: ExprNode
    ) -> tuple[ExprNode, ExprNode]:
        """
        子树交叉: 在两棵树中各随机选一个子树, 互换。

        如果交叉后树深度超过 max_depth, 返回原始父代的拷贝。
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 收集所有非叶节点 (至少有子节点的节点)
        nodes1 = [n for n in child1.all_nodes() if n.children]
        nodes2 = [n for n in child2.all_nodes() if n.children]

        if not nodes1 or not nodes2:
            return child1, child2

        # 随机选择交叉点
        point1 = self._rng.choice(nodes1)
        point2 = self._rng.choice(nodes2)

        # 随机选择要交换的子节点索引
        idx1 = self._rng.randrange(len(point1.children))
        idx2 = self._rng.randrange(len(point2.children))

        # 交换子树
        point1.children[idx1], point2.children[idx2] = (
            point2.children[idx2],
            point1.children[idx1],
        )

        # 检查深度约束
        if child1.depth() > self.max_depth:
            child1 = parent1.copy()
        if child2.depth() > self.max_depth:
            child2 = parent2.copy()

        return child1, child2

    # ------ 变异算子 ------

    def _mutate(self, tree: ExprNode) -> ExprNode:
        """
        变异操作。随机选择一种变异策略:
          1. 点变异 (40%): 替换单个节点的算子或值
          2. 子树变异 (40%): 用随机子树替换一个子树
          3. 窗口期变异 (20%): 修改时序算子的窗口参数
        """
        mutant = tree.copy()
        r = self._rng.random()

        if r < 0.4:
            mutant = self._point_mutate(mutant)
        elif r < 0.8:
            mutant = self._subtree_mutate(mutant)
        else:
            mutant = self._window_mutate(mutant)

        # 深度检查
        if mutant.depth() > self.max_depth:
            return tree.copy()  # 退回原树

        return mutant

    def _point_mutate(self, tree: ExprNode) -> ExprNode:
        """
        点变异: 替换单个节点。
          - 叶节点: 换成另一个特征或常数
          - 内部节点: 换成同元数的另一个算子
        """
        nodes = tree.all_nodes()
        node = self._rng.choice(nodes)

        if node.op == "feature":
            node.value = self._rng.choice(FEATURE_NAMES)
        elif node.op == "const":
            node.value = self._rng.choice(CONSTANTS)
        elif node.op in BINARY_OPS:
            node.op = self._rng.choice(BINARY_OPS)
        elif node.op in UNARY_OPS:
            node.op = self._rng.choice(UNARY_OPS)
        elif node.op in TS_OPS_1ARG:
            node.op = self._rng.choice(TS_OPS_1ARG)
            node.window = self._rng.choice(TS_WINDOWS)
        elif node.op in TS_OPS_2ARG:
            node.op = self._rng.choice(TS_OPS_2ARG)
            node.window = self._rng.choice(TS_WINDOWS)
        elif node.op in CS_OPS:
            node.op = self._rng.choice(CS_OPS)

        return tree

    def _subtree_mutate(self, tree: ExprNode) -> ExprNode:
        """子树变异: 随机选择一个内部节点, 用新的随机子树替换其一个子节点"""
        internal = [n for n in tree.all_nodes() if n.children]
        if not internal:
            return self._random_tree(self.max_depth, "grow")

        node = self._rng.choice(internal)
        idx = self._rng.randrange(len(node.children))
        remaining_depth = self.max_depth - self._node_depth_in_tree(tree, node) - 1
        remaining_depth = max(1, remaining_depth)
        node.children[idx] = self._random_tree(remaining_depth, "grow")

        return tree

    def _window_mutate(self, tree: ExprNode) -> ExprNode:
        """窗口期变异: 修改时序算子的窗口参数"""
        ts_nodes = [
            n for n in tree.all_nodes()
            if n.op in TS_OPS_1ARG or n.op in TS_OPS_2ARG
        ]
        if not ts_nodes:
            return self._point_mutate(tree)  # 无时序节点则退化为点变异

        node = self._rng.choice(ts_nodes)
        node.window = self._rng.choice(TS_WINDOWS)
        return tree

    def _node_depth_in_tree(self, root: ExprNode, target: ExprNode) -> int:
        """计算 target 节点在 root 树中的深度"""
        if root is target:
            return 0
        for c in root.children:
            d = self._node_depth_in_tree(c, target)
            if d >= 0:
                return d + 1
        return -1  # 未找到

    # ------ 精英保留 ------

    def _elitism(
        self,
        population: list[ExprNode],
        fitnesses: list[float],
        n_elite: int,
    ) -> list[ExprNode]:
        """保留适应度最高的 n_elite 个个体"""
        indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        return [population[i].copy() for i in indices[:n_elite]]

    # ------ 主搜索循环 ------

    def run(self) -> list[dict]:
        """
        运行完整的 GP 搜索流程。

        流程:
          1. Warm-Start 初始化种群
          2. 迭代进化 (评估 -> 选择 -> 交叉 -> 变异)
          3. 每代检查是否有新的有效因子 (IC > 阈值 且 与已有因子不冗余)
          4. 早停: 连续 20 代无新因子时终止

        Returns:
            有效因子列表: [{"expression": str, "ic": float, "icir": float, "tree": ExprNode}, ...]
        """
        logger.info("=" * 60)
        logger.info("开始 GP 因子挖掘")
        logger.info("=" * 60)

        # 1. 初始化种群
        population = self._warm_start_population()

        # 早停计数器
        stale_generations = 0
        max_stale = 20

        best_fitness_history: list[float] = []

        for gen in range(self.n_generations):
            # 2. 评估适应度
            fitnesses = []
            for tree in population:
                fitness = self._evaluate(tree)
                fitnesses.append(fitness)

            # 统计本代信息
            valid_fitnesses = [f for f in fitnesses if f > 0]
            gen_best = max(fitnesses) if fitnesses else 0
            gen_mean = np.mean(valid_fitnesses) if valid_fitnesses else 0
            best_fitness_history.append(gen_best)

            logger.info(
                f"第 {gen + 1:3d}/{self.n_generations} 代 | "
                f"最优 IC={gen_best:.4f} | "
                f"均值 IC={gen_mean:.4f} | "
                f"有效个体={len(valid_fitnesses)}/{len(population)} | "
                f"因子库={len(self._factor_pool)}"
            )

            # 3. 检查是否有新的有效因子
            new_factor_found = False
            top_k_idx = sorted(
                range(len(fitnesses)),
                key=lambda i: fitnesses[i],
                reverse=True,
            )[:20]  # 只检查本代 top20

            for idx in top_k_idx:
                if fitnesses[idx] < self.ic_threshold:
                    continue

                tree = population[idx]
                fp = tree._fingerprint()

                # 跳过已见过的表达式
                if fp in self._seen_fingerprints:
                    continue
                self._seen_fingerprints.add(fp)

                # 去冗余检查
                if self._is_redundant(tree):
                    continue

                # 计算完整 IC / ICIR
                ic_result = self._compute_ic_series(tree)
                if ic_result is None:
                    continue

                mean_ic, icir = ic_result
                if abs(mean_ic) < self.ic_threshold:
                    continue

                # 发现新因子!
                expr_str = tree.to_string()
                factor_info = {
                    "expression": expr_str,
                    "ic": mean_ic,
                    "icir": icir,
                    "tree": tree.copy(),
                }
                self._factor_pool.append(factor_info)

                # 缓存因子值
                factor_df = _safe_eval_factor(self.data, tree, "__existing_factor__")
                if factor_df is not None:
                    self._factor_values_cache[fp] = factor_df.select(
                        ["date", "symbol", "__existing_factor__"]
                    )

                new_factor_found = True
                logger.info(
                    f"  >>> 新因子 #{len(self._factor_pool)}: "
                    f"IC={mean_ic:.4f}, ICIR={icir:.4f} | "
                    f"{expr_str}"
                )

            # 4. 早停判断
            if new_factor_found:
                stale_generations = 0
            else:
                stale_generations += 1

            if stale_generations >= max_stale:
                logger.info(
                    f"连续 {max_stale} 代无新因子, 提前终止 (共 {gen + 1} 代)"
                )
                break

            # 5. 生成下一代
            n_elite = max(2, self.population_size // 20)
            new_population = self._elitism(population, fitnesses, n_elite)

            while len(new_population) < self.population_size:
                if self._rng.random() < self.crossover_prob:
                    # 交叉
                    p1 = self._tournament_select(population, fitnesses)
                    p2 = self._tournament_select(population, fitnesses)
                    c1, c2 = self._crossover(p1, p2)
                    new_population.append(c1)
                    if len(new_population) < self.population_size:
                        new_population.append(c2)
                else:
                    # 变异
                    parent = self._tournament_select(population, fitnesses)
                    mutant = self._mutate(parent)
                    new_population.append(mutant)

            population = new_population[: self.population_size]

        # 按 |IC| 降序排列结果
        self._factor_pool.sort(key=lambda x: abs(x["ic"]), reverse=True)

        logger.info("=" * 60)
        logger.info(
            f"GP 搜索完成: 共发现 {len(self._factor_pool)} 个有效因子"
        )
        for i, f in enumerate(self._factor_pool):
            logger.info(
                f"  [{i + 1}] IC={f['ic']:.4f} ICIR={f['icir']:.4f} | "
                f"{f['expression']}"
            )
        logger.info("=" * 60)

        return self._factor_pool

    # ------ 因子计算接口 ------

    def to_polars_expr(
        self,
        tree: ExprNode,
        df: pl.DataFrame,
        col_name: str = "gp_factor",
    ) -> pl.DataFrame:
        """
        将表达式树应用到任意 DataFrame 上, 返回带有因子值列的 DataFrame。

        Parameters:
            tree: 表达式树
            df: 输入数据 (长格式, 含 date, symbol, OHLCV 列)
            col_name: 输出因子列名

        Returns:
            带有 col_name 列的 DataFrame
        """
        df = self._preprocess_data(df)
        result = _safe_eval_factor(df, tree, col_name)
        if result is None:
            raise ValueError(f"无法计算因子: {tree.to_string()}")
        return result

    def get_factor_pool_df(self) -> pl.DataFrame:
        """
        将因子库导出为 Polars DataFrame, 方便查看和分析。

        Returns:
            列: expression, ic, icir, depth, size
        """
        if not self._factor_pool:
            return pl.DataFrame(
                schema={"expression": pl.Utf8, "ic": pl.Float64,
                        "icir": pl.Float64, "depth": pl.Int32, "size": pl.Int32}
            )

        rows = []
        for f in self._factor_pool:
            rows.append({
                "expression": f["expression"],
                "ic": f["ic"],
                "icir": f["icir"],
                "depth": f["tree"].depth(),
                "size": f["tree"].size(),
            })
        return pl.DataFrame(rows)


# ============================================================
# 便捷入口
# ============================================================

def mine_factors(
    data: pl.DataFrame,
    fwd_returns: pl.DataFrame,
    population_size: int = 500,
    n_generations: int = 100,
    max_depth: int = 5,
    ic_threshold: float = 0.02,
    corr_threshold: float = 0.7,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """
    一键式因子挖掘入口函数。

    Parameters:
        data: 原始面板数据 (长格式 Polars DataFrame)
              必须包含列: date, symbol, open, high, low, close, volume
        fwd_returns: 前瞻收益 (长格式, 列: date, symbol, fwd_ret)
        population_size: GP 种群大小
        n_generations: 最大进化代数
        max_depth: 表达式树最大深度
        ic_threshold: 有效因子最低 IC 阈值
        corr_threshold: 去冗余的相关性阈值
        seed: 随机种子
        verbose: 是否输出详细日志

    Returns:
        有效因子列表: [{"expression": str, "ic": float, "icir": float, "tree": ExprNode}, ...]

    使用示例::

        import polars as pl
        from factor_zoo.gp_miner import mine_factors

        # 准备数据 (长格式)
        data = pl.read_parquet("ohlcv.parquet")
        fwd_ret = data.sort(["symbol", "date"]).with_columns(
            (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1)
            .alias("fwd_ret")
        ).select(["date", "symbol", "fwd_ret"])

        # 挖掘因子
        factors = mine_factors(data, fwd_ret, population_size=300, n_generations=50)

        for f in factors:
            print(f"IC={f['ic']:.4f}  ICIR={f['icir']:.4f}  {f['expression']}")
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )

    miner = GPMiner(
        data=data,
        fwd_returns=fwd_returns,
        population_size=population_size,
        n_generations=n_generations,
        max_depth=max_depth,
        ic_threshold=ic_threshold,
        corr_threshold=corr_threshold,
        seed=seed,
    )

    return miner.run()
