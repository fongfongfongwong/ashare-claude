"""
深度学习模型模块 (基于 Apple MLX 框架)
支持 Apple Silicon GPU 加速

包含 8 个模型:
1. MLPModel       - 多层感知机
2. GRUModel       - 门控循环单元
3. LSTMModel      - 长短期记忆网络
4. ALSTMModel     - 注意力 LSTM
5. TransformerModel - Transformer Encoder
6. TCNModel       - 时间卷积网络
7. LocalformerModel - 局部注意力 + 全局注意力
8. GATModel       - 图注意力网络 (简化版)
"""
import logging
import math
import numpy as np
import polars as pl
from typing import Iterator

from .base import BaseModel

logger = logging.getLogger(__name__)

# ============================================================
# 安全导入 MLX
# ============================================================
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # 占位: 避免后续类定义报错
    mx = None
    nn = None
    optim = None


def _check_mlx():
    """检查 MLX 是否可用"""
    if not HAS_MLX:
        raise ImportError(
            "MLX 未安装或当前平台不支持 (仅支持 Apple Silicon)。\n"
            "请执行: pip install mlx"
        )


# ============================================================
# 工具函数
# ============================================================
def _to_mx(arr: np.ndarray) -> "mx.array":
    """numpy -> mlx array (float32)"""
    return mx.array(arr.astype(np.float32))


def _batch_iter(
    X: "mx.array", y: "mx.array", batch_size: int, shuffle: bool = True
) -> Iterator[tuple["mx.array", "mx.array"]]:
    """生成 mini-batch 迭代器"""
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = mx.array(indices[start:end])
        yield X[idx], y[idx]


# ============================================================
# 网络模块定义
# ============================================================

class _MLPNet(nn.Module):
    """3 层全连接网络: input → 256 → 128 → 64 → 1"""

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm(64)
        self.head = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x):
        x = self.dropout(nn.relu(self.bn1(self.fc1(x))))
        x = self.dropout(nn.relu(self.bn2(self.fc2(x))))
        x = self.dropout(nn.relu(self.bn3(self.fc3(x))))
        return self.head(x).squeeze(-1)


class _GRUCell(nn.Module):
    """单个 GRU Cell"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # 更新门
        self.Wz = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # 重置门
        self.Wr = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # 候选隐藏状态
        self.Wh = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def __call__(self, x, h):
        """
        x: (batch, input_dim)
        h: (batch, hidden_dim)
        """
        xh = mx.concatenate([x, h], axis=-1)
        z = mx.sigmoid(self.Wz(xh))   # 更新门
        r = mx.sigmoid(self.Wr(xh))   # 重置门
        xrh = mx.concatenate([x, r * h], axis=-1)
        h_tilde = mx.tanh(self.Wh(xrh))  # 候选状态
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class _GRUNet(nn.Module):
    """2 层 GRU + FC Head"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 多层 GRU
        self.cells = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(_GRUCell(in_dim, hidden_dim))

        self.head = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态
        h = [mx.zeros((batch_size, self.hidden_dim)) for _ in range(self.n_layers)]

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer_idx in range(self.n_layers):
                h[layer_idx] = self.cells[layer_idx](inp, h[layer_idx])
                inp = h[layer_idx]

        # 用最后一层最后时刻的隐藏状态
        out = h[-1]
        return self.head(out).squeeze(-1)


class _LSTMCell(nn.Module):
    """单个 LSTM Cell"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        self.hidden_dim = hidden_dim

    def __call__(self, x, h, c):
        """
        x: (batch, input_dim)
        h: (batch, hidden_dim)
        c: (batch, hidden_dim)
        """
        xh = mx.concatenate([x, h], axis=-1)
        gates = self.gates(xh)
        i = mx.sigmoid(gates[:, :self.hidden_dim])
        f = mx.sigmoid(gates[:, self.hidden_dim:2*self.hidden_dim])
        g = mx.tanh(gates[:, 2*self.hidden_dim:3*self.hidden_dim])
        o = mx.sigmoid(gates[:, 3*self.hidden_dim:])
        c_new = f * c + i * g
        h_new = o * mx.tanh(c_new)
        return h_new, c_new


class _LSTMNet(nn.Module):
    """2 层 LSTM + FC Head"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.cells = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(_LSTMCell(in_dim, hidden_dim))

        self.head = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        h = [mx.zeros((batch_size, self.hidden_dim)) for _ in range(self.n_layers)]
        c = [mx.zeros((batch_size, self.hidden_dim)) for _ in range(self.n_layers)]

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer_idx in range(self.n_layers):
                h[layer_idx], c[layer_idx] = self.cells[layer_idx](
                    inp, h[layer_idx], c[layer_idx]
                )
                inp = h[layer_idx]

        out = h[-1]
        return self.head(out).squeeze(-1)


class _TemporalAttention(nn.Module):
    """时序注意力机制"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_w = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

    def __call__(self, hidden_states):
        """
        hidden_states: (batch, seq_len, hidden_dim)
        返回: (batch, hidden_dim) 加权聚合
        """
        # 计算注意力分数
        energy = mx.tanh(self.attn_w(hidden_states))   # (batch, seq_len, hidden_dim)
        scores = self.attn_v(energy).squeeze(-1)        # (batch, seq_len)
        weights = mx.softmax(scores, axis=-1)            # (batch, seq_len)
        # 加权求和
        context = mx.sum(
            hidden_states * mx.expand_dims(weights, axis=-1), axis=1
        )  # (batch, hidden_dim)
        return context


class _ALSTMNet(nn.Module):
    """LSTM + Temporal Attention"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.cells = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(_LSTMCell(in_dim, hidden_dim))

        self.attention = _TemporalAttention(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        h = [mx.zeros((batch_size, self.hidden_dim)) for _ in range(self.n_layers)]
        c = [mx.zeros((batch_size, self.hidden_dim)) for _ in range(self.n_layers)]

        # 收集所有时步的最后一层隐藏状态
        all_hidden = []
        for t in range(seq_len):
            inp = x[:, t, :]
            for layer_idx in range(self.n_layers):
                h[layer_idx], c[layer_idx] = self.cells[layer_idx](
                    inp, h[layer_idx], c[layer_idx]
                )
                inp = h[layer_idx]
            all_hidden.append(h[-1])

        # (batch, seq_len, hidden_dim)
        hidden_states = mx.stack(all_hidden, axis=1)

        # 注意力加权
        context = self.attention(hidden_states)
        return self.head(context).squeeze(-1)


class _MultiHeadAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def __call__(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = self.W_k(x).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = self.W_v(x).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # (batch, n_heads, seq_len, seq_len)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)

        # (batch, n_heads, seq_len, d_k)
        attn_out = weights @ V
        # 合并多头
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.W_o(attn_out)


class _TransformerEncoderLayer(nn.Module):
    """单层 Transformer Encoder"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = _MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x):
        # 自注意力 + 残差
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        # 前馈 + 残差
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class _PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
        self._pe = mx.array(pe)

    def __call__(self, x):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.shape[1]
        return x + self._pe[:seq_len]


class _TransformerNet(nn.Module):
    """Transformer Encoder + FC Head"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        self.layers = [
            _TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        self.head = nn.Linear(d_model, 1)

    def __call__(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        # 取最后一个时步的输出
        out = x[:, -1, :]
        return self.head(out).squeeze(-1)


class _CausalConv1d(nn.Module):
    """因果卷积: 只看过去的时间步"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def __call__(self, x):
        """
        x: (batch, in_channels, seq_len)
        """
        out = self.conv(x)
        # 截掉未来的 padding
        if self.padding > 0:
            out = out[:, :, :x.shape[2]]
        return out


class _TCNBlock(nn.Module):
    """TCN 残差块: 两层因果卷积 + 残差连接"""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def __call__(self, x):
        """
        x: (batch, channels, seq_len)
        """
        # 第一层
        out = self.conv1(x)
        # LayerNorm 需要 (batch, seq_len, channels)
        out = out.transpose(0, 2, 1)
        out = self.norm1(out)
        out = out.transpose(0, 2, 1)
        out = self.dropout(nn.relu(out))

        # 第二层
        out = self.conv2(out)
        out = out.transpose(0, 2, 1)
        out = self.norm2(out)
        out = out.transpose(0, 2, 1)
        out = self.dropout(nn.relu(out))

        # 残差连接
        return out + x


class _TCNNet(nn.Module):
    """3 层 TCN + FC Head"""

    def __init__(
        self,
        input_dim: int,
        channels: int = 64,
        kernel_size: int = 3,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, channels)
        dilations = [2 ** i for i in range(n_layers)]  # [1, 2, 4]
        self.blocks = [
            _TCNBlock(channels, kernel_size, d, dropout) for d in dilations
        ]
        self.head = nn.Linear(channels, 1)

    def __call__(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        # 投影到 channels 维度
        x = self.input_proj(x)  # (batch, seq_len, channels)
        # Conv1d 需要 (batch, channels, seq_len)
        x = x.transpose(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        # 取最后时步
        out = x[:, :, -1]  # (batch, channels)
        return self.head(out).squeeze(-1)


class _LocalAttention(nn.Module):
    """局部注意力: 每个位置只关注窗口内的邻居"""

    def __init__(self, d_model: int, window_size: int = 5):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def __call__(self, x):
        """
        x: (batch, seq_len, d_model)
        使用全局注意力 + 局部 mask 实现
        """
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 计算注意力分数
        scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(self.d_model)

        # 创建局部 mask: 只允许窗口内的注意力
        mask = np.full((seq_len, seq_len), -1e9, dtype=np.float32)
        half_w = self.window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            mask[i, start:end] = 0.0
        mask_mx = mx.array(mask)

        scores = scores + mask_mx
        weights = mx.softmax(scores, axis=-1)
        return weights @ V


class _LocalformerNet(nn.Module):
    """Localformer: Local Attention + Global Attention + FC Head"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        window_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)

        # 局部注意力层
        self.local_attn = _LocalAttention(d_model, window_size)
        self.norm1 = nn.LayerNorm(d_model)

        # 全局注意力层
        self.global_attn = _MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(d_model, 1)

    def __call__(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # 局部注意力 + 残差
        local_out = self.local_attn(x)
        x = self.norm1(x + self.dropout(local_out))

        # 全局注意力 + 残差
        global_out = self.global_attn(x)
        x = self.norm2(x + self.dropout(global_out))

        # 前馈 + 残差
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        # 最后时步
        out = x[:, -1, :]
        return self.head(out).squeeze(-1)


class _GATLayer(nn.Module):
    """图注意力层: 简化版，不需要预定义图结构"""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, out_dim)
        # 注意力参数: a = [a_l || a_r] ∈ R^{2*d_k} per head
        self.attn_l = nn.Linear(self.d_k, 1, bias=False)
        self.attn_r = nn.Linear(self.d_k, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x):
        """
        x: (n_nodes, in_dim)
        在截面内计算股票间的图注意力
        """
        n_nodes = x.shape[0]

        # 线性变换
        h = self.W(x)  # (n_nodes, out_dim)
        h = h.reshape(n_nodes, self.n_heads, self.d_k)  # (n_nodes, n_heads, d_k)

        # 计算注意力系数
        # e_{ij} = LeakyReLU(a_l^T h_i + a_r^T h_j)
        attn_l = self.attn_l(h)  # (n_nodes, n_heads, 1)
        attn_r = self.attn_r(h)  # (n_nodes, n_heads, 1)

        # (n_nodes, n_heads, 1) + (1, n_heads, n_nodes) = (n_nodes, n_heads, n_nodes)
        e = attn_l + attn_r.transpose(2, 1, 0)
        e = nn.leaky_relu(e, negative_slope=0.2)

        # softmax 归一化
        alpha = mx.softmax(e, axis=-1)  # (n_nodes, n_heads, n_nodes)
        alpha = self.dropout(alpha)

        # 聚合邻居特征
        # (n_nodes, n_heads, n_nodes) @ (n_nodes, n_heads, d_k) -- 需要转置 h
        # h: (n_nodes, n_heads, d_k) -> 对每个 head: (n_nodes, n_nodes) @ (n_nodes, d_k)
        out = mx.zeros((n_nodes, self.n_heads, self.d_k))
        for head_idx in range(self.n_heads):
            a_h = alpha[:, head_idx, :]   # (n_nodes, n_nodes)
            h_h = h[:, head_idx, :]       # (n_nodes, d_k)
            out_h = a_h @ h_h             # (n_nodes, d_k)
            out = out.at[:, head_idx, :].add(out_h)

        # 合并多头: (n_nodes, out_dim)
        out = out.reshape(n_nodes, self.out_dim)
        return nn.elu(out)


class _GATNet(nn.Module):
    """
    简化版图注意力网络
    将截面内的股票视为图中的节点
    不需要预定义图结构，完全通过注意力学习股票间关系
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gat1 = _GATLayer(input_dim, hidden_dim, n_heads, dropout)
        self.gat2 = _GATLayer(hidden_dim, hidden_dim, n_heads, dropout)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x):
        """
        x: (n_nodes, input_dim)
        返回: (n_nodes,) 每只股票的预测值
        """
        x = self.dropout(self.gat1(x))
        x = self.dropout(self.gat2(x))
        return self.head(x).squeeze(-1)


# ============================================================
# MLX 模型基类 (训练循环封装)
# ============================================================
class _MLXBaseModel(BaseModel):
    """
    MLX 深度学习模型的公共基类
    封装训练循环、early stopping、mini-batch 逻辑
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seq_len: int = 1,
        **kwargs,
    ):
        _check_mlx()
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=seq_len,
            **kwargs,
        )
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seq_len = seq_len
        self.model = None  # 子类设置
        self._feature_names: list[str] = []

    def _build_model(self, input_dim: int):
        """子类实现: 构建网络"""
        raise NotImplementedError

    def _prepare_data(self, X: pl.DataFrame) -> np.ndarray:
        """
        准备输入数据
        对于序列模型 (GRU/LSTM/Transformer等): reshape 为 (N, seq_len, D)
        对于非序列模型 (MLP): 保持 (N, D)
        """
        X_np = X.to_numpy().astype(np.float32)
        # 判断是否为序列模型 (非 MLP)
        is_seq_model = not isinstance(self, MLPModel)
        if is_seq_model:
            n_samples, n_features = X_np.shape
            if self.seq_len == 1:
                # 直接 reshape 为 (N, 1, D)
                X_np = X_np.reshape(n_samples, 1, n_features)
            else:
                features_per_step = max(1, n_features // self.seq_len)
                X_np = X_np[:, :features_per_step * self.seq_len]
                X_np = X_np.reshape(n_samples, self.seq_len, features_per_step)
        return X_np

    def _get_input_dim(self, X: pl.DataFrame) -> int:
        """根据 seq_len 计算单步输入维度"""
        n_features = len(X.columns)
        if self.seq_len > 1:
            return n_features // self.seq_len
        return n_features

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """通用训练循环"""
        self._feature_names = X_train.columns
        input_dim = self._get_input_dim(X_train)

        # 构建模型
        self._build_model(input_dim)

        # 准备数据
        X_tr = _to_mx(self._prepare_data(X_train))
        y_tr = _to_mx(y_train.to_numpy().ravel())

        X_v, y_v = None, None
        if X_val is not None and y_val is not None:
            X_v = _to_mx(self._prepare_data(X_val))
            y_v = _to_mx(y_val.to_numpy().ravel())

        optimizer = optim.Adam(learning_rate=self.learning_rate)

        # 定义 loss 函数
        def loss_fn(model, X_batch, y_batch):
            preds = model(X_batch)
            return mx.mean((preds - y_batch) ** 2)

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        best_val_loss = float("inf")
        patience_counter = 0
        best_params = None

        for epoch in range(self.n_epochs):
            # 训练模式
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in _batch_iter(
                X_tr, y_tr, self.batch_size, shuffle=True
            ):
                loss, grads = loss_and_grad_fn(self.model, batch_X, batch_y)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # 验证 + early stopping
            if X_v is not None:
                self.model.eval()
                val_preds = self.model(X_v)
                val_loss = mx.mean((val_preds - y_v) ** 2).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最优参数
                    best_params = self.model.parameters()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"{self.__class__.__name__} epoch {epoch+1}: "
                            f"early stopping (val_loss={val_loss:.6f})"
                        )
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"{self.__class__.__name__} epoch {epoch+1}/{self.n_epochs}: "
                        f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"{self.__class__.__name__} epoch {epoch+1}/{self.n_epochs}: "
                        f"train_loss={avg_train_loss:.6f}"
                    )

        # 恢复最优参数
        if best_params is not None:
            self.model.update(best_params)
            mx.eval(self.model.parameters())

        self._fitted = True
        logger.info(f"{self.__class__.__name__} 训练完成")
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """通用预测"""
        self._check_fitted()
        self.model.eval()
        X_mx = _to_mx(self._prepare_data(X))

        # 分 batch 预测 (避免内存溢出)
        all_preds = []
        n = X_mx.shape[0]
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_preds = self.model(X_mx[start:end])
            all_preds.append(np.array(batch_preds))

        preds = np.concatenate(all_preds)
        return pl.Series("prediction", preds)


# ============================================================
# 8 个具体模型
# ============================================================

class MLPModel(_MLXBaseModel):
    """
    多层感知机 (MLP)
    3 层全连接: 256 → 128 → 64 → 1
    激活函数: ReLU, Dropout=0.3, BatchNorm
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        patience: int = 10,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=1,
            dropout=dropout,
            **kwargs,
        )
        self.dropout = dropout

    def _build_model(self, input_dim: int):
        self.model = _MLPNet(input_dim, self.dropout)
        mx.eval(self.model.parameters())


class GRUModel(_MLXBaseModel):
    """
    GRU 模型
    2 层 GRU (hidden=128) + FC Head
    输入序列长度默认 20
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seq_len: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 2,
        **kwargs,
    ):
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def _build_model(self, input_dim: int):
        self.model = _GRUNet(input_dim, self.hidden_dim, self.n_layers)
        mx.eval(self.model.parameters())


class LSTMModel(_MLXBaseModel):
    """
    LSTM 模型
    2 层 LSTM (hidden=128) + FC Head
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seq_len: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 2,
        **kwargs,
    ):
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def _build_model(self, input_dim: int):
        self.model = _LSTMNet(input_dim, self.hidden_dim, self.n_layers)
        mx.eval(self.model.parameters())


class ALSTMModel(_MLXBaseModel):
    """
    ALSTM: LSTM + Temporal Attention
    用注意力机制对所有时步的隐藏状态加权聚合
    比普通 LSTM 能更好地捕捉关键时间步信息
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seq_len: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 2,
        **kwargs,
    ):
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def _build_model(self, input_dim: int):
        self.model = _ALSTMNet(input_dim, self.hidden_dim, self.n_layers)
        mx.eval(self.model.parameters())


class TransformerModel(_MLXBaseModel):
    """
    Transformer Encoder 模型
    2 层 Encoder (d_model=64, nhead=4) + FC Head
    使用正弦位置编码
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seq_len: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            **kwargs,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout

    def _build_model(self, input_dim: int):
        self.model = _TransformerNet(
            input_dim, self.d_model, self.n_heads,
            self.n_layers, self.d_ff, self.dropout_rate,
        )
        mx.eval(self.model.parameters())


class TCNModel(_MLXBaseModel):
    """
    时间卷积网络 (TCN)
    3 层因果卷积 (kernel=3, dilation=[1,2,4]) + FC Head
    通过膨胀卷积实现指数级感受野增长
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seq_len: int = 20,
        channels: int = 64,
        kernel_size: int = 3,
        n_layers: int = 3,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=seq_len,
            channels=channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.dropout_rate = dropout

    def _build_model(self, input_dim: int):
        self.model = _TCNNet(
            input_dim, self.channels, self.kernel_size,
            self.n_layers, self.dropout_rate,
        )
        mx.eval(self.model.parameters())


class LocalformerModel(_MLXBaseModel):
    """
    Localformer: 局部注意力 + 全局注意力
    局部注意力 (window=5) 捕捉短期模式
    全局注意力捕捉长期依赖
    两者结合比纯 Transformer 更适合金融时序
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seq_len: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        window_size: int = 5,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout,
            **kwargs,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.dropout_rate = dropout

    def _build_model(self, input_dim: int):
        self.model = _LocalformerNet(
            input_dim, self.d_model, self.n_heads,
            self.window_size, self.dropout_rate,
        )
        mx.eval(self.model.parameters())


class GATModel(_MLXBaseModel):
    """
    简化版图注意力网络 (GAT)
    将截面内的股票视为图节点
    不需要预定义图结构，通过注意力自动学习股票间关系

    注意: 此模型的 fit/predict 与其他模型不同
    - 输入不是序列数据，而是截面数据 (每只股票一行)
    - 每个 batch 是一个截面 (同一时间的所有股票)
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        patience: int = 10,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seq_len=1,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout

    def _build_model(self, input_dim: int):
        self.model = _GATNet(
            input_dim, self.hidden_dim, self.n_heads, self.dropout_rate,
        )
        mx.eval(self.model.parameters())

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame = None,
        y_val: pl.Series = None,
    ):
        """
        GAT 训练
        由于 GAT 需要截面数据，这里将所有样本视为一个大图
        使用 mini-batch 随机采样节点子集训练
        """
        self._feature_names = X_train.columns
        input_dim = len(self._feature_names)
        self._build_model(input_dim)

        X_tr = _to_mx(X_train.to_numpy().astype(np.float32))
        y_tr = _to_mx(y_train.to_numpy().ravel().astype(np.float32))

        X_v, y_v = None, None
        if X_val is not None and y_val is not None:
            X_v = _to_mx(X_val.to_numpy().astype(np.float32))
            y_v = _to_mx(y_val.to_numpy().ravel().astype(np.float32))

        optimizer = optim.Adam(learning_rate=self.learning_rate)

        def loss_fn(model, X_batch, y_batch):
            preds = model(X_batch)
            return mx.mean((preds - y_batch) ** 2)

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        best_val_loss = float("inf")
        patience_counter = 0
        best_params = None

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # 随机采样节点子集 (mini-batch of nodes)
            for batch_X, batch_y in _batch_iter(
                X_tr, y_tr, self.batch_size, shuffle=True
            ):
                loss, grads = loss_and_grad_fn(self.model, batch_X, batch_y)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            if X_v is not None:
                self.model.eval()
                val_preds = self.model(X_v)
                val_loss = mx.mean((val_preds - y_v) ** 2).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_params = self.model.parameters()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"GATModel epoch {epoch+1}: early stopping "
                            f"(val_loss={val_loss:.6f})"
                        )
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"GATModel epoch {epoch+1}/{self.n_epochs}: "
                        f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}"
                    )

        if best_params is not None:
            self.model.update(best_params)
            mx.eval(self.model.parameters())

        self._fitted = True
        logger.info("GATModel 训练完成")
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """GAT 预测"""
        self._check_fitted()
        self.model.eval()
        X_mx = _to_mx(X.to_numpy().astype(np.float32))

        # 分 batch 预测
        all_preds = []
        n = X_mx.shape[0]
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_preds = self.model(X_mx[start:end])
            all_preds.append(np.array(batch_preds))

        preds = np.concatenate(all_preds)
        return pl.Series("prediction", preds)
