"""
增强版 Barra 风险模型 (CNE5 参考实现)
- 因子暴露矩阵构建 (country + industry + style)
- 带行业约束的 WLS 因子收益率估计
- 因子协方差矩阵估计 (Newey-West + MC Eigenfactor Adjustment + Vol Regime)
- 特异性风险估计 (结构化模型 + 贝叶斯收缩)
- 纯因子组合构建
- 风险分解与归因
"""
import numpy as np
import pandas as pd
from scipy import linalg


class BarraRiskModel:
    """
    Barra 多因子风险模型

    风险分解: r_i = X_i @ f + epsilon_i
    V = X @ F @ X.T + Delta

    其中:
        X: 因子暴露矩阵 (N x K)
        F: 因子协方差矩阵 (K x K)
        Delta: 特异性风险对角阵 (N x N)
    """

    def __init__(
        self,
        factors: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        industry: pd.Series,
        market_cap: pd.DataFrame,
        halflife: int = 52,
        nw_lags: int = 2,
        vol_halflife: int = 26,
        eigenfactor_mc: int = 100,
        eigenfactor_alpha: float = 1.2,
    ):
        self.factors = factors
        self.returns = returns
        self.industry = industry
        self.market_cap = market_cap
        self.halflife = halflife
        self.nw_lags = nw_lags
        self.vol_halflife = vol_halflife
        self.eigenfactor_mc = eigenfactor_mc
        self.eigenfactor_alpha = eigenfactor_alpha

        # 缓存
        self._ind_dummies = pd.get_dummies(self.industry, dtype=float)
        self._industry_cols = list(self._ind_dummies.columns)

    # ------------------------------------------------------------------
    #  因子暴露矩阵
    # ------------------------------------------------------------------

    def _build_exposure_matrix(self, date) -> pd.DataFrame:
        """构建单期因子暴露矩阵 X (N x K)"""
        exposures = {}

        # 风格因子暴露
        for name, factor_df in self.factors.items():
            if date in factor_df.index:
                exposures[name] = factor_df.loc[date]

        # 行业因子暴露 (哑变量)
        for col in self._ind_dummies.columns:
            exposures[col] = self._ind_dummies[col]

        X = pd.DataFrame(exposures)
        # 国家因子 (截距项)
        X.insert(0, "country", 1.0)

        return X.dropna()

    # ------------------------------------------------------------------
    #  权重工具
    # ------------------------------------------------------------------

    def _exponential_weights(self, n: int, halflife: int = None) -> np.ndarray:
        """指数衰减权重"""
        hl = halflife or self.halflife
        lam = 0.5 ** (1 / hl)
        w = lam ** np.arange(n - 1, -1, -1)
        return w / w.sum()

    # ------------------------------------------------------------------
    #  带行业约束的 WLS 因子收益率估计
    # ------------------------------------------------------------------

    def _get_industry_cap_weights(self, date, stocks) -> np.ndarray:
        """各行业的市值权重 (用于行业约束)"""
        if date not in self.market_cap.index:
            n_ind = len(self._industry_cols)
            return np.ones(n_ind) / n_ind

        caps = self.market_cap.loc[date].reindex(stocks).fillna(0)
        ind_caps = np.zeros(len(self._industry_cols))
        for i, ind in enumerate(self._industry_cols):
            mask = self._ind_dummies.reindex(stocks)[ind].fillna(0).values > 0.5
            ind_caps[i] = caps.values[mask].sum()

        total = ind_caps.sum()
        return ind_caps / total if total > 0 else np.ones(len(ind_caps)) / len(ind_caps)

    def estimate_factor_returns(self, lookback: int = 52) -> pd.DataFrame:
        """
        带行业约束的 WLS 截面回归估计因子收益率

        约束: sum(cap_j * f_Ij) = 0
        使用 Lagrange 乘子法嵌入约束
        """
        dates = self.returns.index[-lookback:]
        factor_returns = {}

        for date_idx, date in enumerate(dates):
            if date_idx == 0:
                continue
            prev_date = dates[date_idx - 1]

            X = self._build_exposure_matrix(prev_date)
            r = self.returns.loc[date]

            common = X.index.intersection(r.dropna().index)
            if len(common) < X.shape[1] + 10:
                continue

            X_val = X.loc[common].values
            r_val = r.loc[common].values
            N, K = X_val.shape

            # 市值加权 WLS
            if prev_date in self.market_cap.index:
                w = np.sqrt(self.market_cap.loc[prev_date, common].fillna(0).values.astype(float))
                w = np.maximum(w, 1e-10)
                w = w / w.sum()
            else:
                w = np.ones(N) / N

            W = np.diag(w)

            # 行业约束矩阵 R: 行业因子收益率的市值加权和为0
            # R (1 x K): R[0, country] = 0, R[0, industries] = cap_weights, R[0, styles] = 0
            ind_cap_w = self._get_industry_cap_weights(prev_date, common)
            R = np.zeros((1, K))
            # 行业因子在 X 中的位置: country(1) + styles(n_style) ... 行业在 style 后面
            style_names = list(self.factors.keys())
            ind_start = 1 + len(style_names)  # country=0, styles, then industries
            # 但实际上顺序是: country, style_factors, industries
            # 需要从 X.columns 中确定位置
            for i, ind_col in enumerate(self._industry_cols):
                if ind_col in X.columns:
                    col_idx = list(X.columns).index(ind_col)
                    R[0, col_idx] = ind_cap_w[i]

            # 带约束的扩展方程组:
            # [X'WX  R'] [f     ]   [X'Wr]
            # [R     0 ] [lambda] = [0   ]
            XtW = X_val.T @ W
            A = np.block([
                [XtW @ X_val, R.T],
                [R, np.zeros((1, 1))]
            ])
            b = np.concatenate([XtW @ r_val, [0.0]])

            try:
                result = linalg.solve(A, b)
                f_hat = result[:K]
            except linalg.LinAlgError:
                # fallback: 无约束 WLS
                try:
                    f_hat = linalg.solve(XtW @ X_val, XtW @ r_val)
                except linalg.LinAlgError:
                    continue

            factor_returns[date] = pd.Series(f_hat, index=X.columns)

        return pd.DataFrame(factor_returns).T

    # ------------------------------------------------------------------
    #  因子协方差矩阵 (NW + MC Eigenfactor + Vol Regime)
    # ------------------------------------------------------------------

    def estimate_factor_covariance(
        self, factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        完整流程:
        1. 指数加权协方差
        2. Newey-West 自相关调整
        3. 蒙特卡洛 Eigenfactor Risk Adjustment
        4. Volatility Regime Adjustment
        """
        T, K = factor_returns.shape
        weights = self._exponential_weights(T)
        f_demean = factor_returns.values - np.average(
            factor_returns.values, weights=weights, axis=0
        )

        # Step 1 + 2: Newey-West 协方差
        cov_nw = self._newey_west_cov(f_demean, weights)

        # Step 3: MC Eigenfactor Adjustment
        cov_eigen = self._eigenfactor_adjustment(cov_nw, T)

        # Step 4: Vol Regime Adjustment
        cov_final = self._vol_regime_adjustment(
            cov_eigen, factor_returns.values, weights
        )

        return pd.DataFrame(
            cov_final,
            index=factor_returns.columns,
            columns=factor_returns.columns,
        )

    def _newey_west_cov(
        self, f_demean: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Newey-West 协方差估计"""
        T, K = f_demean.shape

        # 零阶协方差
        cov = np.zeros((K, K))
        for t in range(T):
            cov += weights[t] * np.outer(f_demean[t], f_demean[t])

        # 滞后协方差
        for lag in range(1, self.nw_lags + 1):
            bartlett_w = 1 - lag / (self.nw_lags + 1)
            gamma = np.zeros((K, K))
            for t in range(lag, T):
                gamma += weights[t] * np.outer(f_demean[t], f_demean[t - lag])
            cov += bartlett_w * (gamma + gamma.T)

        return cov

    def _eigenfactor_adjustment(
        self, cov_nw: np.ndarray, T_sample: int
    ) -> np.ndarray:
        """
        蒙特卡洛 Eigenfactor Risk Adjustment

        通过模拟估计特征值偏差，校正样本协方差矩阵
        对大特征值的低估和小特征值的高估
        """
        K = cov_nw.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(cov_nw)

        # 从大到小排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 保证正值
        eigenvalues = np.maximum(eigenvalues, 1e-12)

        if self.eigenfactor_mc <= 0 or T_sample < K + 5:
            # fallback: 简单的正定调整
            cov_adj = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            return (cov_adj + cov_adj.T) / 2

        # 蒙特卡洛模拟
        rng = np.random.default_rng(42)
        D_k = np.sqrt(eigenvalues)
        bias_sum = np.zeros(K)

        for _ in range(self.eigenfactor_mc):
            # 从 "真实" 分布中模拟
            sim_data = rng.standard_normal((T_sample, K)) * D_k[None, :]
            sim_data = sim_data @ eigenvectors.T

            # 样本协方差的特征值
            sim_cov = np.cov(sim_data.T)
            sim_eig = np.sort(np.linalg.eigvalsh(sim_cov))[::-1]
            sim_eig = np.maximum(sim_eig, 1e-12)

            bias_sum += np.sqrt(sim_eig / eigenvalues)

        bias_ratio = bias_sum / self.eigenfactor_mc

        # 校正: 放大被低估的特征值，缩小被高估的
        adjusted_eigenvalues = eigenvalues * (self.eigenfactor_alpha / np.maximum(bias_ratio, 0.1)) ** 2

        cov_adj = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
        return (cov_adj + cov_adj.T) / 2

    def _vol_regime_adjustment(
        self, cov: np.ndarray, factor_returns: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        波动率状态调整

        B_t = sqrt( sum_k (f_kt^2 / sigma_k^2) / K )
        lambda = EW_mean(B_t^2)
        F_adj = lambda * F
        """
        T, K = factor_returns.shape
        factor_vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))

        B_sq = np.zeros(T)
        for t in range(T):
            standardized = factor_returns[t] / factor_vols
            B_sq[t] = np.mean(standardized ** 2)

        vol_weights = self._exponential_weights(T, halflife=self.vol_halflife)
        lambda_F = np.sum(vol_weights * B_sq)

        # lambda 接近 1 表示模型预测准确, > 1 表示近期波动高于预期
        return lambda_F * cov

    # ------------------------------------------------------------------
    #  特异性风险 (结构化模型 + 贝叶斯收缩)
    # ------------------------------------------------------------------

    def estimate_specific_risk(
        self, factor_returns: pd.DataFrame, lookback: int = 52
    ) -> pd.Series:
        """
        完整的特异性风险估计流程:
        1. 截面回归残差
        2. 指数加权时序方差
        3. 结构化模型估计
        4. 贝叶斯收缩
        """
        dates = self.returns.index[-lookback:]
        residuals = {}

        for date_idx, date in enumerate(dates):
            if date_idx == 0 or date not in factor_returns.index:
                continue
            prev_date = dates[date_idx - 1]

            X = self._build_exposure_matrix(prev_date)
            r = self.returns.loc[date]
            f = factor_returns.loc[date]

            common_stocks = X.index.intersection(r.dropna().index)
            common_factors = X.columns.intersection(f.index)

            if len(common_stocks) < 30:
                continue

            predicted = X.loc[common_stocks, common_factors].values @ f.loc[common_factors].values
            resid = r.loc[common_stocks].values - predicted
            residuals[date] = pd.Series(resid, index=common_stocks)

        resid_df = pd.DataFrame(residuals).T
        if resid_df.empty:
            return pd.Series(dtype=float)

        # Step 1: 指数加权时序方差
        weights = self._exponential_weights(len(resid_df))
        weighted_var = np.average(
            resid_df.fillna(0).values ** 2, weights=weights, axis=0
        )
        sigma_ts = pd.Series(np.sqrt(np.maximum(weighted_var, 1e-12)), index=resid_df.columns)

        # Step 2: 结构化模型 (用市值解释特异性波动率截面差异)
        last_date = dates[-1]
        if last_date in self.market_cap.index:
            caps = self.market_cap.loc[last_date].reindex(resid_df.columns).fillna(1e6)
            log_cap = np.log(caps.values.astype(float))

            # ln(sigma) = a + b * ln(cap) + epsilon
            valid = np.isfinite(log_cap) & (sigma_ts.values > 1e-10)
            if valid.sum() > 30:
                ln_sigma = np.log(sigma_ts.values[valid])
                X_struct = np.column_stack([np.ones(valid.sum()), log_cap[valid]])
                try:
                    beta = np.linalg.lstsq(X_struct, ln_sigma, rcond=None)[0]
                    X_all = np.column_stack([np.ones(len(log_cap)), log_cap])
                    sigma_structural = pd.Series(
                        np.exp(X_all @ beta), index=resid_df.columns
                    )
                except Exception:
                    sigma_structural = sigma_ts.copy()
                    sigma_structural[:] = sigma_ts.median()
            else:
                sigma_structural = sigma_ts.copy()
                sigma_structural[:] = sigma_ts.median()
        else:
            sigma_structural = sigma_ts.copy()
            sigma_structural[:] = sigma_ts.median()

        # Step 3: 贝叶斯收缩
        # v_n ∝ |sigma_ts - sigma_str| / sigma_str
        delta = np.abs(sigma_ts - sigma_structural) / sigma_structural.clip(lower=1e-10)
        shrink_coeff = np.clip(delta, 0, 1)
        sigma_shrunk = shrink_coeff * sigma_structural + (1 - shrink_coeff) * sigma_ts

        return sigma_shrunk * np.sqrt(52)  # 年化

    # ------------------------------------------------------------------
    #  纯因子组合
    # ------------------------------------------------------------------

    def build_pure_factor_portfolio(
        self, target_factor: str, date
    ) -> pd.Series:
        """
        构建纯因子组合

        目标: 对目标因子暴露为1，对其他因子暴露为0
        约束: 权重和为0 (多空组合)

        使用方法: w = X (X'X)^-1 e_k
        """
        X = self._build_exposure_matrix(date)
        if target_factor not in X.columns:
            raise ValueError(f"Factor {target_factor} not in exposure matrix")

        X_val = X.values
        try:
            XtX_inv = linalg.inv(X_val.T @ X_val)
        except linalg.LinAlgError:
            XtX_inv = linalg.pinv(X_val.T @ X_val)

        # 目标暴露向量 e_k
        e_k = np.zeros(X.shape[1])
        k_idx = list(X.columns).index(target_factor)
        e_k[k_idx] = 1.0

        # 纯因子组合权重
        w = X_val @ XtX_inv @ e_k

        weights = pd.Series(w, index=X.index, name=target_factor)
        # 归一化使得多头和空头权重分别和为1
        long_sum = weights[weights > 0].sum()
        short_sum = abs(weights[weights < 0].sum())
        if long_sum > 0 and short_sum > 0:
            scale = 2 / (long_sum + short_sum)
            weights *= scale

        return weights

    # ------------------------------------------------------------------
    #  风险分解与归因
    # ------------------------------------------------------------------

    def full_risk_decomposition(self, portfolio_weights: pd.Series, date) -> dict:
        """
        投资组合风险分解

        Returns:
            total_risk: 组合总风险 (年化波动率)
            factor_risk: 因子风险贡献
            specific_risk: 特异性风险贡献
            factor_exposure: 因子暴露
            style_risk_contrib: 各风格因子的风险贡献
        """
        X = self._build_exposure_matrix(date)
        common = portfolio_weights.index.intersection(X.index)
        w = portfolio_weights.loc[common].values

        # 因子暴露 h = X' w
        exposure = X.loc[common].values.T @ w

        # 因子协方差
        fret = self.estimate_factor_returns()
        F = self.estimate_factor_covariance(fret)
        common_factors = X.columns.intersection(F.index)
        F_val = F.loc[common_factors, common_factors].values

        # 组合因子暴露
        h = X.loc[common, common_factors].values.T @ w

        # 因子方差
        factor_var = h @ F_val @ h

        # 特异性风险
        spec_risk = self.estimate_specific_risk(fret)
        common_spec = pd.Index(common).intersection(spec_risk.index)
        w_spec = portfolio_weights.loc[common_spec].values
        spec_var = np.sum((w_spec * spec_risk.loc[common_spec].values) ** 2)

        total_var = factor_var + spec_var

        # 各因子的边际风险贡献 (MCTR)
        factor_mctr = F_val @ h
        factor_risk_contrib = h * factor_mctr

        # 仅取风格因子
        style_names = list(self.factors.keys())
        style_contrib = {}
        for sname in style_names:
            if sname in common_factors:
                idx = list(common_factors).index(sname)
                style_contrib[sname] = {
                    "exposure": h[idx],
                    "risk_contrib": factor_risk_contrib[idx],
                    "risk_contrib_pct": factor_risk_contrib[idx] / total_var if total_var > 0 else 0,
                }

        return {
            "total_risk": np.sqrt(total_var),
            "factor_risk": np.sqrt(factor_var),
            "specific_risk": np.sqrt(spec_var),
            "factor_exposure": pd.Series(exposure, index=X.columns),
            "factor_risk_pct": factor_var / total_var if total_var > 0 else 0,
            "style_risk_contrib": style_contrib,
        }
