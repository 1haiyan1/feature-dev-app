# -*- coding: utf-8 -*-
"""特征引擎公共工具。

包含：
- train/test 拟合-映射框架（FittedParam）
- 安全除法、时间窗口过滤、观察点计算、top-k 类别等通用函数

设计原则：所有"看分布"的统计量（均值/标准差/分位点/编码映射）一律只在 train 子集拟合，
再 transform 到全体，杜绝数据泄漏。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# train 子集获取
# ----------------------------------------------------------------------------
def get_train_mask(df: pd.DataFrame, data_set_col: str, train_value: str) -> pd.Series:
    """返回标记 train 行的布尔 Series。data_set 列缺失时退化为"全体即 train"。"""
    if data_set_col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[data_set_col].astype(str) == str(train_value)


# ----------------------------------------------------------------------------
# 数值安全工具
# ----------------------------------------------------------------------------
def safe_divide(numerator, denominator, fill: float = 0.0):
    """安全除法：分母为 0 或 NaN 时返回 fill，避免 inf/NaN 污染特征。"""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


# ----------------------------------------------------------------------------
# 时间与窗口
# ----------------------------------------------------------------------------
def ensure_datetime(s: pd.Series) -> pd.Series:
    """统一把时间列转成 datetime（强健解析，结果等价于固定 yyyy-mm-dd 口径）。

    支持 2024-1-5 / 2024/01/05 / 2024.01.05 / 20240105 / 2024年1月5日 / Excel序列号；
    '', 'nan', 'null', 'NaT' 等假空统一转 NaT。无法解析的留 NaT，不会污染窗口计算。
    """
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    txt = s.astype(str).str.strip()
    txt = txt.where(~txt.str.lower().isin(
        ["", "nan", "none", "nat", "null", "na", "n/a", "-"]))
    # 先把分隔符统一成 '-'、去掉中文年月日，避免 pandas 2.x 单一格式推断打架
    norm = (txt.str.replace(r"[./年月]", "-", regex=True)
            .str.replace("日", "", regex=False).str.replace(r"-+$", "", regex=True))
    # format="mixed"：逐元素推断，兼容 2024-1-5 / 2024-02-10 等混合格式
    dt = pd.to_datetime(norm, errors="coerce", format="mixed")

    # 兜底1：8位纯数字按 yyyymmdd
    mask = dt.isna() & txt.notna() & txt.str.match(r"^\d{8}$")
    if mask.any():
        dt.loc[mask] = pd.to_datetime(txt[mask], format="%Y%m%d", errors="coerce")

    # 兜底2：5位纯数字按 Excel 序列号（1899-12-30 起）
    mask = dt.isna() & txt.notna() & txt.str.match(r"^\d{5}$")
    if mask.any():
        dt.loc[mask] = pd.to_datetime(txt[mask].astype(float), unit="D",
                                      origin="1899-12-30", errors="coerce")
    return dt


def compute_obs_point(df: pd.DataFrame, sample_key: str, time_col: str,
                      dateback_col: Optional[str], update_policy: str,
                      update_lag_days: int, fallback_obs_point: str = "per_sample_max"
                      ) -> pd.Series:
    """计算每个样本的观察点（窗口的右端点 = 截至哪一天的流水可用）。

    优先级：
      1) 若提供了样本级 dateback_col：观察点 = 按 update_policy 调整后的 dateback。
      2) 否则：fallback_obs_point 为固定日期 -> 全样本共用该日期；
                                  为 "per_sample_max" -> 每样本最后一笔时间。

    更新策略：
      - "T+N":     观察点 = dateback - update_lag_days 天（含 T+0/T+1/T+2/...）
      - "monthly": 观察点 = dateback 当月的上月最后一天（月更）
    """
    # 路径 1：有 dateback 列，按策略算
    if dateback_col and dateback_col in df.columns:
        db = df.groupby(sample_key)[dateback_col].first()
        db = ensure_datetime(db)
        obs = apply_update_policy(db, update_policy, update_lag_days)
        # dateback 缺失的样本兜底：用该样本最后一笔流水时间
        if obs.isna().any() and time_col and time_col in df.columns:
            fb = df.groupby(sample_key)[time_col].max()
            obs = obs.fillna(fb)
        return obs

    # 路径 2：无 dateback，沿用原 obs_point 逻辑
    if fallback_obs_point and fallback_obs_point != "per_sample_max":
        fixed = pd.to_datetime(fallback_obs_point)
        keys = df[sample_key].dropna().unique()
        return pd.Series(fixed, index=pd.Index(keys, name=sample_key))
    return df.groupby(sample_key)[time_col].max()


def apply_update_policy(dateback: pd.Series, policy: str, lag_days: int) -> pd.Series:
    """根据更新策略把 dateback 调成"可用数据截止日"。"""
    p = (policy or "T+N").lower()
    if p in ("monthly", "month", "月更"):
        # 上月最后一天 = 当月第一天 - 1 天
        return dateback.apply(
            lambda d: (d.replace(day=1) - pd.Timedelta(days=1))
                      if pd.notna(d) else pd.NaT
        )
    # 默认 T+N（T+0/T+1/T+2/...）
    return dateback - pd.Timedelta(days=int(lag_days))


def add_window_helpers(df: pd.DataFrame, sample_key: str, time_col: str,
                       obs: pd.Series) -> pd.DataFrame:
    """给流水加一列 days_ago = 观察点 - 事件时间（天）。供窗口过滤与衰减加权复用。"""
    out = df.copy()
    out[time_col] = ensure_datetime(out[time_col])
    obs_mapped = out[sample_key].map(obs)
    out["_days_ago"] = (obs_mapped - out[time_col]).dt.total_seconds() / 86400.0
    return out


def window_slice(df_with_days: pd.DataFrame, window_days: Optional[int]) -> pd.DataFrame:
    """取窗口内的流水：0 <= days_ago <= window_days。window_days=None 表示全历史。"""
    d = df_with_days["_days_ago"]
    mask = d >= 0
    if window_days is not None:
        mask &= d <= window_days
    return df_with_days[mask]


# ----------------------------------------------------------------------------
# 类别工具
# ----------------------------------------------------------------------------
def fit_top_k_categories(train_df: pd.DataFrame, col: str, k: int) -> List[str]:
    """在 train 上取某维度列频次最高的前 K 个类别值（字符串化）。"""
    vc = train_df[col].astype(str).value_counts()
    return list(vc.head(k).index)


def resolve_categories(train_df: pd.DataFrame, col: str, k: int,
                       fixed_values: Optional[List[str]] = None) -> List[str]:
    """决定某维度列要 pivot 出哪些类别：
    - 若 fixed_values 提供（例如打标列的规则名），直接用，确保**所有定义的标都出列**；
    - 否则按 train 取频次 top-K（兼容旧行为）。
    """
    if fixed_values is not None:
        return [str(v) for v in fixed_values]
    return fit_top_k_categories(train_df, col, k)


# ----------------------------------------------------------------------------
# 流水打标
# ----------------------------------------------------------------------------
def apply_tags(df: pd.DataFrame, rules: List[Dict], tag_col: str = "_tag",
               default_tag: str = "other") -> pd.DataFrame:
    """按规则列表给流水打标，单标·首条命中，未命中归 default_tag。

    每条规则：{"name": "大额", "query": "amount >= 1000"}
    query 使用 pandas.DataFrame.eval 语法（支持列名引用、比较、布尔运算、in 等）。

    设计原则：
    - 规则顺序即优先级；前面命中的不再被后面规则覆盖。
    - 安全：query 在 pandas.eval 沙盒里跑，不接 Python exec。
    - 一行只打一个标，输出列为字符串类型。
    """
    out = df.copy()
    tags = pd.Series([default_tag] * len(out), index=out.index, dtype=object)
    assigned = pd.Series(False, index=out.index)

    for rule in rules or []:
        name = str(rule.get("name") or "").strip()
        query = rule.get("query")
        if not name or not query:
            continue
        try:
            mask = out.eval(query)
        except Exception as e:
            # 规则跑挂了不能让全流程崩；跳过并标记
            mask = pd.Series(False, index=out.index)
        # 只给尚未命中的行打标
        target = mask & (~assigned)
        tags = tags.where(~target, name)
        assigned = assigned | target

    out[tag_col] = tags.astype(str)
    return out


# ----------------------------------------------------------------------------
# 集中度指标
# ----------------------------------------------------------------------------
def hhi(shares: np.ndarray) -> float:
    """赫芬达尔指数 = Σ(占比²)，越大越集中（单一类别独大 -> 接近 1）。"""
    shares = np.asarray(shares, dtype=float)
    s = shares.sum()
    if s <= 0:
        return 0.0
    p = shares / s
    return float(np.square(p).sum())


def gini(values: np.ndarray) -> float:
    """基尼系数（针对非负金额数组），衡量不均衡程度，0=完全均匀。"""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    cum = np.cumsum(v)
    # 基尼 = 1 - 2 * 洛伦兹曲线下面积
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


# ----------------------------------------------------------------------------
# 拟合参数容器
# ----------------------------------------------------------------------------
class FittedParam:
    """封装一次 train 拟合的产物，统一序列化便于 codegen 和复用。"""

    def __init__(self, kind: str, payload: Dict):
        self.kind = kind          # 例如 'zscore' / 'qcut' / 'woe'
        self.payload = payload    # 拟合出来的统计量字典

    def __repr__(self):
        return f"FittedParam(kind={self.kind!r}, keys={list(self.payload)})"
