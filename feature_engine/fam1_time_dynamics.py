# -*- coding: utf-8 -*-
"""Fam1 时间动态：趋势在变好还是变坏。

跨窗口对比（短/长窗口比值）、斜率（按周分桶线性拟合）、波动（变异系数）、
指数衰减加权、recency（距最近/最早事件天数）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import safe_divide, window_slice
from .config import FeatureConfig


def generate(df: pd.DataFrame, cfg: FeatureConfig, train_mask: pd.Series
             ) -> Tuple[pd.DataFrame, List[Dict]]:
    sk = cfg.sample_key
    feats: Dict[str, pd.Series] = {}
    feat_dict: List[Dict] = []

    # ---- 跨窗口比值：相邻窗口的事件数 / 首个度量金额 ----
    win_sorted = sorted(cfg.windows)
    cnt_by_win = {w: window_slice(df, w).groupby(sk).size() for w in win_sorted}
    base_index = df.groupby(sk).size().index
    for short, long in zip(win_sorted[:-1], win_sorted[1:]):
        name = f"f1_cnt_ratio_{short}d_over_{long}d"
        feats[name] = safe_divide(cnt_by_win[short].reindex(base_index, fill_value=0),
                                  cnt_by_win[long].reindex(base_index, fill_value=0))
        feat_dict.append(_d(name, f"近{short}天与近{long}天笔数比（>窗口占比看近期是否提速）"))

    if cfg.measure_cols:
        m0 = cfg.measure_cols[0]
        amt_by_win = {w: window_slice(df, w).groupby(sk)[m0].sum() for w in win_sorted}
        for short, long in zip(win_sorted[:-1], win_sorted[1:]):
            name = f"f1_{m0}_ratio_{short}d_over_{long}d"
            feats[name] = safe_divide(amt_by_win[short].reindex(base_index, fill_value=0),
                                      amt_by_win[long].reindex(base_index, fill_value=0))
            feat_dict.append(_d(name, f"近{short}天与近{long}天 {cfg.col_label(m0)} 金额比"))

    # ---- 斜率与波动：按周分桶的事件数序列 ----
    full = window_slice(df, None).copy()
    full["_week"] = (full["_days_ago"] // 7).astype(int)  # 0=最近一周
    weekly = full.groupby([sk, "_week"]).size().rename("cnt").reset_index()

    def _slope_cv(grp: pd.DataFrame) -> pd.Series:
        # 周序号取负，使时间从早到晚递增，斜率>0 表示越近越活跃
        x = -grp["_week"].to_numpy(dtype=float)
        y = grp["cnt"].to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0] if len(x) >= 2 else 0.0
        mean = y.mean()
        cv = (y.std() / mean) if mean > 0 else 0.0
        return pd.Series({"f1_cnt_weekly_slope": slope, "f1_cnt_weekly_cv": cv})

    if len(weekly):
        sc = weekly.groupby(sk).apply(_slope_cv, include_groups=False)
        for col in sc.columns:
            feats[col] = sc[col]
        feat_dict.append(_d("f1_cnt_weekly_slope", "周事件数线性斜率（正=趋势上升）"))
        feat_dict.append(_d("f1_cnt_weekly_cv", "周事件数变异系数（越大越不稳定）"))

    # ---- 指数衰减加权事件数：近期事件权重更高 ----
    half_life = 30.0  # 半衰期 30 天
    decay = np.power(0.5, full["_days_ago"] / half_life)
    full["_decay"] = decay
    feats["f1_decay_weighted_cnt"] = full.groupby(sk)["_decay"].sum()
    feat_dict.append(_d("f1_decay_weighted_cnt", "指数衰减加权事件数(半衰期30天)"))

    # ---- recency：距最近/最早一次事件天数 ----
    feats["f1_recency_days"] = full.groupby(sk)["_days_ago"].min()
    feats["f1_history_days"] = full.groupby(sk)["_days_ago"].max()
    feat_dict.append(_d("f1_recency_days", "距最近一次事件天数（越小越活跃）"))
    feat_dict.append(_d("f1_history_days", "距最早一次事件天数（历史长度）"))

    out = pd.DataFrame(feats)
    return out, feat_dict


def _d(name: str, desc: str) -> Dict:
    return {"name": name, "family": 1, "family_name": "时间动态", "desc": desc}
