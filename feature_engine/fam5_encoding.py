# -*- coding: utf-8 -*-
"""Fam5 类别编码：把类别变成有判别力的数（需要 label）。

对每个维度列的"样本主类别"做三种编码，全部在 train 上拟合映射表：
- 频率编码：类别在 train 的出现频率
- 目标编码：类别的平滑后坏样本率（带先验平滑）
- WOE：证据权重，并附带 IV（信息值）写入特征说明

未见类别（test 出现 train 没有的）回退到全局先验/0。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import window_slice
from .config import FeatureConfig


def generate(df: pd.DataFrame, cfg: FeatureConfig, train_mask: pd.Series,
             sample_label: pd.Series, base_index: pd.Index
             ) -> Tuple[pd.DataFrame, List[Dict]]:
    sk = cfg.sample_key
    feats: Dict[str, pd.Series] = {}
    feat_dict: List[Dict] = []

    full = window_slice(df, None)
    # 每样本的主类别（众数）
    modal = {c: full.groupby(sk)[c].agg(lambda s: s.astype(str).mode().iloc[0]
                                        if len(s) else "NA").reindex(base_index)
             for c in cfg.dim_cols}

    # 样本级 train 掩码 + 标签对齐
    sample_train = train_mask.groupby(df[sk]).max().reindex(base_index, fill_value=False)
    y = sample_label.reindex(base_index)
    prior = float(y[sample_train].mean())          # 全局坏样本率（train）
    eps = 0.5                                       # WOE 拉普拉斯平滑

    for c in cfg.dim_cols:
        cat = modal[c].astype(str)
        tr_cat = cat[sample_train]
        tr_y = y[sample_train]

        grp = pd.DataFrame({"cat": tr_cat, "y": tr_y}).dropna()
        stat = grp.groupby("cat")["y"].agg(["count", "sum", "mean"])
        stat = stat.rename(columns={"count": "n", "sum": "bad", "mean": "bad_rate"})

        # 频率编码
        freq = (stat["n"] / stat["n"].sum()).to_dict()
        feats[f"f5_{c}_freq"] = cat.map(freq).fillna(0.0)

        # 目标编码（平滑）：(n*bad_rate + m*prior) / (n + m)
        m = cfg.target_smoothing
        te = ((stat["n"] * stat["bad_rate"] + m * prior) / (stat["n"] + m)).to_dict()
        feats[f"f5_{c}_target"] = cat.map(te).fillna(prior)

        # WOE + IV
        total_bad = stat["bad"].sum()
        total_good = (stat["n"] - stat["bad"]).sum()
        dist_bad = (stat["bad"] + eps) / (total_bad + eps * len(stat))
        dist_good = (stat["n"] - stat["bad"] + eps) / (total_good + eps * len(stat))
        woe = np.log(dist_good / dist_bad)
        iv = float(((dist_good - dist_bad) * woe).sum())
        feats[f"f5_{c}_woe"] = cat.map(woe.to_dict()).fillna(0.0)

        feat_dict += [
            _d(f"f5_{c}_freq", f"{c} 主类别的 train 频率"),
            _d(f"f5_{c}_target", f"{c} 主类别的平滑目标编码(坏率,先验{prior:.3f})"),
            _d(f"f5_{c}_woe", f"{c} 主类别 WOE (该维度 IV={iv:.4f})"),
        ]

    out = pd.DataFrame(feats, index=base_index)
    return out, feat_dict


def _d(name: str, desc: str) -> Dict:
    return {"name": name, "family": 5, "family_name": "类别编码", "desc": desc}
