# -*- coding: utf-8 -*-
"""Fam3 相对位置：在人群里排第几。

对样本级特征宽表里的每个数值特征，计算：
- 百分位排名（相对 train 分布的分位，0~1）
- z-score （(x-μ)/σ，μ/σ 取自 train）
- 分位分箱（train 分位切点，test 超界裁剪到端桶）

关键：所有分布统计只在 data_set=='train' 的样本上拟合，再映射到全体，避免泄漏。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig

# 仅对原始数值特征做相对位置，避免对已派生的 rank/zscore 再套娃
_SKIP_SUFFIX = ("_pct", "_z", "_bin")


def generate(feat_df: pd.DataFrame, sample_train_mask: pd.Series, cfg: FeatureConfig
             ) -> Tuple[pd.DataFrame, List[Dict]]:
    feats: Dict[str, pd.Series] = {}
    feat_dict: List[Dict] = []

    num_cols = [c for c in feat_df.columns
                if pd.api.types.is_numeric_dtype(feat_df[c])
                and not c.endswith(_SKIP_SUFFIX)]
    q_grid = np.linspace(0, 1, 101)

    for c in num_cols:
        col = pd.to_numeric(feat_df[c], errors="coerce")
        train_vals = col[sample_train_mask].dropna()
        if train_vals.nunique() < 2:
            continue  # 常数列无相对位置可言

        mu = float(train_vals.mean())
        sigma = float(train_vals.std(ddof=0))
        q_vals = np.quantile(train_vals.to_numpy(), q_grid)

        # 百分位排名：把数值映射到 train 分布的分位
        pct = np.interp(col.to_numpy(dtype=float), q_vals, q_grid,
                        left=0.0, right=1.0)
        feats[f"{c}_pct"] = pd.Series(pct, index=feat_df.index)

        # z-score
        if sigma > 0:
            feats[f"{c}_z"] = (col - mu) / sigma

        # 分位分箱（去重切点，超界裁剪到端桶）
        edges = np.unique(np.quantile(train_vals.to_numpy(),
                                      np.linspace(0, 1, cfg.qcut_bins + 1)))
        if len(edges) >= 3:
            bins = np.clip(np.digitize(col.to_numpy(dtype=float), edges[1:-1]),
                           0, len(edges) - 2)
            feats[f"{c}_bin"] = pd.Series(bins, index=feat_df.index)

        feat_dict += [
            _d(f"{c}_pct", f"{c} 在 train 人群中的百分位(0~1)"),
            _d(f"{c}_z", f"{c} 的 z-score(基于 train 均值方差)"),
            _d(f"{c}_bin", f"{c} 的 train 分位分箱编号"),
        ]

    out = pd.DataFrame(feats, index=feat_df.index)
    # 只保留实际产出的列对应的字典项
    feat_dict = [d for d in feat_dict if d["name"] in out.columns]
    return out, feat_dict


def _d(name: str, desc: str) -> Dict:
    return {"name": name, "family": 3, "family_name": "相对位置", "desc": desc}
