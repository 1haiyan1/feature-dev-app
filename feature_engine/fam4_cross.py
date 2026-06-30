# -*- coding: utf-8 -*-
"""Fam4 交叉组合：两个维度叠加后的信号。

- 数值×数值：对 train 方差最大的若干基础特征，两两做比值/差值。
- 类别×类别：维度两两拼接成联合键，统计每样本联合组合的去重数。
- 类别内统计：样本主类别的金额相对该类别 train 均值的偏离。
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import safe_divide, window_slice
from .config import FeatureConfig

_TOP_N_NUMERIC = 5   # 参与数值交叉的基础特征上限，防列数爆炸


def generate(base_feat_df: pd.DataFrame, df: pd.DataFrame, cfg: FeatureConfig,
             train_mask: pd.Series, sample_train_mask: pd.Series
             ) -> Tuple[pd.DataFrame, List[Dict]]:
    sk = cfg.sample_key
    feats: Dict[str, pd.Series] = {}
    feat_dict: List[Dict] = []
    base_index = base_feat_df.index

    # ---- 数值×数值：取 train 方差最大的特征两两交叉 ----
    num_cols = [c for c in base_feat_df.columns
                if pd.api.types.is_numeric_dtype(base_feat_df[c])]
    if num_cols:
        var = base_feat_df.loc[sample_train_mask, num_cols].var().sort_values(ascending=False)
        chosen = list(var.head(_TOP_N_NUMERIC).index)
        for a, b in combinations(chosen, 2):
            rname = f"f4_ratio__{a}__over__{b}"
            feats[rname] = safe_divide(base_feat_df[a], base_feat_df[b])
            feat_dict.append(_d(rname, f"{a} 与 {b} 的比值"))
            dname = f"f4_diff__{a}__minus__{b}"
            feats[dname] = base_feat_df[a].fillna(0) - base_feat_df[b].fillna(0)
            feat_dict.append(_d(dname, f"{a} 与 {b} 的差值"))

    # ---- 类别×类别：联合键去重数 ----
    full = window_slice(df, None)
    for a, b in combinations(cfg.dim_cols, 2):
        joint = full[a].astype(str) + "|" + full[b].astype(str)
        name = f"f4_joint_nunique__{a}__x__{b}"
        feats[name] = joint.groupby(full[sk]).nunique().reindex(base_index, fill_value=0)
        feat_dict.append(_d(name, f"{cfg.col_label(a)}×{cfg.col_label(b)} 联合组合的去重数"))

    # ---- 类别内统计：主类别金额相对 train 类别均值的偏离 ----
    if cfg.measure_cols and cfg.dim_cols:
        m0 = cfg.measure_cols[0]
        train_events = df[train_mask]
        for c in cfg.dim_cols:
            cat_mean = train_events.groupby(c)[m0].mean()              # train 各类别均值
            global_mean = float(train_events[m0].mean())
            # 每样本的主类别（出现最多的类别）与其平均金额
            modal = full.groupby(sk)[c].agg(lambda s: s.astype(str).mode().iloc[0]
                                            if len(s) else np.nan)
            samp_amt = full.groupby(sk)[m0].mean()
            ref = modal.map(cat_mean).fillna(global_mean)
            name = f"f4_within_{c}_dev_{m0}"
            feats[name] = (samp_amt.reindex(base_index) - ref.reindex(base_index))
            feat_dict.append(_d(name, f"样本{cfg.col_label(m0)}均值相对其主{cfg.col_label(c)}类别均值的偏离"))

    out = pd.DataFrame(feats, index=base_index)
    return out, feat_dict


def _d(name: str, desc: str) -> Dict:
    return {"name": name, "family": 4, "family_name": "交叉组合", "desc": desc}
