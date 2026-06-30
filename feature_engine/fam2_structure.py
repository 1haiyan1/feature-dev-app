# -*- coding: utf-8 -*-
"""Fam2 结构比例：钱花在哪、有多集中。

占比（某类别金额/该维度总金额）、强度比（客单价=金额/笔数）、
集中度（HHI、基尼系数、top1 占比、活跃类别数）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import gini, hhi, safe_divide, window_slice
from .config import FeatureConfig


def generate(df: pd.DataFrame, cfg: FeatureConfig, train_mask: pd.Series
             ) -> Tuple[pd.DataFrame, List[Dict]]:
    sk = cfg.sample_key
    feats: Dict[str, pd.Series] = {}
    feat_dict: List[Dict] = []

    full = window_slice(df, None)
    base_index = full.groupby(sk).size().index
    m0 = cfg.measure_cols[0] if cfg.measure_cols else None

    # ---- 强度比：客单价（首个度量金额 / 笔数）----
    if m0:
        amt = full.groupby(sk)[m0].sum()
        cnt = full.groupby(sk).size()
        name = f"f2_{m0}_per_event"
        feats[name] = safe_divide(amt, cnt)
        feat_dict.append(_d(name, f"单笔平均{m0}（客单价）"))

    # ---- 按维度：占比 + 集中度 ----
    for c in cfg.dim_cols:
        # 每样本-每类别的金额（无度量则用笔数）
        if m0:
            cell = full.groupby([sk, c])[m0].sum()
        else:
            cell = full.groupby([sk, c]).size()
        cell = cell.rename("v").reset_index()

        # 集中度指标（对每个样本的类别分布计算）
        def _conc(grp: pd.DataFrame) -> pd.Series:
            vals = grp["v"].to_numpy(dtype=float)
            total = vals.sum()
            top1 = (vals.max() / total) if total > 0 else 0.0
            return pd.Series({
                f"f2_{c}_hhi": hhi(vals),
                f"f2_{c}_gini": gini(vals),
                f"f2_{c}_top1_share": top1,
                f"f2_{c}_active_cnt": float(len(vals)),
            })

        conc = cell.groupby(sk).apply(_conc, include_groups=False)
        for col in conc.columns:
            feats[col] = conc[col].reindex(base_index)
        feat_dict += [
            _d(f"f2_{c}_hhi", f"{c} 金额集中度 HHI（越大越集中）"),
            _d(f"f2_{c}_gini", f"{c} 金额基尼系数"),
            _d(f"f2_{c}_top1_share", f"{c} 占比最高类别的份额"),
            _d(f"f2_{c}_active_cnt", f"{c} 活跃类别数"),
        ]

        # top-K 类别各自的占比（份额）；打标列用规则名全集，普通列用 train top-K
        if c == cfg.tag_col and cfg.tag_rules:
            tag_vals = [str(r.get("name")) for r in cfg.tag_rules if r.get("name")]
            cat_iter = tag_vals + [cfg.default_tag]
        else:
            cat_iter = list(df[train_mask][c].astype(str).value_counts()
                            .head(cfg.top_k_categories).index)
        for cat in cat_iter:
            cat_amt = cell[cell[c].astype(str) == str(cat)].set_index(sk)["v"]
            total_amt = cell.groupby(sk)["v"].sum()
            name = f"f2_share_{c}={cat}"
            feats[name] = safe_divide(cat_amt.reindex(base_index, fill_value=0),
                                      total_amt.reindex(base_index, fill_value=0))
            feat_dict.append(_d(name, f"{c}={cat} 占该维度总额的比例"))

    out = pd.DataFrame(feats)
    return out, feat_dict


def _d(name: str, desc: str) -> Dict:
    return {"name": name, "family": 2, "family_name": "结构比例", "desc": desc}
