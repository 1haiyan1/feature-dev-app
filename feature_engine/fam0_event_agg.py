# -*- coding: utf-8 -*-
"""Fam0 事件聚合：窗口 × 筛选 × 度量 × 聚合。

回答"这个人某类事件有多少 / 多大"。
对每个时间窗口，按样本主键聚合事件数、各度量列的 sum/mean/max/min/std，
各维度列的去重类别数；并对每个维度的 top-K 类别 pivot 出"分类计数/分类金额"。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import resolve_categories, window_slice
from .config import FeatureConfig

# 各算子对应的 pandas 聚合函数
_MEASURE_AGG_FUNC = {
    "sum": "sum", "mean": "mean", "max": "max", "min": "min", "std": "std",
}


def generate(df: pd.DataFrame, cfg: FeatureConfig, train_mask: pd.Series
             ) -> Tuple[pd.DataFrame, List[Dict]]:
    sk = cfg.sample_key
    feats: Dict[str, pd.Series] = {}
    feat_dict: List[Dict] = []

    train_df = df[train_mask]
    # 预先确定每个维度列的取值集合，保证 train/test 列对齐：
    # - 打标列：用规则名 + default_tag 全集
    # - 普通维度列：用 train 上 top-K
    tag_values = ([str(r.get("name")) for r in (cfg.tag_rules or []) if r.get("name")]
                  + [cfg.default_tag]) if cfg.tag_rules else None
    top_cats = {
        c: resolve_categories(train_df, c, cfg.top_k_categories,
                              fixed_values=tag_values if c == cfg.tag_col else None)
        for c in cfg.dim_cols
    }

    windows: List = list(cfg.windows) + [None]  # None = 全历史
    for w in windows:
        tag = "all" if w is None else f"{w}d"
        win = window_slice(df, w)
        g = win.groupby(sk)

        # 事件计数
        if "count" in cfg.aggs:
            name = f"f0_cnt_{tag}"
            feats[name] = g.size()
            feat_dict.append(_d(name, f"近{tag}事件笔数"))

        # 度量聚合
        for m in cfg.measure_cols:
            for agg in cfg.aggs:
                func = _MEASURE_AGG_FUNC.get(agg)
                if func is None:
                    continue
                name = f"f0_{m}_{agg}_{tag}"
                feats[name] = g[m].agg(func)
                feat_dict.append(_d(name, f"近{tag} {cfg.col_label(m)} 的{agg}"))

        # 去重计数列：窗口内唯一值个数（如机构数、设备数）
        for c in cfg.distinct_cols:
            name = f"f0_{c}_nunique_{tag}"
            feats[name] = g[c].nunique()
            feat_dict.append(_d(name, f"近{tag} {cfg.col_label(c)} 的唯一值个数"))

        # 维度 top-K 类别 pivot：分类计数 + 分类金额
        for c in cfg.dim_cols:
            for cat in top_cats[c]:
                sub = win[win[c].astype(str) == cat]
                gsub = sub.groupby(sk)
                cname = f"f0_cnt_{tag}__{c}={cat}"
                feats[cname] = gsub.size()
                feat_dict.append(_d(cname, f"近{tag} {cfg.col_label(c)}={cat} 的笔数"))
                # 若有第一个度量列，再出该类别的金额和
                if cfg.measure_cols:
                    m0 = cfg.measure_cols[0]
                    aname = f"f0_{m0}sum_{tag}__{c}={cat}"
                    feats[aname] = gsub[m0].sum()
                    feat_dict.append(_d(aname, f"近{tag} {cfg.col_label(c)}={cat} 的{cfg.col_label(m0)}合计"))

    out = pd.DataFrame(feats)
    # 计数类缺失补 0；统计类（mean/std/max/min）保留 NaN 交由下游处理
    cnt_cols = [c for c in out.columns if "_cnt_" in c or c.endswith("_sum_all")
                or "_sum_" in c or "_nunique_" in c]
    out[cnt_cols] = out[cnt_cols].fillna(0)
    return out, feat_dict


def _d(name: str, desc: str) -> Dict:
    return {"name": name, "family": 0, "family_name": "事件聚合", "desc": desc}
