# -*- coding: utf-8 -*-
"""编排器：按配置依次跑各家族，产出样本级特征宽表 + 特征字典。

数据流：
  原始流水 --(加 _days_ago)--> Fam0/1/2 (样本级基础特征)
        基础特征 --> Fam3 相对位置 / Fam4 交叉 / Fam5 编码
        全部 concat --> 宽表(每 sample_key 一行) + data_set + label
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import (fam0_event_agg, fam1_time_dynamics, fam2_structure, fam4_cross)
from .base import (add_window_helpers, apply_tags, compute_obs_point,
                   ensure_datetime, get_train_mask)
from .config import FeatureConfig
from .quality import clean_flow


def run(df_raw: pd.DataFrame, cfg: FeatureConfig
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (特征宽表, 特征字典表)。"""
    _validate(df_raw, cfg)
    df = df_raw.copy()

    # ---- 复合主键：把多列样本主键拼成单一内部键 _skey，输出时再还原 ----
    # sample_key 可为 str 或 list[str]。列表时内部 groupby 用 _skey，
    # 末尾把 _skey 拆回原始几列。原始组件列名记在 skey_parts。
    skey_parts = ([cfg.sample_key] if isinstance(cfg.sample_key, str)
                  else list(cfg.sample_key))
    composite_key = len(skey_parts) > 1
    if composite_key:
        df["_skey"] = _join_key(df, skey_parts)
        cfg = replace(cfg, sample_key="_skey")
    sk = cfg.sample_key

    # ---- 流水打标（按业务规则）----
    # 打完会在 df 上多一列 cfg.tag_col；该列被并入维度列参与 Fam0/2/4/5。
    if cfg.tag_rules:
        df = apply_tags(df, cfg.tag_rules, tag_col=cfg.tag_col,
                        default_tag=cfg.default_tag)
        if cfg.tag_col not in cfg.dim_cols:
            cfg = _copy_cfg_with_dim(cfg, cfg.tag_col)

    # ---- 流水清洗（NaT剔除 / txn_key去重 / 度量列转float）----
    # 放在打标之后、算时间窗之前。清洗报告挂到 df.attrs 供网页展示。
    df, clean_report = clean_flow(df, cfg)
    df.attrs["clean_report"] = clean_report

    # ---- 时间与窗口准备 ----
    if cfg.time_col:
        df[cfg.time_col] = ensure_datetime(df[cfg.time_col])
        obs = compute_obs_point(df, sk, cfg.time_col, cfg.dateback_col,
                                cfg.update_policy, cfg.update_lag_days,
                                cfg.obs_point)
        df = add_window_helpers(df, sk, cfg.time_col, obs)
    else:
        df["_days_ago"] = 0.0  # 无时间列：所有事件视为同一时点

    # ---- 掩码与索引 ----
    train_mask = get_train_mask(df, cfg.data_set_col, cfg.train_value)  # 事件级
    base_index = pd.Index(df[sk].dropna().unique(), name=sk)
    sample_train_mask = (train_mask.groupby(df[sk]).max()
                         .reindex(base_index, fill_value=False).astype(bool))

    families = cfg.enabled_families()
    parts: List[pd.DataFrame] = []
    feat_dict: List[Dict] = []

    # ---- 基础家族 0/1/2 ----
    if 0 in families:
        f, d = fam0_event_agg.generate(df, cfg, train_mask)
        parts.append(f.reindex(base_index)); feat_dict += d
    if 1 in families:
        f, d = fam1_time_dynamics.generate(df, cfg, train_mask)
        parts.append(f.reindex(base_index)); feat_dict += d
    if 2 in families:
        f, d = fam2_structure.generate(df, cfg, train_mask)
        parts.append(f.reindex(base_index)); feat_dict += d

    base_feat_df = (pd.concat(parts, axis=1) if parts
                    else pd.DataFrame(index=base_index))

    # ---- 交叉组合 4 (依赖基础特征) ----
    if 4 in families and len(base_feat_df.columns):
        f, d = fam4_cross.generate(base_feat_df, df, cfg, train_mask, sample_train_mask)
        parts.append(f); feat_dict += d

    wide = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=base_index)
    wide = wide.loc[:, ~wide.columns.duplicated()]

    # ---- 拼回样本级 data_set / label ----
    wide.insert(0, sk, base_index)
    if cfg.data_set_col in df.columns:
        ds = df.groupby(sk)[cfg.data_set_col].first().reindex(base_index)
        wide[cfg.data_set_col] = ds.values
    if cfg.label_col and cfg.label_col in df.columns:
        lb = df.groupby(sk)[cfg.label_col].first().reindex(base_index)
        wide[cfg.label_col] = lb.values

    wide = wide.reset_index(drop=True)

    # ---- 特征名后缀（多策略对比时用）----
    if cfg.feature_suffix:
        protected = {sk, cfg.data_set_col, cfg.label_col}
        rename = {c: c + cfg.feature_suffix for c in wide.columns
                  if c not in protected}
        wide = wide.rename(columns=rename)
        for item in feat_dict:
            item["name"] = item["name"] + cfg.feature_suffix

    # ---- 复合主键还原：_skey 拆回原始几列，放到最前 ----
    if composite_key and "_skey" in wide.columns:
        parts_df = df[["_skey"] + skey_parts].drop_duplicates("_skey").set_index("_skey")
        comp = parts_df.reindex(wide["_skey"].values).reset_index(drop=True)
        wide = pd.concat([comp[skey_parts], wide.drop(columns=["_skey"])], axis=1)

    fdict = _build_feature_dict(feat_dict, wide, cfg, sample_train_mask, base_index)
    return wide, fdict


def _validate(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    sk_parts = ([cfg.sample_key] if isinstance(cfg.sample_key, str)
                else list(cfg.sample_key))
    for c in sk_parts:
        if c not in df.columns:
            raise ValueError(f"样本主键 {c!r} 不在数据列中")
    for c in cfg.dim_cols + cfg.measure_cols + cfg.distinct_cols:
        if c not in df.columns:
            raise ValueError(f"配置列 {c!r} 不在数据列中")
    if 5 in cfg.enabled_families() and not cfg.label_col:
        raise ValueError("Fam5 类别编码需要 label 列，请配置 label_col 或关闭 Fam5")


def _join_key(df: pd.DataFrame, parts: List[str]) -> pd.Series:
    """把多列拼成单一字符串键（用 \\x1f 分隔，避免与正常取值冲突）。"""
    out = df[parts[0]].astype(str)
    for c in parts[1:]:
        out = out.str.cat(df[c].astype(str), sep="\x1f")
    return out


def _sample_label(df: pd.DataFrame, sk: str, label_col: str,
                  base_index: pd.Index) -> pd.Series:
    y = df.groupby(sk)[label_col].first().reindex(base_index)
    return pd.to_numeric(y, errors="coerce")


def _copy_cfg_with_dim(cfg: FeatureConfig, extra_dim: str) -> FeatureConfig:
    """返回一份把 extra_dim 加进 dim_cols 的新 cfg（不修改原 cfg）。"""
    from dataclasses import replace
    new_dims = list(cfg.dim_cols) + [extra_dim]
    return replace(cfg, dim_cols=new_dims)


def _build_feature_dict(feat_dict: List[Dict], wide: pd.DataFrame,
                        cfg: FeatureConfig, sample_train_mask: pd.Series,
                        base_index: pd.Index) -> pd.DataFrame:
    """给每个特征补 dtype / 缺失率 / 全样本均值标准差，方便查看与做数据字典。"""
    rows = []
    for item in feat_dict:
        name = item["name"]
        if name not in wide.columns:
            continue
        col = pd.to_numeric(wide[name], errors="coerce")
        rows.append({
            "特征名": name,
            "家族": f"{item['family']} {item['family_name']}",
            "含义": item["desc"],
            "dtype": str(wide[name].dtype),
            "缺失率": round(float(wide[name].isna().mean()), 4),
            "均值": round(float(col.mean()), 4) if col.notna().any() else np.nan,
            "标准差": round(float(col.std()), 4) if col.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)
