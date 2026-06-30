# -*- coding: utf-8 -*-
"""数据质量模块：跑前体检 + 产出"数据可信度"特征。

inspect_data(df, cfg) 返回结构化报告 dict，覆盖：
  - 时间列健康度（解析失败、哨兵值、时间跨度）
  - 未来事件（time > dateback 的流水，未来信息泄露的源头）
  - 重复流水（txn_key 重复）
  - 样本事件覆盖度分布（每样本笔数、稀疏样本数）

compute_coverage_features(df, cfg, base_index) 产出 fq_* 特征，
作为"数据可信度"特征跟着进宽表，建模时常常很有用。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import ensure_datetime
from .config import FeatureConfig


# 哨兵时间值：早于这个/晚于这个的事件视为异常
_SENTINEL_MIN = pd.Timestamp("1990-01-01")
_SENTINEL_MAX = pd.Timestamp("2099-12-31")

# 度量列里这些字符串视为"假空"，转 float 前先统一替换为真正的缺失值
_DIRTY_NULLS = ["", " ", "nan", "NaN", "NAN", "null", "NULL", "None",
                "none", "NaT", "na", "NA", "N/A", "n/a", "-", "--", "\\N"]


def clean_measures(df: pd.DataFrame, measure_cols: List[str]
                   ) -> Tuple[pd.DataFrame, Dict]:
    """把度量列强制转成 float：先把假空字符串('', 'NAN', 'null'…)转真空，再 to_numeric。

    返回 (清洗后 df, 统计 dict)。只动 measure_cols，不碰其它列。
    """
    out = df.copy()
    stats = {"cols": [], "n_to_nan": 0, "n_coerced": 0}
    for c in measure_cols:
        if c not in out.columns:
            continue
        col = out[c]
        before_na = int(col.isna().sum())
        # object/字符串列：先把脏值字符串清成真空
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            col = col.astype(object).where(~col.astype(str).str.strip().isin(_DIRTY_NULLS))
        num = pd.to_numeric(col, errors="coerce")
        after_na = int(num.isna().sum())
        coerced = max(0, after_na - before_na)  # 本次新增的空（脏值+无法解析）
        out[c] = num.astype(float)
        stats["cols"].append({"col": c, "new_null": coerced, "dtype_to": "float64"})
        stats["n_coerced"] += coerced
    return out, stats


def clean_flow(df: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, Dict]:
    """流水清洗编排：按 cfg 开关依次做 NaT 剔除 → txn_key 去重 → 度量列转 float。

    放在打标之后、算时间窗之前调用。返回 (清洗后 df, 清洗统计 dict)。
    """
    out = df.copy()
    n0 = len(out)
    report = {"n_in": n0}

    # 1) 时间解析失败(NaT)的行剔除
    if cfg.filter_nat_time and cfg.time_col and cfg.time_col in out.columns:
        t = ensure_datetime(out[cfg.time_col])
        n_nat = int(t.isna().sum())
        if n_nat:
            out = out.loc[t.notna()].copy()
        report["n_drop_nat"] = n_nat

    # 2) 流水主键去重（保留首条）；txn_key 可为单列或多列(复合)
    if cfg.drop_duplicate_txn and cfg.txn_key:
        tk = [cfg.txn_key] if isinstance(cfg.txn_key, str) else list(cfg.txn_key)
        tk = [c for c in tk if c in out.columns]
        if tk:
            dup = out.duplicated(subset=tk, keep="first")
            n_dup = int(dup.sum())
            if n_dup:
                out = out.loc[~dup].copy()
            report["n_drop_dup"] = n_dup

    # 3) 度量列转 float（脏字符串转真空）
    if cfg.clean_measure_to_float and cfg.measure_cols:
        out, mstats = clean_measures(out, cfg.measure_cols)
        report["measure"] = mstats

    report["n_out"] = len(out)
    return out, report


def inspect_data(df: pd.DataFrame, cfg: FeatureConfig) -> Dict:
    """对原始流水做体检，返回结构化报告 dict（只读不改 df）。"""
    sk = cfg.sample_key
    tc = cfg.time_col
    dbc = cfg.dateback_col
    n = len(df)

    report = {
        "n_rows": int(n),
        "n_samples": int(df[sk].nunique()) if sk in df.columns else 0,
        "time_health": _check_time_health(df, tc) if tc else None,
        "future_events": _check_future_events(df, sk, tc, dbc) if (tc and dbc) else None,
        "duplicates":   _check_duplicates(df, cfg),
        "coverage":     _check_coverage(df, sk, cfg.sparse_sample_threshold),
        "key_health":   _check_key_health(df, cfg),
    }
    return report


# ----------------------------------------------------------------------------
# 各项检查
# ----------------------------------------------------------------------------
def _check_time_health(df: pd.DataFrame, tc: str) -> Dict:
    if tc not in df.columns:
        return {"available": False}
    t = ensure_datetime(df[tc])
    n_nat = int(t.isna().sum())
    valid = t.dropna()
    return {
        "available": True,
        "n_total": int(len(t)),
        "n_nat": n_nat,
        "nat_pct": round(n_nat / max(1, len(t)) * 100, 3),
        "t_min": valid.min().strftime("%Y-%m-%d") if len(valid) else None,
        "t_max": valid.max().strftime("%Y-%m-%d") if len(valid) else None,
        "span_days": int((valid.max() - valid.min()).days) if len(valid) else 0,
        "n_pre_sentinel": int((valid < _SENTINEL_MIN).sum()),
        "n_post_sentinel": int((valid > _SENTINEL_MAX).sum()),
    }


def _check_future_events(df: pd.DataFrame, sk: str, tc: str, dbc: str) -> Dict:
    """time > 样本 dateback 的流水 = 未来信息泄露，必须警惕。"""
    if dbc not in df.columns or tc not in df.columns:
        return {"available": False}
    t = ensure_datetime(df[tc])
    db = ensure_datetime(df[dbc])
    mask = (t.notna() & db.notna() & (t > db))
    n_evt = int(mask.sum())
    if n_evt == 0:
        return {"available": True, "n_events": 0, "n_samples": 0,
                "pct_events": 0.0, "top_samples": []}
    affected = df.loc[mask, sk].value_counts().head(5)
    return {
        "available": True,
        "n_events": n_evt,
        "n_samples": int(df.loc[mask, sk].nunique()),
        "pct_events": round(n_evt / max(1, len(df)) * 100, 3),
        "top_samples": [{"sample": str(k), "future_cnt": int(v)}
                        for k, v in affected.items()],
    }


def _check_duplicates(df: pd.DataFrame, cfg: FeatureConfig) -> Dict:
    """硬重复：txn_key 重复出现。"""
    if not cfg.txn_key or cfg.txn_key not in df.columns:
        return {"available": False, "txn_key_dups": 0}
    dup_mask = df[cfg.txn_key].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    n_groups = int(df.loc[dup_mask, cfg.txn_key].nunique()) if n_dup else 0
    return {
        "available": True,
        "txn_key_dups": n_dup,
        "txn_key_dup_groups": n_groups,
        "pct_events": round(n_dup / max(1, len(df)) * 100, 3),
    }


def _check_coverage(df: pd.DataFrame, sk: str, sparse_thr: int) -> Dict:
    """每样本事件数分布 + 稀疏样本统计。"""
    if sk not in df.columns:
        return {"available": False}
    per_sample = df.groupby(sk).size()
    buckets = {
        "0 笔": int((per_sample == 0).sum()),
        "1 笔": int((per_sample == 1).sum()),
        "2-5 笔": int(((per_sample >= 2) & (per_sample <= 5)).sum()),
        "6-30 笔": int(((per_sample >= 6) & (per_sample <= 30)).sum()),
        "31-100 笔": int(((per_sample >= 31) & (per_sample <= 100)).sum()),
        "100+ 笔": int((per_sample > 100).sum()),
    }
    return {
        "available": True,
        "n_samples": int(len(per_sample)),
        "n_events_total": int(per_sample.sum()),
        "events_per_sample": {
            "min":    int(per_sample.min()),
            "p25":    int(per_sample.quantile(0.25)),
            "median": int(per_sample.median()),
            "mean":   round(float(per_sample.mean()), 2),
            "p75":    int(per_sample.quantile(0.75)),
            "max":    int(per_sample.max()),
        },
        "buckets": buckets,
        "n_sparse": int((per_sample <= sparse_thr).sum()),
        "sparse_threshold": sparse_thr,
    }


def _check_key_health(df: pd.DataFrame, cfg: FeatureConfig) -> Dict:
    """样本主键 + dateback / data_set / label 的一致性。

    同一 sample_key 出现多个不同 dateback / data_set / label 通常是数据错误，
    runner 默认用 .first() 静默处理，这里把它显式报出来。
    """
    sk = cfg.sample_key
    out = {}
    for label, col in [("dateback", cfg.dateback_col),
                       ("data_set", cfg.data_set_col),
                       ("label", cfg.label_col)]:
        if col and col in df.columns:
            n_inconsistent = int(
                (df.groupby(sk)[col].nunique() > 1).sum()
            )
            out[label] = {"col": col, "n_samples_inconsistent": n_inconsistent}
    return out


# ----------------------------------------------------------------------------
# 数据质量特征
# ----------------------------------------------------------------------------
def compute_coverage_features(df: pd.DataFrame, cfg: FeatureConfig,
                              base_index: pd.Index
                              ) -> Tuple[pd.DataFrame, List[Dict]]:
    """产出样本级"数据可信度"特征 fq_*。

    - fq_event_cnt_total: 该样本累计事件笔数（数据有多丰富）
    - fq_has_any_event:   是否有任何事件 (0/1) —— 区分"无数据"与"行为为 0"
    - fq_unique_days:     有事件的不同天数（活跃宽度）
    - fq_days_span:       首末事件天数差（历史长度）
    """
    sk = cfg.sample_key
    feats: Dict[str, pd.Series] = {}
    fdict: List[Dict] = []

    cnt = df.groupby(sk).size().reindex(base_index, fill_value=0)
    feats["fq_event_cnt_total"] = cnt
    feats["fq_has_any_event"] = (cnt > 0).astype(int)

    if cfg.time_col and cfg.time_col in df.columns:
        t = ensure_datetime(df[cfg.time_col])
        valid = df.assign(_t=t).dropna(subset=["_t"])
        # 活跃天数
        days = valid.groupby(sk)["_t"].apply(lambda s: s.dt.normalize().nunique())
        feats["fq_unique_days"] = days.reindex(base_index, fill_value=0).astype(int)
        # 历史跨度
        span = (valid.groupby(sk)["_t"].max() - valid.groupby(sk)["_t"].min()).dt.days
        feats["fq_days_span"] = span.reindex(base_index, fill_value=0).astype(int)

    for name, desc in [
        ("fq_event_cnt_total", "样本累计事件笔数（数据丰富度）"),
        ("fq_has_any_event",   "是否有任何事件 (区分'无数据'与'行为为0')"),
        ("fq_unique_days",     "有事件的不同天数（活跃宽度）"),
        ("fq_days_span",       "首末事件相距天数（历史长度）"),
    ]:
        if name in feats:
            fdict.append({"name": name, "family": -1, "family_name": "数据质量",
                          "desc": desc})

    return pd.DataFrame(feats, index=base_index), fdict


# ----------------------------------------------------------------------------
# 报告渲染（markdown，便于在网页和 CLI 都能直观看）
# ----------------------------------------------------------------------------
def format_report_markdown(report: Dict) -> str:
    """把报告 dict 渲染成可读的 Markdown，给网页 / CLI 复用。"""
    lines = [
        f"**总行数**: {report.get('n_rows', 0):,} | **样本数**: {report.get('n_samples', 0):,}",
        "",
    ]
    th = report.get("time_health") or {}
    if th.get("available"):
        lines += [
            "### ⏱ 时间列健康度",
            f"- 解析失败 (NaT): **{th['n_nat']:,}** ({th['nat_pct']}%)",
            f"- 时间范围: {th['t_min']} ~ {th['t_max']} （跨 {th['span_days']} 天）",
            f"- 哨兵异常: 早于1990年 {th['n_pre_sentinel']}, 晚于2099年 {th['n_post_sentinel']}",
            "",
        ]

    fe = report.get("future_events") or {}
    if fe.get("available"):
        if fe["n_events"] == 0:
            lines += ["### ✅ 未来事件检查", "- 无 `time > dateback` 的流水", ""]
        else:
            lines += [
                "### ⚠️ 未来事件（time > dateback）—— 风险",
                f"- 受影响事件: **{fe['n_events']:,}** ({fe['pct_events']}%)",
                f"- 受影响样本: **{fe['n_samples']:,}**",
                "- top 受影响样本:",
            ]
            for x in fe["top_samples"]:
                lines.append(f"    - `{x['sample']}`: {x['future_cnt']} 笔")
            lines.append("")

    dup = report.get("duplicates") or {}
    if dup.get("available"):
        if dup["txn_key_dups"] == 0:
            lines += ["### ✅ 流水主键重复", "- 无 `txn_key` 重复", ""]
        else:
            lines += [
                "### ⚠️ 流水主键重复",
                f"- 重复事件: **{dup['txn_key_dups']:,}** ({dup['pct_events']}%)，"
                f"涉及 {dup['txn_key_dup_groups']:,} 组",
                "",
            ]

    cov = report.get("coverage") or {}
    if cov.get("available"):
        eps = cov["events_per_sample"]
        lines += [
            "### 📊 样本事件覆盖度",
            f"- 每样本事件数: min={eps['min']} / p25={eps['p25']} / median={eps['median']}"
            f" / mean={eps['mean']} / p75={eps['p75']} / max={eps['max']}",
            f"- **稀疏样本（≤{cov['sparse_threshold']} 笔）: {cov['n_sparse']:,} 个**",
            "",
        ]

    kh = report.get("key_health") or {}
    has_issue = any(v.get("n_samples_inconsistent", 0) > 0 for v in kh.values())
    if kh and has_issue:
        lines += ["### ⚠️ 样本级字段一致性"]
        for k, v in kh.items():
            if v["n_samples_inconsistent"] > 0:
                lines.append(f"- `{v['col']}` 同一样本出现多个不同值: "
                             f"**{v['n_samples_inconsistent']}** 个样本")
        lines.append("")
    return "\n".join(lines)
