# -*- coding: utf-8 -*-
"""特征引擎配置对象。

把"网页上的所有配置"收敛到一个 FeatureConfig 数据类里，
让特征家族、runner、codegen 三处共用同一份配置，避免参数到处传。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


# 支持的聚合算子（事件聚合 Fam0 使用）
DEFAULT_AGGS = ["count", "sum", "mean", "max", "min", "std", "nunique"]

# 默认时间窗口（单位：天）。观察点向前回看 N 天。
DEFAULT_WINDOWS = [7, 30, 90, 180]


# 一条打标规则的形态：
#   {"name": "大额", "query": "amount >= 1000"}  -> pandas.eval 查询串
#   规则列表顺序即优先级，单标·首条命中；未命中归 default_tag (默认 'other')。
TagRule = Dict[str, str]


@dataclass
class FeatureConfig:
    """一次特征衍生任务的全部配置。"""

    # ---- 主键与核心列 ----
    sample_key: Union[str, List[str]]    # 样本主键（实体级，可复合：多列联合）
    txn_key: Optional[Union[str, List[str]]] = None  # 流水主键（去重用，可复合）
    time_col: Optional[str] = None       # 流水时间列（事件发生日期）
    dateback_col: Optional[str] = None   # 样本回溯日期列（决策日/观察日，样本级）
    data_set_col: str = "data_set"       # train/test 划分列
    label_col: Optional[str] = None      # 标签列（类别编码 Fam5 必需，二分类 0/1）

    # ---- 参与衍生的列 ----
    dim_cols: List[str] = field(default_factory=list)      # 可聚合的分类维度列
    measure_cols: List[str] = field(default_factory=list)  # 数值度量列

    # ---- 衍生参数 ----
    windows: List[int] = field(default_factory=lambda: list(DEFAULT_WINDOWS))
    aggs: List[str] = field(default_factory=lambda: list(DEFAULT_AGGS))
    top_k_categories: int = 5            # 每个维度列取频次最高的前 K 个类别做 pivot
    qcut_bins: int = 10                  # 相对位置 Fam3 分位分箱桶数
    target_smoothing: float = 20.0       # 目标编码平滑系数（先验权重）

    # 观察点策略（仅在 dateback_col 缺失时生效）：
    # "per_sample_max" = 每样本最后一笔时间；或固定全局日期字符串 (如 "2024-06-30")
    obs_point: str = "per_sample_max"

    # ---- 更新策略：决定"截至哪一天的流水可用" ----
    # update_policy: "T+N"  -> 观察点 = dateback - update_lag_days
    #                "monthly" -> 观察点 = dateback 当月的上月最后一天
    # T+0/即时 = T+N 且 update_lag_days=0
    update_policy: str = "T+N"
    update_lag_days: int = 1

    # 特征名后缀（多策略对比时给本次跑的所有特征列加个后缀，如 "_T1" / "_T2" / "_MoM"）
    feature_suffix: str = ""

    # ---- 流水打标（业务标签）----
    # tag_rules：规则列表，每条形如 {"name": "大额", "query": "amount >= 1000"}。
    # 单标·首条命中；未命中归 default_tag。打完会在流水里新增一列 tag_col。
    # 该列会自动并入 dim_cols 参与 Fam0/2/4/5 的衍生，且取值集合 = 规则名∪{default_tag}。
    tag_rules: List[Dict] = field(default_factory=list)
    tag_col: str = "_tag"
    default_tag: str = "other"

    # ---- 数据质量 ----
    # 跑特征前自动做一次质量检查，并按下面开关对流水做剔除：
    # - filter_future_events: 事件时间晚于该样本 dateback 的流水（"未来信息泄露"）剔除
    # - filter_nat_time:      time_col 解析失败成 NaT 的行剔除
    # - drop_duplicate_txn:   按 txn_key 去重（若提供了 txn_key）
    # - emit_quality_features: 产出样本级数据质量特征 fq_event_cnt_total / fq_has_any_event 等
    filter_future_events: bool = True
    filter_nat_time: bool = True
    drop_duplicate_txn: bool = True
    emit_quality_features: bool = True
    # - clean_measure_to_float: 把度量列强制转 float，脏字符串('', 'NAN', 'null' 等)先转真空
    clean_measure_to_float: bool = True
    sparse_sample_threshold: int = 2   # 低于该笔数的样本视为"稀疏"，质量报告会标记

    # ---- 家族开关 ----
    fam0_event_agg: bool = True
    fam1_time_dynamics: bool = True
    fam2_structure: bool = True
    fam3_position: bool = True
    fam4_cross: bool = True
    fam5_encoding: bool = True

    # ---- 数据集取值 ----
    train_value: str = "train"
    test_value: str = "test"

    # 家族编号 -> 开关属性名
    FAMILY_FLAGS = {
        0: "fam0_event_agg",
        1: "fam1_time_dynamics",
        2: "fam2_structure",
        3: "fam3_position",
        4: "fam4_cross",
        5: "fam5_encoding",
    }

    def enabled_families(self) -> List[int]:
        """返回开启的家族编号列表，如 [0, 1, 2]。"""
        return [i for i, flag in self.FAMILY_FLAGS.items() if getattr(self, flag)]
