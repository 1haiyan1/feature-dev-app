# -*- coding: utf-8 -*-
"""特征开发网页 —— Streamlit 入口（纯代码生成版）。

定位：配置器。手填列名与参数 → 生成一份独立可跑的 `generated_features.py`，
拿到你自己的环境/Spark 上跑真实数据。网页本身不读数据、不跑特征。

流程：填列名 → 选维度/度量/窗口/家族/更新策略/清洗 → 下载 Python 脚本。
"""
import pandas as pd
import streamlit as st

from feature_engine.codegen import generate_code
from feature_engine.config import DEFAULT_AGGS, DEFAULT_WINDOWS, FeatureConfig

st.set_page_config(page_title="特征开发网页 · 代码生成", layout="wide")
st.title("🧬 特征开发网页 · 代码生成")
st.caption("填列名 / 维度 / 度量 / 窗口 / 家族 / 清洗 → 生成独立可跑的特征衍生 Python 脚本。"
           "**网页不读数据、不跑特征**，脚本拿去自己环境跑。")


# ---------------------------------------------------------------------------
# 侧栏：衍生参数
# ---------------------------------------------------------------------------
st.sidebar.header("1. 衍生参数")
windows = st.sidebar.multiselect("时间窗口（天）", [7, 14, 30, 60, 90, 180, 365],
                                 default=DEFAULT_WINDOWS)
top_k = st.sidebar.slider("每维度 top-K 类别", 1, 10, 5)
qcut_bins = st.sidebar.slider("相对位置分箱数", 4, 20, 10)

# ---- 更新策略：决定截至哪天的流水可用 ----
st.sidebar.header("2. 更新策略 (基于 dateback)")
preset = st.sidebar.selectbox(
    "可用性时延", ["T+0 即时", "T+1", "T+2", "T+3", "T+7 周更近似", "T+N 自定义",
                   "月更 (上月底之前可用)"], index=1,
    help="观察点 = dateback - 时延。决定特征「截至哪一天」的流水。")
if preset.startswith("T+0"):
    update_policy, update_lag_days = "T+N", 0
elif preset == "T+1":
    update_policy, update_lag_days = "T+N", 1
elif preset == "T+2":
    update_policy, update_lag_days = "T+N", 2
elif preset == "T+3":
    update_policy, update_lag_days = "T+N", 3
elif preset.startswith("T+7"):
    update_policy, update_lag_days = "T+N", 7
elif preset.startswith("T+N"):
    update_policy = "T+N"
    update_lag_days = st.sidebar.slider("自定义时延天数 N", 0, 90, 1)
else:  # 月更
    update_policy, update_lag_days = "monthly", 0

feature_suffix = st.sidebar.text_input(
    "特征名后缀 (可空)", value="",
    help="多策略对比时建议填，如 _T1 / _T2 / _MoM；两次生成后两套列可直接 join 不冲突。")

st.sidebar.header("3. 家族开关")
fam_flags = {
    0: st.sidebar.checkbox("Fam0 事件聚合", True),
    1: st.sidebar.checkbox("Fam1 时间动态", True),
    2: st.sidebar.checkbox("Fam2 结构比例", True),
    3: st.sidebar.checkbox("Fam3 相对位置", True),
    4: st.sidebar.checkbox("Fam4 交叉组合", True),
    5: st.sidebar.checkbox("Fam5 类别编码", True),
}

st.sidebar.header("4. 流水清洗（写进脚本）")
clean_measure = st.sidebar.checkbox("度量列转 float（脏字符串转空）", True)
drop_dup = st.sidebar.checkbox("按流水主键去重", True)
filter_nat = st.sidebar.checkbox("剔除时间解析失败的行", True)


# ---------------------------------------------------------------------------
# 主区：手填列名
# ---------------------------------------------------------------------------
st.subheader("① 列名映射（按你的数据表填）")
st.caption("主键支持**复合键**：多列用英文逗号分隔，如 `mobile,idcard`。其余列单列填写。")
c1, c2, c3 = st.columns(3)
sk = c1.text_input("样本主键 sample_key", "user_id",
                   help="实体级，输出一行一个；可复合，逗号分隔多列")
tk = c2.text_input("流水主键 txn_key", "txn_id",
                   help="事件唯一标识，去重用；可复合，逗号分隔多列")
tc = c3.text_input("流水时间列 time_col", "txn_time", help="事件发生日期")
c4, c5, c6 = st.columns(3)
dbc = c4.text_input("样本回溯日期 dateback", "dateback", help="样本级决策日/观察日")
dsc = c5.text_input("数据集划分列 data_set", "data_set", help="train/test/oot")
lbl = c6.text_input("标签列 label", "label", help="Fam5 需要，二分类 0/1")

dims_txt = st.text_input("维度列（分类，逗号分隔）", "merchant_type,channel,region")
meas_txt = st.text_input("度量列（数值，逗号分隔）", "amount")

dim_cols = [x.strip() for x in dims_txt.split(",") if x.strip()]
measure_cols = [x.strip() for x in meas_txt.split(",") if x.strip()]


def _parse_key(text, default):
    """逗号分隔 -> 单列(str) 或 复合键(list)。空则用 default。"""
    parts = [x.strip() for x in (text or "").split(",") if x.strip()]
    if not parts:
        return default
    return parts[0] if len(parts) == 1 else parts

# ---- 流水打标规则（作为维度列写进脚本）----
with st.expander("🏷 流水打标规则（按业务定义标签，作为维度列写进脚本）", expanded=False):
    st.caption("按顺序匹配，单标·首条命中；未命中归 `other`。query 用 pandas.eval 语法 "
               "（列名直接写不加引号，支持 `>=` `<` `==` `in` `&` `|`）。")
    if "tag_rules" not in st.session_state:
        st.session_state["tag_rules"] = [
            {"name": "大额", "query": "amount >= 500"},
            {"name": "中额", "query": "amount >= 100"},
            {"name": "小额", "query": "amount < 100"},
        ]
    rules_df = pd.DataFrame(st.session_state["tag_rules"])
    edited = st.data_editor(
        rules_df, num_rows="dynamic", use_container_width=True, key="rules_editor",
        column_config={
            "name":  st.column_config.TextColumn("标签名", required=True),
            "query": st.column_config.TextColumn("query 表达式", required=True,
                                                  width="large"),
        },
    )
    if isinstance(edited, pd.DataFrame):
        clean = [r for r in edited.to_dict("records")
                 if str(r.get("name") or "").strip() and str(r.get("query") or "").strip()]
        st.session_state["tag_rules"] = clean
    if st.button("清空规则"):
        st.session_state["tag_rules"] = []
        st.rerun()


def build_cfg() -> FeatureConfig:
    return FeatureConfig(
        sample_key=_parse_key(sk, "sample_key"), txn_key=_parse_key(tk, None),
        time_col=tc or None,
        dateback_col=dbc or None, data_set_col=dsc or "data_set", label_col=lbl or None,
        dim_cols=list(dim_cols), measure_cols=list(measure_cols),
        windows=list(windows) or list(DEFAULT_WINDOWS),
        aggs=list(DEFAULT_AGGS), top_k_categories=top_k, qcut_bins=qcut_bins,
        update_policy=update_policy, update_lag_days=update_lag_days,
        feature_suffix=feature_suffix,
        tag_rules=list(st.session_state.get("tag_rules", [])),
        clean_measure_to_float=clean_measure, drop_duplicate_txn=drop_dup,
        filter_nat_time=filter_nat,
        fam0_event_agg=fam_flags[0], fam1_time_dynamics=fam_flags[1],
        fam2_structure=fam_flags[2], fam3_position=fam_flags[3],
        fam4_cross=fam_flags[4], fam5_encoding=fam_flags[5],
    )


cfg = build_cfg()

# ---------------------------------------------------------------------------
# 生成代码
# ---------------------------------------------------------------------------
st.subheader("② 生成 Python 脚本")

_lag_hint = ("月更（取 dateback 当月的上月底）" if update_policy == "monthly"
             else f"T+{update_lag_days}（观察点 = dateback - {update_lag_days} 天）")
_clean_bits = [b for b, on in [("度量转float", clean_measure), ("去重", drop_dup),
                               ("剔NaT", filter_nat)] if on]
st.info(f"⏱ 更新策略：**{_lag_hint}** ｜ 家族 {sum(fam_flags.values())} 个 ｜ "
        f"清洗：{('、'.join(_clean_bits) or '无')}"
        + (f" ｜ 后缀 `{feature_suffix}`" if feature_suffix else ""))

# 轻量完整度提示（不阻断生成）
_warn = []
if not cfg.sample_key or cfg.sample_key == "sample_key":
    _warn.append("样本主键")
if not cfg.time_col:
    _warn.append("流水时间列（Fam0/1/2 需要）")
if not cfg.dim_cols and not cfg.measure_cols:
    _warn.append("至少 1 个维度列或度量列")
if 5 in cfg.enabled_families() and not cfg.label_col:
    _warn.append("Fam5 需要 label 列（或关闭 Fam5）")
if _warn:
    st.warning("配置还差：**" + "、".join(_warn) + "**。脚本仍可生成，但跑之前请补全。")

code = generate_code(cfg, input_path="your_data.csv")
n_lines = code.count("\n") + 1
st.success(f"✅ 独立可跑脚本：{n_lines} 行 / {len(code)//1024} KB，"
           "**不依赖本项目任何模块**，只要 pandas + numpy。"
           "改脚本顶部 `INPUT_PATH` 指向你的数据，`python generated_features.py` 即可。")
st.download_button("⬇ 下载 generated_features.py", code.encode("utf-8"),
                   "generated_features.py", "text/x-python", type="primary")
with st.expander("👀 预览代码", expanded=False):
    st.code(code, language="python")
