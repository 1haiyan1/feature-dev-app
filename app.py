# -*- coding: utf-8 -*-
"""特征开发网页 —— Streamlit 入口（纯代码生成版）。

定位：配置器。手填列名与参数 → 生成一份独立可跑的 `generated_features.py`，
拿到你自己的环境/Spark 上跑真实数据。网页本身不读数据、不跑特征。

流程：填列名/中文释义 → 选维度/度量/窗口/家族/清洗 → 下载脚本（含数据字典）。
"""
import pandas as pd
import streamlit as st

from feature_engine.codegen import generate_code
from feature_engine.config import DEFAULT_AGGS, DEFAULT_WINDOWS, FeatureConfig

st.set_page_config(page_title="特征开发网页 · 代码生成", layout="wide")
st.title("🧬 特征开发网页 · 代码生成")
st.caption("填列名 / 维度 / 度量 / 窗口 / 家族 / 清洗 → 生成独立可跑的特征衍生 Python 脚本。"
           "**网页不读数据、不跑特征**，脚本拿去自己环境跑，会同时产出特征宽表与数据字典。")


# ---------------------------------------------------------------------------
# 侧栏：衍生参数
# ---------------------------------------------------------------------------
st.sidebar.header("1. 衍生参数")
win_txt = st.sidebar.text_input(
    "时间窗口（天，逗号分隔）", value="7,30,90,180",
    help="自定义任意天数，必须是数字，用英文逗号分隔。如 7,30,90。每个窗口都会回看对应天数的流水做聚合。")
# 解析 + 数字校验
_raw = [x.strip() for x in win_txt.split(",") if x.strip()]
_bad = [x for x in _raw if not x.isdigit() or int(x) <= 0]
if _bad:
    st.sidebar.error(f"窗口必须是正整数，这些不对：{'、'.join(_bad)}")
windows = sorted({int(x) for x in _raw if x.isdigit() and int(x) > 0}) or list(DEFAULT_WINDOWS)

top_k = st.sidebar.slider(
    "每维度 top-K 类别", 1, 10, 5,
    help="每个维度列只取出现最频繁的前 K 个类别，单独衍生它们的笔数/金额特征。"
         "K 越大覆盖的类别越多、特征列也越多；建议 3~5，避免长尾类别产生大量稀疏列。")

# ---- 更新策略：决定截至哪天的流水可用（仅日更）----
st.sidebar.header("2. 更新策略 (基于 dateback)")
preset = st.sidebar.selectbox(
    "可用性时延（日更）", ["T+0 即时", "T+1", "T+2", "T+3", "T+7", "T+N 自定义"], index=1,
    help="观察点 = dateback - N 天。决定特征「截至哪一天」的流水可用。")
if preset.startswith("T+0"):
    update_lag_days = 0
elif preset == "T+1":
    update_lag_days = 1
elif preset == "T+2":
    update_lag_days = 2
elif preset == "T+3":
    update_lag_days = 3
elif preset == "T+7":
    update_lag_days = 7
else:  # T+N 自定义
    update_lag_days = st.sidebar.slider("自定义时延天数 N", 0, 90, 1)
update_policy = "T+N"

feature_suffix = st.sidebar.text_input(
    "特征名后缀 (可空)", value="",
    help="多策略对比时建议填，如 _T1 / _T2；两次生成后两套列可直接 join 不冲突。")

st.sidebar.header("3. 家族开关")
st.sidebar.caption("勾选要生成的特征家族，鼠标悬停看每族做什么。")
_FAM_HELP = {
    0: "事件聚合：每个时间窗内，按样本聚合事件笔数、各度量列的 sum/mean/max/min/std、"
       "维度列的去重类别数，并对 top-K 类别 pivot 出分类笔数与金额。回答「某类事件有多少」。",
    1: "时间动态：跨窗口比值（近短期 vs 近长期）、按周斜率（趋势升降）、变异系数（稳定性）、"
       "指数衰减加权事件数、recency（距最近/最早事件天数）。回答「趋势在变好还是变坏」。",
    2: "结构比例：各类别金额占比、客单价（金额/笔数）、集中度（HHI、基尼、top1 占比、活跃类别数）。"
       "回答「钱花在哪、有多集中」。",
    4: "交叉组合：方差最大的基础特征两两比值/差值；维度两两拼成交叉派生维度（如 渠道×商户类型→线上_餐饮），"
       "对 top-K 组合按窗口算笔数与金额合计；样本金额相对其主类别均值的偏离。回答「两维叠加后的信号」。",
}
fam_flags = {
    0: st.sidebar.checkbox("Fam0 事件聚合", True, help=_FAM_HELP[0]),
    1: st.sidebar.checkbox("Fam1 时间动态", True, help=_FAM_HELP[1]),
    2: st.sidebar.checkbox("Fam2 结构比例", True, help=_FAM_HELP[2]),
    4: st.sidebar.checkbox("Fam4 交叉组合", True, help=_FAM_HELP[4]),
}

st.sidebar.header("4. 流水清洗（写进脚本）")
clean_measure = st.sidebar.checkbox("度量列转 float（脏字符串转空）", True,
                                    help="把度量列强制转数值，'' / 'NAN' / 'null' / '-' 等脏字符串先转为真空。")
drop_dup = st.sidebar.checkbox("按流水主键去重", True,
                               help="按流水主键（可复合）保留首条，去掉重复流水。")
filter_nat = st.sidebar.checkbox("剔除时间解析失败的行", True,
                                 help="时间列无法解析成日期（NaT）的流水行直接剔除。")


# ---------------------------------------------------------------------------
# 主区：列名映射
# ---------------------------------------------------------------------------
st.subheader("① 列名映射（按你的数据表填）")
st.caption("主键支持**复合键**：多列用英文逗号分隔，如 `mobile,idcard`。其余列单列填写。")
c1, c2, c3 = st.columns(3)
sk = c1.text_input("样本主键 sample_key", "user_id",
                   help="实体级，输出一行一个；可复合，逗号分隔多列")
tk = c2.text_input("流水主键 txn_key", "txn_id",
                   help="事件唯一标识，去重用；可复合，逗号分隔多列")
tc = c3.text_input("流水时间列 time_col", "txn_time", help="事件发生日期")
dbc = st.text_input("样本回溯日期 dateback", "dateback",
                    help="样本级决策日/观察日；窗口右端点 = dateback - 时延")

# ---- 维度列 + 中文释义 ----
st.markdown("**维度列（分类，可聚合）** — 填列名与中文释义")
if "dim_rows" not in st.session_state:
    st.session_state["dim_rows"] = [
        {"列名": "merchant_type", "中文释义": "商户类型"},
        {"列名": "channel", "中文释义": "渠道"},
        {"列名": "region", "中文释义": "地区"},
    ]
dim_edit = st.data_editor(
    pd.DataFrame(st.session_state["dim_rows"]), num_rows="dynamic",
    use_container_width=True, key="dim_editor",
    column_config={
        "列名": st.column_config.TextColumn("列名", required=True, help="数据表里的列名"),
        "中文释义": st.column_config.TextColumn("中文释义", help="用于特征描述与数据字典，可空"),
    })

# ---- 度量列 + 中文释义 ----
st.markdown("**度量列（数值）** — 填列名与中文释义")
if "meas_rows" not in st.session_state:
    st.session_state["meas_rows"] = [{"列名": "amount", "中文释义": "交易金额"}]
meas_edit = st.data_editor(
    pd.DataFrame(st.session_state["meas_rows"]), num_rows="dynamic",
    use_container_width=True, key="meas_editor",
    column_config={
        "列名": st.column_config.TextColumn("列名", required=True, help="数据表里的列名"),
        "中文释义": st.column_config.TextColumn("中文释义", help="用于特征描述与数据字典，可空"),
    })

# ---- 去重计数列 + 中文释义 ----
st.markdown("**去重计数列** — 算窗口内唯一值个数（如机构数、设备数、城市数）")
if "dist_rows" not in st.session_state:
    st.session_state["dist_rows"] = [{"列名": "org_id", "中文释义": "机构"}]
dist_edit = st.data_editor(
    pd.DataFrame(st.session_state["dist_rows"]), num_rows="dynamic",
    use_container_width=True, key="dist_editor",
    column_config={
        "列名": st.column_config.TextColumn("列名", required=True, help="如 org_id / device_id"),
        "中文释义": st.column_config.TextColumn("中文释义", help="如 机构 / 设备，可空"),
    })


def _rows_to_cols_aliases(edited):
    """data_editor 结果 -> (列名列表, {列名:中文释义})。"""
    cols, aliases = [], {}
    if isinstance(edited, pd.DataFrame):
        for r in edited.to_dict("records"):
            name = str(r.get("列名") or "").strip()
            if not name:
                continue
            cols.append(name)
            alias = str(r.get("中文释义") or "").strip()
            if alias:
                aliases[name] = alias
    return cols, aliases


dim_cols, dim_aliases = _rows_to_cols_aliases(dim_edit)
measure_cols, meas_aliases = _rows_to_cols_aliases(meas_edit)
distinct_cols, dist_aliases = _rows_to_cols_aliases(dist_edit)
col_aliases = {**dim_aliases, **meas_aliases, **dist_aliases}


def _parse_key(text, default):
    """逗号分隔 -> 单列(str) 或 复合键(list)。空则用 default。"""
    parts = [x.strip() for x in (text or "").split(",") if x.strip()]
    if not parts:
        return default
    return parts[0] if len(parts) == 1 else parts


def build_cfg() -> FeatureConfig:
    return FeatureConfig(
        sample_key=_parse_key(sk, "sample_key"), txn_key=_parse_key(tk, None),
        time_col=tc or None, dateback_col=dbc or None,
        dim_cols=list(dim_cols), measure_cols=list(measure_cols),
        distinct_cols=list(distinct_cols),
        col_aliases=dict(col_aliases),
        windows=list(windows) or list(DEFAULT_WINDOWS),
        aggs=list(DEFAULT_AGGS), top_k_categories=top_k,
        update_policy=update_policy, update_lag_days=update_lag_days,
        feature_suffix=feature_suffix,
        clean_measure_to_float=clean_measure, drop_duplicate_txn=drop_dup,
        filter_nat_time=filter_nat,
        fam0_event_agg=fam_flags[0], fam1_time_dynamics=fam_flags[1],
        fam2_structure=fam_flags[2], fam3_position=False,
        fam4_cross=fam_flags[4], fam5_encoding=False,
    )


cfg = build_cfg()

# ---------------------------------------------------------------------------
# 生成代码
# ---------------------------------------------------------------------------
st.subheader("② 生成 Python 脚本")

_clean_bits = [b for b, on in [("度量转float", clean_measure), ("去重", drop_dup),
                               ("剔NaT", filter_nat)] if on]
st.info(f"⏱ 更新策略：**T+{update_lag_days}（观察点 = dateback - {update_lag_days} 天）** ｜ "
        f"窗口 {windows} ｜ 家族 {sum(fam_flags.values())} 个 ｜ "
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
if _warn:
    st.warning("配置还差：**" + "、".join(_warn) + "**。脚本仍可生成，但跑之前请补全。")

code = generate_code(cfg, input_path="your_data.csv")
n_lines = code.count("\n") + 1
st.success(f"✅ 独立可跑脚本：{n_lines} 行 / {len(code)//1024} KB，**只依赖 pandas + numpy**。"
           "改脚本顶部 `INPUT_PATH` 指向你的数据，`python generated_features.py` 即可。"
           "跑完产出**特征宽表** + **数据字典**（特征名/家族/含义/dtype/缺失率/均值/标准差）。")
st.download_button("⬇ 下载 generated_features.py", code.encode("utf-8"),
                   "generated_features.py", "text/x-python", type="primary")
with st.expander("👀 预览代码", expanded=False):
    st.code(code, language="python")
