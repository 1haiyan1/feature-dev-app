# -*- coding: utf-8 -*-
"""代码生成：把当前配置 + 整个引擎源码导出成一份独立可跑的 pandas 脚本。

设计目标：
- 生成的文件是**单一 .py**，只依赖 pandas / numpy，**不依赖本项目任何模块**。
- 把 feature_engine/ 下的全部源码按依赖顺序内联进来，自动保持与引擎本体一致。
- 顶部放可读的 CONFIG 字典，方便二次调参；底部 main() 直接读 CSV/Parquet 跑特征。
"""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from .config import FeatureConfig

_ENGINE_DIR = Path(__file__).parent

# 按依赖顺序拼装：config -> base -> fam0..fam5 -> runner
_FILE_ORDER = [
    ("config", "config.py"),
    ("base",   "base.py"),
    ("quality", "quality.py"),
    ("fam0",   "fam0_event_agg.py"),
    ("fam1",   "fam1_time_dynamics.py"),
    ("fam2",   "fam2_structure.py"),
    ("fam4",   "fam4_cross.py"),
    ("runner", "runner.py"),
]

# runner 里 `fam0_event_agg.generate(...)` -> `fam0_generate(...)`
_MODULE_REWRITES = {
    "fam0_event_agg.generate":  "fam0_generate",
    "fam1_time_dynamics.generate": "fam1_generate",
    "fam2_structure.generate":  "fam2_generate",
    "fam4_cross.generate":      "fam4_generate",
}

_FAM_DESC = {
    0: "事件聚合（窗口×筛选×度量×聚合）",
    1: "时间动态（跨窗口比值、斜率、衰减、recency）",
    2: "结构比例（占比、HHI、基尼、客单价）",
    4: "交叉组合（数值×数值、类别联合、类别内偏离）",
}


def _strip_module_header(src: str) -> str:
    """删除单文件顶部的编码声明、shebang、import / from 语句、模块级 docstring。

    每个源文件原本各自有自己的 imports；合并后由生成文件顶部统一一份。
    """
    lines = src.splitlines()
    out = []
    i = 0
    # 跳过编码声明、shebang、模块 docstring（位于所有 import 之前）
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1; continue
        if s.startswith('"""') or s.startswith("'''"):
            quote = s[:3]
            # 单行 docstring
            if s.count(quote) >= 2 and len(s) > 3:
                i += 1; continue
            # 多行 docstring，跳到结束
            i += 1
            while i < len(lines) and quote not in lines[i]:
                i += 1
            i += 1
            continue
        break

    # 余下部分：再过滤掉所有 import / from 语句（含多行括号形式）
    j = 0
    rest = lines[i:]
    while j < len(rest):
        s = rest[j].lstrip()
        if s.startswith("from __future__"):
            j += 1; continue
        if s.startswith("import ") or s.startswith("from "):
            # 多行 import：吃到右括号闭合（或当行就闭合）
            paren = rest[j].count("(") - rest[j].count(")")
            j += 1
            while paren > 0 and j < len(rest):
                paren += rest[j].count("(") - rest[j].count(")")
                j += 1
            continue
        out.append(rest[j])
        j += 1
    return "\n".join(out).strip("\n")


def _rename_generate(src: str, fam_id: str) -> str:
    """fam* 模块里的 `def generate(` 改为 `def famN_generate(`，便于内联后无歧义。"""
    return re.sub(r"^def generate\(", f"def {fam_id}_generate(", src, flags=re.MULTILINE)


def _rename_private_helpers(src: str, fam_id: str) -> str:
    """每个 fam 文件里都有同名的 `_d(...)` 私有 helper，内联到同一模块后会相互覆盖。
    这里给每个文件的 `_d` 加上 fam 后缀，彻底避免冲突。"""
    new_name = f"_d_{fam_id}"
    # 定义处
    src = re.sub(r"^def _d\(", f"def {new_name}(", src, flags=re.MULTILINE)
    # 调用处（` _d(` 或 `\n_d(` 或行首 `_d(`）
    src = re.sub(r"(?<![A-Za-z0-9_])_d\(", f"{new_name}(", src)
    return src


def _rewrite_module_refs(src: str) -> str:
    """runner.py 里 `fam0_event_agg.generate(...)` 改为 `fam0_generate(...)`。"""
    for old, new in _MODULE_REWRITES.items():
        src = src.replace(old, new)
    return src


def _bundle_engine() -> str:
    """读出 feature_engine/ 全部源码，剥导入、改名、合并成一段可粘贴的代码。"""
    parts = []
    for fam_id, fname in _FILE_ORDER:
        src = (_ENGINE_DIR / fname).read_text(encoding="utf-8")
        src = _strip_module_header(src)
        if fam_id.startswith("fam"):
            src = _rename_generate(src, fam_id)
            src = _rename_private_helpers(src, fam_id)
        if fam_id == "runner":
            src = _rewrite_module_refs(src)
        parts.append(f"# ============================================================\n"
                     f"#  {fname}\n"
                     f"# ============================================================\n"
                     f"{src.strip()}")
    return "\n\n\n".join(parts)


def _format_config(cfg: FeatureConfig) -> str:
    """把 FeatureConfig 序列化成可读的字面量字典代码。"""
    lines = ["CONFIG = {"]
    for k, v in asdict(cfg).items():
        lines.append(f"    {k!r}: {v!r},")
    lines.append("}")
    return "\n".join(lines)


def generate_code(cfg: FeatureConfig, input_path: str = "your_data.csv",
                  output_path: str = "features.parquet",
                  dict_path: str = "features_dict.csv") -> str:
    """生成完整独立的 pandas 脚本字符串。"""
    cfg_block = _format_config(cfg)
    fam_notes = "\n".join(f"#   - Fam{i}: {_FAM_DESC[i]}"
                          for i in cfg.enabled_families() if i in _FAM_DESC)
    engine = _bundle_engine()
    return f'''# -*- coding: utf-8 -*-
"""自动生成的特征衍生脚本 —— 独立可跑，仅依赖 pandas / numpy。

本脚本由「特征开发网页」导出，内联了完整特征引擎代码，
**不依赖任何外部自定义模块**，可单文件发给他人直接运行。

启用的特征家族：
{fam_notes}

更新策略：{cfg.update_policy}（lag_days={cfg.update_lag_days}）
说明：均为纯统计类特征，不依赖样本划分与标签。

用法：
    pip install pandas numpy
    python {Path(output_path).stem}.py
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# 配置（可直接修改下面参数二次开发）
# ============================================================================
{cfg_block}

INPUT_PATH = r"{input_path}"
OUTPUT_PATH = r"{output_path}"
DICT_PATH = r"{dict_path}"


{engine}


# ============================================================================
# 主入口
# ============================================================================
def _load(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    df = _load(INPUT_PATH)
    cfg = FeatureConfig(**CONFIG)
    wide, feat_dict = run(df, cfg)
    print(f"特征宽表: {{wide.shape[0]}} 行 x {{wide.shape[1]}} 列")
    print(feat_dict.head(20).to_string(index=False))
    if OUTPUT_PATH.lower().endswith(".parquet"):
        wide.to_parquet(OUTPUT_PATH, index=False)
    else:
        wide.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"已保存特征宽表 -> {{OUTPUT_PATH}}")
    # 数据字典：特征名 / 家族 / 含义 / dtype / 缺失率 / 均值 / 标准差
    dict_path = DICT_PATH
    feat_dict.to_csv(dict_path, index=False, encoding="utf-8-sig")
    print(f"已保存数据字典 -> {{dict_path}}（共 {{len(feat_dict)}} 个特征）")


if __name__ == "__main__":
    main()
'''
