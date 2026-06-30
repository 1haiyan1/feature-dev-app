# 特征开发网页 · 代码生成 (Feature Dev Web Tool)

把"流水（事件流）→ 样本级特征宽表"的衍生逻辑配置化：在网页上填列名、选维度/度量/窗口/特征家族/清洗策略，
一键生成一份**独立可跑的 Python 脚本** `generated_features.py`，拿到自己的环境/Spark 上跑真实数据。

> 定位：**配置器 + 代码生成器**。网页本身不读数据、不跑特征，只产出脚本。

## 在线使用

部署后访问对应网址即可，无需安装。

## 本地运行

```powershell
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

## 使用流程

1. **填列名映射**：样本主键 / 流水主键 / 时间列 / 回溯日期 dateback / data_set / label。
   - 主键支持**复合键**：多列用英文逗号分隔，如 `mobile,idcard`。
2. **填维度列、度量列**（逗号分隔）。
3. 左侧栏选：时间窗口、特征家族开关、更新策略（T+N / 月更）、流水清洗开关。
4. （可选）展开"流水打标规则"，按业务定义标签作为维度列。
5. 点 **下载 generated_features.py**。

## 生成脚本怎么跑

脚本是单文件、**只依赖 pandas + numpy**，不依赖本项目任何模块：

```powershell
pip install pandas numpy
# 改脚本顶部 INPUT_PATH 指向你的数据文件
python generated_features.py
```

## 特征家族

| 家族 | 回答什么 | 做法 |
|---|---|---|
| 0 事件聚合 | 某类事件有多少 | 窗口×筛选×度量×聚合 |
| 1 时间动态 | 趋势变好还是变坏 | 跨窗口比值、周斜率、衰减加权、recency |
| 2 结构比例 | 钱花在哪、多集中 | 占比、客单价、HHI、基尼、top1 占比 |
| 3 相对位置 | 人群里排第几 | train 分位排名、z-score、分位分箱 |
| 4 交叉组合 | 两维叠加的信号 | 数值×数值、类别联合、类别内偏离 |
| 5 类别编码 | 类别变判别力 | 频率/目标/WOE 编码（含 IV） |

## 关键设计

- **防数据泄漏**：所有"看分布"的统计（分位、z-score、排名、WOE、目标编码）只在 `data_set=='train'` 上拟合，再映射到全体。
- **可用性时延**：用 `dateback - 时延` 作窗口右端点，支持 T+0 / T+1 / T+N / 月更，贴近生产"数据何时可用"。
- **复合主键**：样本主键 / 流水主键均支持多列联合，内部拼键计算、输出还原原始列。
- **日期归一**：`2024/1/5`、`2024.03.18`、`20240422`、`2024年5月20日`、Excel 序列号等统一解析，假空转空。
- **流水清洗**：度量列强制转 float（脏字符串转空）、按流水主键去重、剔除时间解析失败行——清洗逻辑一并写进导出脚本。

## 目录结构

```
feature_dev_app/
  app.py                  # Streamlit 入口（配置 + 生成代码）
  feature_engine/
    config.py             # FeatureConfig 配置对象
    base.py               # 公共工具 + train 拟合框架 + 日期解析
    quality.py            # 流水清洗（度量转float / 去重 / 剔NaT）
    fam0..fam5_*.py       # 各特征家族实现
    runner.py             # 编排 -> 宽表 + 特征字典
    codegen.py            # 导出独立可复现脚本
  requirements.txt
```

## 部署到 Streamlit Community Cloud

1. 把本文件夹推到 GitHub 仓库（可私有）。
2. 登录 https://share.streamlit.io ，用 GitHub 授权。
3. 选仓库 → 主文件填 `app.py` → Deploy。
4. 几分钟后得到公网网址 `https://<name>.streamlit.app`。
