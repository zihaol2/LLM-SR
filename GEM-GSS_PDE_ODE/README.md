# GEM-GSS：历史 PDE 版本

Archived PDE implementation of graph-guided symbolic discovery. This source-only
release preserves the two historical source profiles behind the eight-task PDE
summary dated 2026-11-15 (mean structural Jaccard **0.9375**).

本目录用于上传 GitHub，共 **50 个文件**。包含算法源码、配置和八张冻结初始图；
不包含观测数组、实验输出、模型答案、API 对话、图片、基线代码或虚拟环境。
它不是后来加入 ODE 构图、每方程 25 次历史拟合或预拟合实验的开发版本。

## 版本范围

93.75% 是八个任务的平均结构 Jaccard，不是八个系统全部恢复的比例，也不是
本次重新运行取得的结果。历史汇总来自以下两个接口版本：

| 自动选择的源码配置 | 案例 | 历史版本 |
|---|---|---|
| `pde047` | T01–T05、T2D | 0.4.7 |
| `pde049` | T3D、TSC1 | 0.4.9 |

两版共享 25 个模块，六个不同文件保存在 `gem_gss/versions/pde047/`。
入口在导入算法前选择对应版本；不是把后来的改动混入旧实验。
`gem_gss/assets/release_manifest.json` 保存原始源码哈希、导出源码哈希、
数据/初始图哈希、每题协议及打包改动。没有保存方程结果或评估答案。

保持原算法的构图规则、轮转、历史搜索、前沿、数值拟合和最终选择。
发布封装只处理命令入口、数据目录、缓存目录和源码核验；后来的按方程测试记录、
LLM 满项替换修复、ODE 新图和各种预拟合原型均不在本包。

## 安装与无数据检查

推荐 Python 3.12。依赖按历史数值环境固定在 `pyproject.toml` 中。

```bash
python -m pip install -e .
python -B -m gem_gss check
python -B -m gem_gss list
python -B -m unittest discover -s tests -v
```

这些检查不调用 LLM。测试使用临时合成观测和模拟规划回复，不需要真实数据或密钥。
也可使用安装后的命令 `gem-gss-pde` 替代 `python -m gem_gss`。

## 观测数据单独存放

输入为 observation-only NPZ：坐标数组、`t`、各观测场，不包含解析真解或时间导数标签。
文件名为 `T01.npz` 等；原数据哈希保存在发布清单中，运行时核对。
数据不随代码分发，没有数据也能安装、检查源码和运行离线测试。

如果你保留了原本的本地研究目录，可以从本目录执行：

```bash
python tools/import_data.py --workspace .. --output ../GEM-GSS_PDE_data
python -B -m gem_gss doctor --case T04 --data-dir ../GEM-GSS_PDE_data
```

如果已经另有八个按案例命名的 NPZ 文件：

```bash
python tools/import_data.py --source /path/to/observations --output ../GEM-GSS_PDE_data
```

导入器核对哈希，并拒绝覆盖不同数据；不读取或复制任何运行结果。
`--cases T04 T05` 可以只导入选定案例。这里的工作区导入是可选便利入口，
算法运行不依赖原研究目录。

## 运行

在环境变量中设置 `GEM_GSS_API_KEY`、`GEM_GSS_BASE_URL`、`GEM_GSS_MODEL`。
`.env.example` 仅是示例，不会自动加载。

```bash
python -m gem_gss run --case T04 --data-dir ../GEM-GSS_PDE_data --output ../GEM-GSS_PDE_runs
python -m gem_gss run --case T2D --data-dir ../GEM-GSS_PDE_data --output ../GEM-GSS_PDE_runs
```

自动恢复该案例的历史源码配置、紧凑提示词、`global_1pct` 覆盖协议和数值后端。
T2D 使用原 `compiled_chain_2d`，其他案例使用 `reference`。
磁盘缓存默认写入当前目录的 `.gem_gss_cache/`；可用 `GEM_GSS_CACHE_DIR` 改到外部目录。
不同源码配置使用不同缓存子目录。需要手动选择版本时使用 `--profile pde047` 或 `pde049`；
一个 Python 进程内不切换已导入的源码配置。

**历史预算**：20 轮；每轮展示一个初始节点组，共十组、两遍；每个输出从展示组中选择
约 10% 进行覆盖拟合。历史重组每轮 **总计 25 次，多个输出共同分享**，
不是后来 ODE 版本的每输出 25 次。覆盖与在线删项/简化拟合另计。
最终 LLM 从已有前沿模型中选择，不进行终端新拟合。最多八项、前沿最多十个模型。

运行会发起付费 API 请求并把输出写到指定目录；打包检查和离线测试不会。
结果目录不在本发布包内。不要把历史原始运行目录作为新封装的续跑目录；
此封装记录了新的入口/路径元数据，不修改原存档协议来强行续跑。

## 代码阅读顺序

| 层 | 文件 | 职责 |
|---|---|---|
| 入口与配置 | `entry.py`、`cli.py`、`config.py` | 源码配置、数据路径、命令和运行协议 |
| 表达式与图 | `grammar.py`、`profile_rules.py`、`builder.py`、`constructors.py`、`deduplicate.py`、`graph.py` | 语法、有限构图、等价去重、节点注册 |
| 搜索循环 | `search.py`、`hypothesis.py`、`rotation.py`、`recombine.py` | 覆盖提案、四类历史重组、执行顺序 |
| 模型保留 | `memory.py`、`frontier.py` | 全历史档案与最多十个前沿模型 |
| 数值计算 | `core.py`、`fitting.py`、`numerics.py` | 观测/窗口特征、变量投影与拟合 |
| 二维后端 | `chain_numerics.py`、`derivative_program.py`、`fused_numerics.py` | 历史数值后端；公开入口使用已归档的 reference/compiled 配置 |
| 在线修正 | `refinement.py`、`structural_rules.py`、`structure.py` | 删项、多项式替换、已有核心简化 |
| LLM 交互 | `api.py`、`compression.py`、`wire.py`、`feedback.py`、`final_selection.py` | 请求、压缩、执行反馈和终选 |

初始图是构图产物/运行输入，不是拟合结果；图中没有正确性标签。
本包不提供离线真解评估器，也不将评估参考方程导入发现过程。
Jaccard 的历史记录可定位版本，不能保证再次调用不确定性 LLM 得到完全相同结果。

## 自定义案例

使用 `--case-json case.json --graph graph.json.gz`。声明中包含 `id`、`fields`、
`coordinates`、`prior`、`data_path` 和 `data_sha256`，数据路径相对声明文件。
图使用同一语法，由 `builder.Graph`/`Registry` 等现有接口构建；新增案例需提供自己的
初始图，不自动借用其他案例的图。入口默认使用 `pde049`，支持现有 1D/2D/3D 观测适配。

## 来源与许可说明

源码由本地 GEM-GSS PDE 研究归档整理，原模块哈希见发布清单。
没有另行替作者或第三方授予整包 MIT/Apache 许可证。
观测数据及第三方基线没有随包分发，使用者需要自行准备有权使用的数据。

tokenizer 来自 `deepseek-ai/DeepSeek-V4.1-Flash`，归档 revision 为
`314113f8a7dadb388215396ba5fbee2c610a04d6`，只用于本地输入长度计数。
压缩前 tokenizer JSON SHA-256：
`c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`。
其原授权适用；本包不分发模型权重。第三方 Python 库通过依赖安装，不复制其源码。
