# GEM-GSS PDE

图引导的偏微分方程符号发现代码，统一采用 **0.4.9** 实现，发布版本为
`0.4.9.post2`。全部八个案例共用一套源码，不再按案例切换 0.4.7/0.4.9。

保留观测数据校验、冻结初始图、轮转覆盖、历史重组、数值拟合和前沿终选流程。
发布包包含源码、八张初始图和本地 tokenizer，不包含观测数组、实验输出或模型答案。

历史八任务平均结构 Jaccard **0.9375** 来自原归档中的两版实现，
**不是本次统一版本重新运行的成绩**。本次验证使用合成数据与模拟规划；
真实数据和 LLM 实验需要重新运行。原始输入与历史来源哈希保存在
`gem_gss/assets/release_manifest.json`。

## 安装与离线检查

推荐 Python 3.12，数值依赖固定在 `pyproject.toml`。

```bash
python -m pip install -e .
python -B -m gem_gss check
python -B -m gem_gss list
python -B -m unittest discover -s tests -v
```

这些命令不调用 LLM，测试也不需要真实数据或密钥。
安装后可用 `gem-gss-pde` 替代 `python -m gem_gss`。

## 准备数据

观测文件使用 NPZ，键只能包含声明的空间坐标、`t` 和各观测场；
数组顺序为 `(x, [y, z,] t)`。每条空间轴至少六个点，有限、递增且等距；
时间轴至少两个有限递增点，各场必须有限且与网格形状一致。
不接收解析真解或时间导数标签。

从已有观测目录导入并核对历史哈希：

```bash
python tools/import_data.py --source /path/to/observations --output ../GEM-GSS_PDE_data
python -m gem_gss doctor --case T04 --data-dir ../GEM-GSS_PDE_data
```

文件名为 `T01.npz` 等；`--cases T04 T05` 可只导入指定案例。
导入器保留了 `--workspace` 入口，可从原研究目录读取观测文件。

## 运行

在 shell 中设置 `GEM_GSS_API_KEY`、`GEM_GSS_BASE_URL`、`GEM_GSS_MODEL`。
`.env.example` 仅提供示例，不会自动加载。

```bash
python -m gem_gss run --case T04 --data-dir ../GEM-GSS_PDE_data --output ../GEM-GSS_PDE_runs
python -m gem_gss run --case T2D --data-dir ../GEM-GSS_PDE_data --output ../GEM-GSS_PDE_runs --numerical-backend compiled_chain_2d
```

默认使用 `reference` 后端。`compiled_chain_2d` 可显式用于名为 `u` 的二维标量场，
不再依赖旧版本配置或案例名称。固定算子沿用参考特征；参数算子使用链式导数，
奇异/非有限分支回退到参考实现。缓存按观测、实际候选图与数值源码哈希隔离，
默认位于 `.gem_gss_cache/pde049/`，可用 `GEM_GSS_CACHE_DIR` 指定目录。

默认协议为 20 轮、十组初始节点展示两遍，每轮每个输出选择展示组中约 10% 的节点
进行覆盖测试。历史重组每轮**总共最多 25 次拟合，由全部输出共同分享**；
覆盖与在线修正另计。父方程和输出方程最多八项，前沿最多十个模型。
终选只从已拟合前沿中选择，不追加拟合。规划与终选合计最多 21 次 HTTP 请求。

输入压缩默认开启，可用 `--no-compact-prompt` 关闭；旧的半组覆盖策略仍可通过
`--coverage-schedule legacy_half` 显式启用。`--history-fits` 调整每轮共享的历史拟合预算。

运行记录实际源码、数据、图、tokenizer、环境和配置。相同协议和输出目录可续跑；
每轮先冻结数值结果与准确提示词，再发请求，已记录的请求不会自动重发。
协议变化时应使用新输出目录；**旧的 0.4.7/0.4.9 归档运行目录不可直接续跑**。
`--profile` 与一次性后端迁移参数已移除。

## 自定义案例与三维构图

```bash
python -m gem_gss run --case-json case.json --graph graph.json --output runs
```

声明包含 `id`、`fields`、`coordinates`、`prior`、`data_path`，
可包含 `metadata`、`data_sha256`、`graph_path`。相对路径以声明文件为基准。
不指定数据哈希时记录当前文件哈希。自定义案例需提供自己的初始图。

可离线构建 1D/2D/3D 图；以下调用不会访问 API 或观测数据：

```python
from gem_gss.builder import build
from gem_gss.core import dump

seeds = {"L0": [], "G0": []}  # 可填入自己的表达式种子
_, graph = build(seeds, fields=["u"], coordinates=["x", "y", "z"])
dump("graph.json", graph)
```

3D 在线构图新增了方向导数、混合导数、局部变换、通量、调制与统一导数闭包。
冻结图仍保持原始字节及哈希；新构建的图不保证与历史图的节点编号或数量一致。

## 代码结构

| 模块 | 职责 |
|---|---|
| `entry.py`、`cli.py`、`config.py` | 单一入口、路径、运行协议与参数 |
| `grammar.py`、`profile_rules.py` | 表达式语法与执行约束 |
| `builder.py`、`constructors.py`、`deduplicate.py`、`graph.py` | 构图、等价分析与在线节点注册 |
| `search.py` | 初始化、单字段拟合、轮次提交、断点恢复与终选 |
| `hypothesis.py`、`recombine.py` | 覆盖轮转、显式修订、历史重组与探针 |
| `memory.py`、`frontier.py` | 历史档案与受限前沿 |
| `core.py`、`fitting.py`、`chain_numerics.py`、`derivative_program.py` | 观测特征、拟合与数值后端 |
| `refinement.py`、`structural_rules.py` | 在线删项、替换与结构修正 |
| `api.py`、`compression.py`、`wire.py`、`feedback.py`、`final_selection.py` | 请求审计、提示词压缩、反馈与终选 |

`SETTINGS` 只保存运行参数；算法规则与修改记录见本文和 `CHANGELOG.md`。
压缩保留实际节点 ID、系数精度、候选集合与明确的空间导数多重索引，
每次发送前检查精确解码、token 数和提示词哈希。每次请求仍自包含必要规则，
不依赖服务端对话记忆。最终选择提示词保持原版。

开发检查与发布清单更新：

```bash
python -m pip install -e '.[dev]'
python -m black --check .
python -m ruff check .
python tools/release.py
python -m gem_gss check
```

## 来源与许可

源码由提供的 GEM-GSS PDE 研究归档重构。本包没有新增 MIT/Apache 整包许可；
正式开源前应由权利人确认许可。数据和第三方模型权重均未分发。

本地 tokenizer 来自 `deepseek-ai/DeepSeek-V4.1-Flash`，归档 revision 为
`314113f8a7dadb388215396ba5fbee2c610a04d6`，仅用于输入长度计数；其原授权适用。
未压缩 tokenizer JSON 的 SHA-256 为
`c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`。
