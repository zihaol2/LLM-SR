# GEM-GSS PDE

图引导的偏微分方程符号发现代码,任务平均结构 Jaccard **0.9375** 来自原归档中的两版实现，

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


## 运行

在 shell 中设置 `GEM_GSS_API_KEY`、`GEM_GSS_BASE_URL`、`GEM_GSS_MODEL`。
`.env.example` 仅提供示例，不会自动加载。

```bash
python -m gem_gss run --case T04 --data-dir ../GEM-GSS_PDE_data --output ../GEM-GSS_PDE_runs
python -m gem_gss run --case T2D --data-dir ../GEM-GSS_PDE_data --output ../GEM-GSS_PDE_runs --numerical-backend compiled_chain_2d
```



默认协议为 20 轮、十组初始节点展示两遍，每轮每个输出选择展示组中约 10% 的节点
进行覆盖测试。历史重组每轮**总共最多 25 次拟合，由全部输出共同分享**；
覆盖与在线修正另计。父方程和输出方程最多八项，前沿最多十个模型。
终选只从已拟合前沿中选择，不追加拟合。

输入压缩token默认开启，可用 `--no-compact-prompt` 关闭；




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




