# 消融：基线 PDESR × 同样的半先验 prompt

## 这个臂要回答什么

把主臂（`../code`）用的那段半先验 prompt **原封不动**交给基线 PDESR，但不给它
设计好的机制库，也不告诉它变量角色。看基线自己能做到什么程度。

所以两个臂的 prompt **只差一段**：主臂在 spec 前面多了
`### LIBRARY GUIDANCE (...)` 那一块（约 9.8k 字符）；这个臂什么都没有。

两个臂的 spec 是**同一个字节**：

| 文件 | sha1 |
|---|---|
| `../code/spec_traffic_flow_bottleneck.txt` | `5d0c79a019e995778fbb1119ec070dcaa00012dd` |
| `spec_baseline_prior.txt`（本目录） | `5d0c79a019e995778fbb1119ec070dcaa00012dd` |

先验文本本身只说结构（沿 x1 收窄的通道、由相对梯度驱动的预期、自限性运输），
不含任何学科名词，也不出现 traffic / road / vehicle / density。这一点两个臂一致。

## 每次运行都会落一份实际发出去的 prompt

`<LogDir>/search_prompt.txt` 写的是这次运行**第一个**样本收到的完整 prompt。要验证
"只差库那一段"，直接 diff：

```powershell
# 主臂（带库）
diff ..\code\..\results\search_prompt.txt .\results\search_prompt.txt
```

## 运行

```powershell
cd 消融_基线_同prompt
pwsh run.ps1 -MaxSamples 50
```

不传 `--design_library`、不传 `--design_from`，所以库为空、搜索提示里没有 guidance。
`RoleIdentify` 默认关：没有库设计这一步就没有统计量读出的 Evidence 1，角色识别那边
只剩方程与母体，证据不完整，放在这里比较意义不大。要看也可以：

```powershell
pwsh run.ps1 -MaxSamples 50 -RoleIdentify true
```

## 对比时要注意

命中率这东西方差很大。同一个库、同一份 prompt，16 号里的三次自由搜索分别在第 3、
第 18、以及 12 个样本内**没**命中真解。要下结论，两个臂的样本数要一致，最好各跑几遍。

---

## 结果：配对的 100 样本对比（2026-10-05）

两个臂同时跑，各 100 样本预算、各一次，**先验文本相同、只有库这一项不同**。

| | 消融：无库 | 主臂：带库（`../../results_main`） |
|---|---|---|
| 预算 / 实际落盘 | 100 / 88 | 100 / 94 |
| 有效样本 | 84 | 88 |
| 平流项写成散度 `∂ₓ(f1·speed)` | **0（0%）** | **87（99%）** |
| 落到地板（<1e-6） | **0** | **83** |
| 中位 MSE | 3.637e-04 | **3.128e-11** |
| 最好 MSE | 3.158e-04 | **3.128e-11** |

主臂的 rank 1 就是真解（拟合系数 `[0.5, 1.0, 1.0, 0.05, 1.0]`）：

```python
geometry  = 1.0 - params[0] * np.tanh(params[1] * x[:, 0:1])
crowding  = 1.0 - params[2] * f1
flux      = f1 * (geometry * crowding)
advection = -d_axis(flux, 0, 1)
f1_t_pred = advection + params[3] * d_axis(d_axis(f1, 0, 1) / f1, 0, 1)
```

角色识别给出的母体是 **Lighthill-Whitham-Richards（含瓶颈与预期项）**，
变量 traffic density，置信度 high。

消融臂的 rank 1（MSE 3.158e-04, complexity 25）：

```python
channel_profile = np.tanh(x[:, 0:1])
transport = -params[0] * (1 - params[1] * channel_profile) * d_axis(f1, 0, 1) / (1 + params[2] * f1)
relative_gradient = d_axis(f1, 0, 1) / (f1 + params[3])
f1_t_pred = transport - params[4] * d_axis(relative_gradient, 0, 1)
```

平流是 `<因子>·∂ₓf1`（非守恒），通量里没有 `f1` 因子；剖面用的是裸 `tanh`。
**84 个有效样本无一例外**，所以这一臂不是"没搜够"，是候选空间里从来没有
`∂ₓ(f1·speed)` 这个形状。

> 第一次运行（50 预算 / 41 样本，最好 2.800e-04）归档在 `results_run1_budget50/`，
> 结论相同。

## prompt 一致性的实测

主臂那次运行的 `../../results_main/search_prompt.txt` 是它**第一个**样本收到的完整
prompt。把它里面那段 `### LIBRARY GUIDANCE ...` 到 `### Current Pareto Frontier`
之前的内容删掉，得到的就是本目录的 `results/search_prompt.txt`，**逐字节相同**：

| | 字符数 |
|---|---|
| 主臂第一个 prompt | 14079 |
| 其中库 guidance 段 | 7056（占 50%） |
| 删掉后 | 7023 |
| 消融臂 prompt | 7023 |

两份 prompt 都不含 traffic / road / vehicle / density 等学科名词。
