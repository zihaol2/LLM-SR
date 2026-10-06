# 三种 prompt 的对比：实名 / 完全匿名 / 半匿名（少量先验）

同一个问题（交通流瓶颈）、同一份数据、同一个模型（`deepseek-v4-flash`），
只有"告诉模型多少关于变量的信息"在变。本文只做文字对比：先给三种 prompt 的原文，
再逐项说明差别，最后给出四种设定下的结果作为依据。

---

## 一、三种 prompt 的原文

### ① 实名（原版 LLM-PDESR）

文件：`01_原版复现_LLM-PDESR/specs/specification_Traffic_Flow_Bottleneck_numpy.txt`

```text
Find the mathematical function skeleton that represents the time derivative of the
field variable (u_t), given data on spatial coordinates (x) and the field variable
(u) in a traffic flow with a spatial bottleneck and anticipation-driven relative
gradient diffusion.
```

一句话。**学科与现象都点名**（traffic flow、bottleneck、anticipation），
但**不说**这些现象在数学上长什么样——没有讲相对梯度与绝对梯度的区别，
没讲瓶颈剖面是"基准值被削减"还是"剖面本身"，也没讲自限性。

### ② 完全匿名

文件：`01_原版复现_LLM-PDESR/specs/specification_Traffic_Flow_Bottleneck_numpy_anonymous.txt`

```text
Find the mathematical function skeleton that represents the time derivative of the
field variable (u_t), given data on spatial coordinates (x) and the field variable
(u) of an UNKNOWN one-dimensional dynamical system. Nothing about the physical
meaning of u or x is provided on purpose: the symbols carry no interpretation beyond
their numerical values, and the data comes from a numerical simulation and is noise
free.
```

连"有瓶颈"这样的结构都不给。模型只知道：一维、一个场、有时间导数、无噪声。

### ③ 半匿名（少量先验，现在的方法）

文件：`带少量先验的LLMPDESR/code/spec_traffic_flow_bottleneck.txt`

```text
Find the mathematical function skeleton that represents the time derivative of the
field variable (f1_t), given data on the spatial coordinate (x1) and the field
variable (f1) in a one-dimensional transport system with a spatial bottleneck and
anticipation-driven relative gradient diffusion.

Only the subject-area nouns are withheld. The structural prior is kept: f1 is the
transported field; the domain narrows along x1 (a spatial bottleneck); and the response
includes anticipation driven by the field's RELATIVE gradient -- how steep it is
compared with how much of it is there -- not by its absolute slope.

Transport is self-limiting: unobstructed, the field is carried at full speed, and each
of two effects -- the channel narrowing along x1, and the field's own build-up -- can
only cut that speed down, never negative and never reversed. The channel is widest at
one end of x1 and tightens towards the other, so its reduction is a smooth, positive,
one-sided function of x1: a baseline value reduced by a profile, not the profile alone.
Crowding acts through a positive factor of f1 alone. The two reductions do not limit
each other -- each cuts the speed by the same fraction whatever the other is -- so the
field is carried at one speed both apply to together. The data is noise free.
```

**学科名词全部隐去，过程讲清楚**

---

## 二、逐项对比

### 2.1 给了什么、隐去了什么

| 信息 | ① 实名 | ② 完全匿名 | ③ 半匿名 |
|---|---|---|---|
| 变量是什么（车流密度 / 交通） | **给** | 不给 | 不给 |
| "存在空间瓶颈"这一结构 | 给（`bottleneck` 一词） | 不给 | **给（描述为"通道沿 x1 收窄"）** |
| "存在预期/前视效应"这一结构 | 给（`anticipation` 一词） | 不给 | **给，并指明由相对梯度驱动** |
| 符号名 | `u`, `x` | `u`, `x` | `f1`, `x1` |
| 能直接推断学科的词 | 有（traffic flow） | 无 | **无**（traffic / road / vehicle / density 全部 0 次） |
| 先验长度 | 1 句 | 3 句（全是"没有信息"的声明） | **3 段（约 1300 字符）** |

### 2.2 三者的定性差别

- **① 给结论，不给过程。** 这句话等于告诉模型"这是交通流里的方程"，但方程的形态完全靠模型自己联想。
  好处是模型可以调用领域直觉（例如"车流密度有上限"），坏处是这样得到的结果**无法与具体数据对应**——
  它是在回忆交通模型，不是在拟合这条数据。
- **② 什么都不给。** prompt 里能拿到的只有"1D、单场、无噪声"。模型只能靠通用先验
  （扩散、平流、Burgers、多项式反应……）去凑，位置依赖的结构根本不会自己冒出来。
- **③ 不给结论，给过程。** 把"方程该长什么样"的物理约束讲清楚，但不告诉它这是什么学科。
  这样模型既不被学科名牵着走，又知道该往哪些结构上写。**这是唯一一种"信息量与可控性都保留"的设定。**


## 三、四种设定下的结果（交通流瓶颈）

| 方法 | 变量 | 库 | 样本 | 有效 | 最好 MSE |
|---|---|---|---|---|---|
| 基线 实名 | 告知 | 无 | 501 | 389 | 4.51e-07 |
| 有库 实名 | 告知 | 有（6 条，API） | 101 | 93 | 4.90e-11 |
| 基线 匿名 | 隐藏 | 无 | 101 | 92 | 1.34e-04 |
| 有库 匿名 | 隐藏 | 有（9 条，API） | 101 | 95 | 1.05e-04 |
| **基线 半匿名** | 隐藏（给结构） | 无 | 88 | 84 | 3.16e-04 |
| **有库 半匿名** | 隐藏（给结构） | 有（9 条，guidance） | 94 | 88 | **3.13e-11** |

真解档位 = 3.128e-11（弱形式损失地板）。半匿名主臂 88 个有效样本里有 **83 个**落在地板上。

从这张表能读出四件事：

1. **库只有在有结构信息时才起作用。** 同样"加库"：实名时 4.51e-07 → 4.90e-11（约 10⁴ 倍）；
   完全匿名时 1.34e-04 → 1.05e-04（1.27 倍，几乎没用）。
2. **结构信息可以替代变量名。** 半匿名（不给名字、只给结构）+ 库 = 3.13e-11，与实名 + 库同级，
   两者都到了真解档。
3. **光给名字也不够。** 匿名 + 库 的最优式子是 `f²` 代 `f(1−f)`、`sin/cos` 代 `tanh`、
   频率靠猜、且完全缺 `∂ₓ(fₓ/f)`；实名基线虽然结构对了，但前视项写成
   `∂ₓ((1−u)·∂ₓ(u/(1+u)))`，与真解不同形，停在 4.51e-07。
4. **半匿名基线是最差的一档（3.16e-04）。** 既没有名字、也没有结构引导，
   说明 3.13e-11 不是"prompt 变长了"带来的，而是结构先验与机制库共同起的作用。


