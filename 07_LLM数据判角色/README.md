# 07 — 只给数据，让 LLM 判断变量角色

独立实验：只把数据的统计证据喂给 LLM，让它以物理
专家身份从统计规律直接推断每个场（f1, f2, ...）代表什么物理量。

- `roles_probe.py`：对 5 个方程（Topography_Chemotaxis / Morphogenesis /
  Forced_Swift_Hohenberg / Traffic_Flow_Bottleneck / Predator_Prey）各做 5 次
  （类似 5 个种子），默认模型 `deepseek-v4-flash`。
  原始回答写入 `roles_raw.json`，整理后的对照表写入 `roles_summary.md`。


数据来自 `原版复现的LLM-PDES`。

运行：

```bash
python roles_probe.py                       # 5 个方程 × 5 次，deepseek-v4-flash
python roles_probe.py deepseek-v4-flash 3   # 改成 3 次
```
