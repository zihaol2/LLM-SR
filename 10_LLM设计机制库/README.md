# 09 — 实名（告诉变量）的物理引导 PDESR

和 08 是同一套东西：原版 PDESR + 物理机制库 + LLM 专家循环（可提新 primitive：
`DIV` 除法、`T` tanh、`coord` 任意坐标函数，仍是固定算子）。

唯一区别：**不匿名**。spec 的提示词和基线 PDESR 一样，明确告诉 LLM 变量是什么
（f1 = 道路上的车辆密度 u，路宽沿 x1 收窄，etc.）。

目录是平的：所有代码、spec、数据都直接放在这个文件夹里，没有 src/ specs/ data/ 子目录。
只有运行结果写进 `results/`（TensorBoard 需要一个目录）。

```bash
pwsh run.ps1 -MaxSamples 100 -Grow
```
