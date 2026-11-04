# GEM-GSS 中文使用说明

这是当前方法的独立整理版。旧实验目录依赖已移除；合并了重复终选、剪枝和历史交叉模块。
包内保留六题观测、原初始图、初始化提案、tokenizer、精简结果和离线测试。
不包含真实密钥、API历史全文、旧检查点、角色识别或未完成的三维搜索。

## 安装与无API检查

需要 Python 3.12 或更新版本，仅使用CPU。

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-repro.txt
python -m pip install -e .
python -m gem_gss doctor --verify-results
python -m unittest discover -s tests -v
```

仅安装 `pip install -e .` 也可运行；复现实验数值时推荐固定版本。
检查命令不调用API；冻结方程回放不重新拟合参数。模拟闭环测试中的LLM请求全部被替换。

## 正式搜索

```powershell
$env:GEM_GSS_API_KEY = "你的密钥"
$env:GEM_GSS_BASE_URL = "https://你的接口/v1"
$env:GEM_GSS_MODEL = "你的模型名"
python -m gem_gss run --case T01 --output runs/standard
```

六题ID：T01、T02、T03、T04、T05、T2D。分别启动命令，不在同一进程并行改全局设置。
默认20轮，每轮一次API，另一次最终选择，共21次；使用缓存初始图，不额外初始化。
模型和接口必须显式设置，程序不从旧工程查找密钥，也不会自动读取 `.env`。
原模型别名 `deepseek-flash-1017` 是否有效由接口商决定，当前清理包没有调用新API验证接口。

相同命令可续跑；已保留的请求不会自动重发。异常终止留下锁文件时，应先确认旧进程已停止，
再手动移除该次运行的 `run.lock`。更换代码、配置或数值后端使用新输出目录。

## 当前方法

1. 初始图按通用构图和带定义域条件的去重规则建立。
2. 初始可选机制固定分十组、访问两遍；每轮至少半数进入拟合尝试，第二遍优先补测。
3. 同一次LLM响应可提出最多4个新表达式；验证后复用已有节点或在线扩图。
4. 维护多个方程集合，每轮最多12次普通局部、20次历史重组；强制测试预算另计。
5. 外系数用线性最小二乘，内部参数用有界非线性最小二乘；二维默认保留已验收的保护式加速。
6. 搜索冻结后，统一比较删项、完整多项式前缀和已有子表达式替换，最多十候选，一次最终LLM选择。
7. 检查选择与禁用项一致性；不在最终选择后偷偷改方程或追加调用。

搜索中最多8个机制/输出；终端中间拟合可暂时超过8项，剪枝后再检查最终容量。
图中并非所有子表达式都是候选；覆盖保证只针对初始可选机制，也不等于所有组合覆盖。

## 构图与消融

```powershell
python -m gem_gss build-graph --case T01 --output graph.json
python -m gem_gss run --case T03 --ablation A --output runs/A
python -m gem_gss run --case T03 --ablation B --output runs/B
python -m gem_gss run --case T03 --ablation AB --output runs/AB
```

A去掉LLM的一跳邻居/边输入，保留程序图和历史；B将强制比例改为四分之一并失去全覆盖保证。
本清理包的所有变体统一使用最新终端规则；随包的历史T03消融发生在子表达式规则扩展之前，
不是本次清理版的新跑结果。三维仅有此前构图诊断，本包不宣称已支持、验证三维发现。

## 结果与评价

```powershell
python -m gem_gss evaluate --case T01 --result runs/standard/T01/final_result.json
```

只有显式评价命令读取参考方程；搜索、拟合和终选不读取答案或已发布方程。
六题当前保存的结构Jaccard均为1，但来自开发算例，T05包含规则扩展后的额外终选，
不能宣称严格盲测或任何模型都能复现。其他限制、成本和消融失败详见 [结果说明](docs/RESULTS.md)。

直接将本目录内容上传为GitHub仓库根目录即可。完整命令、跨平台环境设置、缓存和来源说明见
[English README](README.md) 与 [来源说明](THIRD_PARTY_NOTICES.md)。
