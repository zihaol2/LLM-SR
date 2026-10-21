# 第三方来源与许可说明

本发布包由用户提供的本地研究工程整理，未声称所有文件均为新原创代码。

## LLM-PDESR 代码

`gems_pde/vendor/llmpdesr/buffer.py` 与 `code_manipulation.py` 来自本地 `01_原版复现_LLM-PDESR/src/`；轻量 `config.py` 保留其所需的 `ExperienceBufferConfig`。包内导入和 logging 作了可移植性适配，Pareto 更新/提示核心保留。源文件哈希见 `artifacts/source_provenance.json`。

本地来源目录的 README 自述为 “Official Code Implementation”，但所提供目录中没有发现 LICENSE、可确认的上游仓库地址或完整版权声明。因此本包不臆造作者、上游链接或许可类型，也不为这些第三方内容重新授予 MIT/Apache 等许可证。公开分发时应沿用实际来源允许的许可与署名要求；当前材料不足以替代上游授权证明。

## 观测数据

`gems_pde/resources/data/` 中五个 .mat 原样来自上述本地来源的数据目录，SHA-256 已记录。所提供材料未包含可确认的数据再分发许可。这里保留来源信息，不替数据所有者授予新许可。

## 研究实现与结果

其余研究代码由本地 38/39/42/44 等版本提取、整理；历史结果、先验和方法图随本项目提供。没有为整个混合来源包添加一份覆盖全部内容的统一 LICENSE，避免错误覆盖第三方权利。

这些说明记录实际已知的来源及缺项，不影响代码的本地运行。仓库维护者可以在确认实际权利后补充正式 LICENSE、上游引用和数据说明。
