# 首版验证记录

日期：2026-11-01（Asia/Shanghai）。目标：验证首版软件能实际运行，并明确哪些结论尚无证据。

## 实际执行

| 项目 | 方法 | 本机结果 |
|---|---|---|
| 逻辑与接口回归 | `python -m pytest --junitxml=artifacts/tests.xml` | **49 passed**；无跳过；含真实 ONNX CPU 适配测试 |
| 词法检索与拒答 | `fieldvision evaluate --output artifacts/baseline_eval.json` | 20/20 自编回归题通过；15 条有答案、5 条域外拒答；不是盲测或业务正确率 |
| ONNX 适配 | 测试中临时生成 `[1,3,8,8]` Identity 模型，CPU，2 次预热、5 次计时 | 实际执行且返回非零计时、模型/输入哈希与运行时；不是视觉模型或 Jetson 测试 |
| 模型 HTTP 适配 | HTTPX MockTransport 检查请求、引用失效、超时 | 仅模拟传输，无真实供应商请求或费用发生 |
| 网页 | 本机启动后浏览器操作问答、实验比较、遥测诊断 | 三模块可操作；实验比较正确发现 `split_sha256` 不一致 |

本机运行：Windows，Python 3.12.14。基础与可选测试依赖精确版本见 `requirements-lock.txt`。测试产生的绝对临时路径及本机环境日志不随开源版本发布。

## 远程补验

[GitHub Actions run 41301828210](https://github.com/monsterduan/codex-project/actions/runs/41301828210) 对代码提交 `8f3f9fab429e2592b1b1990581ec5cc177e0be10` 的两个任务均已完成并成功：Windows/Python 3.12 与 Ubuntu/Python 3.12。依赖安装、回归与 CPU smoke test、评测输出和产物上传步骤均成功。这补齐了首版 Linux 软件回归，不扩展为真实模型或设备验收。

本机还构建了 wheel 安装包并检查其归档，确认包含网页、评测样例及 CSV 示例；没有把文档存在等同于运行效果。

## 指标口径

检索 hit@3 和 MRR@3 仅检查预期来源是否进入返回片段，不评价回答语义正确性。20 条题与知识资料一起编写，存在明显开发集偏差。它们用于检测代码回归，不能代表真实领域问答能力。

测试中的模型响应由人工 fixture 提供，不能算真实模型效果评测。语义检索融合只测试排序算法；还没有使用真实 embedding 模型验证召回。LoRA 没有训练，真实视觉模型没有基准结论，真机闭环没有验收。

## 已知限制

- 文档只支持 UTF-8 Markdown/Text；不支持 PDF OCR、表格理解或图像理解。
- 证据门槛是保守的词法覆盖规则；不能可靠识别所有领域内无答案、冲突和提示注入。
- 引用校验检查编号存在，不能保证模型结论被引用蕴含。
- 知识库在启动时载入，不是实时监控数据库；多用户隔离、任务队列、持久索引未实现。
- 实验身份、数据清单和 measured 标记由输入方提供；目前不能独立验证远端采集真实性。
- 当前依赖在导入 LangGraph 时出现一条上游 `LangChainPendingDeprecationWarning`，测试不失败；项目未使用其 checkpoint 持久化功能。
- GitHub Actions 配置存在不等于远程已跑通；首版实际通过记录见上方远程补验，后续提交仍以各自 Actions 结果为准。

后续验收按 [13 周路线](roadmap-13-weeks.md) 逐项补齐，不能将计划内容提前写成履历成绩。
