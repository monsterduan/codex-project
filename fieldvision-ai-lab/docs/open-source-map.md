# 开源方向与个人增量

选型依据为项目官方仓库或文档，查询日期 2026-11-01。使用社区组件解决共性问题，把自己的贡献放在视觉工程证据、对照实验和诊断工作流，而不是重新命名上游项目。

| 开源项目 / 方向 | 与本项目的关系 | 自己应完成的部分 |
|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) / 状态工作流 | v0.1 实际依赖；首版固定 1.0.2，依赖组合锁定在 requirements-lock.txt | 检索门槛、引用检查、异常路径、状态与测试 |
| [LlamaIndex](https://github.com/run-llama/llama_index) / RAG 框架 | 已调研，首版未依赖；避免同时引入两套编排框架 | 后续仅在文档解析/索引确有收益时引入并对照 |
| [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding) / 中文语义检索 | 后续候选；首版没有下载权重或报告其效果 | 领域文档、查询格式、词法/语义/融合对照和误检分析 |
| [Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/model.html) | 已实现可选本地 encode 适配；首版实际模型验收待完成 | 批处理、缓存、模型版本、查询提示与正确归一化 |
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) / 小模型 LoRA | 第 9 周可选实验工具；未实现或训练 | 自有结构化诊断数据、源级划分、基线、过拟合与失败分析 |
| [ONNX Runtime](https://onnxruntime.ai/docs/api/python/api_summary.html) / 推理基准 | 已实现可选 adapter 和 CPU Identity smoke test | 真正视觉模型的输入、前后处理一致性、设备实测和可复现报告 |

不建议本阶段另起通用聊天机器人、套壳 PDF 问答、复杂多智能体平台、从零训练大模型或同时学习多个前端框架。它们会挤占真实数据与评测的时间，且不能自然体现原有视觉工程优势。

这些是技术选择与阶段安排，不声称上游项目、华为或其他机构认可本项目。
