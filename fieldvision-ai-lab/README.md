# FieldVision AI Lab

**Evidence-first AI applications for vision experiments and edge diagnostics.**

面向视觉工程的 AI 应用项目组合：让回答能找到出处，让实验结论能回到原始数据，让设备诊断承认未知。场景来自目标检测训练和边缘视觉部署，技术重点是检索问答、数据分析、效果评估和可复现的软件工程。

> 当前版本 **v0.1：可运行工程基线**。默认不调用大模型；默认知识与实验数据均为自编教学/合成数据。它不代表真实设备部署、模型微调或现场性能已经完成。

## 三个互相连接的项目模块

| 模块 | 实际实现 | 个人工程贡献 | 后续能力目标 |
|---|---|---|---|
| **EvidenceRAG** | LangGraph 检索→证据检查→摘录/模型回答；中文词法检索；来源行号与 SHA256；评测 CLI；可配置兼容模型 API | 原文溯源、低证据拒答、错误引用拦截、模型失败明确返回错误状态 | 本地语义检索实测、混合检索对照、重排序、独立问题集与回答事实核查 |
| **RunLens** | 读取 YOLO results.csv；保留同一最佳轮次指标；识别无效行、续训轮次、对照混杂因素 | 防止“多列最大值拼接”、历史版本误归因、把验证差值写成消融结论 | 接入有权公开的真实实验、代码/数据版本清单、多种子对照分析 |
| **EdgeGuard** | 缺失证据诊断、模型哈希核对、结构化遥测分析；可选真实 ONNX Runtime 基准入口 | 区分采集记录、演示阈值、模型耗时与端到端性能 | 真实目标检测模型基准、独立测试集精度核对、目标设备测量 |

三者先在一个仓库共享数据规范、评测与界面。稳定后再拆成独立包，避免三个互不相连的演示仓库。

## 快速开始

首版验证环境为 Python 3.12；其他 Python 版本尚未验收。以下命令在**本目录**执行。

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
fieldvision serve
```

打开 <http://127.0.0.1:8765>。接口说明在 `/docs`。默认只监听本机。

```bash
fieldvision ask "单应性矩阵为什么需要非共线对应点？"
fieldvision evaluate --output artifacts/baseline_eval.json
fieldvision run fieldvision/sample_data/runs/baseline
fieldvision compare fieldvision/sample_data/runs/baseline fieldvision/sample_data/runs/candidate --declared-change "model configuration"
fieldvision diagnose fieldvision/sample_data/telemetry.json
python -m pytest
```

演示候选实验故意改了数据划分，比较结果应标记 `split_sha256` 混杂。演示数据增加 3 个百分点并不是模型提升证据。

安装时依赖范围由 `pyproject.toml` 定义；`requirements-lock.txt` 记录首版验证时的精确依赖。复现首版可先安装该文件再执行 `pip install --no-deps -e .`。可选语义/硬件依赖单独安装。

## 接入真实模型服务

只有显式配置才发送问题与检索片段到该地址；默认不进行远程模型调用。支持兼容 `/chat/completions` 的服务，例如本地服务或按供应商官方文档配置的 Qwen 服务。端点、模型名和凭据均由你提供，仓库不预设付费账号。

```powershell
$env:FIELDVISION_LLM_BASE_URL="http://127.0.0.1:8000/v1"
$env:FIELDVISION_LLM_MODEL="your-served-model"
$env:FIELDVISION_LLM_API_KEY=""  # 本地无需鉴权时留空
fieldvision serve
```

非本机服务要求 HTTPS。复制 `.env.example` 只作配置参考：程序读取环境变量，**不会自动加载 `.env`**。当前远程服务适配通过模拟传输测试，未把模拟测试称为真实供应商调用验收。

模型回答检查引用是否存在，但不自动证明语义被引用支持。词法证据门槛也不等于可靠的语义拒答器；领域内缺失答案、文档冲突和提示注入需要后续专项评测。

## 可选：本地语义检索

```bash
pip install -e ".[semantic]"
```

将 `FIELDVISION_EMBEDDING_PATH` 指向**已自行下载并核对许可的本地 SentenceTransformer 模型目录**，然后启动服务。适配器设置 `local_files_only=True`、`trust_remote_code=False`，不会隐式下载或执行仓库远程模型代码。RRF 融合词法与语义排名。

Qwen3-Embedding 是研究候选，不是首版已测结论。不同模型的查询提示、指令格式与长度限制需要依据对应模型卡配置。首版该可选适配器仅提供通用 `encode`，没有声称已达到 Qwen 官方检索效果。

## 可选：真正执行 ONNX 基准

```bash
pip install -e ".[benchmark]"
fieldvision benchmark model.onnx inputs.npz --provider CPUExecutionProvider --iterations 50 --output artifacts/benchmark.json
```

`inputs.npz` 的键必须与模型输入名一致；文件不能包含对象数组。输出有模型/输入哈希、运行时和平台、预热次数、原始耗时与 p50/p95。指定提供程序不存在时直接报错，不静默假装 GPU 已运行。运行时仍可能按算子回退；报告包含实际会话提供程序，不能据此声称所有算子都在 GPU 上执行。

CPU Identity 小模型 smoke test 只验证基准适配器，不代表 YOLO、Jetson 或昇腾的推理性能。摄像头、预处理、后处理、传输和执行器时延必须另外测量。

## 评测与证据

- `tests/`：CSV、引用、HTTP、未知状态和工作流边界测试。
- `fieldvision/sample_data/eval_cases.json`：20 条自编回归题；不是独立测试集。它们与知识库一起编写，没有采用外部标注或盲评。
- [首版验证记录](docs/validation.md)：记录实际运行的命令和未完成项。
- [三个月学习与交付路线](docs/roadmap-13-weeks.md)：每周 25–30 小时，12 周建设、1 周缓冲与面试。
- [岗位匹配与可行性](docs/feasibility.md)：目标是 AI 应用工程实习的可验证能力，不承诺录用。
- [开源技术选型](docs/open-source-map.md)、[架构](docs/architecture.md)、[数据规范](docs/data-contracts.md)。

## 数据与许可证

本项目代码 MIT。第三方库和模型分别遵守其许可证，见 [THIRD_PARTY.md](THIRD_PARTY.md)。没有复制 YOLO 训练框架、模型权重或实验记录器原始代码。样例文本由本项目编写，CSV/遥测由本项目构造。

个人原始数据、API 配置、真实设备位置、未获许可的图像和第三方检测报告不随项目发布。真实实验接入请放在已忽略的 `data/private/` 下，仅导出已确认可公开的数据。

本项目由开发工具辅助实现；仓库代码与测试不能替代使用者对 Python、检索、模型调用及测量方法的理解。学习检查点要求能够独立修改、解释和复现关键模块。
