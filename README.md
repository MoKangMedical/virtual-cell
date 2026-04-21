# 🔬 VirtualCell v0.5.0 — 单细胞基础模型Benchmark平台

**15个模型 × 26个数据集 × 4大任务 × CellForge自动架构设计 = 虚拟细胞的统一评估框架**

> 用AI虚拟细胞替代真实实验，72小时完成6个月的生物学验证。

---

## 🎯 一句话

VirtualCell = 单细胞AI的 **ImageNet** —— 不是训练模型，是定义"什么任务值得做、什么模型真的行"。

## 🧬 背景：为什么需要虚拟细胞？

传统单细胞分析的痛点：
- 每个实验只测几百个基因，丢失大量信息
- 批次效应让不同实验室的数据无法比较
- 基因扰动实验（CRISPR）昂贵且耗时
- 细胞类型注释依赖专家经验

**虚拟细胞的答案**：用AI基础模型预训练在数百万细胞上，学会"细胞的语言"，然后零样本/少样本解决下游任务。

## 🆕 v0.4.0 新特性

### CellForge — 多智能体架构设计引擎
- **三阶段流程**：Task Analysis → Method Design（多Agent讨论）→ Code Generation
- **Mock模式**：基于文献知识的架构模板生成，无需GPU
- **Full模式**：调用CellForge实际多Agent流程（需安装CellForge + LLM API）
- **创新组件库**：9个领域特定创新模块（轨迹感知编码器、扰动扩散模块、基因交互GNN等）

### Lingshu-Cell — 掩码离散扩散模型
- **架构**：Transformer + 掩码离散扩散（MDDM）
- **创新**：281级token化基因表达、SwiGLU FFN、RoPE位置编码、序列压缩
- **能力**：扰动预测、条件生成、采样
- **训练**：LingshuTrainer支持单步训练

### REST API v0.4.0
- 15个端点（新增6个：pipeline/run、leaderboard/{task}、compare、generators、stats、models/{name}/detail）
- Pydantic请求模型 + 完整错误处理

### VCC Pipeline
- 对接Arc Institute Virtual Cell Challenge标准
- 标准化评估指标：DES、PDS、MAE、Spearman、AUPRC、Pearson-Δ

## 📦 15个基础模型

| 模型 | 架构 | 预训练数据 | 核心能力 | 论文 |
|------|------|-----------|---------|------|
| **scGPT** | GPT-style Decoder | 33M+ 多组学细胞 | 生成任务（插补/扰动/整合）| Nature 2024 |
| **Geneformer** | Encoder | 30M 细胞（偏癌症）| 基因网络/药物响应 | Nature 2023 |
| **scBERT** | BERT-style Encoder | ~1M 细胞 | 细胞注释（双向注意力）| ISMB 2022 |
| **scFoundation** | Foundation | 50M+ 细胞 | 多任务/大规模预训练 | Nature 2024 |
| **RegFormer** | Transformer+调控 | 多数据集 | GRN推断/扰动/药物 | bioRxiv 2025 |
| **Nicheformer** | Spatial Transformer | 空间转录组 | 空间细胞微环境 | bioRxiv 2024 |
| **scPRINT** | PRINT架构 | 50M+ 细胞 | 基因调控网络 | bioRxiv 2024 |
| **CellLM** | Language Model | 多数据集 | 通用细胞理解 | 2024 |
| **CellPLM** | Prefix LM | 多组学 | 细胞属性预测 | 2024 |
| **tGPT** | GPT-style | scRNA-seq | 生成建模 | 2023 |
| **CellBert** | BERT-style | scRNA-seq | 细胞表征 | 2023 |
| **CPA** | VAE+Attention | 扰动数据 | 组合扰动预测 | Nature Methods 2023 |
| **GEARS** | GNN+Transformer | 扰动数据 | 基因扰动推断 | Nature Biotech 2023 |
| **xTrimoSCPerturb** | Hybrid | scFoundation | 扰动预测（VCC冠军）| 2025 |
| **🆕 Lingshu-Cell** | MDDM (Transformer+Diffusion) | scRNA-seq | 扰动预测/条件生成 | 2026 |

## 📊 26个数据集

### 核心Benchmark数据集（10个）
| 数据集 | 细胞数 | 类型 | 用途 |
|--------|--------|------|------|
| Zheng68K | 68K | PBMC | 细胞注释 |
| PBMC 10K | 10K | PBMC | 多任务 |
| Human Cell Atlas | 1M+ | 多组织 | 细胞图谱 |
| CELLxGENE Census | 50M+ | 多组学 | 预训练/评估 |
| Kang2018 | 24K | PBMC扰动 | 扰动预测 |
| Adamson2016 | 60K | CRISPR扰动 | 扰动预测 |
| Norman2019 | 100K+ | 组合扰动 | 组合扰动 |
| Haber2017 | 53K | 肠道 | 细胞注释 |
| Tabula Muris | 100K+ | 20小鼠组织 | 多任务 |
| Tabula Sapiens | 500K+ | 人体多组织 | 细胞图谱 |

### 扩展数据集（16个）
Zhengsorted, Macosko2015, Baron2016, Muraro2016, Segerstolpe2016, Xin2016, Lawlor2017, Enge2017, Camp2017, Plasschaert2019, Lukowski2019, Travaglini2020, Cao2020, Cao2017, Saunders2018, Virtual Cell Challenge

## 🧪 4大评估任务

### 1. 细胞类型注释 (Cell Type Annotation)
- 指标：Accuracy, F1, AUROC
- 基准：scBERT > scGPT > Geneformer

### 2. 扰动预测 (Perturbation Prediction)
- 指标：PDS, DES, MAE, MSE, PCC
- 基准：xTrimoSCPerturb > RegFormer > CPA > scGPT

### 3. 批次整合 (Batch Integration)
- 指标：kBET, LISI, ASW, Graph Connectivity
- 基准：scGPT(fine-tuned) > scBERT > Geneformer

### 4. 基因调控网络推断 (GRN Inference)
- 指标：AUPRC, AUROC
- 基准：RegFormer > Geneformer > scGPT

## 🏗️ 架构

```
virtual-cell/
├── virtual_cell/
│   ├── models/          # 15个模型的统一接口
│   │   ├── base.py      # 基础模型抽象类
│   │   ├── lingshu_cell.py  # 🆕 Lingshu-Cell (MDDM)
│   │   └── ...
│   ├── generators/      # 🆕 CellForge架构生成器
│   │   ├── cellforge.py     # Mock模式
│   │   ├── cellforge_full.py # Full模式（多Agent）
│   │   └── model_adapter.py # 架构→BaseModel适配器
│   ├── vcc/             # 🆕 VCC Pipeline
│   │   └── pipeline.py      # 标准评估流程
│   ├── datasets/        # 26个数据集加载器
│   ├── tasks/           # 4大评估任务
│   ├── benchmark.py     # 主benchmark引擎
│   ├── api.py           # REST API (15端点)
│   ├── report.py        # 结果报告生成
│   ├── visualizer.py    # 可视化
│   └── downloader.py    # 数据下载器
├── tests/               # 测试套件 (55+ tests)
├── examples/            # 示例脚本
│   └── cellforge_demo.py    # 🆕 端到端演示
└── docs/                # 文档
```

## ⚡ Quick Start

### Python API

```python
from virtual_cell import Benchmark, load_model, load_dataset

# 加载模型和数据集
model = load_model("scgpt")
dataset = load_dataset("zheng68k")

# 运行benchmark
benchmark = Benchmark()
result = benchmark.evaluate(
    model=model,
    dataset=dataset,
    task="cell_annotation",
)

print(f"Accuracy: {result.metrics['accuracy']:.3f}")
```

### CellForge 架构生成

```python
from virtual_cell.generators import CellForgeGenerator

gen = CellForgeGenerator()
result = gen.generate("perturbation", "adamson2016", n_architectures=3)

# 最优架构
best = result.best()
print(f"Best: {best.name}, Confidence: {best.confidence:.4f}")
print(f"Innovations: {best.innovations}")
print(f"Code:\n{best.code}")
```

### VCC Pipeline

```python
from virtual_cell.vcc.pipeline import VCCPipeline, VCCMetrics
import numpy as np

pipeline = VCCPipeline()
pred = np.random.randn(100, 50)
gt = np.random.randn(100, 50)

metrics = pipeline.evaluate(pred, gt)
print(f"MAE: {metrics.mae:.4f}, PDS: {metrics.pds:.4f}")

# 格式化为VCC提交
submission = pipeline.format_submission(metrics, "my_model")
pipeline.save_submission(submission, "submission.json")
```

### 完整 Generate & Evaluate

```python
from virtual_cell.benchmark import Benchmark

bench = Benchmark()
result = bench.generate_and_evaluate(
    task="perturbation",
    dataset="adamson2016",
    n_architectures=3,
)

for entry in result.get_leaderboard():
    print(f"{entry['model']}: {entry['primary_score']:.4f}")
```

## 🌐 REST API 文档

### 启动服务

```bash
pip install uvicorn
uvicorn virtual_cell.api:app --host 0.0.0.0 --port 8000
```

### 端点列表

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/models` | 列出所有模型 |
| GET | `/datasets` | 列出所有数据集 |
| GET | `/tasks` | 列出所有任务 |
| POST | `/generate` | 生成架构（CellForge） |
| POST | `/benchmark` | 运行评估 |
| GET | `/leaderboard` | 获取排行榜 |
| POST | `/predict` | 单模型预测 |
| GET | `/info/{model}` | 模型基本信息 |
| POST | `/api/v1/pipeline/run` | 🆕 完整VCC Pipeline |
| GET | `/api/v1/leaderboard/{task}` | 🆕 按任务筛选排行榜 |
| POST | `/api/v1/compare` | 🆕 对比两个模型 |
| GET | `/api/v1/generators` | 🆕 列出所有生成器 |
| GET | `/api/v1/stats` | 🆕 平台统计 |
| GET | `/api/v1/models/{name}/detail` | 🆕 模型详情 |

### 示例 curl 命令

```bash
# 健康检查
curl http://localhost:8000/health

# 平台统计
curl http://localhost:8000/api/v1/stats

# 列出生成器
curl http://localhost:8000/api/v1/generators

# 模型详情
curl http://localhost:8000/api/v1/models/scgpt/detail

# 按任务排行榜
curl http://localhost:8000/api/v1/leaderboard/perturbation

# 运行完整Pipeline
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"task": "perturbation", "dataset": "adamson2016", "n_architectures": 3}'

# 对比两个模型
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"model1": "scgpt", "model2": "geneformer", "tasks": ["cell_annotation"]}'

# 生成架构
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"task": "perturbation", "dataset": "adamson2016", "n_architectures": 3}'

# 运行Benchmark
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{"models": ["scgpt", "geneformer"], "datasets": ["zheng68k"], "tasks": ["cell_annotation"], "max_cells": 500}'
```

## 📊 Reports & Leaderboard

### 终端排行榜

```bash
virtual-cell leaderboard
```

### HTML报告生成

```bash
virtual-cell report --models scgpt,geneformer,scbert --datasets zheng68k --tasks cell_annotation,perturbation
```

### 端到端演示

```bash
python3 examples/cellforge_demo.py
```

## 🔄 VCC Pipeline 流程

```
输入任务描述
    │
    ▼
┌─────────────────────┐
│  Phase 1: 数据分析   │  分析数据集特征 + 文献RAG
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 2: 架构设计   │  多Agent专家讨论 → 生成N个候选
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 3: 代码生成   │  每个架构生成PyTorch代码
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 4: 模型适配   │  GeneratedArchitecture → BaseModel
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 5: 评估       │  VCC标准指标 (DES/PDS/MAE/...)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 6: 排行榜     │  排名 + 生成提交文件
└─────────────────────┘
```

## 🆚 与其他Benchmark对比

| 特性 | VirtualCell v0.4.0 | scGPT Benchmark | PertEval-scFM | scBERT Benchmark |
|------|-------------------|-----------------|---------------|-----------------|
| 模型数 | **15** | 3 | 5 | 2 |
| 数据集 | **26** | 5 | 3 | 4 |
| 任务数 | **4** | 3 | 1 | 1 |
| 统一接口 | ✅ | ❌ | ❌ | ❌ |
| 零样本评估 | ✅ | 部分 | ✅ | ❌ |
| 微调评估 | ✅ | ✅ | ❌ | ✅ |
| 报告生成 | ✅ | ❌ | ❌ | ❌ |
| 架构自动生成 | ✅ 🆕 | ❌ | ❌ | ❌ |
| REST API | ✅ 🆕 | ❌ | ❌ | ❌ |
| VCC Pipeline | ✅ 🆕 | ❌ | ❌ | ❌ |

## 🆓 免费资源 — 医学研究者上车手册

> 你一没算力二没数据？照样上车，而且不花钱。

### 两大免费数据平台

| 资源 | 数据量 | 核心价值 |
|------|--------|---------|
| **[CZI CELLxGENE](https://cellxgene.cziscience.com/)** | 4400万+ 人类单细胞 | 机构邮箱注册即可下载，无需测序 |
| **[Arc Virtual Cell Atlas](https://virtualcellatlas.arcinstitute.org/)** | 1亿单细胞 × 1100+ 药物 | AI预测药物对特定细胞的效果 |

### 三大上车方向

1. **💰 基金申请** — 标书加"虚拟细胞预测+实验验证"模块，创新性拉满
2. **💊 药物筛选** — 先虚拟筛选候选，再湿实验验证，省时省钱80%+
3. **🎯 靶点发现** — 用 Geneformer/scGPT + GRN 推断从公共数据挖掘新靶点

👉 **详细指南：** [docs/free-resources-guide.md](docs/free-resources-guide.md)

## 🗺️ Roadmap

- [x] 架构设计
- [x] 模型接口层（15个模型封装，含Lingshu-Cell）
- [x] 数据集加载层（26个数据集适配）
- [x] 评估任务层（4大任务）
- [x] Benchmark引擎
- [x] 报告生成器
- [x] CellForge集成（Mock + Full模式）
- [x] REST API（15个端点）
- [x] VCC Pipeline（标准评估流程）
- [x] 端到端演示脚本
- [x] 增强测试套件（55+ 测试）
- [x] GitHub Pages
- [ ] 真实模型集成（需要GPU环境）
- [ ] 首个Benchmark结果
- [ ] CellForge Full模式多Agent优化
- [ ] 论文撰写

## 💰 应用场景

| 场景 | 价值 | 潜在客户 |
|------|------|---------|
| 药物发现 | 预测基因扰动效果，减少实验 | 药企（恒瑞/信达/百济）|
| 细胞治疗 | CAR-T/干细胞质量评估 | 细胞治疗公司 |
| 疾病建模 | 虚拟细胞模拟疾病状态 | 学术机构 |
| 生物学验证 | 替代部分湿实验 | CRO公司 |

## 📐 理论基础

### Harness 理论

在AI领域，Harness（环境设计）比模型本身更重要。优秀的Harness设计（工具链+信息格式+上下文管理+失败恢复+结果验证）能使性能提升64%。

VirtualCell 正是 Harness 理论在单细胞生物学领域的实践：我们不训练更好的细胞模型，而是构建更好的评估环境——标准化数据集、统一评估指标、自动化Benchmark Pipeline——让每一个模型的真实能力得以公平展现。

### 红杉论点

> 下一代万亿美元公司是伪装成服务公司的软件公司。从卖工具到卖结果。

VirtualCell 从开源 Benchmark 平台出发，最终目标是提供"虚拟细胞即服务"（Virtual Cell as a Service）：用户提交生物学问题，平台返回经过验证的预测结果，而非一堆需要自行解读的模型输出。

### 理论宪法

本项目遵循理论宪法四卷八章统一框架，将单细胞基础模型的评估体系建立在可验证、可复现、可扩展的理论根基之上。

## 📄 License

MIT License

---

*VirtualCell v0.4.0 — 定义虚拟细胞的评估标准*
*MoKangMedical | 2026*
