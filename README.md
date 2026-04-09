# 🔬 VirtualCell — 单细胞基础模型Benchmark平台

**26个数据集 × 14个模型 × 4大任务 = 虚拟细胞的统一评估框架**

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

## 📦 14个基础模型

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
| 数据集 | 用途 | 来源 |
|--------|------|------|
| Zhengsorted | 细胞分选验证 | 10X Genomics |
| Macosko2015 | 视网膜细胞 | Drop-seq |
| Baron2016 | 胰腺 | inDrop |
| Muraro2016 | 胰腺 | CEL-seq2 |
| Segerstolpe2016 | 胰腺 | Smart-seq2 |
| Xin2016 | 胰腺 | SMARTer |
| Lawlor2017 | 胰腺 | Fluidigm C1 |
| Enge2017 | 胰腺衰老 | inDrop |
| Camp2017 | 大脑类器官 | Drop-seq |
| Plasschaert2019 | 肺上皮 | 10X |
| Lukowski2019 | 视网膜 | 10X |
| Travaglini2020 | 肺 | 10X |
| Cao2020 | 斑马鱼 | sci-RNA-seq3 |
| Cao2017 | 蠕虫发育 | sci-RNA-seq |
| Saunders2018 | 小鼠大脑 | Drop-seq |
| Virtual Cell Challenge | H1 hESCs扰动 | Arc Institute |

## 🧪 4大评估任务

### 1. 细胞类型注释 (Cell Type Annotation)
- 指标：Accuracy, F1, AUROC
- 基准：scBERT > scGPT > Geneformer

### 2. 扰动预测 (Perturbation Prediction)
- 指标：PDS (Perturbation Discrimination Score), DES, MAE, MSE, PCC
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
│   ├── models/          # 14个模型的统一接口
│   │   ├── base.py      # 基础模型抽象类
│   │   ├── scgpt.py     # scGPT 封装
│   │   ├── geneformer.py # Geneformer 封装
│   │   ├── scbert.py    # scBERT 封装
│   │   └── ...          # 其他11个模型
│   ├── datasets/        # 26个数据集加载器
│   │   ├── base.py      # 数据集抽象类
│   │   ├── census.py    # CELLxGENE Census
│   │   ├── atlas.py     # Human Cell Atlas
│   │   └── ...          # 其他数据集
│   ├── tasks/           # 4大评估任务
│   │   ├── annotation.py    # 细胞注释
│   │   ├── perturbation.py  # 扰动预测
│   │   ├── integration.py   # 批次整合
│   │   └── grn.py           # GRN推断
│   ├── utils/           # 工具函数
│   ├── benchmark.py     # 主benchmark引擎
│   └── report.py        # 结果报告生成
├── scripts/             # 运行脚本
├── tests/               # 测试
├── docs/                # 文档
└── examples/            # 示例
```

## ⚡ Quick Start

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

print(f"Accuracy: {result.accuracy:.3f}")
print(f"F1 Score: {result.f1:.3f}")
```

## 🆚 与其他Benchmark对比

| 特性 | VirtualCell | scGPT Benchmark | PertEval-scFM | scBERT Benchmark |
|------|------------|-----------------|---------------|-----------------|
| 模型数 | **14** | 3 | 5 | 2 |
| 数据集 | **26** | 5 | 3 | 4 |
| 任务数 | **4** | 3 | 1 | 1 |
| 统一接口 | ✅ | ❌ | ❌ | ❌ |
| 零样本评估 | ✅ | 部分 | ✅ | ❌ |
| 微调评估 | ✅ | ✅ | ❌ | ✅ |
| 报告生成 | ✅ | ❌ | ❌ | ❌ |

## 💰 应用场景

| 场景 | 价值 | 潜在客户 |
|------|------|---------|
| 药物发现 | 预测基因扰动效果，减少实验 | 药企（恒瑞/信达/百济）|
| 细胞治疗 | CAR-T/干细胞质量评估 | 细胞治疗公司 |
| 疾病建模 | 虚拟细胞模拟疾病状态 | 学术机构 |
| 生物学验证 | 替代部分湿实验 | CRO公司 |

## 📊 Reports & Leaderboard

### 终端排行榜

```bash
# 查看所有评估结果的排行榜
virtual-cell leaderboard
```

### HTML报告生成

```bash
# 运行benchmark并生成HTML报告（热力图 + 排行榜）
virtual-cell report --models scgpt,geneformer,scbert --datasets zheng68k --tasks cell_annotation,perturbation

# 输出：
#   benchmark_report_heatmap.html
#   benchmark_report_leaderboard.html
```

### 模型对比

```bash
# 对比两个模型，生成HTML对比报告
virtual-cell compare scgpt geneformer --output comparison.html
```

### 模型详情

```bash
# 查看单个模型详细信息
virtual-cell info scgpt
```

### 🌐 GitHub Pages

交互式排行榜已部署到 GitHub Pages：

**🔗 [VirtualCell Benchmark Leaderboard](https://mokangmedical.github.io/virtual-cell/)**

功能包括：
- 🏆 按任务筛选的排行榜
- 📊 模型 × 任务热力图（Canvas绘制）
- 📋 点击模型名展开所有数据集得分
- 📱 响应式布局，支持移动端

## 🗺️ Roadmap

- [x] 架构设计
- [x] 模型接口层（14个模型封装）
- [x] 数据集加载层（26个数据集适配）
- [x] 评估任务层（4大任务）
- [x] Benchmark引擎
- [x] 报告生成器
- [ ] 真实模型集成（需要GPU环境）
- [ ] 首个Benchmark结果
- [ ] 论文撰写
- [x] GitHub Pages

## 📄 License

MIT License

---

*VirtualCell — 定义虚拟细胞的评估标准*
*MoKangMedical | 2026*
