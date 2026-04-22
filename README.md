# 🔬 VirtualCell

> 单细胞基础模型Benchmark平台 — 15个模型×26个数据集×6大任务

[![Python](https://img.shields.io/badge/python-3.9+-green.svg)]()
[![Models](https://img.shields.io/badge/Models-15-blue.svg)]()
[![Tasks](https://img.shields.io/badge/Tasks-6-orange.svg)]()

## 一句话定义

**VirtualCell 是AI虚拟细胞的统一评测平台。** 追踪最新单细胞基础模型，提供标准化Benchmark，帮助研究者快速比较和选择模型。

> 💡 细胞是生命的最小完整单元。如果我们能用AI精确模拟每个细胞的行为，就等于掌握了生命的基本语言。VirtualCell正在为这个未来建立测量标准。

---

## 🎯 解决什么问题

| 痛点 | 现状 | VirtualCell |
|------|------|-------------|
| 模型分散 | 每篇论文一个GitHub，格式各异 | **统一接口** |
| 无法比较 | 不同论文用不同数据集/指标 | **标准化Benchmark** |
| 跟踪困难 | 领域发展太快，每月有新模型 | **持续更新追踪** |
| 门槛高 | 需要理解每种架构才能使用 | **一键调用** |

---

## 📊 模型覆盖（15个）

### Transformer/GPT架构
| 模型 | 机构 | 参数 | 预训练数据 | 核心能力 |
|------|------|------|-----------|----------|
| **scGPT** | U Toronto | 100M | 33M细胞 | 通用迁移学习 |
| **Geneformer** | Broad Institute | 95M | 30M细胞 | 基因网络建模 |
| **scBERT** | PKU | 100M | 1.2M细胞 | 双向编码 |
| **scFoundation** | MSR | 100M | 50M细胞 | 多任务统一 |
| **RegFormer** | 华大基因 | 75M | 25M细胞 | 调控网络 |
| **Nicheformer** | CZ Biohub | 110M | 35M细胞 | 空间转录组 |
| **scPrint** | GNosis | 78M | 50M细胞 | 基因印记 |
| **TGPT** | 复旦 | 80M | 10M细胞 | 转录组GPT |
| **CellBert** | HUJI | 90M | 15M细胞 | 序列建模 |
| **CellPLM** | GenBio AI | 120M | 20M细胞 | 多组学PLM |

### 其他架构
| 模型 | 机构 | 架构 | 核心能力 |
|------|------|------|----------|
| **ScBERT** | PKU | BERT | 双向编码 |
| **CellLM** | IAS | 语言模型 | 细胞语言 |
| **Lingshu-Cell** | 灵枢团队 | 掩码离散扩散 | VCC H1领先 |
| **UCell** | Caltech | U-Net | 空间解析 |
| **🆕 Squidiff** | 维也纳大学 | **扩散模型** | Nature Methods 2026封面 |

---

## 🧪 Benchmark任务（6大类）

| 任务 | 数据集数 | 描述 |
|------|---------|------|
| 细胞类型注释 | 8 | 自动识别细胞类型 |
| 基因扰动预测 | 6 | 预测基因敲除/过表达效果 |
| 批次效应校正 | 4 | 消除技术批次差异 |
| 基因调控网络 | 4 | 推断基因间调控关系 |
| **🆕 细胞分化** | 2 | Squidiff专长 |
| **🆕 药物响应** | 2 | Squidiff专长 |

---

## 🔬 Harness理论

VirtualCell 的核心价值是**Benchmark Harness**设计：

```
Benchmark Harness = 统一数据接口 + 标准化评估流程 + 可重复实验框架 + 模型注册表
```

- 不同模型、不同架构、不同预训练数据 → 同一套评估标准
- 让研究者专注于模型创新，而非数据工程
- Harness比模型更有价值——模型会过时，Benchmark标准持续进化

---

## 🚀 快速开始

```bash
git clone https://github.com/MoKangMedical/virtual-cell.git
cd virtual-cell
pip install -r requirements.txt
python -m virtual_cell.benchmark --model scgpt --task cell_type
```

---

## 📄 License

MIT License
