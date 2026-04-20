# 🆓 虚拟细胞免费资源指南 — 医学研究者上车手册

> 虚拟细胞火得一塌糊涂：Cell封面、西湖大学、扎克伯格砸钱。但很多医学研究者问：**我一没算力二没数据，怎么上车？**
>
> 答案是：**你现在就可以，而且不花钱。**

---

## 📦 两大免费资源

### 1. CZI 虚拟细胞平台

**Chan Zuckerberg Initiative (CZI)** 开放了超过 **4400万个人类单细胞数据**。

| 项目 | 详情 |
|------|------|
| **数据量** | 44,000,000+ 人类单细胞 |
| **注册要求** | 机构邮箱（edu/org） |
| **费用** | 完全免费 |
| **用途** | 细胞图谱、扰动研究、多组学分析 |
| **平台** | CELLxGENE + CZI Cell Census |

**🔗 入口：**
- CELLxGENE: https://cellxgene.cziscience.com/
- CZI Cell Census: https://chanzuckerberg.github.io/cellxgene-census/

**适合你做什么：**
- ✅ 直接下载处理好的单细胞矩阵（h5ad格式）
- ✅ 按细胞类型/组织/疾病筛选子集
- ✅ 不用自己养细胞、不用测序，数据已经帮你准备好了
- ✅ 与 VirtualCell Benchmark 数据集无缝对接

### 2. Arc 虚拟细胞图谱

**Arc Institute** 的虚拟细胞图谱包含 **1亿个单细胞** 对 **1100多种药物** 的反应数据。

| 项目 | 详情 |
|------|------|
| **数据量** | 100,000,000+ 单细胞 |
| **药物覆盖** | 1,100+ 种药物扰动 |
| **核心能力** | AI预测药物对特定细胞的效果 |
| **最佳模型** | xTrimoSCPerturb (VCC冠军) |
| **用途** | 药物筛选、靶点发现、组合扰动预测 |

**🔗 入口：**
- Arc Virtual Cell Atlas: https://virtualcellatlas.arcinstitute.org/
- Virtual Cell Challenge: https://virtualcellchallenge.org/

**适合你做什么：**
- ✅ 研究某种药对某种癌细胞的效果 → 直接查
- ✅ AI帮你预测扰动结果 → 省掉大量湿实验
- ✅ 支持组合扰动（A药+B药+C基因敲除）
- ✅ 结果可直接用于论文图表

---

## 🧪 医学研究者三大上车方向

### 方向一：基金申请 💰

**痛点：** 国自然/省基金创新性不够，方法学陈旧
**解法：** 在本子里加"虚拟细胞预测+实验验证"模块

**操作步骤：**
1. 在 CZI 平台下载你的目标细胞类型数据
2. 用 VirtualCell Benchmark 评估 2-3 个基础模型
3. 选出最优模型做扰动预测
4. 设计湿实验验证预测结果

**本子写法示例：**
> "本项目拟采用虚拟细胞基础模型（scGPT/Geneformer），基于4400万公开单细胞数据，预测[靶点]对[细胞类型]的扰动效应，再通过CRISPR实验验证预测结果。"

**创新性直接拉满** —— 目前绝大多数国自然标书还没有这个套路。

### 方向二：药物筛选 💊

**痛点：** 新药研发失败率高，筛选成本巨大
**解法：** 先在虚拟细胞上跑一轮，再做实验

**操作流程：**
```
目标：A药对B肿瘤的效果

Step 1: Arc平台查询已知药物-细胞反应数据
        ↓
Step 2: VirtualCell Benchmark 对比模型预测
        ↓
Step 3: 筛选 Top 5-10 候选
        ↓
Step 4: 湿实验验证（工作量减少80%+）
        ↓
Step 5: 论文/专利
```

**省时省钱：** 6个月的筛选工作 → 1周虚拟筛选 + 1个月验证

### 方向三：靶点发现 🎯

**痛点：** 不知道从哪里找新靶点
**解法：** 用 TwinCell 等模型从公共数据里挖掘

**操作步骤：**
1. 在 CZI/CELLxGENE 定义你的疾病细胞群体
2. 用 Geneformer 或 scGPT 做差异分析
3. 用 GRN 推断（RegFormer）找上游调控因子
4. 交叉验证 → 候选靶点列表

**进阶玩法：**
- 用 VirtualCell 的扰动预测模块验证靶点
- 用 Arc 平台的药物数据做反向筛选
- 输出可直接用于基金标书/SCI论文

---

## 🔗 资源汇总

| 资源 | 链接 | 免费数据量 | 最佳用途 |
|------|------|-----------|---------|
| CZI CELLxGENE | https://cellxgene.cziscience.com/ | 4400万细胞 | 数据下载、细胞图谱 |
| Arc Virtual Cell Atlas | https://virtualcellatlas.arcinstitute.org/ | 1亿细胞 | 药物反应、扰动预测 |
| Virtual Cell Challenge | https://virtualcellchallenge.org/ | 基准数据 | 模型评估、论文复现 |
| VirtualCell Benchmark | https://github.com/MoKangMedical/virtual-cell | 15模型×26数据集 | 统一评估框架 |
| scGPT (预训练权重) | https://github.com/bowang-lab/scGPT | 开源模型 | 零样本细胞注释 |
| Geneformer | https://huggingface.co/ctheodoris/Geneformer | 开源模型 | 基因网络分析 |

---

## 📝 FAQ

**Q: 我需要 GPU 吗？**
A: 不一定。CZI/Arc 的数据查询不需要GPU。小规模推理可以用 Google Colab 免费GPU（T4）。大规模训练才需要。

**Q: 我需要会编程吗？**
A: 基础的 Python + scanpy 就够了。VirtualCell 提供了统一API，不用自己对接每个模型。

**Q: 虚拟细胞预测靠谱吗？**
A: 目前扰动预测的 PDS/DES 指标在 0.6-0.8 区间，不能替代实验，但能大幅缩小候选范围。**预测+验证 > 纯实验**。

**Q: 我的领域适合吗？**
A: 只要有单细胞数据的领域都适合：肿瘤、免疫、神经、心血管、代谢病、罕见病……

---

## 🚀 快速开始

```bash
# 安装 VirtualCell
pip install virtual-cell

# 列出可用模型
virtual-cell models

# 下载 CZI 公开数据集
virtual-cell download --source cellxgene --cell-type "T cell" --tissue "lung"

# 运行扰动预测
virtual-cell benchmark --models scgpt,geneformer --task perturbation --dataset kang2018

# 生成报告
virtual-cell report --output my_report.html
```

---

*VirtualCell — 让每个医学研究者都能上车虚拟细胞*
*MoKangMedical | 2026*
