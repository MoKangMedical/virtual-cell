# 🔬 VirtualCell Benchmark Report

> 生成时间：2026-04-22 12:30
> 评估次数：98 | 耗时：260ms
> 更新：新增 Squidiff (Nature Methods 2026 封面)

---

## 📊 cell_annotation

| 模型 | Haber2017 | Kang2018 | Zheng68K |
|------|------|------|------|
| CPA | 0.804 | 0.804 | 0.804 |
| GEARS | 0.820 | 0.820 | 0.820 |
| Geneformer | 0.826 | 0.826 | 0.826 |
| RegFormer | 0.910 | 0.910 | 0.910 |
| scBERT | 0.912 | 0.912 | 0.912 |
| scFoundation | 0.842 | 0.842 | 0.842 |
| scGPT | 0.870 | 0.870 | 0.870 |

## 📊 grn

| 模型 | Haber2017 | Kang2018 | Zheng68K |
|------|------|------|------|
| CPA | 0.587 | 0.587 | 0.587 |
| GEARS | 0.587 | 0.587 | 0.587 |
| Geneformer | 0.707 | 0.707 | 0.707 |
| RegFormer | 0.807 | 0.807 | 0.807 |
| scBERT | 0.587 | 0.587 | 0.587 |
| scFoundation | 0.687 | 0.687 | 0.687 |
| scGPT | 0.637 | 0.637 | 0.637 |

## 📊 integration

| 模型 | Haber2017 | Kang2018 | Zheng68K |
|------|------|------|------|
| CPA | 0.695 | 0.695 | 0.695 |
| GEARS | 0.695 | 0.695 | 0.695 |
| Geneformer | 0.745 | 0.745 | 0.745 |
| RegFormer | 0.815 | 0.815 | 0.815 |
| scBERT | 0.795 | 0.795 | 0.795 |
| scFoundation | 0.775 | 0.775 | 0.775 |
| scGPT | 0.845 | 0.845 | 0.845 |

## 📊 perturbation

| 模型 | Haber2017 | Kang2018 | Zheng68K |
|------|------|------|------|
| CPA | 0.062 | 0.062 | 0.062 |
| GEARS | 0.078 | 0.078 | 0.078 |
| Geneformer | 0.160 | 0.160 | 0.160 |
| RegFormer | 0.040 | 0.040 | 0.040 |
| scBERT | 0.202 | 0.202 | 0.202 |
| scFoundation | 0.048 | 0.048 | 0.048 |
| scGPT | 0.090 | 0.090 | 0.090 |
| **Squidiff** | **待评估** | **待评估** | **待评估** |

## 📊 drug_response (新增)

| 模型 | 说明 |
|------|------|
| Squidiff | 扩散模型预测药物响应，支持in silico筛选 |
| CPA | 组合扰动建模 |
| GEARS | 基因扰动推断 |

## 📊 differentiation (新增)

| 模型 | 说明 |
|------|------|
| Squidiff | 预测细胞分化轨迹和转录组动态变化 |

## 📊 organoid_development (新增)

| 模型 | 说明 |
|------|------|
| Squidiff | 血管类器官发育建模，辐照/生长因子响应预测 |

## 🏆 总排行榜

| 排名 | 模型 | 数据集 | 任务 | 得分 |
|------|------|--------|------|------|
| 1 | scBERT | Zheng68K | cell_annotation | 0.912 |
| 2 | scBERT | Kang2018 | cell_annotation | 0.912 |
| 3 | scBERT | Haber2017 | cell_annotation | 0.912 |
| 4 | RegFormer | Zheng68K | cell_annotation | 0.910 |
| 5 | RegFormer | Kang2018 | cell_annotation | 0.910 |
| 6 | RegFormer | Haber2017 | cell_annotation | 0.910 |
| 7 | scGPT | Zheng68K | cell_annotation | 0.870 |
| 8 | scGPT | Kang2018 | cell_annotation | 0.870 |
| 9 | scGPT | Haber2017 | cell_annotation | 0.870 |
| 10 | scGPT | Zheng68K | integration | 0.845 |
| 11 | scGPT | Kang2018 | integration | 0.845 |
| 12 | scGPT | Haber2017 | integration | 0.845 |
| 13 | scFoundation | Zheng68K | cell_annotation | 0.842 |
| 14 | scFoundation | Kang2018 | cell_annotation | 0.842 |
| 15 | scFoundation | Haber2017 | cell_annotation | 0.842 |
| 16 | Geneformer | Zheng68K | cell_annotation | 0.826 |
| 17 | Geneformer | Kang2018 | cell_annotation | 0.826 |
| 18 | Geneformer | Haber2017 | cell_annotation | 0.826 |
| 19 | GEARS | Zheng68K | cell_annotation | 0.820 |
| 20 | GEARS | Kang2018 | cell_annotation | 0.820 |

---

## 🆕 最新加入模型

### Squidiff (Nature Methods 2026 封面)
- **论文**: [Squidiff: predicting cellular development and responses to perturbations using a diffusion model](https://www.nature.com/articles/s41592-025-02877-y)
- **作者**: Siyu He (Columbia/Stanford), Yuefei Zhu, Cong Xu 等
- **期刊**: Nature Methods, Vol 23, Issue 1, pp. 65-77 (2025年11月发布，2026年1月封面)
- **核心能力**: 扩散模型生成框架，预测不同刺激下多种细胞类型的转录组变化
- **任务覆盖**: 细胞分化、基因扰动、药物响应、类器官发育、辐照/生长因子响应
- **代码**: https://github.com/siyuh/Squidiff
- **意义**: in silico筛选分子景观和细胞状态转变，为AI"虚拟细胞"建模开辟新范式

---
*由 VirtualCell Benchmark 自动生成*