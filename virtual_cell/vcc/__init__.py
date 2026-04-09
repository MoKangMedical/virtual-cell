"""
VCC Pipeline — Virtual Cell Challenge标准评估流程

对接Arc Institute的Virtual Cell Challenge标准：
- 数据预处理（H1 hESCs, CRISPRi扰动）
- 标准化评估指标（DES, PDS, MAE, Spearman, AUPRC, Pearson-Δ）
- 结果格式化（可直接提交到VCC Leaderboard）
"""

from virtual_cell.vcc.pipeline import VCCMetrics, VCCPipeline

__all__ = ["VCCMetrics", "VCCPipeline"]
