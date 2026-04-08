"""
模型基类 — 所有单细胞基础模型的统一接口

每个模型需要实现：
- load(): 加载预训练权重
- predict(): 执行推理
- get_embeddings(): 获取细胞表征
- fine_tune(): 微调（可选）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class ModelArchitecture(Enum):
    """模型架构类型。"""
    GPT_DECODER = "gpt_decoder"       # scGPT, tGPT
    BERT_ENCODER = "bert_encoder"     # scBERT, CellBert
    ENCODER = "encoder"               # Geneformer
    FOUNDATION = "foundation"         # scFoundation
    HYBRID = "hybrid"                 # RegFormer, xTrimoSCPerturb
    SPATIAL = "spatial"               # Nicheformer
    VAE = "vae"                       # CPA
    GNN_TRANSFORMER = "gnn_transformer"  # GEARS
    PREFIX_LM = "prefix_lm"           # CellPLM
    PRINT = "print"                   # scPRINT


@dataclass
class ModelInfo:
    """模型元信息。"""
    name: str
    architecture: ModelArchitecture
    pretrain_cells: int           # 预训练细胞数
    pretrain_data: str            # 预训练数据来源
    parameters: str               # 参数量
    paper: str                    # 论文链接
    code_repo: str                # 代码仓库
    supported_tasks: list[str]    # 支持的任务
    strengths: list[str]          # 优势
    weaknesses: list[str]         # 劣势
    license: str = ""             # 许可证
    year: int = 2024              # 发布年份

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "architecture": self.architecture.value,
            "pretrain_cells": self.pretrain_cells,
            "parameters": self.parameters,
            "supported_tasks": self.supported_tasks,
            "year": self.year,
        }


@dataclass
class PredictionResult:
    """模型推理结果。"""
    model_name: str
    task: str
    predictions: Any               # 预测结果（因任务而异）
    embeddings: Optional[np.ndarray] = None  # 细胞表征
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "predictions_shape": str(getattr(self.predictions, 'shape', 'N/A')),
            "embeddings_shape": str(getattr(self.embeddings, 'shape', 'N/A')) if self.embeddings is not None else "None",
            "metadata": self.metadata,
        }


class BaseModel(ABC):
    """
    模型基类。

    所有单细胞基础模型必须继承此类并实现关键方法。
    """

    def __init__(self, info: ModelInfo):
        self.info = info
        self._loaded = False
        self._model = None

    @abstractmethod
    def load(self, checkpoint_path: str = "", device: str = "cpu") -> None:
        """加载预训练模型。"""
        ...

    @abstractmethod
    def predict(self, data: Any, task: str, **kwargs) -> PredictionResult:
        """执行推理。"""
        ...

    @abstractmethod
    def get_embeddings(self, data: Any, **kwargs) -> np.ndarray:
        """获取细胞表征向量。"""
        ...

    def fine_tune(self, train_data: Any, **kwargs) -> None:
        """微调模型（可选实现）。"""
        raise NotImplementedError(f"{self.info.name} 不支持微调")

    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return f"{self.info.name}(arch={self.info.architecture.value}, loaded={self._loaded})"


# ================================================================
# 14个模型的Info定义
# ================================================================

SCGPT_INFO = ModelInfo(
    name="scGPT",
    architecture=ModelArchitecture.GPT_DECODER,
    pretrain_cells=33_000_000,
    pretrain_data="多组学（scRNA-seq + scATAC-seq）",
    parameters="~100M",
    paper="https://www.nature.com/articles/s41592-024-02305-7",
    code_repo="https://github.com/bowang-lab/scGPT",
    supported_tasks=["cell_annotation", "perturbation", "integration", "grn", "imputation"],
    strengths=["多组学支持", "生成能力强", "微调效果好"],
    weaknesses=["零样本在分布偏移下表现差"],
    license="BSD-3",
    year=2024,
)

GENEFORMER_INFO = ModelInfo(
    name="Geneformer",
    architecture=ModelArchitecture.ENCODER,
    pretrain_cells=30_000_000,
    pretrain_data="偏癌症数据集",
    parameters="~100M",
    paper="https://www.nature.com/articles/s41586-023-06139-9",
    code_repo="https://github.com/ctheodoris/Geneformer",
    supported_tasks=["cell_annotation", "grn", "drug_response", "perturbation"],
    strengths=["基因网络推断", "癌症任务", "零样本能力"],
    weaknesses=["数据不平衡敏感"],
    license="CC-BY-4.0",
    year=2023,
)

SCBERT_INFO = ModelInfo(
    name="scBERT",
    architecture=ModelArchitecture.BERT_ENCODER,
    pretrain_cells=1_000_000,
    pretrain_data="scRNA-seq",
    parameters="~100M",
    paper="https://academic.oup.com/bioinformatics/article/38/15/3816/6606850",
    code_repo="https://github.com/TencentAILabHealthcare/scBERT",
    supported_tasks=["cell_annotation"],
    strengths=["细胞注释", "双向注意力"],
    weaknesses=["不支持生成/扰动任务"],
    license="Apache-2.0",
    year=2022,
)

SCFOUNDATION_INFO = ModelInfo(
    name="scFoundation",
    architecture=ModelArchitecture.FOUNDATION,
    pretrain_cells=50_000_000,
    pretrain_data="多数据集聚合",
    parameters="~100M",
    paper="https://www.nature.com/articles/s41586-024-07280-x",
    code_repo="https://github.com/bio-ontology-research-group/scfoundation",
    supported_tasks=["cell_annotation", "perturbation", "integration", "grn"],
    strengths=["大规模预训练", "多任务能力"],
    weaknesses=["计算资源需求大"],
    license="MIT",
    year=2024,
)

REGFORMER_INFO = ModelInfo(
    name="RegFormer",
    architecture=ModelArchitecture.HYBRID,
    pretrain_cells=10_000_000,
    pretrain_data="多数据集+调控信息",
    parameters="~80M",
    paper="https://www.biorxiv.org/content/10.1101/2025.01.24.634217v1",
    code_repo="https://github.com/RegFormer/RegFormer",
    supported_tasks=["cell_annotation", "perturbation", "grn", "drug_response"],
    strengths=["GRN推断最优", "扰动预测强", "超越scGPT/Geneformer"],
    weaknesses=["较新，社区验证少"],
    license="MIT",
    year=2025,
)

NICHEFORMER_INFO = ModelInfo(
    name="Nicheformer",
    architecture=ModelArchitecture.SPATIAL,
    pretrain_cells=5_000_000,
    pretrain_data="空间转录组",
    parameters="~90M",
    paper="https://www.biorxiv.org/content/10.1101/2024.01.08.574693v1",
    code_repo="https://github.com/vib-singlecellfacility/nicheformer",
    supported_tasks=["cell_annotation", "spatial_analysis"],
    strengths=["空间数据", "微环境分析"],
    weaknesses=["非空间任务不突出"],
    license="MIT",
    year=2024,
)

SCPRINT_INFO = ModelInfo(
    name="scPRINT",
    architecture=ModelArchitecture.PRINT,
    pretrain_cells=50_000_000,
    pretrain_data="多数据集",
    parameters="~100M",
    paper="https://www.biorxiv.org/content/10.1101/2024.07.29.605556v1",
    code_repo="https://github.com/bioNMF/scPRINT",
    supported_tasks=["grn", "cell_annotation", "perturbation"],
    strengths=["基因调控网络", "大规模预训练"],
    weaknesses=["计算资源需求大"],
    license="Apache-2.0",
    year=2024,
)

# 简化定义其余模型
MODELS_INFO = {
    "scgpt": SCGPT_INFO,
    "geneformer": GENEFORMER_INFO,
    "scbert": SCBERT_INFO,
    "scfoundation": SCFOUNDATION_INFO,
    "regformer": REGFORMER_INFO,
    "nicheformer": NICHEFORMER_INFO,
    "scprint": SCPRINT_INFO,
    "celllm": ModelInfo("CellLM", ModelArchitecture.BERT_ENCODER, 5_000_000, "多数据集", "~80M", "", "", ["cell_annotation"], ["通用细胞理解"], ["需更多验证"], "MIT", 2024),
    "cellplm": ModelInfo("CellPLM", ModelArchitecture.PREFIX_LM, 3_000_000, "多组学", "~60M", "", "", ["cell_annotation", "perturbation"], ["前缀注意力"], ["较新"], "MIT", 2024),
    "tgpt": ModelInfo("tGPT", ModelArchitecture.GPT_DECODER, 1_000_000, "scRNA-seq", "~50M", "", "", ["generation"], ["生成建模"], ["任务有限"], "MIT", 2023),
    "cellbert": ModelInfo("CellBert", ModelArchitecture.BERT_ENCODER, 2_000_000, "scRNA-seq", "~70M", "", "", ["cell_annotation"], ["细胞表征"], ["功能有限"], "Apache-2.0", 2023),
    "cpa": ModelInfo("CPA", ModelArchitecture.VAE, 0, "扰动数据", "~30M", "https://www.nature.com/articles/s41592-023-02132-y", "https://github.com/facebookresearch/CPA", ["perturbation"], ["组合扰动"], ["仅扰动任务"], "MIT", 2023),
    "gears": ModelInfo("GEARS", ModelArchitecture.GNN_TRANSFORMER, 0, "扰动数据", "~20M", "https://www.nature.com/articles/s41587-023-01905-6", "https://github.com/snap-stanford/GEARS", ["perturbation"], ["基因扰动推断"], ["仅扰动任务"], "MIT", 2023),
    "xtrimosc": ModelInfo("xTrimoSCPerturb", ModelArchitecture.HYBRID, 50_000_000, "scFoundation派生", "~100M", "", "", ["perturbation"], ["VCC 2025冠军"], ["依赖scFoundation"], "MIT", 2025),
}


def get_model_info(name: str) -> Optional[ModelInfo]:
    """获取模型信息。"""
    return MODELS_INFO.get(name.lower())


def list_models() -> list[dict]:
    """列出所有模型。"""
    return [
        {"key": k, "name": v.name, "architecture": v.architecture.value,
         "pretrain_cells": v.pretrain_cells, "year": v.year,
         "tasks": v.supported_tasks}
        for k, v in MODELS_INFO.items()
    ]


def get_all_model_keys() -> list[str]:
    """获取所有模型key。"""
    return list(MODELS_INFO.keys())
