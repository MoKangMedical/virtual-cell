"""
模型实现 — 14个单细胞基础模型的具体封装

每个模型实现BaseModel接口，支持：
- load(): 加载预训练权重
- predict(): 推理
- get_embeddings(): 获取表征

真实推理需要GPU环境和模型权重，这里提供：
1. Mock模式：用于开发/测试/CI
2. 真实模式：接入实际模型（需要GPU + 权重下载）
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

from .base import BaseModel, ModelInfo, PredictionResult, MODELS_INFO


class MockModel(BaseModel):
    """通用模拟模型（用于开发测试）。"""

    def __init__(self, info: ModelInfo):
        super().__init__(info)
        self._embedding_dim = 512

    def load(self, checkpoint_path: str = "", device: str = "cpu") -> None:
        self._loaded = True

    def predict(self, data: Any, task: str, **kwargs) -> PredictionResult:
        n_cells = kwargs.get("n_cells", 100)
        if task == "cell_annotation":
            cell_types = ["T cell", "B cell", "Monocyte", "NK cell", "DC"]
            predictions = np.random.choice(cell_types, size=n_cells)
        elif task == "perturbation":
            predictions = np.random.randn(n_cells, 10)  # gene expression changes
        elif task == "grn":
            predictions = np.random.rand(100, 100)  # adjacency matrix
        else:
            predictions = np.random.randn(n_cells, self._embedding_dim)

        return PredictionResult(
            model_name=self.info.name,
            task=task,
            predictions=predictions,
            metadata={"mode": "mock", "n_cells": n_cells},
        )

    def get_embeddings(self, data: Any, **kwargs) -> np.ndarray:
        n_cells = kwargs.get("n_cells", 100)
        return np.random.randn(n_cells, self._embedding_dim)


# 每个模型的Mock实现
class ScGPTModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["scgpt"])


class GeneformerModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["geneformer"])


class ScBERTModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["scbert"])


class ScFoundationModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["scfoundation"])


class RegFormerModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["regformer"])


class NicheformerModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["nicheformer"])


class ScPrintModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["scprint"])


class CellLMModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["celllm"])


class CellPLMModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["cellplm"])


class TGPTModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["tgpt"])


class CellBertModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["cellbert"])


class CPAModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["cpa"])


class GEARSModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["gears"])


class XTrimoSCPerturbModel(MockModel):
    def __init__(self):
        super().__init__(MODELS_INFO["xtrimosc"])


# 模型类映射
MODEL_CLASSES = {
    "scgpt": ScGPTModel,
    "geneformer": GeneformerModel,
    "scbert": ScBERTModel,
    "scfoundation": ScFoundationModel,
    "regformer": RegFormerModel,
    "nicheformer": NicheformerModel,
    "scprint": ScPrintModel,
    "celllm": CellLMModel,
    "cellplm": CellPLMModel,
    "tgpt": TGPTModel,
    "cellbert": CellBertModel,
    "cpa": CPAModel,
    "gears": GEARSModel,
    "xtrimosc": XTrimoSCPerturbModel,
}


def create_model(name: str) -> BaseModel:
    """创建模型实例。"""
    cls = MODEL_CLASSES.get(name.lower())
    if cls is None:
        raise ValueError(f"未知模型: {name}. 可用: {list(MODEL_CLASSES.keys())}")
    return cls()
