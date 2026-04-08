"""
VirtualCell — 单细胞基础模型Benchmark平台

统一评估14个基础模型在26个数据集上的4大任务表现。
"""

__version__ = "0.1.0"
__author__ = "MoKangMedical"

from .benchmark import Benchmark, BenchmarkResult
from .models.base import BaseModel, ModelInfo
from .datasets.base import BaseDataset, DatasetInfo
from .tasks import (
    CellAnnotationTask, PerturbationPredictionTask,
    BatchIntegrationTask, GRNInferenceTask, TaskResult,
)
from .registry import ModelRegistry, DatasetRegistry, load_model, load_dataset
from .report import BenchmarkReport

__all__ = [
    "Benchmark", "BenchmarkResult",
    "BaseModel", "ModelInfo",
    "BaseDataset", "DatasetInfo",
    "CellAnnotationTask", "PerturbationPredictionTask",
    "BatchIntegrationTask", "GRNInferenceTask", "TaskResult",
    "ModelRegistry", "DatasetRegistry",
    "load_model", "load_dataset",
    "BenchmarkReport",
]
