"""
生成器基类 — 所有架构生成器的统一接口
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GeneratedArchitecture:
    """生成的架构描述。"""
    name: str                          # 模型名称
    task: str                          # 目标任务
    dataset: str                       # 目标数据集
    architecture_type: str             # 架构类型（如"transformer+diffusion"）
    layers: list[dict]                 # 层结构描述
    hyperparams: dict                  # 超参数
    innovations: list[str]             # 创新组件
    code: str = ""                     # 生成的PyTorch代码
    design_rationale: str = ""         # 设计推理过程
    confidence: float = 0.0            # 设计置信度

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "dataset": self.dataset,
            "architecture_type": self.architecture_type,
            "layers": self.layers,
            "hyperparams": self.hyperparams,
            "innovations": self.innovations,
            "confidence": self.confidence,
            "design_rationale": self.design_rationale,
            "has_code": bool(self.code),
        }


@dataclass
class GenerationResult:
    """生成结果。"""
    architectures: list[GeneratedArchitecture] = field(default_factory=list)
    task_analysis: dict = field(default_factory=dict)
    design_history: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def best(self) -> Optional[GeneratedArchitecture]:
        """返回置信度最高的架构。"""
        if not self.architectures:
            return None
        return max(self.architectures, key=lambda a: a.confidence)

    def to_dict(self) -> dict:
        return {
            "n_architectures": len(self.architectures),
            "architectures": [a.to_dict() for a in self.architectures],
            "task_analysis": self.task_analysis,
            "metadata": self.metadata,
        }


class BaseGenerator(ABC):
    """所有架构生成器的基类。"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(
        self,
        task: str,
        dataset: str,
        n_architectures: int = 3,
        **kwargs,
    ) -> GenerationResult:
        """生成N个候选架构。"""
        ...

    @abstractmethod
    def describe(self) -> str:
        """描述生成器能力。"""
        ...
