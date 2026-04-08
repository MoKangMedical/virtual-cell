"""
评估任务 — 4大任务的统一接口

1. 细胞注释 (Cell Annotation)
2. 扰动预测 (Perturbation Prediction)
3. 批次整合 (Batch Integration)
4. 基因调控网络推断 (GRN Inference)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskResult:
    """任务评估结果。"""
    task_name: str
    model_name: str
    dataset_name: str
    metrics: dict[str, float]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task_name,
            "model": self.model_name,
            "dataset": self.dataset_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        metrics_str = " | ".join(f"{k}: {v:.3f}" for k, v in self.metrics.items())
        return f"[{self.task_name}] {self.model_name} @ {self.dataset_name}: {metrics_str}"


class BaseTask(ABC):
    """评估任务基类。"""

    def __init__(self, name: str, metrics: list[str]):
        self.name = name
        self.metrics = metrics

    @abstractmethod
    def evaluate(self, model, dataset, **kwargs) -> TaskResult:
        """执行评估。"""
        ...


# ================================================================
# 任务1：细胞类型注释
# ================================================================

class CellAnnotationTask(BaseTask):
    """
    细胞类型注释任务。

    指标：Accuracy, F1 (macro/weighted), AUROC
    """

    def __init__(self):
        super().__init__("cell_annotation", ["accuracy", "f1_macro", "f1_weighted", "auroc"])

    def evaluate(self, model, dataset, **kwargs) -> TaskResult:
        pred_result = model.predict(None, task="cell_annotation", n_cells=kwargs.get("n_cells", 100))
        predictions = pred_result.predictions

        # Mock评估：随机生成ground truth并计算指标
        n = len(predictions)
        cell_types = list(set(predictions))
        rng = np.random.RandomState(42)
        ground_truth = rng.choice(cell_types, size=n)

        # 模拟不同模型的准确率差异
        noise_rate = {"scgpt": 0.15, "geneformer": 0.2, "scbert": 0.12, "scfoundation": 0.18,
                      "regformer": 0.1, "nicheformer": 0.22}.get(model.info.name.lower(), 0.2)
        noisy_pred = ground_truth.copy()
        flip_idx = rng.choice(n, size=int(n * noise_rate), replace=False)
        noisy_pred[flip_idx] = rng.choice(cell_types, size=len(flip_idx))

        # 手动计算指标（无需sklearn）
        acc = float(np.mean(ground_truth == noisy_pred))

        # F1 macro
        f1_scores = []
        for ct in cell_types:
            tp = np.sum((noisy_pred == ct) & (ground_truth == ct))
            fp = np.sum((noisy_pred == ct) & (ground_truth != ct))
            fn = np.sum((noisy_pred != ct) & (ground_truth == ct))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        f1_macro = float(np.mean(f1_scores))

        return TaskResult(
            task_name=self.name,
            model_name=model.info.name,
            dataset_name=dataset.info.name,
            metrics={"accuracy": acc, "f1_macro": f1_macro, "auroc": acc * 0.95 + rng.uniform(0, 0.05)},
            metadata={"n_cells": n, "n_cell_types": len(cell_types)},
        )


# ================================================================
# 任务2：扰动预测
# ================================================================

class PerturbationPredictionTask(BaseTask):
    """
    基因扰动预测任务。

    指标：MSE, MAE, PCC (Pearson Correlation Coefficient), PDS
    """

    def __init__(self):
        super().__init__("perturbation", ["mse", "mae", "pcc", "pds"])

    def evaluate(self, model, dataset, **kwargs) -> TaskResult:
        n_genes = 100
        pred_result = model.predict(None, task="perturbation", n_cells=n_genes)
        predictions = pred_result.predictions

        # 模拟评估
        rng = np.random.RandomState(42)
        ground_truth = rng.randn(*predictions.shape)

        noise_scale = {"scgpt": 0.3, "geneformer": 0.4, "cpa": 0.25, "gears": 0.28,
                       "regformer": 0.2, "xtrimosc": 0.15}.get(model.info.name.lower(), 0.35)
        noisy_pred = ground_truth + rng.randn(*predictions.shape) * noise_scale

        mse = np.mean((ground_truth - noisy_pred) ** 2)
        mae = np.mean(np.abs(ground_truth - noisy_pred))
        pcc = np.corrcoef(ground_truth.flatten(), noisy_pred.flatten())[0, 1]
        pds = max(0, 1 - mse / np.var(ground_truth))  # 简化PDS

        return TaskResult(
            task_name=self.name,
            model_name=model.info.name,
            dataset_name=dataset.info.name,
            metrics={"mse": mse, "mae": mae, "pcc": pcc, "pds": pds},
            metadata={"n_perturbations": n_genes},
        )


# ================================================================
# 任务3：批次整合
# ================================================================

class BatchIntegrationTask(BaseTask):
    """
    批次整合任务。

    指标：kBET, LISI, ASW, Graph Connectivity
    """

    def __init__(self):
        super().__init__("integration", ["kbet", "lisi", "asw", "graph_connectivity"])

    def evaluate(self, model, dataset, **kwargs) -> TaskResult:
        n_cells = 500
        embeddings = model.get_embeddings(None, n_cells=n_cells)

        # 模拟批次效应评估
        rng = np.random.RandomState(42)
        batch_labels = rng.choice(3, size=n_cells)

        # 基于模型的模拟评分
        quality = {"scgpt": 0.85, "geneformer": 0.75, "scbert": 0.80,
                    "scfoundation": 0.78}.get(model.info.name.lower(), 0.7)
        noise = rng.uniform(-0.05, 0.05, 4)

        kbet = np.clip(quality + noise[0], 0, 1)  # 1-ideal
        lisi = np.clip(quality * 0.9 + noise[1], 0, 1)
        asw = np.clip(quality * 1.1 + noise[2], 0, 1)
        gc = np.clip(quality * 0.95 + noise[3], 0, 1)

        return TaskResult(
            task_name=self.name,
            model_name=model.info.name,
            dataset_name=dataset.info.name,
            metrics={"kbet": kbet, "lisi": lisi, "asw": asw, "graph_connectivity": gc},
            metadata={"n_cells": n_cells, "n_batches": 3},
        )


# ================================================================
# 任务4：基因调控网络推断
# ================================================================

class GRNInferenceTask(BaseTask):
    """
    基因调控网络推断任务。

    指标：AUPRC, AUROC
    """

    def __init__(self):
        super().__init__("grn", ["auprc", "auroc"])

    def evaluate(self, model, dataset, **kwargs) -> TaskResult:
        pred_result = model.predict(None, task="grn")
        adj_matrix = pred_result.predictions

        # 模拟评估
        rng = np.random.RandomState(42)
        n_genes = adj_matrix.shape[0]
        true_grn = (rng.rand(n_genes, n_genes) > 0.95).astype(float)  # 稀疏真实网络

        quality = {"regformer": 0.82, "geneformer": 0.72, "scgpt": 0.65,
                    "scprint": 0.75}.get(model.info.name.lower(), 0.6)
        noise = rng.uniform(-0.05, 0.05, 2)

        auprc = np.clip(quality + noise[0], 0, 1)
        auroc = np.clip(quality * 1.05 + noise[1], 0, 1)

        return TaskResult(
            task_name=self.name,
            model_name=model.info.name,
            dataset_name=dataset.info.name,
            metrics={"auprc": auprc, "auroc": auroc},
            metadata={"n_genes": n_genes, "sparsity": true_grn.sum() / true_grn.size},
        )


# 任务注册
TASK_REGISTRY = {
    "cell_annotation": CellAnnotationTask,
    "perturbation": PerturbationPredictionTask,
    "integration": BatchIntegrationTask,
    "grn": GRNInferenceTask,
}


def get_task(name: str) -> BaseTask:
    """获取任务实例。"""
    cls = TASK_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"未知任务: {name}. 可用: {list(TASK_REGISTRY.keys())}")
    return cls()


def list_tasks() -> list[dict]:
    """列出所有任务。"""
    return [
        {"key": k, "name": v().name, "metrics": v().metrics}
        for k, v in TASK_REGISTRY.items()
    ]
