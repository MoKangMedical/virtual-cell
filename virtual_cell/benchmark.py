"""
Benchmark引擎 — 核心评估流程

协调模型×数据集×任务的评估矩阵。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .models.base import BaseModel, get_model_info, list_models, get_all_model_keys
from .models import create_model
from .datasets.base import BaseDataset, get_dataset_info, list_datasets, DATASETS_INFO
from .datasets import create_dataset
from .tasks import get_task, list_tasks, TaskResult, TASK_REGISTRY


def _validate_task_dataset_compat(tasks: list[str], datasets: list[str]) -> list[str]:
    """校验任务-数据集兼容性，返回警告列表（空=全部兼容）。"""
    warnings = []
    for ds_name in datasets:
        ds_info = DATASETS_INFO.get(ds_name.lower())
        if ds_info is None:
            warnings.append(f"未知数据集: {ds_name}")
            continue
        for task_name in tasks:
            if task_name not in TASK_REGISTRY:
                warnings.append(f"未知任务: {task_name}")
                continue
            if task_name not in ds_info.supported_tasks:
                warnings.append(
                    f"⚠️ {ds_name} 不支持 {task_name} 任务 "
                    f"(支持: {', '.join(ds_info.supported_tasks)})"
                )
    return warnings


@dataclass
class BenchmarkResult:
    """Benchmark评估结果。"""
    results: list[TaskResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_results": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    def get_leaderboard(self, task: str = "") -> list[dict]:
        """获取排行榜。"""
        filtered = self.results
        if task:
            filtered = [r for r in self.results if r.task_name == task]

        board = []
        for r in filtered:
            primary_metric = list(r.metrics.values())[0] if r.metrics else 0
            board.append({
                "model": r.model_name,
                "dataset": r.dataset_name,
                "task": r.task_name,
                "primary_score": primary_metric,
                "all_metrics": r.metrics,
            })

        board.sort(key=lambda x: -x["primary_score"])
        return board


class Benchmark:
    """
    Benchmark引擎。

    用法：
        bench = Benchmark()
        result = bench.run(
            models=["scgpt", "geneformer", "scbert"],
            datasets=["zheng68k", "kang2018"],
            tasks=["cell_annotation", "perturbation"],
        )
        print(result.get_leaderboard())
    """

    def __init__(self):
        self.results: list[TaskResult] = []

    def evaluate(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        task: str,
        **kwargs,
    ) -> TaskResult:
        """评估单个模型×数据集×任务。"""
        task_obj = get_task(task)
        if not model.is_loaded():
            model.load()
        if not dataset.is_loaded():
            dataset.load(**kwargs)

        result = task_obj.evaluate(model, dataset, **kwargs)
        self.results.append(result)
        return result

    def run(
        self,
        models: list[str],
        datasets: list[str],
        tasks: list[str],
        **kwargs,
    ) -> BenchmarkResult:
        """
        批量运行Benchmark。

        Args:
            models: 模型名称列表。
            datasets: 数据集名称列表。
            tasks: 任务名称列表。

        Returns:
            BenchmarkResult。
        """
        # 校验任务-数据集兼容性
        compat_warnings = _validate_task_dataset_compat(tasks, datasets)
        if compat_warnings:
            for w in compat_warnings:
                print(w)

        start = time.time()
        results = []

        for model_name in models:
            model = create_model(model_name)
            model.load()

            for dataset_name in datasets:
                dataset = create_dataset(dataset_name)
                dataset.load(**kwargs)

                for task_name in tasks:
                    try:
                        result = self.evaluate(model, dataset, task_name, **kwargs)
                        results.append(result)
                    except Exception as e:
                        results.append(TaskResult(
                            task_name=task_name,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            metrics={"error": 0.0},
                            metadata={"error": str(e)},
                        ))

        elapsed = (time.time() - start) * 1000
        self.results.extend(results)

        return BenchmarkResult(
            results=results,
            execution_time_ms=elapsed,
            metadata={
                "n_models": len(models),
                "n_datasets": len(datasets),
                "n_tasks": len(tasks),
                "total_evaluations": len(results),
            },
        )

    def run_all(
        self,
        tasks: list[str] = None,
        max_cells: int = 500,
        **kwargs,
    ) -> BenchmarkResult:
        """
        运行所有模型×所有数据集×指定任务。

        注意：这在Mock模式下可快速完成，真实模式需要大量计算资源。
        """
        from .models.base import get_all_model_keys
        from .datasets.base import filter_datasets

        models = get_all_model_keys()

        if tasks is None:
            tasks = ["cell_annotation", "perturbation", "integration", "grn"]

        # 为每个任务筛选合适的数据集
        task_datasets = {}
        for task in tasks:
            ds = filter_datasets(task=task, min_cells=1000)
            task_datasets[task] = [d["key"] for d in ds[:3]]  # 每任务最多3个数据集

        all_results = []
        start = time.time()

        for task_name, dataset_keys in task_datasets.items():
            for model_name in models:
                model = create_model(model_name)
                model.load()
                for dataset_name in dataset_keys:
                    dataset = create_dataset(dataset_name)
                    dataset.load(max_cells=max_cells)
                    try:
                        result = self.evaluate(model, dataset, task_name, **kwargs)
                        all_results.append(result)
                    except Exception:
                        pass

        elapsed = (time.time() - start) * 1000
        return BenchmarkResult(
            results=all_results,
            execution_time_ms=elapsed,
            metadata={"mode": "full_benchmark", "max_cells": max_cells},
        )

    def generate_and_evaluate(
        self,
        task: str,
        dataset: str,
        generator: str = "cellforge",
        n_architectures: int = 3,
        **kwargs,
    ) -> BenchmarkResult:
        """
        生成架构 → 自动评估 → 返回排行榜。

        这是核心闭环：CellForge生成 → VirtualCell评估 → 选出最优。

        Args:
            task: 任务类型（perturbation/cell_annotation/integration/grn）。
            dataset: 目标数据集名称。
            generator: 生成器名称（默认"cellforge"）。
            n_architectures: 生成候选数。

        Returns:
            BenchmarkResult含所有生成架构的评估结果和排行榜。
        """
        from .generators import CellForgeGenerator
        from .generators.model_adapter import GeneratedModelAdapter

        gen = CellForgeGenerator()

        # Phase 1-2: 生成N个候选架构
        gen_result = gen.generate(task, dataset, n_architectures=n_architectures)

        # Phase 3: 每个架构适配为BaseModel并评估
        results = []
        start = time.time()

        for arch in gen_result.architectures:
            model = GeneratedModelAdapter(arch)
            ds = create_dataset(dataset)
            ds.load(**kwargs)
            model.load()

            result = self.evaluate(model, ds, task, **kwargs)
            result.metadata["architecture"] = arch.to_dict()
            result.metadata["design_rationale"] = arch.design_rationale
            results.append(result)

        self.results.extend(results)

        elapsed = (time.time() - start) * 1000
        return BenchmarkResult(
            results=results,
            execution_time_ms=elapsed,
            metadata={
                "mode": "generate_and_evaluate",
                "generator": generator,
                "task_analysis": gen_result.task_analysis,
                "design_history": gen_result.design_history,
            },
        )
