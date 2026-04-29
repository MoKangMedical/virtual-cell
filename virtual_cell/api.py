"""
VirtualCell REST API

提供以下接口：
- GET  /health           健康检查
- GET  /models           列出所有模型
- GET  /datasets         列出所有数据集
- GET  /tasks            列出所有任务
- POST /generate         生成架构（CellForge）
- POST /benchmark        运行评估
- GET  /leaderboard      获取排行榜
- POST /predict          单模型预测
- GET  /info/{model}     模型详情
- POST /pipeline/run     完整VCC Pipeline
- GET  /leaderboard/{task} 按任务筛选排行榜
- POST /compare          对比两个模型
- GET  /generators       列出所有生成器
- GET  /stats            平台统计
- GET  /models/{name}/detail 模型详情（含论文/架构/参数量）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="VirtualCell API",
    version="0.4.0",
    description="单细胞基础模型Benchmark平台API",
)


def _benchmark_snapshot_path() -> Path:
    """Locate the bundled benchmark snapshot relative to the package root."""
    return Path(__file__).resolve().parents[1] / "docs" / "benchmark.json"


def _leaderboard_entry_from_result(result: dict) -> dict:
    """Normalize a persisted benchmark result into leaderboard shape."""
    metrics = result.get("metrics", {}) or {}
    primary_score = next(
        (
            float(value)
            for value in metrics.values()
            if isinstance(value, (int, float))
        ),
        0.0,
    )
    return {
        "model": result.get("model", ""),
        "dataset": result.get("dataset", ""),
        "task": result.get("task", ""),
        "primary_score": primary_score,
        "all_metrics": metrics,
    }


def _load_persisted_leaderboard(task: Optional[str] = None) -> list[dict]:
    """Load leaderboard rows from the bundled benchmark snapshot if present."""
    snapshot_path = _benchmark_snapshot_path()
    if not snapshot_path.exists():
        return []

    with snapshot_path.open() as handle:
        payload = json.load(handle)

    raw_results: list[dict]
    if isinstance(payload, dict):
        raw_results = payload.get("results", []) or []
    elif isinstance(payload, list):
        raw_results = payload
    else:
        return []

    board = []
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        if task and result.get("task") != task:
            continue
        board.append(_leaderboard_entry_from_result(result))

    board.sort(key=lambda item: -item["primary_score"])
    return board


# ====== Request/Response Models ======


class GenerateRequest(BaseModel):
    task: str  # perturbation/cell_annotation/integration/grn
    dataset: str
    n_architectures: int = 3
    mode: str = "mock"  # mock/full


class BenchmarkRequest(BaseModel):
    models: list[str] = []
    datasets: list[str] = []
    tasks: list[str] = []
    max_cells: int = 500


class PredictRequest(BaseModel):
    model: str
    task: str
    dataset: str
    n_cells: int = 500


class PipelineRunRequest(BaseModel):
    task: str = Field(..., description="任务类型: perturbation/cell_annotation/integration/grn")
    dataset: str = Field(..., description="目标数据集名称")
    n_architectures: int = Field(default=3, ge=1, le=10, description="生成候选架构数")
    max_cells: int = Field(default=500, ge=1, description="最大细胞数")
    mode: str = Field(default="mock", description="运行模式: mock/full")


class CompareRequest(BaseModel):
    model1: str = Field(..., description="第一个模型名称")
    model2: str = Field(..., description="第二个模型名称")
    datasets: list[str] = Field(default=[], description="数据集列表")
    tasks: list[str] = Field(default=[], description="任务列表")
    max_cells: int = Field(default=500, ge=1)


# ====== Endpoints ======


@app.get("/health")
async def health():
    from virtual_cell import __version__
    from virtual_cell.registry import ModelRegistry, DatasetRegistry

    return {
        "status": "healthy",
        "version": __version__,
        "models": len(ModelRegistry.list()),
        "datasets": len(DatasetRegistry.list()),
    }


@app.get("/models")
async def list_models():
    from virtual_cell.registry import ModelRegistry

    return ModelRegistry.list()


@app.get("/datasets")
async def list_datasets():
    from virtual_cell.registry import DatasetRegistry

    return DatasetRegistry.list()


@app.get("/tasks")
async def list_tasks():
    return [
        {
            "name": "cell_annotation",
            "display": "细胞类型注释",
            "metrics": ["accuracy", "f1_macro"],
        },
        {
            "name": "perturbation",
            "display": "扰动预测",
            "metrics": ["mse", "mae", "pcc", "pds"],
        },
        {
            "name": "integration",
            "display": "批次整合",
            "metrics": ["kbet", "lisi", "asw"],
        },
        {
            "name": "grn",
            "display": "基因调控网络推断",
            "metrics": ["auprc", "auroc"],
        },
    ]


@app.post("/generate")
async def generate_architecture(req: GenerateRequest):
    from virtual_cell.generators import CellForgeGenerator

    if req.mode == "full":
        from virtual_cell.generators.cellforge_full import CellForgeFullGenerator

        gen = CellForgeFullGenerator()
    else:
        gen = CellForgeGenerator()

    result = gen.generate(req.task, req.dataset, n_architectures=req.n_architectures)

    return {
        "task_analysis": result.task_analysis,
        "architectures": [a.to_dict() for a in result.architectures],
        "design_history": result.design_history,
    }


@app.post("/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.registry import ModelRegistry, DatasetRegistry

    models = req.models or [m["key"] for m in ModelRegistry.list()[:5]]
    datasets = req.datasets or [d["key"] for d in DatasetRegistry.list()[:3]]
    tasks = req.tasks or ["cell_annotation", "perturbation"]

    bench = Benchmark()
    result = bench.run(models=models, datasets=datasets, tasks=tasks, max_cells=req.max_cells)

    return {
        "leaderboard": result.get_leaderboard(),
        "n_results": len(result.results),
        "execution_time_ms": result.execution_time_ms,
    }


@app.get("/leaderboard")
async def get_leaderboard(task: Optional[str] = None, top_n: int = 20):
    lb = _load_persisted_leaderboard(task)
    return {"leaderboard": lb[:top_n], "task": task or "all"}


@app.post("/predict")
async def predict(req: PredictRequest):
    from virtual_cell.registry import load_model
    from virtual_cell.datasets import create_dataset

    model = load_model(req.model)
    dataset = create_dataset(req.dataset)
    dataset.load(max_cells=req.n_cells)
    model.load()

    result = model.predict(dataset, req.task, n_cells=req.n_cells)

    return {
        "model": result.model_name,
        "task": result.task,
        "n_predictions": len(result.predictions),
        "metadata": result.metadata,
    }


@app.get("/info/{model_name}")
async def model_info(model_name: str):
    from virtual_cell.registry import ModelRegistry

    models = {m["key"]: m for m in ModelRegistry.list()}
    if model_name not in models:
        raise HTTPException(404, f"Model {model_name} not found")
    return models[model_name]


# ====== New Endpoints (v0.4.0) ======


@app.post("/api/v1/pipeline/run")
async def run_pipeline(req: PipelineRunRequest):
    """运行完整VCC Pipeline：生成架构 → 适配 → 评估 → 排行榜。"""
    from virtual_cell.benchmark import Benchmark

    valid_tasks = ["perturbation", "cell_annotation", "integration", "grn"]
    if req.task not in valid_tasks:
        raise HTTPException(400, f"Invalid task: {req.task}. Valid: {valid_tasks}")

    bench = Benchmark()
    result = bench.generate_and_evaluate(
        task=req.task,
        dataset=req.dataset,
        n_architectures=req.n_architectures,
        max_cells=req.max_cells,
    )

    leaderboard = result.get_leaderboard()
    return {
        "status": "completed",
        "task": req.task,
        "dataset": req.dataset,
        "n_architectures": req.n_architectures,
        "leaderboard": leaderboard,
        "n_results": len(result.results),
        "execution_time_ms": result.execution_time_ms,
        "design_history": result.metadata.get("design_history", []),
        "task_analysis": result.metadata.get("task_analysis", {}),
    }


@app.get("/api/v1/leaderboard/{task}")
async def get_leaderboard_by_task(task: str, top_n: int = 20):
    """按任务筛选排行榜。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.registry import ModelRegistry, DatasetRegistry

    valid_tasks = ["perturbation", "cell_annotation", "integration", "grn"]
    if task not in valid_tasks:
        raise HTTPException(400, f"Invalid task: {task}. Valid: {valid_tasks}")

    bench = Benchmark()
    models = [m["key"] for m in ModelRegistry.list()[:5]]
    datasets = [d["key"] for d in DatasetRegistry.list()[:3]]
    result = bench.run(models=models, datasets=datasets, tasks=[task], max_cells=200, max_genes=50)
    lb = result.get_leaderboard(task)
    return {"task": task, "leaderboard": lb[:top_n], "total_entries": len(lb)}


@app.post("/api/v1/compare")
async def compare_models(req: CompareRequest):
    """对比两个模型的benchmark结果。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.registry import ModelRegistry, DatasetRegistry

    # Validate models exist
    available_models = {m["key"] for m in ModelRegistry.list()}
    if req.model1 not in available_models:
        raise HTTPException(404, f"Model '{req.model1}' not found. Available: {sorted(available_models)}")
    if req.model2 not in available_models:
        raise HTTPException(404, f"Model '{req.model2}' not found. Available: {sorted(available_models)}")

    datasets = req.datasets or [d["key"] for d in DatasetRegistry.list()[:3]]
    tasks = req.tasks or ["cell_annotation", "perturbation"]

    bench = Benchmark()
    result = bench.run(
        models=[req.model1, req.model2],
        datasets=datasets,
        tasks=tasks,
        max_cells=req.max_cells,
    )

    lb = result.get_leaderboard()

    # Split results by model
    m1_results = [e for e in lb if e["model"] == req.model1]
    m2_results = [e for e in lb if e["model"] == req.model2]

    m1_avg = sum(e["primary_score"] for e in m1_results) / max(1, len(m1_results))
    m2_avg = sum(e["primary_score"] for e in m2_results) / max(1, len(m2_results))

    winner = req.model1 if m1_avg > m2_avg else req.model2 if m2_avg > m1_avg else "tie"

    return {
        "model1": {"name": req.model1, "avg_score": round(m1_avg, 4), "results": m1_results},
        "model2": {"name": req.model2, "avg_score": round(m2_avg, 4), "results": m2_results},
        "winner": winner,
        "tasks_compared": tasks,
        "datasets_compared": datasets,
        "n_evaluations": len(result.results),
    }


@app.get("/api/v1/generators")
async def list_generators():
    """列出所有架构生成器。"""
    return {
        "generators": [
            {
                "name": "CellForge",
                "mode": "mock",
                "description": "基于文献知识的架构模板生成，无需GPU",
                "supported_tasks": ["perturbation", "cell_annotation", "integration", "grn"],
                "endpoint": "/generate",
            },
            {
                "name": "CellForgeFull",
                "mode": "full",
                "description": "多Agent LLM驱动的完整架构设计流程",
                "supported_tasks": ["perturbation", "cell_annotation", "integration", "grn"],
                "endpoint": "/generate",
                "requirements": ["CellForge安装", "LLM API密钥"],
            },
        ],
        "total": 2,
    }


@app.get("/api/v1/stats")
async def platform_stats():
    """平台统计信息。"""
    from virtual_cell import __version__
    from virtual_cell.registry import ModelRegistry, DatasetRegistry

    models = ModelRegistry.list()
    datasets = DatasetRegistry.list()

    return {
        "platform": "VirtualCell",
        "version": __version__,
        "n_models": len(models),
        "n_datasets": len(datasets),
        "n_tasks": 4,
        "n_generators": 2,
        "model_names": [m["name"] for m in models],
        "dataset_names": [d["name"] for d in datasets],
        "supported_tasks": ["cell_annotation", "perturbation", "integration", "grn"],
    }


@app.get("/api/v1/models/{model_name}/detail")
async def model_detail(model_name: str):
    """模型详情（含论文/架构/参数量）。"""
    from virtual_cell.registry import ModelRegistry, load_model

    models = {m["key"]: m for m in ModelRegistry.list()}
    if model_name not in models:
        raise HTTPException(404, f"Model '{model_name}' not found. Available: {sorted(models.keys())}")

    base_info = models[model_name]

    # Try to get additional detail from the model instance
    try:
        model = load_model(model_name)
        info = model.info
        detail = {
            "key": model_name,
            "name": info.name,
            "architecture": info.architecture.value if hasattr(info.architecture, 'value') else str(info.architecture),
            "pretrain_cells": info.pretrain_cells,
            "pretrain_data": info.pretrain_data,
            "parameters": info.parameters,
            "paper": info.paper,
            "code_repo": info.code_repo,
            "supported_tasks": info.supported_tasks,
            "strengths": info.strengths,
            "weaknesses": info.weaknesses,
            "license": info.license,
            "year": info.year,
        }
    except Exception:
        detail = base_info

    return detail
