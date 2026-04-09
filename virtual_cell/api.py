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
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="VirtualCell API",
    version="0.3.0",
    description="单细胞基础模型Benchmark平台API",
)


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
    from virtual_cell.benchmark import Benchmark

    bench = Benchmark()

    import json
    import os

    bench_file = "/root/virtual-cell/docs/benchmark.json"
    if os.path.exists(bench_file):
        with open(bench_file) as f:
            data = json.load(f)
            for r in data:
                pass  # Use existing data

    lb = bench.get_leaderboard(task)
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
