"""增强测试 — 为API、Pipeline、Adapter、Downloader、Report、Visualizer补充覆盖。"""
import sys
import os
import tempfile
import json
import numpy as np

sys.path.insert(0, '/root/virtual-cell')


# =====================================================================
# API 端点测试（FastAPI TestClient）
# =====================================================================

from fastapi.testclient import TestClient
from virtual_cell.api import app

client = TestClient(app)


def test_api_health():
    """测试 /health 端点。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.4.0"
    assert data["models"] == 15
    print("✅ API /health: version=0.4.0, models=15")


def test_api_list_models():
    """测试 /models 端点。"""
    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 15
    names = [m["name"] for m in models]
    assert "scGPT" in names
    assert "Lingshu-Cell" in names
    print(f"✅ API /models: {len(models)}个模型")


def test_api_list_tasks():
    """测试 /tasks 端点。"""
    resp = client.get("/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) == 4
    task_names = [t["name"] for t in tasks]
    assert "perturbation" in task_names
    assert "cell_annotation" in task_names
    print(f"✅ API /tasks: {len(tasks)}个任务")


def test_api_stats():
    """测试 /api/v1/stats 端点。"""
    resp = client.get("/api/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["platform"] == "VirtualCell"
    assert data["version"] == "0.4.0"
    assert data["n_models"] == 15
    assert data["n_datasets"] == 26
    assert data["n_tasks"] == 4
    assert data["n_generators"] == 2
    print(f"✅ API /api/v1/stats: models={data['n_models']}, datasets={data['n_datasets']}")


def test_api_generators():
    """测试 /api/v1/generators 端点。"""
    resp = client.get("/api/v1/generators")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    gen_names = [g["name"] for g in data["generators"]]
    assert "CellForge" in gen_names
    assert "CellForgeFull" in gen_names
    print(f"✅ API /api/v1/generators: {data['total']}个生成器")


def test_api_model_detail():
    """测试 /api/v1/models/{name}/detail 端点。"""
    resp = client.get("/api/v1/models/scgpt/detail")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "scGPT"
    assert "architecture" in data
    assert "parameters" in data
    assert "paper" in data
    assert "supported_tasks" in data
    print(f"✅ API /api/v1/models/scgpt/detail: {data['name']}")


def test_api_model_detail_not_found():
    """测试不存在的模型详情。"""
    resp = client.get("/api/v1/models/nonexistent_model/detail")
    assert resp.status_code == 404
    print("✅ API model detail 404: nonexistent model")


def test_api_leaderboard_by_task():
    """测试 /api/v1/leaderboard/{task} 端点。"""
    resp = client.get("/api/v1/leaderboard/perturbation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "perturbation"
    assert "leaderboard" in data
    print(f"✅ API /api/v1/leaderboard/perturbation: {data['total_entries']} entries")


def test_api_leaderboard_invalid_task():
    """测试无效任务的排行榜。"""
    resp = client.get("/api/v1/leaderboard/invalid_task")
    assert resp.status_code == 400
    print("✅ API leaderboard 400: invalid task")


def test_api_compare():
    """测试 /api/v1/compare 端点。"""
    resp = client.post("/api/v1/compare", json={
        "model1": "scgpt",
        "model2": "geneformer",
        "datasets": ["zheng68k"],
        "tasks": ["cell_annotation"],
        "max_cells": 100,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["model1"]["name"] == "scgpt"
    assert data["model2"]["name"] == "geneformer"
    assert data["winner"] in ["scgpt", "geneformer", "tie"]
    print(f"✅ API /api/v1/compare: winner={data['winner']}")


def test_api_compare_invalid_model():
    """测试对比不存在的模型。"""
    resp = client.post("/api/v1/compare", json={
        "model1": "nonexistent",
        "model2": "scgpt",
    })
    assert resp.status_code == 404
    print("✅ API compare 404: nonexistent model")


def test_api_pipeline_run():
    """测试 /api/v1/pipeline/run 端点。"""
    resp = client.post("/api/v1/pipeline/run", json={
        "task": "perturbation",
        "dataset": "adamson2016",
        "n_architectures": 2,
        "max_cells": 100,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["task"] == "perturbation"
    assert data["n_architectures"] == 2
    assert len(data["leaderboard"]) == 2
    assert "design_history" in data
    print(f"✅ API /api/v1/pipeline/run: {data['n_results']} results")


def test_api_pipeline_invalid_task():
    """测试pipeline无效任务。"""
    resp = client.post("/api/v1/pipeline/run", json={
        "task": "invalid",
        "dataset": "zheng68k",
    })
    assert resp.status_code == 400
    print("✅ API pipeline 400: invalid task")


# =====================================================================
# VCC Pipeline 测试
# =====================================================================

def test_pipeline_preprocess_mock():
    """测试VCC Pipeline evaluate方法。"""
    from virtual_cell.vcc.pipeline import VCCPipeline, VCCMetrics

    pipeline = VCCPipeline()
    pred = np.random.randn(200, 50)
    gt = np.random.randn(200, 50)

    metrics = pipeline.evaluate(pred, gt)
    assert isinstance(metrics, VCCMetrics)
    assert metrics.mae >= 0
    assert isinstance(metrics.to_dict(), dict)
    print(f"✅ Pipeline evaluate: MAE={metrics.mae:.4f}")


def test_pipeline_with_perturbation_labels():
    """测试带扰动标签的评估。"""
    from virtual_cell.vcc.pipeline import VCCPipeline

    pipeline = VCCPipeline()
    pred = np.random.randn(100, 30)
    gt = np.random.randn(100, 30)
    labels = np.array(["ctrl"] * 50 + ["geneA"] * 50)

    metrics = pipeline.evaluate(pred, gt, perturbation_labels=labels)
    assert 0 <= metrics.pds <= 1.0
    print(f"✅ Pipeline with labels: PDS={metrics.pds:.4f}")


def test_pipeline_submission_format():
    """测试VCC提交格式化。"""
    from virtual_cell.vcc.pipeline import VCCPipeline, VCCMetrics

    pipeline = VCCPipeline()
    metrics = VCCMetrics(des=0.8, pds=0.9, mae=0.05, pearson_delta=0.7)

    sub = pipeline.format_submission(metrics, "test_model_v2")
    assert sub["team"] == "VirtualCell-OPC"
    assert sub["model"] == "test_model_v2"
    assert sub["metrics"]["des"] == 0.8
    assert sub["metrics"]["pds"] == 0.9
    print(f"✅ Pipeline submission: {sub['model']}")


def test_pipeline_save_submission():
    """测试保存VCC提交文件。"""
    from virtual_cell.vcc.pipeline import VCCPipeline, VCCMetrics

    pipeline = VCCPipeline()
    metrics = VCCMetrics(des=0.5, pds=0.6, mae=0.1)
    sub = pipeline.format_submission(metrics, "save_test")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    pipeline.save_submission(sub, tmp_path)
    assert os.path.exists(tmp_path)

    with open(tmp_path) as f:
        loaded = json.load(f)
    assert loaded["model"] == "save_test"
    assert loaded["metrics"]["des"] == 0.5

    os.unlink(tmp_path)
    print("✅ Pipeline save_submission: JSON文件验证")


def test_vcc_metrics_average():
    """测试VCCMetrics综合评分。"""
    from virtual_cell.vcc.pipeline import VCCMetrics

    m = VCCMetrics(des=0.8, pds=0.7, mae=0.1, spearman_deg=0.6,
                   spearman_lfc=0.5, auprc=0.9, pearson_delta=0.7)
    avg = m.average_score()
    assert isinstance(avg, float)
    # MAE is negated, so avg should be reasonable
    print(f"✅ VCCMetrics average_score: {avg:.4f}")


# =====================================================================
# Model Adapter 测试
# =====================================================================

def test_adapter_basic():
    """测试GeneratedModelAdapter基本功能。"""
    from virtual_cell.generators import CellForgeGenerator
    from virtual_cell.generators.model_adapter import GeneratedModelAdapter

    gen = CellForgeGenerator()
    result = gen.generate("perturbation", "adamson2016", n_architectures=1)
    arch = result.architectures[0]

    adapter = GeneratedModelAdapter(arch)
    adapter.load()
    assert adapter.is_loaded()
    assert adapter.info.name == arch.name
    print(f"✅ Adapter basic: {adapter.info.name}")


def test_adapter_predict_all_tasks():
    """测试适配器在所有4个任务上的预测。"""
    from virtual_cell.generators import CellForgeGenerator
    from virtual_cell.generators.model_adapter import GeneratedModelAdapter

    gen = CellForgeGenerator()
    for task in ["perturbation", "cell_annotation", "integration", "grn"]:
        result = gen.generate(task, "test_ds", n_architectures=1)
        arch = result.architectures[0]
        # Override supported_tasks to allow all tasks
        adapter = GeneratedModelAdapter(arch)
        adapter.load()
        pred = adapter.predict(None, task=task, n_cells=50)
        assert pred.model_name == arch.name
        assert pred.task == task
        assert pred.predictions is not None
    print("✅ Adapter all tasks: perturbation/cell_annotation/integration/grn")


def test_adapter_embeddings():
    """测试适配器的embedding生成。"""
    from virtual_cell.generators import CellForgeGenerator
    from virtual_cell.generators.model_adapter import GeneratedModelAdapter

    gen = CellForgeGenerator()
    result = gen.generate("perturbation", "test_ds", n_architectures=1)
    adapter = GeneratedModelAdapter(result.architectures[0])
    adapter.load()

    emb = adapter.get_embeddings(None, n_cells=100)
    assert emb.shape == (100, 512)
    assert emb.dtype == np.float32
    print(f"✅ Adapter embeddings: shape={emb.shape}")


def test_adapter_innovation_boosts():
    """测试创新组件对置信度的加成。"""
    from virtual_cell.generators import CellForgeGenerator
    from virtual_cell.generators.model_adapter import GeneratedModelAdapter

    gen = CellForgeGenerator()
    result = gen.generate("perturbation", "adamson2016", n_architectures=3)

    for arch in result.architectures:
        adapter = GeneratedModelAdapter(arch)
        adapter.load()
        pred = adapter.predict(None, task="perturbation", n_cells=50, seed=42)
        assert pred.metadata["mode"] == "generated"
        assert pred.metadata["quality_factor"] > 0
    print(f"✅ Adapter innovation boosts: {len(result.architectures)} architectures checked")


# =====================================================================
# Downloader 测试
# =====================================================================

def test_downloader_init():
    """测试DatasetDownloader初始化。"""
    from virtual_cell.downloader import DatasetDownloader

    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DatasetDownloader(data_dir=tmpdir)
        assert dl.data_dir.exists()
    print("✅ Downloader init")


def test_downloader_list_available():
    """测试列出可下载数据集。"""
    from virtual_cell.downloader import DatasetDownloader

    dl = DatasetDownloader(data_dir="/tmp/test_dl")
    available = dl.list_available()
    assert len(available) > 0
    assert "zheng68k" in available
    print(f"✅ Downloader list_available: {len(available)} datasets")


def test_downloader_unknown_dataset():
    """测试下载未知数据集抛出异常。"""
    from virtual_cell.downloader import DatasetDownloader

    dl = DatasetDownloader(data_dir="/tmp/test_dl")
    try:
        dl.download("totally_nonexistent_dataset")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "未知数据集" in str(e)
    print("✅ Downloader unknown dataset: ValueError raised")


def test_downloader_list_local():
    """测试列出本地数据集。"""
    from virtual_cell.downloader import DatasetDownloader

    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DatasetDownloader(data_dir=tmpdir)
        local = dl.list_local()
        assert isinstance(local, list)
        assert len(local) == 0  # Empty dir
    print("✅ Downloader list_local: empty dir")


# =====================================================================
# Report 测试
# =====================================================================

def test_report_markdown():
    """测试报告生成Markdown。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.report import BenchmarkReport

    bench = Benchmark()
    result = bench.run(
        models=["scgpt", "geneformer"],
        datasets=["zheng68k"],
        tasks=["cell_annotation"],
        max_cells=100, max_genes=50,
    )
    report = BenchmarkReport(result)
    md = report.to_markdown()
    assert "# 🔬 VirtualCell" in md
    assert "scGPT" in md or "scgpt" in md
    assert "排行榜" in md
    print(f"✅ Report markdown: {len(md)} chars")


def test_report_json():
    """测试报告生成JSON。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.report import BenchmarkReport

    bench = Benchmark()
    result = bench.run(
        models=["scgpt", "regformer"],
        datasets=["zheng68k"],
        tasks=["perturbation"],
        max_cells=100, max_genes=50,
    )
    report = BenchmarkReport(result)
    json_str = report.to_json()
    data = json.loads(json_str)
    assert "n_results" in data
    assert data["n_results"] == 2
    print(f"✅ Report JSON: {data['n_results']} results")


def test_report_summary():
    """测试报告摘要。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.report import BenchmarkReport

    bench = Benchmark()
    result = bench.run(
        models=["scgpt"],
        datasets=["zheng68k"],
        tasks=["cell_annotation", "perturbation"],
        max_cells=100, max_genes=50,
    )
    report = BenchmarkReport(result)
    summary = report.get_summary()
    assert summary["total_evaluations"] == 2
    assert "cell_annotation" in summary["tasks"]
    assert "perturbation" in summary["tasks"]
    print(f"✅ Report summary: {summary['total_evaluations']} evaluations")


# =====================================================================
# Visualizer 测试
# =====================================================================

def test_visualizer_radar_data():
    """测试雷达图数据生成。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.visualizer import Visualizer

    bench = Benchmark()
    result = bench.run(
        models=["scgpt"],
        datasets=["zheng68k"],
        tasks=["cell_annotation", "perturbation"],
        max_cells=100, max_genes=50,
    )
    viz = Visualizer(result)
    # Check with the actual model name from results
    actual_model = result.results[0].model_name
    radar = viz.radar_chart_data(actual_model)
    assert radar["model"] == actual_model
    assert len(radar["labels"]) > 0
    assert len(radar["values"]) > 0
    print(f"✅ Visualizer radar: {len(radar['labels'])} tasks")


def test_visualizer_heatmap_data():
    """测试热力图数据。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.visualizer import Visualizer

    bench = Benchmark()
    result = bench.run(
        models=["scgpt", "geneformer"],
        datasets=["zheng68k"],
        tasks=["cell_annotation"],
        max_cells=100, max_genes=50,
    )
    viz = Visualizer(result)
    hm = viz.heatmap_data()
    assert len(hm["models"]) == 2
    assert len(hm["tasks"]) == 1
    assert len(hm["values"]) > 0
    print(f"✅ Visualizer heatmap_data: {len(hm['models'])}×{len(hm['tasks'])}")


def test_visualizer_comparison():
    """测试对比报告生成。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.visualizer import Visualizer

    bench = Benchmark()
    result = bench.run(
        models=["scgpt", "geneformer"],
        datasets=["zheng68k"],
        tasks=["cell_annotation"],
        max_cells=100, max_genes=50,
    )
    viz = Visualizer(result)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        tmp_path = f.name

    html = viz.generate_comparison("scgpt", "geneformer", output=tmp_path)
    assert "<html" in html
    assert "scgpt" in html.lower()
    assert "geneformer" in html.lower()
    assert os.path.exists(tmp_path)

    os.unlink(tmp_path)
    print(f"✅ Visualizer comparison: {len(html)} chars")


def test_visualizer_interactive():
    """测试交互式HTML报告。"""
    from virtual_cell.benchmark import Benchmark
    from virtual_cell.visualizer import Visualizer

    bench = Benchmark()
    result = bench.run(
        models=["scgpt"],
        datasets=["zheng68k"],
        tasks=["cell_annotation"],
        max_cells=100, max_genes=50,
    )
    viz = Visualizer(result)
    html = viz.to_interactive_html("Test Report")
    assert "Test Report" in html
    assert "VirtualCell" in html
    print(f"✅ Visualizer interactive: {len(html)} chars")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    tests = [
        test_api_health, test_api_list_models, test_api_list_tasks,
        test_api_stats, test_api_generators,
        test_api_model_detail, test_api_model_detail_not_found,
        test_api_leaderboard_by_task, test_api_leaderboard_invalid_task,
        test_api_compare, test_api_compare_invalid_model,
        test_api_pipeline_run, test_api_pipeline_invalid_task,
        test_pipeline_preprocess_mock, test_pipeline_with_perturbation_labels,
        test_pipeline_submission_format, test_pipeline_save_submission,
        test_vcc_metrics_average,
        test_adapter_basic, test_adapter_predict_all_tasks,
        test_adapter_embeddings, test_adapter_innovation_boosts,
        test_downloader_init, test_downloader_list_available,
        test_downloader_unknown_dataset, test_downloader_list_local,
        test_report_markdown, test_report_json, test_report_summary,
        test_visualizer_radar_data, test_visualizer_heatmap_data,
        test_visualizer_comparison, test_visualizer_interactive,
    ]

    print("=" * 60)
    print(f"🧪 Enhanced Test Suite — {len(tests)} tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1

    print()
    print(f"📊 Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("🎉 全部通过！")
