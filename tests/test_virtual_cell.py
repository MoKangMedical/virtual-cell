"""VirtualCell 测试套件。"""

import sys
import os
import tempfile

sys.path.insert(0, '/root/virtual-cell')

from virtual_cell import (
    Benchmark, load_model, load_dataset,
    BenchmarkReport, ModelRegistry, DatasetRegistry,
)


def test_model_registry():
    models = ModelRegistry.list()
    assert len(models) == 15, f"期望15个模型，实际{len(models)}"
    print(f"✅ 模型注册: {len(models)}个模型")


def test_dataset_registry():
    datasets = DatasetRegistry.list()
    assert len(datasets) == 26, f"期望26个数据集，实际{len(datasets)}"
    print(f"✅ 数据集注册: {len(datasets)}个数据集")


def test_load_model():
    model = load_model("scgpt")
    assert model.info.name == "scGPT"
    model.load()
    assert model.is_loaded()
    print(f"✅ 模型加载: {model}")


def test_load_dataset():
    ds = load_dataset("zheng68k")
    assert ds.info.name == "Zheng68K"
    ds.load(max_cells=100, max_genes=50)
    assert ds.is_loaded()
    splits = ds.get_splits()
    assert splits.train_X.shape[0] > 0
    print(f"✅ 数据集加载: {ds}")


def test_single_evaluation():
    model = load_model("scgpt")
    model.load()
    ds = load_dataset("zheng68k")
    ds.load(max_cells=100, max_genes=50)

    bench = Benchmark()
    result = bench.evaluate(model, ds, "cell_annotation")
    assert "accuracy" in result.metrics
    print(f"✅ 单次评估: {result.summary()}")


def test_benchmark_run():
    bench = Benchmark()
    result = bench.run(
        models=["scgpt", "geneformer", "scbert"],
        datasets=["zheng68k"],
        tasks=["cell_annotation", "perturbation"],
        max_cells=100, max_genes=50,
    )
    assert len(result.results) == 6  # 3 models × 2 tasks
    leaderboard = result.get_leaderboard()
    assert len(leaderboard) > 0
    print(f"✅ Benchmark运行: {len(result.results)}次评估")
    print(f"   排行榜Top 3:")
    for entry in leaderboard[:3]:
        print(f"   {entry['model']} @ {entry['task']}: {entry['primary_score']:.3f}")


def test_report_generation():
    bench = Benchmark()
    result = bench.run(
        models=["scgpt", "regformer"],
        datasets=["zheng68k"],
        tasks=["cell_annotation"],
        max_cells=100, max_genes=50,
    )
    report = BenchmarkReport(result)
    md = report.to_markdown()
    assert "# 🔬 VirtualCell" in md
    print(f"✅ 报告生成: {len(md)}字符")


def test_task_filter():
    perturbation_datasets = DatasetRegistry.filter(task="perturbation")
    assert len(perturbation_datasets) > 0
    print(f"✅ 任务筛选: perturbation有{len(perturbation_datasets)}个数据集")


def test_visualizer_heatmap():
    """测试Visualizer.generate_heatmap生成HTML热力图。"""
    from virtual_cell.visualizer import Visualizer

    bench = Benchmark()
    result = bench.run(
        models=["scgpt", "geneformer"],
        datasets=["zheng68k"],
        tasks=["cell_annotation", "perturbation"],
        max_cells=100, max_genes=50,
    )

    viz = Visualizer(result)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        tmp_path = f.name

    html = viz.generate_heatmap(output_path=tmp_path)
    assert "<html" in html
    assert "Model" in html or "model" in html
    assert os.path.exists(tmp_path)
    with open(tmp_path) as f:
        content = f.read()
    assert len(content) > 500

    os.unlink(tmp_path)
    print(f"✅ Visualizer heatmap: {len(html)}字符")


def test_visualizer_leaderboard():
    """测试Visualizer.generate_leaderboard_html生成排行榜HTML。"""
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

    html = viz.generate_leaderboard_html(output_path=tmp_path)
    assert "<html" in html
    assert "Leaderboard" in html or "leaderboard" in html or "排行" in html
    assert os.path.exists(tmp_path)

    os.unlink(tmp_path)
    print(f"✅ Visualizer leaderboard: {len(html)}字符")


def test_cli_leaderboard():
    """测试CLI leaderboard命令。"""
    from virtual_cell.cli import _cmd_leaderboard

    # Just ensure it doesn't crash — output goes to stdout
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_leaderboard()

    output = buf.getvalue()
    assert "排行榜" in output or "Leaderboard" in output or "scGPT" in output
    print(f"✅ CLI leaderboard: 输出{len(output)}字符")


def test_cli_report():
    """测试CLI report命令生成HTML文件。"""
    from virtual_cell.cli import _cmd_report
    import argparse

    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = os.path.join(tmpdir, "test_report")

        args = argparse.Namespace(
            models="scgpt,geneformer",
            datasets="zheng68k",
            tasks="cell_annotation",
            max_cells=100,
            max_genes=50,
            output=output_prefix,
        )

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            _cmd_report(args)

        heatmap_path = f"{output_prefix}_heatmap.html"
        leaderboard_path = f"{output_prefix}_leaderboard.html"

        assert os.path.exists(heatmap_path), f"热力图文件未生成: {heatmap_path}"
        assert os.path.exists(leaderboard_path), f"排行榜文件未生成: {leaderboard_path}"

        with open(heatmap_path) as f:
            assert len(f.read()) > 100
        with open(leaderboard_path) as f:
            assert len(f.read()) > 100

    print(f"✅ CLI report: HTML文件生成成功")


def test_lingshu_cell_model():
    """测试Lingshu-Cell作为第15个模型。"""
    from virtual_cell.models import create_model
    model = create_model("lingshu")
    assert model.info.name == "Lingshu-Cell"
    model.load()
    assert model.is_loaded()
    result = model.predict(None, task="perturbation", n_cells=100)
    assert result.metadata.get("model_type") == "masked_discrete_diffusion"
    print(f"✅ Lingshu-Cell模型: {model.info.name}")


def test_cellforge_generation():
    """测试CellForge架构生成。"""
    from virtual_cell.generators import CellForgeGenerator, CellForgeConfig

    config = CellForgeConfig(mode="mock", seed=42)
    gen = CellForgeGenerator(config)

    result = gen.generate("perturbation", "adamson2016", n_architectures=3)
    assert len(result.architectures) == 3
    assert result.best() is not None
    assert result.best().confidence > 0.5
    assert all(a.task == "perturbation" for a in result.architectures)
    assert all(a.code for a in result.architectures)
    print(f"✅ CellForge生成: {len(result.architectures)}个架构, 最优置信度={result.best().confidence:.4f}")


def test_cellforge_all_tasks():
    """测试CellForge在所有4个任务上的生成。"""
    from virtual_cell.generators import CellForgeGenerator

    gen = CellForgeGenerator()
    for task in ["perturbation", "cell_annotation", "integration", "grn"]:
        result = gen.generate(task, "test_dataset", n_architectures=2)
        assert len(result.architectures) == 2
        assert all(a.task == task for a in result.architectures)
        assert result.task_analysis["key_challenges"]
    print(f"✅ CellForge全任务: 4个任务各2个架构")


def test_generated_model_adapter():
    """测试生成架构适配为BaseModel。"""
    from virtual_cell.generators import CellForgeGenerator
    from virtual_cell.generators.model_adapter import GeneratedModelAdapter

    gen = CellForgeGenerator()
    result = gen.generate("perturbation", "adamson2016", n_architectures=1)
    arch = result.architectures[0]

    model = GeneratedModelAdapter(arch)
    model.load()
    assert model.is_loaded()
    pred = model.predict(None, task="perturbation", n_cells=100)
    assert pred.predictions is not None
    print(f"✅ GeneratedModelAdapter: {model.info.name}")


def test_generate_and_evaluate():
    """测试完整的生成→评估闭环。"""
    from virtual_cell.benchmark import Benchmark

    bench = Benchmark()
    result = bench.generate_and_evaluate(
        task="perturbation",
        dataset="adamson2016",
        n_architectures=2,
        max_cells=100,
        max_genes=50,
    )
    assert len(result.results) == 2
    lb = result.get_leaderboard()
    assert len(lb) == 2
    assert result.metadata["mode"] == "generate_and_evaluate"
    print(f"✅ Generate & Evaluate: {len(result.results)}次评估")


def test_cli_generate():
    """测试CLI generate命令。"""
    import argparse
    import io
    from contextlib import redirect_stdout
    from virtual_cell.cli import _cmd_generate

    args = argparse.Namespace(
        task="perturbation",
        dataset="adamson2016",
        n=2,
        max_cells=100,
        output="test_gen_report",
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_generate(args)

    output = buf.getvalue()
    assert "CellForge" in output
    assert "生成" in output
    print(f"✅ CLI generate: 输出{len(output)}字符")


def test_model_count():
    """测试模型总数（14+1=15）。"""
    from virtual_cell.registry import ModelRegistry
    models = ModelRegistry.list()
    assert len(models) == 15, f"期望15个模型，实际{len(models)}"
    names = [m["name"] for m in models]
    assert "Lingshu-Cell" in names
    print(f"✅ 模型总数: {len(models)} (含Lingshu-Cell)")


if __name__ == "__main__":
    print("=" * 50)
    print("🧪 VirtualCell Test Suite v0.3.0")
    print("=" * 50)

    test_model_registry()
    test_dataset_registry()
    test_load_model()
    test_load_dataset()
    test_single_evaluation()
    test_benchmark_run()
    test_report_generation()
    test_task_filter()
    test_visualizer_heatmap()
    test_visualizer_leaderboard()
    test_cli_leaderboard()
    test_cli_report()
    test_lingshu_cell_model()
    test_cellforge_generation()
    test_cellforge_all_tasks()
    test_generated_model_adapter()
    test_generate_and_evaluate()
    test_cli_generate()
    test_model_count()

    print()
    print("🎉 全部19个测试通过！")
