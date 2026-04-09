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
    assert len(models) == 14, f"期望14个模型，实际{len(models)}"
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


if __name__ == "__main__":
    print("=" * 50)
    print("🧪 VirtualCell Test Suite")
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

    print()
    print("🎉 全部测试通过！")
