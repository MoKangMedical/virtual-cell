#!/usr/bin/env python3
"""
VirtualCell 完整Benchmark演示

运行14个模型 × 多个数据集 × 4大任务的完整评估。
"""

import sys
sys.path.insert(0, '/root/virtual-cell')

from virtual_cell import Benchmark, BenchmarkReport


def main():
    print("=" * 60)
    print("🔬 VirtualCell — 单细胞基础模型完整Benchmark")
    print("=" * 60)
    print()

    bench = Benchmark()

    # 选取代表性模型和数据集
    models = ["scgpt", "geneformer", "scbert", "scfoundation", "regformer", "cpa", "gears"]
    datasets = ["zheng68k", "kang2018", "haber2017"]
    tasks = ["cell_annotation", "perturbation", "integration", "grn"]

    print(f"📊 模型: {len(models)}个")
    print(f"📊 数据集: {len(datasets)}个")
    print(f"📊 任务: {len(tasks)}个")
    print(f"📊 总评估: {len(models) * len(datasets) * len(tasks)}次")
    print()

    # 运行Benchmark
    result = bench.run(
        models=models,
        datasets=datasets,
        tasks=tasks,
        max_cells=200,
        max_genes=100,
    )

    print(f"✅ 完成! 耗时: {result.execution_time_ms:.0f}ms")
    print(f"   评估次数: {len(result.results)}")
    print()

    # 生成报告
    report = BenchmarkReport(result)

    # 保存Markdown
    md = report.to_markdown()
    with open("/root/virtual-cell/docs/benchmark_report.md", "w") as f:
        f.write(md)
    print(f"📄 Markdown报告: docs/benchmark_report.md ({len(md)}字符)")

    # 保存JSON
    json_report = report.to_json()
    with open("/root/virtual-cell/docs/benchmark_report.json", "w") as f:
        f.write(json_report)
    print(f"📄 JSON报告: docs/benchmark_report.json")

    # 打印排行榜
    print()
    print("🏆 总排行榜 Top 10:")
    print("-" * 60)
    leaderboard = result.get_leaderboard()
    for i, entry in enumerate(leaderboard[:10], 1):
        print(f"  {i:2d}. {entry['model']:15s} | {entry['task']:20s} | {entry['dataset']:15s} | {entry['primary_score']:.3f}")

    # 按任务分组展示
    print()
    for task in tasks:
        print(f"📊 {task}:")
        task_lb = result.get_leaderboard(task)
        for entry in task_lb[:5]:
            print(f"   {entry['model']:15s}: {entry['primary_score']:.3f}")
        print()

    # 摘要
    summary = report.get_summary()
    print(f"📋 摘要: {summary['total_evaluations']}次评估, {summary['execution_time_ms']:.0f}ms")


if __name__ == "__main__":
    main()
