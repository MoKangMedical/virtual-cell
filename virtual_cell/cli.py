"""
CLI — 命令行接口

用法：
    virtual-cell list models          # 列出所有模型
    virtual-cell list datasets        # 列出所有数据集
    virtual-cell list tasks           # 列出所有任务
    virtual-cell run --models scgpt,geneformer --datasets zheng68k --tasks cell_annotation
    virtual-cell report --input results.json --output report.md
    virtual-cell leaderboard          # 终端表格排行榜
    virtual-cell compare MODEL1 MODEL2  # 对比两个模型
    virtual-cell info MODEL           # 模型详细信息
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="virtual-cell",
        description="🔬 VirtualCell — 单细胞基础模型Benchmark平台",
    )
    sub = parser.add_subparsers(dest="command")

    # list 子命令
    list_parser = sub.add_parser("list", help="列出模型/数据集/任务")
    list_parser.add_argument("what", choices=["models", "datasets", "tasks"], help="列出什么")

    # run 子命令
    run_parser = sub.add_parser("run", help="运行Benchmark")
    run_parser.add_argument("--models", "-m", default="scgpt,geneformer,scbert", help="模型列表(逗号分隔)")
    run_parser.add_argument("--datasets", "-d", default="zheng68k", help="数据集列表(逗号分隔)")
    run_parser.add_argument("--tasks", "-t", default="cell_annotation", help="任务列表(逗号分隔)")
    run_parser.add_argument("--max-cells", type=int, default=500, help="最大细胞数")
    run_parser.add_argument("--max-genes", type=int, default=500, help="最大基因数")
    run_parser.add_argument("--output", "-o", default="", help="输出报告路径")
    run_parser.add_argument("--format", "-f", choices=["md", "json", "both"], default="both", help="报告格式")

    # info 子命令
    info_parser = sub.add_parser("info", help="查看模型详情")
    info_parser.add_argument("name", help="模型名称")

    # leaderboard 子命令
    sub.add_parser("leaderboard", help="终端表格排行榜")

    # report 子命令
    report_parser = sub.add_parser("report", help="运行benchmark并生成HTML报告")
    report_parser.add_argument("--models", "-m", default="scgpt,geneformer,scbert", help="模型列表(逗号分隔)")
    report_parser.add_argument("--datasets", "-d", default="zheng68k", help="数据集列表(逗号分隔)")
    report_parser.add_argument("--tasks", "-t", default="cell_annotation", help="任务列表(逗号分隔)")
    report_parser.add_argument("--max-cells", type=int, default=500, help="最大细胞数")
    report_parser.add_argument("--max-genes", type=int, default=500, help="最大基因数")
    report_parser.add_argument("--output", "-o", default="benchmark_report", help="输出文件前缀")

    # compare 子命令
    compare_parser = sub.add_parser("compare", help="对比两个模型")
    compare_parser.add_argument("model1", help="第一个模型名称")
    compare_parser.add_argument("model2", help="第二个模型名称")
    compare_parser.add_argument("--datasets", "-d", default="zheng68k", help="数据集列表(逗号分隔)")
    compare_parser.add_argument("--tasks", "-t", default="cell_annotation,perturbation,integration,grn", help="任务列表(逗号分隔)")
    compare_parser.add_argument("--max-cells", type=int, default=500, help="最大细胞数")
    compare_parser.add_argument("--output", "-o", default="comparison.html", help="输出HTML路径")

    args = parser.parse_args()

    if args.command == "list":
        _cmd_list(args.what)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "info":
        _cmd_info(args.name)
    elif args.command == "leaderboard":
        _cmd_leaderboard()
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "compare":
        _cmd_compare(args)
    else:
        parser.print_help()


def _cmd_list(what: str):
    from .registry import ModelRegistry, DatasetRegistry
    from .tasks import list_tasks

    if what == "models":
        models = ModelRegistry.list()
        print(f"\n📦 共 {len(models)} 个模型:\n")
        print(f"{'Key':15s} {'Name':20s} {'Architecture':15s} {'Cells':>12s} {'Year':>5s}")
        print("-" * 70)
        for m in models:
            print(f"{m['key']:15s} {m['name']:20s} {m['architecture']:15s} {m['pretrain_cells']:>12,} {m['year']:>5d}")

    elif what == "datasets":
        datasets = DatasetRegistry.list()
        print(f"\n📊 共 {len(datasets)} 个数据集:\n")
        print(f"{'Key':20s} {'Name':20s} {'Cells':>10s} {'Genes':>8s} {'Types':>6s} {'Year':>5s}")
        print("-" * 72)
        for d in datasets:
            print(f"{d['key']:20s} {d['name']:20s} {d['n_cells']:>10,} {d['n_genes']:>8,} {d['n_cell_types']:>6d} {d['year']:>5d}")

    elif what == "tasks":
        tasks = list_tasks()
        print(f"\n🧪 共 {len(tasks)} 个任务:\n")
        for t in tasks:
            print(f"  {t['key']:25s} → {', '.join(t['metrics'])}")


def _cmd_run(args):
    from .benchmark import Benchmark
    from .report import BenchmarkReport

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"\n🔬 VirtualCell Benchmark")
    print(f"   模型: {', '.join(models)}")
    print(f"   数据集: {', '.join(datasets)}")
    print(f"   任务: {', '.join(tasks)}")
    print()

    bench = Benchmark()
    result = bench.run(
        models=models, datasets=datasets, tasks=tasks,
        max_cells=args.max_cells, max_genes=args.max_genes,
    )

    print(f"✅ 完成! {len(result.results)}次评估, {result.execution_time_ms:.0f}ms\n")

    report = BenchmarkReport(result)

    # 排行榜
    print("🏆 排行榜:")
    for i, entry in enumerate(result.get_leaderboard()[:15], 1):
        print(f"  {i:2d}. {entry['model']:15s} | {entry['task']:20s} | {entry['dataset']:15s} | {entry['primary_score']:.4f}")

    # 保存报告
    output_base = args.output or "benchmark"
    if args.format in ("md", "both"):
        md_path = f"{output_base}.md"
        with open(md_path, "w") as f:
            f.write(report.to_markdown())
        print(f"\n📄 报告已保存: {md_path}")
    if args.format in ("json", "both"):
        json_path = f"{output_base}.json"
        with open(json_path, "w") as f:
            f.write(report.to_json())
        print(f"📄 报告已保存: {json_path}")


def _cmd_info(name: str):
    from .registry import ModelRegistry

    info = ModelRegistry.info(name)
    if info is None:
        print(f"❌ 未找到模型: {name}")
        return
    print(f"\n🧬 {info.name}")
    print(f"   架构: {info.architecture.value}")
    print(f"   预训练: {info.pretrain_cells:,} 细胞")
    print(f"   参数量: {info.parameters}")
    print(f"   年份: {info.year}")
    print(f"   支持任务: {', '.join(info.supported_tasks)}")
    print(f"   优势: {', '.join(info.strengths)}")
    print(f"   劣势: {', '.join(info.weaknesses)}")
    if info.paper:
        print(f"   论文: {info.paper}")
    if info.code_repo:
        print(f"   代码: {info.code_repo}")


def _cmd_leaderboard():
    """从benchmark.json读取数据，终端表格显示排行榜。"""
    import os
    bench_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "benchmark.json")
    if not os.path.exists(bench_path):
        # Try relative to cwd
        bench_path = os.path.join("docs", "benchmark.json")
    if not os.path.exists(bench_path):
        print("❌ 未找到 docs/benchmark.json，请先运行 benchmark。")
        return

    with open(bench_path) as f:
        data = json.load(f)

    results = data["results"]

    # Compute primary scores
    def primary_score(task, metrics):
        if task == "cell_annotation":
            return metrics.get("accuracy", 0)
        elif task == "perturbation":
            return metrics.get("pcc", 0)
        elif task == "integration":
            vals = [metrics.get(k) for k in ("kbet", "lisi", "asw", "graph_connectivity")]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else 0
        elif task == "grn":
            return metrics.get("auprc", 0)
        return 0

    entries = []
    for r in results:
        ps = primary_score(r["task"], r["metrics"])
        entries.append({**r, "primary": ps})

    entries.sort(key=lambda x: -x["primary"])

    print(f"\n🏆 VirtualCell 排行榜 ({len(entries)} 条评估结果)\n")
    print(f"{'#':>3s}  {'Model':18s} {'Task':20s} {'Dataset':15s} {'Score':>8s}")
    print("-" * 70)
    for i, e in enumerate(entries[:30], 1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{marker}{i:2d}  {e['model']:18s} {e['task']:20s} {e['dataset']:15s} {e['primary']:8.4f}")

    # Summary by model
    print(f"\n📊 模型平均分:")
    model_scores = {}
    for e in entries:
        model_scores.setdefault(e["model"], []).append(e["primary"])
    model_avg = [(m, sum(s)/len(s), len(s)) for m, s in model_scores.items()]
    model_avg.sort(key=lambda x: -x[1])
    for m, avg, n in model_avg:
        print(f"  {m:18s}  avg={avg:.4f}  ({n} evaluations)")


def _cmd_report(args):
    """运行benchmark并生成HTML报告。"""
    from .benchmark import Benchmark
    from .visualizer import Visualizer

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"\n🔬 运行Benchmark并生成HTML报告...")
    bench = Benchmark()
    result = bench.run(
        models=models, datasets=datasets, tasks=tasks,
        max_cells=args.max_cells, max_genes=args.max_genes,
    )
    print(f"✅ {len(result.results)}次评估完成")

    viz = Visualizer(result)

    heatmap_path = f"{args.output}_heatmap.html"
    leaderboard_path = f"{args.output}_leaderboard.html"

    viz.generate_heatmap(output_path=heatmap_path)
    viz.generate_leaderboard_html(output_path=leaderboard_path)

    print(f"📄 热力图: {heatmap_path}")
    print(f"📄 排行榜: {leaderboard_path}")


def _cmd_compare(args):
    """对比两个模型，生成HTML对比报告。"""
    from .benchmark import Benchmark
    from .visualizer import Visualizer

    models = [args.model1, args.model2]
    datasets = [d.strip() for d in args.datasets.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"\n⚔️ 对比 {args.model1} vs {args.model2} ...")
    bench = Benchmark()
    result = bench.run(
        models=models, datasets=datasets, tasks=tasks,
        max_cells=args.max_cells, max_genes=500,
    )
    print(f"✅ {len(result.results)}次评估完成")

    viz = Visualizer(result)
    viz.generate_comparison(args.model1, args.model2, output=args.output)
    print(f"📄 对比报告: {args.output}")


if __name__ == "__main__":
    main()
