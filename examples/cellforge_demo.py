#!/usr/bin/env python3
"""
CellForge 端到端演示脚本

展示VirtualCell v0.4.0的完整功能：
1. CellForge生成3个候选架构（perturbation任务）
2. 架构适配为BaseModel
3. 在数据集上运行评估
4. 输出排行榜对比

用法：python3 examples/cellforge_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 70)
    print("🔬 VirtualCell v0.4.0 — CellForge 端到端演示")
    print("=" * 70)
    print()

    # ================================================================
    # Phase 1: CellForge架构生成
    # ================================================================
    print("📐 Phase 1: CellForge 架构生成")
    print("-" * 50)

    from virtual_cell.generators import CellForgeGenerator, CellForgeConfig

    config = CellForgeConfig(mode="mock", seed=42)
    gen = CellForgeGenerator(config)

    print(f"  生成器: {gen.describe()}")
    print()

    task = "perturbation"
    dataset = "adamson2016"
    n_arch = 3

    print(f"  任务: {task}")
    print(f"  数据集: {dataset}")
    print(f"  候选架构数: {n_arch}")
    print()

    gen_result = gen.generate(task, dataset, n_architectures=n_arch)

    print(f"  ✅ 生成完成！共 {len(gen_result.architectures)} 个候选架构")
    print()

    for i, arch in enumerate(gen_result.architectures, 1):
        print(f"  ┌─ 架构 #{i}: {arch.name}")
        print(f"  │  类型: {arch.architecture_type}")
        print(f"  │  置信度: {arch.confidence:.4f}")
        print(f"  │  层数: {len(arch.layers)}")
        print(f"  │  创新组件: {', '.join(arch.innovations) if arch.innovations else '无'}")
        print(f"  │  设计推理: {arch.design_rationale[:80]}...")
        print(f"  └─")
        print()

    # ================================================================
    # Phase 2: 架构适配为BaseModel
    # ================================================================
    print("🔧 Phase 2: 架构适配 → BaseModel")
    print("-" * 50)

    from virtual_cell.generators.model_adapter import GeneratedModelAdapter

    adapters = []
    for arch in gen_result.architectures:
        adapter = GeneratedModelAdapter(arch)
        adapter.load()
        adapters.append(adapter)
        print(f"  ✅ {adapter.info.name} → BaseModel (loaded={adapter.is_loaded()})")

    print()

    # ================================================================
    # Phase 3: 在数据集上运行评估
    # ================================================================
    print("📊 Phase 3: Benchmark 评估")
    print("-" * 50)

    from virtual_cell.benchmark import Benchmark

    bench = Benchmark()
    result = bench.generate_and_evaluate(
        task=task,
        dataset=dataset,
        n_architectures=n_arch,
        max_cells=200,
        max_genes=100,
    )

    print(f"  ✅ 评估完成！共 {len(result.results)} 次评估，耗时 {result.execution_time_ms:.0f}ms")
    print()

    # ================================================================
    # Phase 4: 排行榜输出
    # ================================================================
    print("🏆 Phase 4: 排行榜对比")
    print("-" * 50)

    leaderboard = result.get_leaderboard()

    print(f"  {'排名':<6} {'模型':<40} {'任务':<15} {'得分':>8}")
    print(f"  {'─'*6} {'─'*40} {'─'*15} {'─'*8}")

    for i, entry in enumerate(leaderboard, 1):
        print(f"  #{i:<5} {entry['model']:<40} {entry['task']:<15} {entry['primary_score']:>8.4f}")

    print()

    # 最优架构详情
    if leaderboard:
        best = leaderboard[0]
        print(f"  🥇 最优架构: {best['model']}")
        print(f"     主指标得分: {best['primary_score']:.4f}")
        print(f"     所有指标: {best['all_metrics']}")

    print()

    # ================================================================
    # Phase 5: 与已知模型对比（可选）
    # ================================================================
    print("⚔️ Phase 5: 与已知模型对比")
    print("-" * 50)

    try:
        bench2 = Benchmark()
        compare_result = bench2.run(
            models=["scgpt", "cpa"],
            datasets=[dataset],
            tasks=[task],
            max_cells=200,
            max_genes=100,
        )

        compare_lb = compare_result.get_leaderboard()
        print(f"  已知模型排行榜:")
        for i, entry in enumerate(compare_lb, 1):
            print(f"    #{i} {entry['model']:<20} {entry['primary_score']:.4f}")

        # 合并排行榜
        all_entries = leaderboard + compare_lb
        all_entries.sort(key=lambda x: -x["primary_score"])

        print()
        print(f"  🏆 合并排行榜（CellForge生成 + 已知模型）:")
        print(f"  {'排名':<6} {'模型':<40} {'得分':>8}")
        print(f"  {'─'*6} {'─'*40} {'─'*8}")
        for i, entry in enumerate(all_entries[:10], 1):
            marker = " 🆕" if "CellForge" in entry["model"] else ""
            print(f"  #{i:<5} {entry['model']:<40} {entry['primary_score']:>8.4f}{marker}")

    except Exception as e:
        print(f"  ⚠️ 对比跳过: {e}")

    print()
    print("=" * 70)
    print("🎉 CellForge 端到端演示完成！")
    print("=" * 70)
    print()
    print("💡 提示:")
    print("  - 使用 REST API: POST /api/v1/pipeline/run")
    print("  - 查看生成器: GET /api/v1/generators")
    print("  - 平台统计: GET /api/v1/stats")


if __name__ == "__main__":
    main()
