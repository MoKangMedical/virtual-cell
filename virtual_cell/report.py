"""
报告生成器 — Benchmark结果的结构化报告
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .benchmark import BenchmarkResult


class BenchmarkReport:
    """Benchmark报告生成器。"""

    def __init__(self, result: BenchmarkResult):
        self.result = result

    def to_markdown(self) -> str:
        """生成Markdown报告。"""
        lines = [
            "# 🔬 VirtualCell Benchmark Report",
            "",
            f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"> 评估次数：{len(self.result.results)} | 耗时：{self.result.execution_time_ms:.0f}ms",
            "",
            "---",
            "",
        ]

        # 按任务分组
        tasks = set(r.task_name for r in self.result.results)
        for task in sorted(tasks):
            task_results = [r for r in self.result.results if r.task_name == task]
            lines.append(f"## 📊 {task}")
            lines.append("")

            # 构建表格
            models = sorted(set(r.model_name for r in task_results))
            datasets = sorted(set(r.dataset_name for r in task_results))
            metrics_set = set()
            for r in task_results:
                metrics_set.update(r.metrics.keys())
            metrics = sorted(metrics_set)

            # 表头
            header = "| 模型 | " + " | ".join(datasets) + " |"
            sep = "|------|" + "|".join(["------"] * len(datasets)) + "|"
            lines.append(header)
            lines.append(sep)

            # 数据行
            for model in models:
                row = f"| {model} |"
                for ds in datasets:
                    match = [r for r in task_results if r.model_name == model and r.dataset_name == ds]
                    if match:
                        primary = list(match[0].metrics.values())[0]
                        row += f" {primary:.3f} |"
                    else:
                        row += " — |"
                lines.append(row)

            lines.append("")

        # 排行榜
        lines.append("## 🏆 总排行榜")
        lines.append("")
        leaderboard = self.result.get_leaderboard()
        lines.append("| 排名 | 模型 | 数据集 | 任务 | 得分 |")
        lines.append("|------|------|--------|------|------|")
        for i, entry in enumerate(leaderboard[:20], 1):
            lines.append(
                f"| {i} | {entry['model']} | {entry['dataset']} | "
                f"{entry['task']} | {entry['primary_score']:.3f} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("*由 VirtualCell Benchmark 自动生成*")
        return "\n".join(lines)

    def to_json(self) -> str:
        """生成JSON报告。"""
        return json.dumps(self.result.to_dict(), ensure_ascii=False, indent=2)

    def get_summary(self) -> dict[str, Any]:
        """获取摘要统计。"""
        if not self.result.results:
            return {}

        tasks = {}
        for r in self.result.results:
            if r.task_name not in tasks:
                tasks[r.task_name] = {"n_evaluations": 0, "models": set(), "datasets": set()}
            tasks[r.task_name]["n_evaluations"] += 1
            tasks[r.task_name]["models"].add(r.model_name)
            tasks[r.task_name]["datasets"].add(r.dataset_name)

        return {
            "total_evaluations": len(self.result.results),
            "tasks": {k: {**v, "models": list(v["models"]), "datasets": list(v["datasets"])} for k, v in tasks.items()},
            "execution_time_ms": self.result.execution_time_ms,
        }
