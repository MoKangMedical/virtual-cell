"""
可视化模块 — Benchmark结果的交互式图表

支持：
- 排行榜柱状图
- 模型对比雷达图
- 任务热力图
- 数据集分布图
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .benchmark import BenchmarkResult


class Visualizer:
    """Benchmark结果可视化。"""

    def __init__(self, result: BenchmarkResult):
        self.result = result

    def leaderboard_html(self, task: str = "", top_n: int = 20) -> str:
        """生成排行榜HTML（纯前端，无依赖）。"""
        lb = self.result.get_leaderboard(task)[:top_n]

        rows = ""
        for i, entry in enumerate(lb, 1):
            score = entry["primary_score"]
            bar_width = int(score * 100)
            color = "#10b981" if score > 0.8 else "#f59e0b" if score > 0.5 else "#ef4444"
            rows += f"""
            <tr>
                <td class="rank">#{i}</td>
                <td class="model">{entry['model']}</td>
                <td class="task">{entry['task']}</td>
                <td class="dataset">{entry['dataset']}</td>
                <td class="score">
                    <div class="bar-container">
                        <div class="bar" style="width:{bar_width}%;background:{color}"></div>
                        <span class="bar-label">{score:.3f}</span>
                    </div>
                </td>
            </tr>"""

        return f"""
        <table class="leaderboard">
            <thead>
                <tr>
                    <th>排名</th><th>模型</th><th>任务</th><th>数据集</th><th>得分</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """

    def radar_chart_data(self, model_name: str) -> dict:
        """生成雷达图数据（模型在4个任务上的表现）。"""
        task_scores = {}
        for r in self.result.results:
            if r.model_name == model_name:
                primary = list(r.metrics.values())[0]
                if r.task_name not in task_scores:
                    task_scores[r.task_name] = []
                task_scores[r.task_name].append(primary)

        return {
            "model": model_name,
            "labels": list(task_scores.keys()),
            "values": [sum(v)/len(v) for v in task_scores.values()],
        }

    def heatmap_data(self) -> dict:
        """生成热力图数据（模型×任务的平均得分）。"""
        matrix = {}
        models = set()
        tasks = set()

        for r in self.result.results:
            models.add(r.model_name)
            tasks.add(r.task_name)
            key = (r.model_name, r.task_name)
            if key not in matrix:
                matrix[key] = []
            matrix[key].append(list(r.metrics.values())[0])

        return {
            "models": sorted(models),
            "tasks": sorted(tasks),
            "values": {
                f"{m}|{t}": sum(matrix.get((m, t), [0])) / max(1, len(matrix.get((m, t), [1])))
                for m in sorted(models) for t in sorted(tasks)
            },
        }

    @staticmethod
    def _get_primary_score(task: str, metrics: dict) -> float:
        """根据任务类型计算主指标得分。"""
        if task == "cell_annotation":
            return metrics.get("accuracy", 0.0)
        elif task == "perturbation":
            return metrics.get("pcc", 0.0)
        elif task == "integration":
            vals = [metrics.get(k) for k in ("kbet", "lisi", "asw", "graph_connectivity")]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else 0.0
        elif task == "grn":
            return metrics.get("auprc", 0.0)
        return 0.0

    @staticmethod
    def _color_for_score(score: float) -> str:
        """返回颜色：绿=高, 红=低。"""
        if score > 0.8:
            return "#10b981"
        elif score > 0.5:
            return "#f59e0b"
        return "#ef4444"

    def generate_heatmap(self, results: list[dict] | None = None, output_path: str = "heatmap.html") -> str:
        """模型×任务得分热力图 → HTML文件。

        Args:
            results: 结果列表（dict格式），默认从self.result生成。
            output_path: 输出HTML路径。

        Returns:
            生成的HTML内容。
        """
        if results is None:
            results = [r.to_dict() for r in self.result.results]

        # Build matrix: model × task average primary score
        matrix: dict[tuple[str, str], list[float]] = {}
        models_set: set[str] = set()
        tasks_set: set[str] = set()

        for r in results:
            model = r["model_name"] if "model_name" in r else r.get("model", "")
            task = r["task_name"] if "task_name" in r else r.get("task", "")
            metrics = r.get("metrics", {})
            models_set.add(model)
            tasks_set.add(task)
            key = (model, task)
            matrix.setdefault(key, []).append(self._get_primary_score(task, metrics))

        models = sorted(models_set)
        tasks = sorted(tasks_set)

        # Build table rows
        rows_html = ""
        for model in models:
            cells = ""
            for task in tasks:
                vals = matrix.get((model, task), [0.0])
                avg = sum(vals) / len(vals)
                color = self._color_for_score(avg)
                cells += f'<td style="background:{color};color:#fff;text-align:center;padding:8px;font-weight:600">{avg:.3f}</td>'
            rows_html += f"<tr><td style='padding:8px;font-weight:600;color:#00d2ff'>{model}</td>{cells}</tr>"

        headers = "".join(f"<th style='padding:8px;text-align:center'>{t}</th>" for t in tasks)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VirtualCell Heatmap</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 2rem; }}
h1 {{ color: #00d2ff; text-align: center; }}
table {{ border-collapse: collapse; margin: 1rem auto; }}
th {{ background: #16213e; color: #aaa; font-size: .85rem; }}
</style>
</head>
<body>
<h1>📊 Model × Task Heatmap</h1>
<table>
<thead><tr><th>Model</th>{headers}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)

        return html

    def generate_leaderboard_html(self, results: list[dict] | None = None, output_path: str = "leaderboard.html") -> str:
        """完整排行榜HTML文件。

        Args:
            results: 结果列表（dict格式），默认从self.result生成。
            output_path: 输出HTML路径。

        Returns:
            生成的HTML内容。
        """
        if results is None:
            results = [r.to_dict() for r in self.result.results]

        # Compute primary scores and sort
        entries = []
        for r in results:
            model = r["model_name"] if "model_name" in r else r.get("model", "")
            task = r["task_name"] if "task_name" in r else r.get("task", "")
            dataset = r["dataset_name"] if "dataset_name" in r else r.get("dataset", "")
            metrics = r.get("metrics", {})
            primary = self._get_primary_score(task, metrics)
            entries.append({"model": model, "task": task, "dataset": dataset, "primary": primary, "metrics": metrics})

        entries.sort(key=lambda x: -x["primary"])

        rows = ""
        for i, e in enumerate(entries, 1):
            color = self._color_for_score(e["primary"])
            bar_w = int(e["primary"] * 100)
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in e["metrics"].items() if isinstance(v, (int, float)))
            rows += f"""<tr>
<td style="color:#f59e0b;font-weight:700">#{i}</td>
<td style="font-weight:600;color:#00d2ff">{e['model']}</td>
<td>{e['task']}</td>
<td style="color:#aaa">{e['dataset']}</td>
<td><div style="display:flex;align-items:center;gap:6px">
<div style="flex:1;background:#253a5e;border-radius:4px;height:20px;overflow:hidden">
<div style="width:{bar_w}%;background:{color};height:100%;border-radius:4px"></div>
</div><span style="font-weight:600;color:{color}">{e['primary']:.4f}</span>
</div></td>
<td style="font-size:.8rem;color:#aaa">{metrics_str}</td>
</tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VirtualCell Leaderboard</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 2rem; }}
h1 {{ color: #00d2ff; text-align: center; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
th {{ text-align: left; padding: .6rem; color: #aaa; border-bottom: 1px solid #253a5e; font-size: .8rem; text-transform: uppercase; }}
td {{ padding: .6rem; border-bottom: 1px solid rgba(255,255,255,.05); }}
tr:hover {{ background: rgba(0,210,255,.05); }}
</style>
</head>
<body>
<h1>🏆 VirtualCell Leaderboard</h1>
<p style="text-align:center;color:#aaa">Ranked by primary task metric</p>
<table>
<thead><tr><th>#</th><th>Model</th><th>Task</th><th>Dataset</th><th>Score</th><th>All Metrics</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)

        return html

    def generate_comparison(self, model1_name: str, model2_name: str, results: list[dict] | None = None, output: str = "comparison.html") -> str:
        """两模型对比报告 → HTML文件。

        Args:
            model1_name: 第一个模型名称。
            model2_name: 第二个模型名称。
            results: 结果列表（dict格式），默认从self.result生成。
            output: 输出HTML路径。

        Returns:
            生成的HTML内容。
        """
        if results is None:
            results = [r.to_dict() for r in self.result.results]

        m1_data: dict[str, list[dict]] = {}
        m2_data: dict[str, list[dict]] = {}

        for r in results:
            model = r["model_name"] if "model_name" in r else r.get("model", "")
            task = r["task_name"] if "task_name" in r else r.get("task", "")
            metrics = r.get("metrics", {})
            entry = {"task": task, "metrics": metrics, "primary": self._get_primary_score(task, metrics)}
            if model.lower() == model1_name.lower():
                m1_data.setdefault(task, []).append(entry)
            elif model.lower() == model2_name.lower():
                m2_data.setdefault(task, []).append(entry)

        all_tasks = sorted(set(list(m1_data.keys()) + list(m2_data.keys())))

        rows = ""
        for task in all_tasks:
            m1_avg = sum(e["primary"] for e in m1_data.get(task, [])) / max(1, len(m1_data.get(task, [1])))
            m2_avg = sum(e["primary"] for e in m2_data.get(task, [])) / max(1, len(m2_data.get(task, [1])))
            winner = model1_name if m1_avg > m2_avg else model2_name if m2_avg > m1_avg else "Tie"
            winner_color = "#10b981" if winner != "Tie" else "#f59e0b"
            rows += f"""<tr>
<td>{task}</td>
<td style="font-weight:600;color:{self._color_for_score(m1_avg)}">{m1_avg:.4f}</td>
<td style="font-weight:600;color:{self._color_for_score(m2_avg)}">{m2_avg:.4f}</td>
<td style="font-weight:700;color:{winner_color}">{winner}</td>
</tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Compare: {model1_name} vs {model2_name}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 2rem; }}
h1 {{ color: #00d2ff; text-align: center; }}
h1 span {{ color: #a78bfa; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
th {{ text-align: left; padding: .6rem; color: #aaa; border-bottom: 1px solid #253a5e; font-size: .8rem; text-transform: uppercase; }}
td {{ padding: .6rem; border-bottom: 1px solid rgba(255,255,255,.05); }}
tr:hover {{ background: rgba(0,210,255,.05); }}
.vs {{ font-size: 1.5rem; text-align: center; color: #f59e0b; margin: .5rem 0; }}
</style>
</head>
<body>
<h1>⚔️ {model1_name} <span>vs</span> {model2_name}</h1>
<p class="vs">Head-to-Head Comparison</p>
<table>
<thead><tr><th>Task</th><th>{model1_name}</th><th>{model2_name}</th><th>Winner</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>"""

        with open(output, "w") as f:
            f.write(html)

        return html

    def to_interactive_html(self, title: str = "VirtualCell Benchmark") -> str:
        """生成完整的交互式HTML报告。"""
        leaderboard = self.leaderboard_html()
        heatmap = self.heatmap_data()

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
h1 {{ font-size: 2.5rem; background: linear-gradient(135deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }}
.subtitle {{ color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }}
.stat-card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #334155; }}
.stat-number {{ font-size: 2rem; font-weight: 700; color: #06b6d4; }}
.stat-label {{ color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }}
.card h2 {{ color: #f1f5f9; font-size: 1.3rem; margin-bottom: 1rem; }}
.leaderboard {{ width: 100%; border-collapse: collapse; }}
.leaderboard th {{ text-align: left; padding: 0.75rem; color: #94a3b8; border-bottom: 1px solid #334155; font-size: 0.85rem; text-transform: uppercase; }}
.leaderboard td {{ padding: 0.75rem; border-bottom: 1px solid #1e293b; }}
.leaderboard tr:hover {{ background: #334155; }}
.rank {{ color: #f59e0b; font-weight: 700; width: 50px; }}
.model {{ font-weight: 600; color: #06b6d4; }}
.task {{ color: #a78bfa; }}
.dataset {{ color: #94a3b8; }}
.bar-container {{ position: relative; background: #334155; border-radius: 4px; height: 24px; min-width: 200px; }}
.bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
.bar-label {{ position: absolute; right: 8px; top: 3px; font-size: 0.8rem; font-weight: 600; }}
.heatmap {{ display: grid; gap: 2px; margin-top: 1rem; }}
.heatmap-cell {{ padding: 0.5rem; text-align: center; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
footer {{ text-align: center; color: #475569; margin-top: 3rem; padding: 1rem; }}
</style>
</head>
<body>
<div class="container">
    <h1>🔬 {title}</h1>
    <p class="subtitle">单细胞基础模型的统一评估标准 — 14个模型 × 26个数据集 × 4大任务</p>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">14</div>
            <div class="stat-label">基础模型</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">26</div>
            <div class="stat-label">数据集</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">4</div>
            <div class="stat-label">评估任务</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(self.result.results)}</div>
            <div class="stat-label">评估次数</div>
        </div>
    </div>

    <div class="card">
        <h2>🏆 排行榜</h2>
        {leaderboard}
    </div>

    <div class="card">
        <h2>📊 热力图 — 模型 × 任务</h2>
        <div id="heatmap"></div>
    </div>
</div>
<footer>VirtualCell Benchmark — MoKangMedical | 2026</footer>
</body>
</html>"""
