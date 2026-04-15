"""
决策检查点 — Decision Checkpoint 机制

为CLI/API操作提供：
1. 干运行模式 (dry-run)：预演操作但不执行
2. 危险操作确认：高资源/破坏性操作前的用户确认
3. 阶段间校验：Pipeline阶段间的数据完整性验证
4. 操作审计日志：记录所有关键决策
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class RiskLevel(Enum):
    """操作风险等级。"""
    LOW = "low"          # 只读查询
    MEDIUM = "medium"    # 单模型单数据集评估
    HIGH = "high"        # 多模型×多数据集，资源密集
    CRITICAL = "critical"  # run_all / 全量pipeline


@dataclass
class CheckpointResult:
    """检查点执行结果。"""
    passed: bool
    stage: str
    message: str
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class OperationPlan:
    """操作执行计划（dry-run产物）。"""
    operation: str
    risk_level: RiskLevel
    models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    estimated_evaluations: int = 0
    estimated_time_label: str = ""
    warnings: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """人类可读的操作摘要。"""
        lines = [
            f"📋 操作计划: {self.operation}",
            f"   风险等级: {self.risk_level.value.upper()}",
            f"   模型 ({len(self.models)}): {', '.join(self.models)}",
            f"   数据集 ({len(self.datasets)}): {', '.join(self.datasets)}",
            f"   任务 ({len(self.tasks)}): {', '.join(self.tasks)}",
            f"   预计评估次数: {self.estimated_evaluations}",
            f"   预计耗时: {self.estimated_time_label}",
        ]
        if self.warnings:
            lines.append("   ⚠️ 警告:")
            for w in self.warnings:
                lines.append(f"      - {w}")
        if self.checkpoints:
            lines.append("   🔒 检查点:")
            for c in self.checkpoints:
                lines.append(f"      - {c}")
        return "\n".join(lines)


def assess_risk(
    operation: str,
    models: list[str],
    datasets: list[str],
    tasks: list[str],
) -> RiskLevel:
    """根据操作规模评估风险等级。"""
    n_evals = len(models) * len(datasets) * len(tasks)

    if operation in ("list", "info", "leaderboard"):
        return RiskLevel.LOW
    elif n_evals <= 1:
        return RiskLevel.MEDIUM
    elif n_evals <= 9:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


def build_plan(
    operation: str,
    models: list[str],
    datasets: list[str],
    tasks: list[str],
    **extra,
) -> OperationPlan:
    """构建操作执行计划。"""
    n_evals = len(models) * len(datasets) * len(tasks)
    risk = assess_risk(operation, models, datasets, tasks)

    # 预估耗时
    if n_evals <= 1:
        time_label = "< 1分钟 (Mock模式)"
    elif n_evals <= 6:
        time_label = "1-5分钟"
    elif n_evals <= 20:
        time_label = "5-15分钟"
    else:
        time_label = "15分钟+"

    warnings = []
    if n_evals > 10:
        warnings.append(f"将执行 {n_evals} 次评估，可能消耗较多资源")
    if "all" in models:
        warnings.append("包含全部模型，评估量较大")

    checkpoints = ["输入参数校验"]
    if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        checkpoints.append("用户确认 (或 --yes 跳过)")
    checkpoints.extend(["阶段间数据完整性校验", "结果输出前验证"])

    return OperationPlan(
        operation=operation,
        risk_level=risk,
        models=models,
        datasets=datasets,
        tasks=tasks,
        estimated_evaluations=n_evals,
        estimated_time_label=time_label,
        warnings=warnings,
        checkpoints=checkpoints,
    )


def confirm_operation(plan: OperationPlan, auto_yes: bool = False) -> bool:
    """危险操作确认。低风险直接通过。"""
    if plan.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
        return True
    if auto_yes:
        print(f"⚡ --yes 已跳过确认（风险: {plan.risk_level.value}）")
        return True

    print(plan.summary())
    print()
    resp = input("确认执行？[y/N] ").strip().lower()
    return resp in ("y", "yes")


def validate_stage_input(stage: str, data: dict) -> CheckpointResult:
    """阶段间输入校验。"""
    required_keys = {
        "preprocess": ["expression_matrix", "gene_names"],
        "evaluate": ["predictions", "ground_truth"],
        "generate": ["task", "dataset"],
        "report": ["results"],
    }

    required = required_keys.get(stage, [])
    missing = [k for k in required if k not in data]

    if missing:
        return CheckpointResult(
            passed=False,
            stage=stage,
            message=f"阶段 '{stage}' 缺少必需字段: {', '.join(missing)}",
            details={"missing": missing},
        )

    # 形状校验
    if stage == "evaluate":
        pred = data.get("predictions")
        gt = data.get("ground_truth")
        if pred is not None and gt is not None:
            import numpy as np
            if isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray):
                if pred.shape != gt.shape:
                    return CheckpointResult(
                        passed=False,
                        stage=stage,
                        message=f"predictions 形状 {pred.shape} 与 ground_truth {gt.shape} 不匹配",
                        details={"pred_shape": pred.shape, "gt_shape": gt.shape},
                    )

    return CheckpointResult(
        passed=True,
        stage=stage,
        message=f"阶段 '{stage}' 校验通过",
    )


def validate_stage_output(stage: str, result: Any) -> CheckpointResult:
    """阶段间输出校验 — 确保阶段输出合理。"""
    if stage == "preprocess":
        if isinstance(result, dict):
            if "n_cells" in result and result["n_cells"] == 0:
                return CheckpointResult(
                    passed=False, stage=stage,
                    message="预处理后细胞数为0，数据可能异常",
                )
    elif stage == "evaluate":
        if hasattr(result, "metrics"):
            if "error" in result.metrics:
                return CheckpointResult(
                    passed=False, stage=stage,
                    message=f"评估出错: {result.metadata.get('error', 'unknown')}",
                )
    elif stage == "generate":
        if isinstance(result, dict):
            archs = result.get("architectures", [])
            if not archs:
                return CheckpointResult(
                    passed=False, stage=stage,
                    message="架构生成结果为空",
                )

    return CheckpointResult(passed=True, stage=stage, message=f"阶段 '{stage}' 输出校验通过")


def audit_log(operation: str, plan: OperationPlan, result: str, log_path: str = ""):
    """操作审计日志。"""
    if not log_path:
        log_path = str(Path.home() / ".virtual_cell" / "audit.jsonl")

    entry = {
        "timestamp": time.time(),
        "operation": operation,
        "risk_level": plan.risk_level.value,
        "models": plan.models,
        "datasets": plan.datasets,
        "tasks": plan.tasks,
        "n_evaluations": plan.estimated_evaluations,
        "result": result,
    }

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
