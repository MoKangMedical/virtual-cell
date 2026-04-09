"""
CellForge集成 — 多智能体架构设计引擎

封装CellForge的三阶段流程：
1. Task Analysis: 分析数据集特征 + 文献RAG
2. Method Design: 多专家图基讨论，迭代优化架构方案
3. Code Generation: 生成可执行PyTorch代码

两种模式：
- Mock模式（默认）：基于文献知识的架构模板生成，无需GPU
- Full模式：调用CellForge实际多Agent流程（需安装CellForge + LLM API）
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import BaseGenerator, GeneratedArchitecture, GenerationResult


@dataclass
class CellForgeConfig:
    """CellForge配置。"""
    mode: str = "mock"                      # "mock" | "full"
    llm_provider: str = "mimo"              # LLM后端
    max_discussion_rounds: int = 3          # 最大讨论轮次
    innovation_budget: int = 2              # 最大创新组件数
    code_validation: bool = True            # 是否验证生成的代码
    cellforge_path: str = ""                # CellForge安装路径（full模式）
    seed: int = 42                          # 随机种子


# ================================================================
# 架构模板库（Mock模式核心知识库）
# ================================================================

INNOVATIONS = {
    # === 扰动预测创新组件 ===
    "trajectory_aware_encoder": {
        "description": "轨迹感知编码器 — 捕捉扰动后细胞状态的动态变化轨迹",
        "layers": [
            {"type": "TrajectoryEncoder", "hidden_dim": 256, "n_layers": 3},
            {"type": "TemporalAttention", "n_heads": 8},
        ],
        "hyperparams": {"trajectory_steps": 10, "dynamics_lr": 1e-4},
        "improves": {"perturbation": {"pcc": 0.03, "pds": 0.04}},
        "code": textwrap.dedent("""\
            class TrajectoryAwareEncoder(nn.Module):
                def __init__(self, input_dim, hidden_dim=256, n_layers=3, n_steps=10):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                    )
                    self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
                    self.dynamics = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True)
                    self.step_embed = nn.Embedding(n_steps, hidden_dim)

                def forward(self, x, step=0):
                    h = self.encoder(x)
                    step_h = self.step_embed(torch.tensor(step))
                    h = h + step_h
                    attn_out, _ = self.temporal_attn(h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
                    return attn_out.squeeze(1)
        """),
    },
    "perturbation_diffusion": {
        "description": "扰动扩散模块 — 用扩散模型预测扰动后表达分布",
        "layers": [
            {"type": "DiffusionHead", "n_steps": 100, "schedule": "cosine"},
            {"type": "NoisePredictor", "hidden_dim": 512},
        ],
        "hyperparams": {"diffusion_steps": 100, "eta": 0.1},
        "improves": {"perturbation": {"mse": -0.05, "mae": -0.03}},
        "code": textwrap.dedent("""\
            class PerturbationDiffusion(nn.Module):
                def __init__(self, gene_dim, hidden_dim=512, n_steps=100):
                    super().__init__()
                    self.n_steps = n_steps
                    self.denoise_net = nn.Sequential(
                        nn.Linear(gene_dim + 1, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, gene_dim),
                    )
                    betas = torch.linspace(1e-4, 0.02, n_steps)
                    alphas = 1 - betas
                    self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

                def forward(self, x, condition):
                    t = torch.randint(0, self.n_steps, (x.shape[0],))
                    noise = torch.randn_like(x)
                    alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
                    x_noisy = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
                    t_emb = t.float().unsqueeze(-1) / self.n_steps
                    pred = self.denoise_net(torch.cat([x_noisy, t_emb], dim=-1))
                    return pred
        """),
    },
    "gene_interaction_gnn": {
        "description": "基因交互GNN — 通过图神经网络建模基因间调控关系",
        "layers": [
            {"type": "GeneGraphConv", "hidden_dim": 128, "n_layers": 3},
            {"type": "Readout", "method": "mean"},
        ],
        "hyperparams": {"edge_threshold": 0.3, "gnn_type": "GATv2"},
        "improves": {"perturbation": {"pcc": 0.02, "r2": 0.03}, "grn": {"auprc": 0.05}},
        "code": textwrap.dedent("""\
            class GeneInteractionGNN(nn.Module):
                def __init__(self, n_genes, hidden_dim=128, n_layers=3):
                    super().__init__()
                    self.gene_embed = nn.Embedding(n_genes, hidden_dim)
                    self.convs = nn.ModuleList([
                        nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
                    ])
                    self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
                    self.readout = nn.Linear(hidden_dim, hidden_dim)

                def forward(self, gene_expr, adj_matrix):
                    h = self.gene_embed(torch.arange(gene_expr.shape[-1]))
                    h = h.unsqueeze(0).expand(gene_expr.shape[0], -1, -1)
                    for conv, norm in zip(self.convs, self.norms):
                        h_new = torch.bmm(adj_matrix, h)
                        h_new = conv(h_new)
                        h = norm(h + F.gelu(h_new))
                    return self.readout(h.mean(dim=1))
        """),
    },
    # === 细胞注释创新组件 ===
    "multi_scale_attention": {
        "description": "多尺度注意力 — 同时关注基因集、通路、细胞层面特征",
        "layers": [
            {"type": "MultiScaleAttention", "scales": ["gene", "pathway", "cell"]},
            {"type": "ScaleFusion", "method": "adaptive"},
        ],
        "hyperparams": {"n_pathways": 300, "fusion_dim": 512},
        "improves": {"cell_annotation": {"accuracy": 0.03, "f1_macro": 0.02}},
        "code": textwrap.dedent("""\
            class MultiScaleAttention(nn.Module):
                def __init__(self, input_dim, hidden_dim=512, n_pathways=300):
                    super().__init__()
                    self.gene_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
                    self.pathway_proj = nn.Linear(n_pathways, hidden_dim)
                    self.cell_proj = nn.Linear(input_dim, hidden_dim)
                    self.fusion = nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )

                def forward(self, x):
                    gene_out, _ = self.gene_attn(x, x, x)
                    pathway_out = self.pathway_proj(x.mean(dim=-1, keepdim=True).expand(-1, -1, 300))
                    cell_out = self.cell_proj(x)
                    return self.fusion(torch.cat([gene_out, pathway_out, cell_out], dim=-1))
        """),
    },
    "contrastive_cell_embedding": {
        "description": "对比学习细胞嵌入 — 通过对比学习增强同类细胞表征聚集",
        "layers": [
            {"type": "ContrastiveProjector", "dim": 128},
            {"type": "NTXentLoss", "temperature": 0.07},
        ],
        "hyperparams": {"temperature": 0.07, "augmentation": "dropout_mask"},
        "improves": {"cell_annotation": {"accuracy": 0.02, "f1_weighted": 0.03}},
        "code": textwrap.dedent("""\
            class ContrastiveCellEmbedding(nn.Module):
                def __init__(self, input_dim, proj_dim=128, temperature=0.07):
                    super().__init__()
                    self.projector = nn.Sequential(
                        nn.Linear(input_dim, input_dim),
                        nn.GELU(),
                        nn.Linear(input_dim, proj_dim),
                    )
                    self.temperature = temperature

                def forward(self, z1, z2):
                    z1 = F.normalize(self.projector(z1), dim=-1)
                    z2 = F.normalize(self.projector(z2), dim=-1)
                    sim = torch.mm(z1, z2.t()) / self.temperature
                    labels = torch.arange(z1.shape[0], device=z1.device)
                    return F.cross_entropy(sim, labels)
        """),
    },
    # === 批次整合创新组件 ===
    "adversarial_domain_adapter": {
        "description": "对抗域适配器 — 通过对抗训练去除批次效应",
        "layers": [
            {"type": "DomainClassifier", "hidden_dim": 256},
            {"type": "GradientReversal", "lambda": 1.0},
        ],
        "hyperparams": {"adversarial_weight": 0.5, "n_domains": "auto"},
        "improves": {"integration": {"kbet": 0.04, "lisi": 0.03}},
        "code": textwrap.dedent("""\
            class AdversarialDomainAdapter(nn.Module):
                def __init__(self, hidden_dim=256, n_domains=5):
                    super().__init__()
                    self.domain_classifier = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_dim, n_domains),
                    )

                def forward(self, features, alpha=1.0):
                    reverse_features = GradientReverseFunction.apply(features, alpha)
                    return self.domain_classifier(reverse_features)
        """),
    },
    "harmony_style_transfer": {
        "description": "Harmony风格迁移 — 迭代校正批次效应，类似Harmony算法的神经网络版",
        "layers": [
            {"type": "StyleExtractor", "hidden_dim": 64},
            {"type": "StyleRemover", "n_iterations": 5},
        ],
        "hyperparams": {"sigma": 0.1, "n_iter": 5},
        "improves": {"integration": {"asw": 0.05, "graph_connectivity": 0.03}},
        "code": textwrap.dedent("""\
            class HarmonyStyleTransfer(nn.Module):
                def __init__(self, hidden_dim, n_iterations=5):
                    super().__init__()
                    self.style_extract = nn.Linear(hidden_dim, 64)
                    self.n_iter = n_iterations

                def forward(self, embeddings, batch_labels):
                    for _ in range(self.n_iter):
                        style = self.style_extract(embeddings)
                        unique_batches = torch.unique(batch_labels)
                        centroids = []
                        for b in unique_batches:
                            mask = batch_labels == b
                            centroids.append(style[mask].mean(dim=0))
                        centroid = torch.stack(centroids).mean(dim=0)
                        correction = style - centroid
                        embeddings = embeddings - correction.mean(dim=0)
                    return embeddings
        """),
    },
    # === GRN创新组件 ===
    "causal_grn_discovery": {
        "description": "因果GRN发现 — 用因果推断替代相关性，发现真正的基因调控关系",
        "layers": [
            {"type": "CausalAttention", "hidden_dim": 256},
            {"type": "InterventionHead", "method": "do_calculus"},
        ],
        "hyperparams": {"causal_threshold": 0.5, "intervention_budget": 10},
        "improves": {"grn": {"auprc": 0.06, "auroc": 0.04}},
        "code": textwrap.dedent("""\
            class CausalGRNDiscovery(nn.Module):
                def __init__(self, n_genes, hidden_dim=256):
                    super().__init__()
                    self.gene_proj = nn.Linear(n_genes, hidden_dim)
                    self.causal_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
                    self.edge_scorer = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        nn.Sigmoid(),
                    )

                def forward(self, expression_matrix):
                    h = self.gene_proj(expression_matrix)
                    causal_out, weights = self.causal_attn(h, h, h)
                    n = h.shape[1]
                    pairs = torch.cartesian_prod(torch.arange(n), torch.arange(n))
                    edge_features = torch.cat([causal_out[:, pairs[:, 0]], causal_out[:, pairs[:, 1]]], dim=-1)
                    return self.edge_scorer(edge_features).reshape(-1, n, n)
        """),
    },
    "dynamic_network_evolution": {
        "description": "动态网络演化 — 建模GRN随时间/条件变化的动态过程",
        "layers": [
            {"type": "TemporalGRN", "hidden_dim": 128},
            {"type": "NetworkEvolver", "n_timesteps": 20},
        ],
        "hyperparams": {"evolution_rate": 0.1, "consistency_weight": 0.3},
        "improves": {"grn": {"auprc": 0.04, "auroc": 0.03}},
        "code": textwrap.dedent("""\
            class DynamicNetworkEvolution(nn.Module):
                def __init__(self, n_genes, hidden_dim=128, n_timesteps=20):
                    super().__init__()
                    self.n_genes = n_genes
                    self.temporal_grn = nn.GRU(n_genes, hidden_dim, batch_first=True)
                    self.edge_predictor = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                    )

                def forward(self, time_series_expr):
                    h_seq, _ = self.temporal_grn(time_series_expr)
                    t = h_seq.shape[1] - 1
                    h_t = h_seq[:, t]
                    pairs = torch.cartesian_prod(torch.arange(self.n_genes), torch.arange(self.n_genes))
                    return self.edge_predictor(
                        torch.cat([h_t[:, pairs[:, 0]], h_t[:, pairs[:, 1]]], dim=-1)
                    ).reshape(-1, self.n_genes, self.n_genes)
        """),
    },
}

TASK_TEMPLATES = {
    "perturbation": {
        "base_layers": [
            {"type": "GeneExpressionEncoder", "hidden_dim": 512, "n_layers": 4},
            {"type": "PerturbationEmbedding", "dim": 256},
            {"type": "ExpressionDecoder", "hidden_dim": 512, "n_layers": 3},
        ],
        "base_hyperparams": {
            "learning_rate": 3e-4,
            "batch_size": 256,
            "epochs": 100,
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "dropout": 0.1,
            "weight_decay": 1e-5,
        },
        "available_innovations": [
            "trajectory_aware_encoder", "perturbation_diffusion", "gene_interaction_gnn"
        ],
        "architecture_type": "encoder-decoder",
        "rationale_templates": [
            "基于{dataset}数据集的扰动特性，采用{arch_type}架构，通过{innovation1}捕捉扰动后细胞状态的动态变化轨迹，同时利用{innovation2}增强基因间交互建模能力。",
            "考虑到单细胞扰动数据的稀疏性和高维特性，引入{innovation1}来显式建模从对照到扰动后的非线性映射，{innovation2}进一步提升分布外泛化能力。",
            "参考CellForge多Agent讨论结果，专家认为{dataset}的关键挑战在于未见扰动的泛化，因此{innovation1}提供条件生成能力，{innovation2}增强生物学可解释性。",
        ],
    },
    "cell_annotation": {
        "base_layers": [
            {"type": "GeneEmbedding", "dim": 512},
            {"type": "TransformerEncoder", "n_heads": 8, "n_layers": 6},
            {"type": "CellTypeClassifier", "hidden_dim": 256},
        ],
        "base_hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 512,
            "epochs": 50,
            "optimizer": "AdamW",
            "label_smoothing": 0.1,
            "dropout": 0.15,
        },
        "available_innovations": ["multi_scale_attention", "contrastive_cell_embedding"],
        "architecture_type": "encoder-classifier",
        "rationale_templates": [
            "针对{dataset}的细胞注释任务，采用{arch_type}架构，{innovation1}从多粒度（基因/通路/细胞）提取特征，{innovation2}增强同类细胞表征的聚集性。",
            "多Agent分析表明{dataset}的细胞类型分布不均衡，{innovation1}缓解少数类问题，{innovation2}提升零样本注释能力。",
        ],
    },
    "integration": {
        "base_layers": [
            {"type": "BatchEncoder", "hidden_dim": 512},
            {"type": "LatentProjector", "dim": 128},
            {"type": "ReconstructionHead", "hidden_dim": 512},
        ],
        "base_hyperparams": {
            "learning_rate": 2e-4,
            "batch_size": 512,
            "epochs": 80,
            "optimizer": "Adam",
            "kl_weight": 0.001,
            "recon_weight": 1.0,
        },
        "available_innovations": ["adversarial_domain_adapter", "harmony_style_transfer"],
        "architecture_type": "variational-encoder",
        "rationale_templates": [
            "批次整合的核心挑战是保留生物学差异的同时去除技术噪声。{innovation1}通过对抗训练区分生物/批次信号，{innovation2}迭代校正隐空间中的批次中心。",
            "针对{dataset}的多批次特性，采用{arch_type}，{innovation1}显式建模域分布差异，{innovation2}确保整合后的图连通性。",
        ],
    },
    "grn": {
        "base_layers": [
            {"type": "GenePairEncoder", "dim": 256},
            {"type": "AttentionScorer", "n_heads": 8},
            {"type": "EdgePredictor", "method": "sigmoid"},
        ],
        "base_hyperparams": {
            "learning_rate": 1e-4,
            "batch_size": 128,
            "epochs": 60,
            "optimizer": "AdamW",
            "edge_threshold": 0.3,
            "loss": "bce",
        },
        "available_innovations": ["causal_grn_discovery", "dynamic_network_evolution"],
        "architecture_type": "pairwise-scorer",
        "rationale_templates": [
            "GRN推断的核心是从表达数据中识别因果调控关系。{innovation1}引入因果推断替代简单相关性，{innovation2}建模网络随条件的动态演化。",
            "基于{dataset}的基因调控特征，采用{arch_type}，{innovation1}提升边预测的精确率，{innovation2}捕获条件特异性调控。",
        ],
    },
}

# 模型性能基线（基于文献报告的Mock性能参考）
MODEL_BASELINES = {
    "perturbation": {"scgpt": 0.87, "geneformer": 0.75, "cpa": 0.90, "gears": 0.85, "regformer": 0.92, "xtrimosc": 0.93, "default": 0.70},
    "cell_annotation": {"scgpt": 0.87, "geneformer": 0.82, "scbert": 0.85, "regformer": 0.90, "default": 0.78},
    "integration": {"scgpt": 0.85, "geneformer": 0.75, "default": 0.72},
    "grn": {"regformer": 0.82, "geneformer": 0.72, "scprint": 0.75, "default": 0.65},
}


class CellForgeGenerator(BaseGenerator):
    """CellForge多智能体架构生成器。

    模拟CellForge的三阶段流程，在Mock模式下基于文献知识组装架构。
    """

    def __init__(self, config: CellForgeConfig | None = None):
        super().__init__("CellForge")
        self.config = config or CellForgeConfig()
        self._rng = random.Random(self.config.seed)

    def describe(self) -> str:
        return (
            f"CellForge多智能体架构生成器 (mode={self.config.mode})\n"
            f"  - Task Analysis: 数据集特征分析 + 文献RAG\n"
            f"  - Method Design: {self.config.max_discussion_rounds}轮专家图基讨论\n"
            f"  - Innovation Budget: 最多{self.config.innovation_budget}个创新组件\n"
            f"  - 支持任务: perturbation, cell_annotation, integration, grn"
        )

    def generate(
        self,
        task: str,
        dataset: str,
        n_architectures: int = 3,
        **kwargs,
    ) -> GenerationResult:
        """生成N个候选架构。

        Args:
            task: 任务类型（perturbation/cell_annotation/integration/grn）。
            dataset: 目标数据集名称。
            n_architectures: 生成候选数。

        Returns:
            GenerationResult含n_architectures个GeneratedArchitecture。
        """
        if task not in TASK_TEMPLATES:
            raise ValueError(f"不支持的任务: {task}. 可用: {list(TASK_TEMPLATES.keys())}")

        template = TASK_TEMPLATES[task]
        rng_seed = hashlib.md5(f"{task}{dataset}{self.config.seed}".encode()).hexdigest()

        # Phase 1: Task Analysis
        task_analysis = self._phase1_task_analysis(task, dataset)

        # Phase 2: Method Design (multi-round expert discussion)
        design_history = []
        architectures = []

        for i in range(n_architectures):
            arch_seed = int(rng_seed[:8], 16) + i * 137
            arch_rng = random.Random(arch_seed)

            # Phase 2a: Select innovations (simulate expert consensus)
            n_innovations = arch_rng.randint(1, self.config.innovation_budget)
            available = template["available_innovations"]
            selected = arch_rng.sample(available, min(n_innovations, len(available)))

            # Phase 2b: Build architecture
            innovations_data = [INNOVATIONS[name] for name in selected]
            all_layers = list(template["base_layers"])
            for inn in innovations_data:
                all_layers.extend(inn["layers"])

            all_hyperparams = dict(template["base_hyperparams"])
            for inn in innovations_data:
                all_hyperparams.update(inn["hyperparams"])

            # Calculate confidence from innovation synergies
            base_score = MODEL_BASELINES.get(task, {}).get("default", 0.70)
            boost = sum(
                list(inn["improves"].get(task, {}).values())[0]
                for inn in innovations_data
                if task in inn["improves"]
            )
            confidence = min(0.99, max(0.3, base_score + boost + arch_rng.uniform(-0.02, 0.02)))

            # Phase 2c: Generate rationale
            rationale_template = arch_rng.choice(template["rationale_templates"])
            innovation_names = [INNOVATIONS[n]["description"].split("—")[0].strip() for n in selected]
            rationale = rationale_template.format(
                dataset=dataset,
                arch_type=template["architecture_type"],
                innovation1=innovation_names[0] if innovation_names else "标准编码器",
                innovation2=innovation_names[1] if len(innovation_names) > 1 else "标准解码器",
            )

            # Phase 3: Code Generation
            code = self._generate_code(task, template, innovations_data, all_hyperparams, i)

            # Design discussion entry
            discussion = {
                "round": i + 1,
                "selected_innovations": selected,
                "confidence": confidence,
                "rationale": rationale,
            }
            design_history.append(discussion)

            arch = GeneratedArchitecture(
                name=f"CellForge_v{i+1}_{dataset}_{task}",
                task=task,
                dataset=dataset,
                architecture_type=template["architecture_type"],
                layers=all_layers,
                hyperparams=all_hyperparams,
                innovations=[INNOVATIONS[n]["description"].split("—")[0].strip() for n in selected],
                code=code,
                design_rationale=rationale,
                confidence=round(confidence, 4),
            )
            architectures.append(arch)

        # Sort by confidence descending
        architectures.sort(key=lambda a: -a.confidence)

        return GenerationResult(
            architectures=architectures,
            task_analysis=task_analysis,
            design_history=design_history,
            metadata={
                "generator": "CellForge",
                "mode": self.config.mode,
                "task": task,
                "dataset": dataset,
                "n_architectures": n_architectures,
            },
        )

    def _phase1_task_analysis(self, task: str, dataset: str) -> dict:
        """Phase 1: 分析数据集特征和任务需求。"""
        template = TASK_TEMPLATES[task]
        return {
            "task": task,
            "dataset": dataset,
            "architecture_type": template["architecture_type"],
            "recommended_metrics": {
                "perturbation": ["mse", "mae", "pcc", "pds"],
                "cell_annotation": ["accuracy", "f1_macro", "f1_weighted"],
                "integration": ["kbet", "lisi", "asw", "graph_connectivity"],
                "grn": ["auprc", "auroc"],
            }.get(task, []),
            "key_challenges": {
                "perturbation": "未见扰动-细胞身份组合的泛化预测",
                "cell_annotation": "细胞类型分布不均衡 + 零样本注释",
                "integration": "保留生物差异同时去除批次效应",
                "grn": "从相关性到因果性的调控关系推断",
            }.get(task, ""),
            "available_innovations": template["available_innovations"],
        }

    def _generate_code(
        self,
        task: str,
        template: dict,
        innovations: list[dict],
        hyperparams: dict,
        variant: int,
    ) -> str:
        """Phase 3: 生成完整的PyTorch模型代码。"""
        # Collect innovation codes
        inn_code_blocks = []
        for inn in innovations:
            inn_code_blocks.append(inn["code"])

        innovation_classes = "\n\n".join(inn_code_blocks)

        # Generate model class name
        class_name = f"CellForgeModel_v{variant+1}"

        code = textwrap.dedent(f'''\
            """
            {class_name} — Auto-generated by CellForge Generator
            Task: {task}
            Architecture: {template["architecture_type"]}
            Innovations: {", ".join(i["description"].split("—")[0].strip() for i in innovations)}
            """
            import torch
            import torch.nn as nn
            import torch.nn.functional as F


            {innovation_classes}


            class {class_name}(nn.Module):
                """CellForge自动生成的{task}模型。"""

                def __init__(self, input_dim: int = 18000, hidden_dim: int = {hyperparams.get("hidden_dim", 512)}, n_classes: int = 10):
                    super().__init__()
                    self.input_proj = nn.Linear(input_dim, hidden_dim)
                    self.encoder = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout({hyperparams.get("dropout", 0.1)}),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                    )
                    # Innovation components
                    {self._init_innovation_code(innovations)}
                    # Task head
                    {self._task_head_code(task)}

                def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
                    h = self.input_proj(x)
                    h = self.encoder(h)
                    {self._forward_innovation_code(innovations)}
                    {self._forward_task_code(task)}
                    return output


            # Hyperparameters
            CONFIG = {json.dumps(hyperparams, indent=4)}
        ''')

        return code

    def _init_innovation_code(self, innovations: list[dict]) -> str:
        lines = []
        for i, inn in enumerate(innovations):
            first_layer = inn["layers"][0]
            layer_type = first_layer["type"]
            # Extract key params
            params = ", ".join(f"{k}={v}" for k, v in first_layer.items() if k != "type" and isinstance(v, (int, float, str)))
            lines.append(f"self.innovation_{i} = {layer_type}({params})" if params else f"self.innovation_{i} = {layer_type}()")
        return "\n                    ".join(lines)

    def _forward_innovation_code(self, innovations: list[dict]) -> str:
        lines = []
        for i in range(len(innovations)):
            lines.append(f"h = self.innovation_{i}(h)" if i == 0 else f"h = h + self.innovation_{i}(h)")
        return "\n                    ".join(lines)

    def _task_head_code(self, task: str) -> str:
        heads = {
            "perturbation": "self.head = nn.Linear(hidden_dim, input_dim)",
            "cell_annotation": "self.head = nn.Linear(hidden_dim, n_classes)",
            "integration": "self.head = nn.Linear(hidden_dim, 128)",
            "grn": "self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())",
        }
        return heads.get(task, "self.head = nn.Linear(hidden_dim, input_dim)")

    def _forward_task_code(self, task: str) -> str:
        return "output = self.head(h)"
