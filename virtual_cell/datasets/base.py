"""
数据集基类 — 26个单细胞数据集的统一接口

每个数据集需要实现：
- load(): 加载数据
- get_splits(): 获取训练/验证/测试划分
- get_metadata(): 获取元信息
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class DatasetType(Enum):
    """数据集类型。"""
    SCRNA_SEQ = "scrna_seq"         # 单细胞RNA测序
    SPATIAL = "spatial"              # 空间转录组
    PERTURBATION = "perturbation"    # 扰动实验
    MULTI_OMICS = "multi_omics"      # 多组学
    ATLAS = "atlas"                  # 细胞图谱


@dataclass
class DatasetInfo:
    """数据集元信息。"""
    name: str
    dataset_type: DatasetType
    n_cells: int                    # 细胞数
    n_genes: int                    # 基因数
    n_cell_types: int               # 细胞类型数
    tissues: list[str]              # 组织来源
    organisms: list[str]            # 物种
    technology: str                 # 测序技术
    paper: str                      # 论文链接
    download_url: str               # 下载链接
    supported_tasks: list[str]      # 支持的任务
    description: str = ""
    year: int = 2020

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.dataset_type.value,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "n_cell_types": self.n_cell_types,
            "tissues": self.tissues,
            "organisms": self.organisms,
            "year": self.year,
        }


@dataclass
class DataSplit:
    """数据集划分。"""
    train_X: np.ndarray
    train_y: Optional[np.ndarray]
    val_X: Optional[np.ndarray] = None
    val_y: Optional[np.ndarray] = None
    test_X: Optional[np.ndarray] = None
    test_y: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


class BaseDataset(ABC):
    """数据集基类。"""

    def __init__(self, info: DatasetInfo):
        self.info = info
        self._loaded = False
        self._data = None

    @abstractmethod
    def load(self, data_dir: str = "", **kwargs) -> None:
        """加载数据集。"""
        ...

    @abstractmethod
    def get_splits(self, split_ratio: float = 0.8, seed: int = 42) -> DataSplit:
        """获取数据划分。"""
        ...

    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return f"{self.info.name}(cells={self.info.n_cells:,}, loaded={self._loaded})"


# ================================================================
# 26个数据集定义
# ================================================================

DATASETS_INFO = {
    # 核心Benchmark数据集（10个）
    "zheng68k": DatasetInfo(
        name="Zheng68K",
        dataset_type=DatasetType.SCRNA_SEQ,
        n_cells=68_579, n_genes=20_000, n_cell_types=10,
        tissues=["PBMC"], organisms=["Human"],
        technology="10X Chromium",
        paper="https://www.nature.com/articles/ncomms14049",
        download_url="https://support.10xgenomics.com/single-cell-gene-expression/datasets",
        supported_tasks=["cell_annotation", "integration"],
        description="PBMC 68K cells, 细胞注释经典benchmark",
        year=2017,
    ),
    "pbmc10k": DatasetInfo(
        name="PBMC 10K",
        dataset_type=DatasetType.SCRNA_SEQ,
        n_cells=10_000, n_genes=15_000, n_cell_types=8,
        tissues=["PBMC"], organisms=["Human"],
        technology="10X Chromium",
        paper="https://www.10xgenomics.com/",
        download_url="https://www.10xgenomics.com/",
        supported_tasks=["cell_annotation", "integration", "perturbation"],
        description="PBMC 10K cells, 多任务benchmark",
        year=2019,
    ),
    "hca": DatasetInfo(
        name="Human Cell Atlas",
        dataset_type=DatasetType.ATLAS,
        n_cells=1_000_000, n_genes=25_000, n_cell_types=500,
        tissues=["多组织"], organisms=["Human"],
        technology="多平台",
        paper="https://www.nature.com/articles/s41591-023-02325-0",
        download_url="https://cellxgene.census.science/",
        supported_tasks=["cell_annotation", "integration", "grn"],
        description="人类细胞图谱，覆盖多组织",
        year=2023,
    ),
    "cellxgene_census": DatasetInfo(
        name="CELLxGENE Census",
        dataset_type=DatasetType.ATLAS,
        n_cells=50_000_000, n_genes=60_000, n_cell_types=1000,
        tissues=["多组织"], organisms=["Human", "Mouse"],
        technology="多平台",
        paper="https://www.nature.com/articles/s41592-023-02132-y",
        download_url="https://cellxgene.census.science/",
        supported_tasks=["cell_annotation", "integration", "grn", "perturbation"],
        description="5000万+细胞的统一数据集，预训练数据源",
        year=2024,
    ),
    "kang2018": DatasetInfo(
        name="Kang2018",
        dataset_type=DatasetType.PERTURBATION,
        n_cells=24_614, n_genes=15_000, n_cell_types=8,
        tissues=["PBMC"], organisms=["Human"],
        technology="10X Chromium",
        paper="https://www.nature.com/articles/nbt.4042",
        download_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583",
        supported_tasks=["perturbation", "cell_annotation"],
        description="IFN-β扰动PBMC数据集",
        year=2018,
    ),
    "adamson2016": DatasetInfo(
        name="Adamson2016",
        dataset_type=DatasetType.PERTURBATION,
        n_cells=60_000, n_genes=12_000, n_cell_types=1,
        tissues=["K562细胞系"], organisms=["Human"],
        technology="Perturb-seq",
        paper="https://www.cell.com/cell/fulltext/S0092-8674(16)31082-0",
        download_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546",
        supported_tasks=["perturbation"],
        description="CRISPR扰动K562细胞",
        year=2016,
    ),
    "norman2019": DatasetInfo(
        name="Norman2019",
        dataset_type=DatasetType.PERTURBATION,
        n_cells=100_000, n_genes=15_000, n_cell_types=1,
        tissues=["K562细胞系"], organisms=["Human"],
        technology="Perturb-seq",
        paper="https://www.science.org/doi/10.1126/science.aax4438",
        download_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344",
        supported_tasks=["perturbation"],
        description="组合基因扰动，Virtual Cell Challenge参考",
        year=2019,
    ),
    "haber2017": DatasetInfo(
        name="Haber2017",
        dataset_type=DatasetType.SCRNA_SEQ,
        n_cells=53_193, n_genes=15_000, n_cell_types=15,
        tissues=["肠道"], organisms=["Mouse"],
        technology="inDrop",
        paper="https://www.nature.com/articles/nature24489",
        download_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92332",
        supported_tasks=["cell_annotation"],
        description="小鼠肠道单细胞图谱",
        year=2017,
    ),
    "tabula_muris": DatasetInfo(
        name="Tabula Muris",
        dataset_type=DatasetType.ATLAS,
        n_cells=100_000, n_genes=20_000, n_cell_types=200,
        tissues=["20个小鼠组织"], organisms=["Mouse"],
        technology="10X + Smart-seq2",
        paper="https://www.nature.com/articles/s41586-018-0590-4",
        download_url="https://tabula-muris.ds.czbiohub.org/",
        supported_tasks=["cell_annotation", "integration"],
        description="小鼠细胞图谱，20个组织",
        year=2018,
    ),
    "tabula_sapiens": DatasetInfo(
        name="Tabula Sapiens",
        dataset_type=DatasetType.ATLAS,
        n_cells=500_000, n_genes=25_000, n_cell_types=400,
        tissues=["人体多组织"], organisms=["Human"],
        technology="10X Chromium",
        paper="https://www.science.org/doi/10.1126/science.abl4896",
        download_url="https://tabula-sapiens-portal.ds.czbiohub.org/",
        supported_tasks=["cell_annotation", "integration", "grn"],
        description="人体细胞图谱，近50万细胞",
        year=2022,
    ),

    # 扩展数据集（16个）
    "zhengsorted": DatasetInfo("Zhengsorted", DatasetType.SCRNA_SEQ, 20_000, 15_000, 10, ["PBMC"], ["Human"], "10X FACS", "", "", ["cell_annotation"], "细胞分选验证", 2017),
    "macosko2015": DatasetInfo("Macosko2015", DatasetType.SCRNA_SEQ, 49_000, 18_000, 30, ["视网膜"], ["Mouse"], "Drop-seq", "", "", ["cell_annotation"], "视网膜细胞类型", 2015),
    "baron2016": DatasetInfo("Baron2016", DatasetType.SCRNA_SEQ, 8_569, 17_000, 14, ["胰腺"], ["Human"], "inDrop", "", "", ["cell_annotation"], "人胰腺细胞图谱", 2016),
    "muraro2016": DatasetInfo("Muraro2016", DatasetType.SCRNA_SEQ, 2_122, 19_000, 9, ["胰腺"], ["Human"], "CEL-seq2", "", "", ["cell_annotation"], "胰腺内分泌细胞", 2016),
    "segerstolpe2016": DatasetInfo("Segerstolpe2016", DatasetType.SCRNA_SEQ, 2_394, 22_000, 14, ["胰腺"], ["Human"], "Smart-seq2", "", "", ["cell_annotation"], "胰腺单细胞", 2016),
    "xin2016": DatasetInfo("Xin2016", DatasetType.SCRNA_SEQ, 1_492, 20_000, 6, ["胰腺"], ["Human"], "SMARTer", "", "", ["cell_annotation"], "胰腺β细胞", 2016),
    "lawlor2017": DatasetInfo("Lawlor2017", DatasetType.SCRNA_SEQ, 2_209, 20_000, 8, ["胰腺"], ["Human"], "Fluidigm C1", "", "", ["cell_annotation"], "胰腺单细胞", 2017),
    "enge2017": DatasetInfo("Enge2017", DatasetType.SCRNA_SEQ, 2_544, 20_000, 7, ["胰腺"], ["Human"], "inDrop", "", "", ["cell_annotation"], "胰腺衰老", 2017),
    "camp2017": DatasetInfo("Camp2017", DatasetType.SCRNA_SEQ, 65_000, 18_000, 25, ["大脑类器官"], ["Human"], "Drop-seq", "", "", ["cell_annotation"], "大脑类器官发育", 2017),
    "plasschaert2019": DatasetInfo("Plasschaert2019", DatasetType.SCRNA_SEQ, 25_000, 15_000, 10, ["肺上皮"], ["Human"], "10X", "", "", ["cell_annotation"], "肺上皮细胞", 2019),
    "lukowski2019": DatasetInfo("Lukowski2019", DatasetType.SCRNA_SEQ, 20_000, 15_000, 12, ["视网膜"], ["Human"], "10X", "", "", ["cell_annotation"], "人视网膜", 2019),
    "travaglini2020": DatasetInfo("Travaglini2020", DatasetType.SCRNA_SEQ, 100_000, 20_000, 50, ["肺"], ["Human"], "10X", "", "", ["cell_annotation", "grn"], "人肺单细胞图谱", 2020),
    "cao2020": DatasetInfo("Cao2020", DatasetType.SCRNA_SEQ, 200_000, 25_000, 80, ["多组织"], ["Zebrafish"], "sci-RNA-seq3", "", "", ["cell_annotation"], "斑马鱼发育", 2020),
    "cao2017": DatasetInfo("Cao2017", DatasetType.SCRNA_SEQ, 42_000, 15_000, 30, ["多组织"], ["C. elegans"], "sci-RNA-seq", "", "", ["cell_annotation"], "蠕虫发育", 2017),
    "saunders2018": DatasetInfo("Saunders2018", DatasetType.SCRNA_SEQ, 70_000, 20_000, 100, ["大脑"], ["Mouse"], "Drop-seq", "", "", ["cell_annotation", "grn"], "小鼠大脑图谱", 2018),
    "vcc2025": DatasetInfo("Virtual Cell Challenge", DatasetType.PERTURBATION, 300_000, 20_000, 1, ["H1 hESCs"], ["Human"], "CRISPRi", "https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up", "", ["perturbation"], "Virtual Cell Challenge 2025，300个CRISPRi扰动", 2025),
}


def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """获取数据集信息。"""
    return DATASETS_INFO.get(name.lower())


def list_datasets() -> list[dict]:
    """列出所有数据集。"""
    return [
        {"key": k, "name": v.name, "n_cells": v.n_cells, "n_genes": v.n_genes,
         "n_cell_types": v.n_cell_types, "type": v.dataset_type.value,
         "tissues": v.tissues, "year": v.year}
        for k, v in DATASETS_INFO.items()
    ]


def get_all_dataset_keys() -> list[str]:
    """获取所有数据集key。"""
    return list(DATASETS_INFO.keys())


def filter_datasets(
    task: str = "",
    organism: str = "",
    min_cells: int = 0,
) -> list[dict]:
    """按条件筛选数据集。"""
    results = []
    for k, v in DATASETS_INFO.items():
        if task and task not in v.supported_tasks:
            continue
        if organism and organism not in v.organisms:
            continue
        if v.n_cells < min_cells:
            continue
        results.append({"key": k, "name": v.name, "n_cells": v.n_cells, "tasks": v.supported_tasks})
    return results
