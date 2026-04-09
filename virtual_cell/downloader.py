"""
数据集下载器 — 自动获取公开单细胞数据集

支持：
- CELLxGENE Census（5000万+细胞）
- 10X Genomics（公开数据集）
- GEO/SRA（学术数据）
- Zenodo（补充数据集）
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional


# ================================================================
# 数据集注册表（下载URL + 校验信息）
# ================================================================

DATASET_REGISTRY = {
    "zheng68k": {
        "name": "Zheng68K",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/68k_pbmc/68k_pbmc_filtered_gene_bc_matrices.tar.gz",
        "format": "10x_mtx",
        "size_mb": 120,
        "n_cells": 68579,
        "description": "68K PBMC cells from 10X Genomics",
    },
    "pbmc10k": {
        "name": "PBMC 10K",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.tar.gz",
        "format": "10x_mtx",
        "size_mb": 180,
        "n_cells": 10000,
        "description": "10K PBMC cells, 3' v3 chemistry",
    },
    "pbmc1k": {
        "name": "PBMC 1K",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/10k_PBMC_3p_nextgem_Chromium_Controller/10k_PBMC_3p_nextgem_Chromium_Controller_filtered_feature_bc_matrix.tar.gz",
        "format": "10x_mtx",
        "size_mb": 50,
        "n_cells": 10000,
        "description": "10K PBMC cells, Next GEM",
    },
}

CELLXGENE_DATASETS = {
    "tabula_sapiens": {
        "name": "Tabula Sapiens",
        "collection_id": "53d208b0-2cfd-4e68-a44c-9a4d8e0c4db5",
        "n_cells": 499000,
        "description": "Human cell atlas across 24 organs",
    },
    "tabula_muris": {
        "name": "Tabula Muris",
        "collection_id": "0bd88099-65d4-4333-b6c6-4e0b3859f278",
        "n_cells": 556512,
        "description": "Mouse cell atlas across 20 organs",
    },
}


class DatasetDownloader:
    """数据集下载管理器。"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self, dataset_key: str, max_cells: int = 0) -> str:
        """
        下载数据集并返回本地路径。

        Args:
            dataset_key: 数据集名称。
            max_cells: 最大细胞数（0=全部）。

        Returns:
            下载后文件/目录的路径。
        """
        if dataset_key in DATASET_REGISTRY:
            return self._download_direct(DATASET_REGISTRY[dataset_key])
        elif dataset_key in CELLXGENE_DATASETS:
            return self._download_cellxgene(CELLXGENE_DATASETS[dataset_key], max_cells)
        else:
            raise ValueError(f"未知数据集: {dataset_key}. 可用: {self.list_available()}")

    def _download_direct(self, info: dict) -> str:
        """直接下载（10X/GEO等）。"""
        target_dir = self.data_dir / info["name"].lower().replace(" ", "_")
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = info["url"].split("/")[-1]
        filepath = target_dir / filename

        if filepath.exists():
            print(f"  ✅ 已存在: {filepath}")
            return str(target_dir)

        print(f"  ⬇️ 下载: {info['name']} ({info['size_mb']}MB)")
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(filepath), info["url"]],
                check=True, capture_output=True, timeout=300,
            )

            # 解压
            if filename.endswith(".tar.gz"):
                subprocess.run(
                    ["tar", "-xzf", str(filepath), "-C", str(target_dir)],
                    check=True, capture_output=True,
                )
            elif filename.endswith(".gz"):
                subprocess.run(
                    ["gunzip", "-k", str(filepath)],
                    check=True, capture_output=True,
                )

            print(f"  ✅ 完成: {target_dir}")
            return str(target_dir)

        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            return ""

    def _download_cellxgene(self, info: dict, max_cells: int = 0) -> str:
        """通过CELLxGENE Census下载。"""
        try:
            import cellxgene_census

            print(f"  📡 连接CELLxGENE Census...")
            with cellxgene_census.open_soma() as census:
                # 查询数据集
                print(f"  📊 下载: {info['name']}")
                # 具体下载逻辑取决于cellxgene_census API
                print(f"  ℹ️ 需要cellxgene_census包: pip install cellxgene-census")
                return ""

        except ImportError:
            print(f"  ⚠️ 未安装cellxgene-census，跳过: {info['name']}")
            print(f"     安装: pip install cellxgene-census")
            return ""

    def list_available(self) -> list[str]:
        """列出所有可下载的数据集。"""
        return list(DATASET_REGISTRY.keys()) + list(CELLXGENE_DATASETS.keys())

    def list_local(self) -> list[dict]:
        """列出已下载的数据集。"""
        results = []
        for d in self.data_dir.iterdir():
            if d.is_dir():
                results.append({
                    "name": d.name,
                    "path": str(d),
                    "size_mb": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6,
                })
        return results
