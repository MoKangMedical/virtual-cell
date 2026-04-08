"""
数据集加载器 — 支持AnnData/h5ad + Mock降级

真实模式：scanpy + anndata 加载 h5ad/csv 文件
Mock模式：生成模拟数据（开发/测试）
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Optional

from .base import BaseDataset, DatasetInfo, DataSplit, DATASETS_INFO

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


class AnnDataDataset(BaseDataset):
    """
    真实数据集加载器（需要scanpy+anndata）。

    支持格式：
    - .h5ad (AnnData HDF5)
    - .csv (基因表达矩阵)
    - 10X .h5 (CellRanger输出)
    - 10X mtx目录
    """

    def __init__(self, info: DatasetInfo):
        super().__init__(info)
        self._adata = None

    def load(self, data_dir: str = "", **kwargs) -> None:
        """加载真实单细胞数据。"""
        if not HAS_SCANPY:
            raise ImportError("需要安装scanpy和anndata: pip install scanpy anndata")

        data_path = Path(data_dir) if data_dir else Path(".")

        # 尝试多种文件格式
        for ext in [".h5ad", ".h5", ".csv"]:
            candidate = data_path / f"{self.info.name.lower()}{ext}"
            if candidate.exists():
                self._load_file(str(candidate), **kwargs)
                return

        # 尝试10X mtx目录
        mtx_dir = data_path / self.info.name.lower()
        if mtx_dir.exists() and (mtx_dir / "matrix.mtx.gz").exists():
            self._adata = sc.read_10x_mtx(str(mtx_dir))
            self._post_process(**kwargs)
            return

        # 没有找到文件，降级到Mock
        self._load_mock(**kwargs)

    def _load_file(self, filepath: str, **kwargs) -> None:
        """从文件加载。"""
        max_cells = kwargs.get("max_cells", 0)

        if filepath.endswith(".h5ad"):
            self._adata = sc.read_h5ad(filepath)
        elif filepath.endswith(".h5"):
            self._adata = sc.read_10x_h5(filepath)
        elif filepath.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(filepath, index_col=0)
            self._adata = ad.AnnData(df.values)
            self._adata.var_names = df.columns.tolist()
            self._adata.obs_names = df.index.tolist()

        self._post_process(max_cells=max_cells)

    def _post_process(self, **kwargs) -> None:
        """数据预处理。"""
        max_cells = kwargs.get("max_cells", 0)
        if max_cells and self._adata.n_obs > max_cells:
            sc.pp.subsample(self._adata, n_obs=max_cells)

        # 基本QC
        sc.pp.filter_cells(self._adata, min_genes=200)
        sc.pp.filter_genes(self._adata, min_cells=3)

        # 归一化
        sc.pp.normalize_total(self._adata, target_sum=1e4)
        sc.pp.log1p(self._adata)

        self._loaded = True

    def _load_mock(self, **kwargs) -> None:
        """降级到Mock数据。"""
        mock = MockDataset(self.info)
        mock.load(**kwargs)
        self._data = mock._data
        self._loaded = True

    def get_splits(self, split_ratio: float = 0.8, seed: int = 42) -> DataSplit:
        if not self._loaded:
            raise RuntimeError("数据集未加载")

        rng = np.random.RandomState(seed)

        if self._adata is not None:
            X = self._adata.X.toarray() if hasattr(self._adata.X, 'toarray') else self._adata.X
            n = X.shape[0]
            idx = rng.permutation(n)
            split = int(n * split_ratio)

            # 尝试获取细胞类型标签
            y = None
            for col in ["cell_type", "celltype", "cell_ontology_class", "labels"]:
                if col in self._adata.obs.columns:
                    y = self._adata.obs[col].values
                    break

            return DataSplit(
                train_X=X[idx[:split]],
                train_y=y[idx[:split]] if y is not None else None,
                val_X=X[idx[split:]],
                val_y=y[idx[split:]] if y is not None else None,
            )
        else:
            # Mock数据
            X = self._data["X"]
            y = self._data["cell_types"]
            n = X.shape[0]
            idx = rng.permutation(n)
            split = int(n * split_ratio)
            return DataSplit(
                train_X=X[idx[:split]], train_y=y[idx[:split]],
                val_X=X[idx[split:]], val_y=y[idx[split:]],
            )


class MockDataset(BaseDataset):
    """Mock数据集（无scanpy时使用）。"""

    def __init__(self, info: DatasetInfo):
        super().__init__(info)

    def load(self, data_dir: str = "", **kwargs) -> None:
        n = min(self.info.n_cells, kwargs.get("max_cells", 1000))
        g = min(self.info.n_genes, kwargs.get("max_genes", 500))
        rng = np.random.RandomState(42)

        # 模拟泊松分布的计数数据（更像真实scRNA-seq）
        base_expr = rng.exponential(scale=1.0, size=(n, g))
        # 添加批次效应
        n_batches = 3
        batch_labels = rng.choice(n_batches, size=n)
        for b in range(n_batches):
            mask = batch_labels == b
            base_expr[mask] *= rng.uniform(0.8, 1.2, size=g)

        self._data = {
            "X": base_expr.astype(np.float32),
            "gene_names": [f"gene_{i}" for i in range(g)],
            "cell_types": rng.choice(
                [f"type_{i}" for i in range(min(self.info.n_cell_types, 10))], size=n
            ),
            "batch_labels": batch_labels,
        }
        self._loaded = True

    def get_splits(self, split_ratio: float = 0.8, seed: int = 42) -> DataSplit:
        if not self._loaded:
            raise RuntimeError("数据集未加载")
        rng = np.random.RandomState(seed)
        X = self._data["X"]
        y = self._data["cell_types"]
        n = X.shape[0]
        idx = rng.permutation(n)
        split = int(n * split_ratio)
        return DataSplit(
            train_X=X[idx[:split]], train_y=y[idx[:split]],
            val_X=X[idx[split:]], val_y=y[idx[split:]],
        )


def create_dataset(name: str, **kwargs) -> BaseDataset:
    """创建数据集实例。"""
    info = DATASETS_INFO.get(name.lower())
    if info is None:
        raise ValueError(f"未知数据集: {name}. 可用: {list(DATASETS_INFO.keys())}")

    if HAS_SCANPY:
        return AnnDataDataset(info)
    return MockDataset(info)
