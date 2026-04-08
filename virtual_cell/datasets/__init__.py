"""
数据集加载器 — Mock实现

Mock模式用于开发/测试，生成模拟单细胞数据。
真实模式需要AnnData/h5ad文件。
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

from .base import BaseDataset, DatasetInfo, DataSplit, DATASETS_INFO


class MockDataset(BaseDataset):
    """通用模拟数据集。"""

    def __init__(self, info: DatasetInfo):
        super().__init__(info)

    def load(self, data_dir: str = "", **kwargs) -> None:
        """生成模拟数据。"""
        n = min(self.info.n_cells, kwargs.get("max_cells", 1000))
        g = min(self.info.n_genes, kwargs.get("max_genes", 500))
        self._data = {
            "X": np.random.randn(n, g).astype(np.float32),
            "gene_names": [f"gene_{i}" for i in range(g)],
            "cell_types": np.random.choice(
                [f"type_{i}" for i in range(min(self.info.n_cell_types, 10))], size=n
            ),
            "batch_labels": np.random.choice(["batch_1", "batch_2", "batch_3"], size=n),
        }
        self._loaded = True

    def get_splits(self, split_ratio: float = 0.8, seed: int = 42) -> DataSplit:
        if not self._loaded:
            raise RuntimeError("数据集未加载，先调用 load()")

        rng = np.random.RandomState(seed)
        X = self._data["X"]
        y = self._data["cell_types"]
        n = X.shape[0]
        idx = rng.permutation(n)
        split = int(n * split_ratio)

        return DataSplit(
            train_X=X[idx[:split]],
            train_y=y[idx[:split]],
            val_X=X[idx[split:]],
            val_y=y[idx[split:]],
        )


def create_dataset(name: str, **kwargs) -> BaseDataset:
    """创建数据集实例。"""
    info = DATASETS_INFO.get(name.lower())
    if info is None:
        raise ValueError(f"未知数据集: {name}. 可用: {list(DATASETS_INFO.keys())}")
    return MockDataset(info)
