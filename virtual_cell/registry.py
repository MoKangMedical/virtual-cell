"""
注册中心 — 统一模型和数据集的加载接口
"""

from .models import create_model
from .models.base import get_model_info, list_models, get_all_model_keys, MODELS_INFO
from .datasets import create_dataset
from .datasets.base import get_dataset_info, list_datasets, get_all_dataset_keys, DATASETS_INFO, filter_datasets
from .tasks import get_task, list_tasks, TASK_REGISTRY


class ModelRegistry:
    """模型注册中心。"""

    @staticmethod
    def get(name: str):
        return create_model(name)

    @staticmethod
    def info(name: str):
        return get_model_info(name)

    @staticmethod
    def list():
        return list_models()


class DatasetRegistry:
    """数据集注册中心。"""

    @staticmethod
    def get(name: str, **kwargs):
        return create_dataset(name, **kwargs)

    @staticmethod
    def info(name: str):
        return get_dataset_info(name)

    @staticmethod
    def list():
        return list_datasets()

    @staticmethod
    def filter(task: str = "", organism: str = "", min_cells: int = 0):
        return filter_datasets(task=task, organism=organism, min_cells=min_cells)


def load_model(name: str):
    """快捷加载模型。"""
    return create_model(name)


def load_dataset(name: str, **kwargs):
    """快捷加载数据集。"""
    return create_dataset(name, **kwargs)
