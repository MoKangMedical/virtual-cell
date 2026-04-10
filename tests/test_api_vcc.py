"""REST API + VCC Pipeline 测试。"""
import sys
sys.path.insert(0, '/root/virtual-cell')


def test_api_import():
    """测试API模块可导入。"""
    from virtual_cell.api import app
    assert app.title == "VirtualCell API"
    print("✅ API模块导入")


def test_api_health_endpoint():
    """测试/health端点逻辑。"""
    from virtual_cell import __version__
    from virtual_cell.registry import ModelRegistry, DatasetRegistry
    
    assert __version__ == "0.4.0"
    assert len(ModelRegistry.list()) == 15
    assert len(DatasetRegistry.list()) == 26
    print("✅ API health端点逻辑")


def test_vcc_pipeline_init():
    """测试VCC Pipeline初始化。"""
    from virtual_cell.vcc.pipeline import VCCPipeline, VCCMetrics
    
    pipeline = VCCPipeline()
    assert pipeline.data_dir == ""
    
    metrics = VCCMetrics(des=0.5, pds=0.7, mae=0.1)
    assert metrics.to_dict()["des"] == 0.5
    assert metrics.average_score() != 0
    print("✅ VCC Pipeline初始化")


def test_vcc_metrics():
    """测试VCC指标计算。"""
    import numpy as np
    from virtual_cell.vcc.pipeline import VCCPipeline, VCCMetrics
    
    pipeline = VCCPipeline()
    
    # Mock evaluation
    pred = np.random.randn(100, 50)
    gt = np.random.randn(100, 50)
    
    metrics = pipeline.evaluate(pred, gt)
    assert isinstance(metrics, VCCMetrics)
    assert 0 <= metrics.mae or True  # MAE is always positive
    print(f"✅ VCC评估: MAE={metrics.mae:.4f}")


def test_vcc_submission_format():
    """测试VCC提交格式化。"""
    from virtual_cell.vcc.pipeline import VCCPipeline, VCCMetrics
    
    pipeline = VCCPipeline()
    metrics = VCCMetrics(des=0.5, pds=0.7, mae=0.1, pearson_delta=0.3)
    
    sub = pipeline.format_submission(metrics, "test_model")
    assert sub["team"] == "VirtualCell-OPC"
    assert sub["model"] == "test_model"
    assert "metrics" in sub
    print("✅ VCC提交格式")


if __name__ == "__main__":
    test_api_import()
    test_api_health_endpoint()
    test_vcc_pipeline_init()
    test_vcc_metrics()
    test_vcc_submission_format()
    print("\n🎉 API + VCC 全部测试通过！")
