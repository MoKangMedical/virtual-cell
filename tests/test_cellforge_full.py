"""CellForge Full模式测试。"""
import sys
sys.path.insert(0, '/root/virtual-cell')

from virtual_cell.generators.cellforge_full import LLMAgent, CellForgeFullConfig


def test_llm_agent_init():
    """测试Agent初始化。"""
    config = CellForgeFullConfig(api_key="test", base_url="http://localhost:8000/v1")
    agent = LLMAgent(config, "You are a test agent.")
    assert agent.system_prompt == "You are a test agent."
    assert agent.history == []
    print("✅ LLM Agent初始化")


def test_cellforge_full_config():
    """测试配置。"""
    config = CellForgeFullConfig()
    assert config.max_discussion_rounds == 3
    assert config.model == "mimo"
    print("✅ CellForge Full Config")


def test_parser():
    """测试设计解析器。"""
    from virtual_cell.generators.cellforge_full import CellForgeFullGenerator
    gen = CellForgeFullGenerator()

    test_json = '''
    {
        "name": "TestModel",
        "layers": [{"type": "Linear"}],
        "innovations": ["Test innovation"],
        "code": "import torch\\nclass Test(nn.Module): pass",
        "rationale": "Test rationale",
        "hyperparams": {"lr": 1e-4}
    }
    '''
    arch = gen._parse_design(test_json, "perturbation", "test_ds", 0)
    assert arch.name == "TestModel"
    assert arch.task == "perturbation"
    assert "import torch" in arch.code
    print("✅ 设计解析器")


def test_describe():
    """测试describe方法。"""
    from virtual_cell.generators.cellforge_full import CellForgeFullGenerator
    gen = CellForgeFullGenerator()
    desc = gen.describe()
    assert "CellForge Full Mode" in desc
    assert "Proposer/Reviewer/Synthesizer" in desc
    print("✅ describe方法")


if __name__ == "__main__":
    test_llm_agent_init()
    test_cellforge_full_config()
    test_parser()
    test_describe()
    print("\n🎉 CellForge Full 测试全部通过！")
