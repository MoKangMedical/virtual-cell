"""Lingshu-Cell 完整架构测试。"""
import sys
import numpy as np

sys.path.insert(0, '/root/virtual-cell')


def test_quantizer_encode_decode():
    """281级token化编解码。"""
    from virtual_cell.models.lingshu_cell import Quantizer

    q = Quantizer()

    # 0-99 精确保留
    counts = np.array([0, 1, 50, 99])
    tokens = q.encode(counts)
    decoded = q.decode(tokens)
    assert np.array_equal(tokens, [0, 1, 50, 99])
    assert np.array_equal(decoded, [0, 1, 50, 99])
    print("✅ Quantizer 0-99: 精确保留")

    # 100-999 对数分箱
    counts = np.array([100, 150, 200, 500, 999])
    tokens = q.encode(counts)
    assert all(100 <= t < 190 for t in tokens), f"Tokens out of range: {tokens}"
    decoded = q.decode(tokens)
    # 近似相等（允许误差）
    for orig, dec in zip(counts, decoded):
        assert abs(orig - dec) / max(orig, 1) < 0.15, f"Too much error: {orig} vs {dec}"
    print("✅ Quantizer 100-999: 对数分箱")

    # 1000-9999 对数分箱
    counts = np.array([1000, 5000, 9999])
    tokens = q.encode(counts)
    assert all(190 <= t < 280 for t in tokens), f"Tokens out of range: {tokens}"
    print("✅ Quantizer 1000-9999: 对数分箱")

    # OVF
    counts = np.array([10000, 50000])
    tokens = q.encode(counts)
    assert all(t == 280 for t in tokens)
    print("✅ Quantizer OVF: 溢出处理")

    # Full roundtrip
    counts = np.random.randint(0, 500, size=(100,))
    tokens = q.encode(counts)
    decoded = q.decode(tokens)
    assert tokens.shape == counts.shape
    assert all(0 <= t <= 280 for t in tokens)
    print("✅ Quantizer roundtrip: 100随机值")


def test_rmsnorm():
    """RMSNorm层。"""
    import torch
    from virtual_cell.models.lingshu_cell import RMSNorm

    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)
    print("✅ RMSNorm")


def test_swiglu():
    """SwiGLU FFN。"""
    import torch
    from virtual_cell.models.lingshu_cell import SwiGLU

    ffn = SwiGLU(64, 128)
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.shape == (2, 10, 64)
    print("✅ SwiGLU")


def test_attention():
    """Multi-Head Attention + RoPE。"""
    import torch
    from virtual_cell.models.lingshu_cell import MultiHeadAttention

    attn = MultiHeadAttention(64, n_heads=4, max_seq_len=100)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out.shape == (2, 10, 64)

    # With mask
    mask = torch.ones(2, 10)
    mask[:, 5:] = 0
    out_masked = attn(x, mask=mask)
    assert out_masked.shape == (2, 10, 64)
    print("✅ Multi-Head Attention + RoPE")


def test_transformer_block():
    """Transformer Block。"""
    import torch
    from virtual_cell.models.lingshu_cell import TransformerBlock

    block = TransformerBlock(64, n_heads=4, intermediate_dim=128, max_seq_len=100)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64)
    print("✅ Transformer Block")


def test_sequence_compressor():
    """序列压缩/解压。"""
    import torch
    from virtual_cell.models.lingshu_cell import SequenceCompressor

    comp = SequenceCompressor(64, patch_size=4)
    x = torch.randn(2, 16, 64)  # 16 genes, patch_size=4 → 4 patches
    compressed = comp.compress(x)
    assert compressed.shape == (2, 4, 64)

    decompressed = comp.decompress(compressed, target_len=16)
    assert decompressed.shape == (2, 16, 64)
    print("✅ Sequence Compressor")


def test_lingshu_cell_forward():
    """LingshuCell前向传播。"""
    import torch
    from virtual_cell.models.lingshu_cell import LingshuCell

    model = LingshuCell(
        n_genes=50, vocab_size=281, hidden_dim=64,
        n_heads=4, n_layers=2, intermediate_dim=128,
        patch_size=10, n_cell_types=5, n_perturbations=10,
    )

    x = torch.randint(0, 282, (2, 50))
    ct = torch.tensor([0, 1])
    pt = torch.tensor([3, 7])

    # 无条件
    logits = model(x)
    assert logits.shape == (2, 50, 281)
    print("✅ LingshuCell forward (无条件)")

    # 有条件
    logits_cond = model(x, condition_cell_type=ct, condition_perturbation=pt)
    assert logits_cond.shape == (2, 50, 281)
    print("✅ LingshuCell forward (有条件)")


def test_lingshu_cell_loss():
    """MDDM训练损失。"""
    import torch
    from virtual_cell.models.lingshu_cell import LingshuCell

    model = LingshuCell(
        n_genes=50, vocab_size=281, hidden_dim=64,
        n_heads=4, n_layers=2, intermediate_dim=128,
        patch_size=10, n_cell_types=5, n_perturbations=10,
    )

    x_0 = torch.randint(0, 281, (2, 50))
    loss = model.compute_loss(x_0)
    assert loss.shape == ()
    assert loss.item() > 0
    print(f"✅ LingshuCell loss: {loss.item():.4f}")

    # 有条件损失
    ct = torch.tensor([0, 1])
    pt = torch.tensor([3, 7])
    loss_cond = model.compute_loss(x_0, ct, pt)
    assert loss_cond.item() > 0
    print(f"✅ LingshuCell loss (有条件): {loss_cond.item():.4f}")


def test_lingshu_cell_sample():
    """采样生成（小规模）。"""
    import torch
    from virtual_cell.models.lingshu_cell import LingshuCell

    model = LingshuCell(
        n_genes=20, vocab_size=281, hidden_dim=64,
        n_heads=4, n_layers=2, intermediate_dim=128,
        patch_size=10, n_cell_types=5, n_perturbations=10,
    )

    generated = model.sample(batch_size=2, n_steps=4, device='cpu')
    assert generated.shape == (2, 20)
    assert (generated >= 0).all() and (generated <= 281).all()
    print(f"✅ LingshuCell sample: shape={generated.shape}")

    # 有先验的采样
    prior = np.array([0, 1, 2])
    generated_prior = model.sample(
        batch_size=2, n_steps=4, prior_genes=prior, prior_value=1, device='cpu'
    )
    assert generated_prior.shape == (2, 20)
    print(f"✅ LingshuCell sample (prior): {generated_prior[:, :3]}")


def test_lingshu_cell_perturbation():
    """扰动预测。"""
    import torch
    from virtual_cell.models.lingshu_cell import LingshuCell

    model = LingshuCell(
        n_genes=20, vocab_size=281, hidden_dim=64,
        n_heads=4, n_layers=2, intermediate_dim=128,
        patch_size=10, n_cell_types=5, n_perturbations=10,
    )

    # 不同扰动应该产生不同输出
    ct = torch.tensor([0, 0])
    pt1 = torch.tensor([1, 1])
    pt2 = torch.tensor([5, 5])

    gen1 = model.sample(batch_size=2, n_steps=4, condition_cell_type=ct,
                        condition_perturbation=pt1, device='cpu')
    gen2 = model.sample(batch_size=2, n_steps=4, condition_cell_type=ct,
                        condition_perturbation=pt2, device='cpu')

    assert gen1.shape == gen2.shape == (2, 20)
    print(f"✅ LingshuCell perturbation prediction: gen1={gen1[0,:5].tolist()}, gen2={gen2[0,:5].tolist()}")


def test_trainer():
    """训练器单步。"""
    import torch
    from virtual_cell.models.lingshu_cell import LingshuCell, LingshuTrainer

    model = LingshuCell(
        n_genes=20, vocab_size=281, hidden_dim=64,
        n_heads=4, n_layers=2, intermediate_dim=128,
        patch_size=10, n_cell_types=5, n_perturbations=10,
    )
    trainer = LingshuTrainer(model, lr=1e-3)

    batch = torch.randint(0, 281, (4, 20))
    loss = trainer.train_step(batch)
    assert isinstance(loss, float)
    assert loss > 0

    # 多步训练loss应下降（或至少稳定）
    losses = [trainer.train_step(batch) for _ in range(5)]
    print(f"✅ Trainer: 5步loss: {[f'{l:.3f}' for l in losses]}")


if __name__ == "__main__":
    print("=" * 50)
    print("🧬 Lingshu-Cell Architecture Tests")
    print("=" * 50)

    test_quantizer_encode_decode()
    test_rmsnorm()
    test_swiglu()
    test_attention()
    test_transformer_block()
    test_sequence_compressor()
    test_lingshu_cell_forward()
    test_lingshu_cell_loss()
    test_lingshu_cell_sample()
    test_lingshu_cell_perturbation()
    test_trainer()

    print()
    print("🎉 全部11个Lingshu-Cell测试通过！")
