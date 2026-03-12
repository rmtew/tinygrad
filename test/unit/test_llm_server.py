import unittest
from unittest.mock import patch
from tinygrad import Tensor, UOp
from tinygrad.engine.schedule import schedule_cache

class TestTransformerGenerate(unittest.TestCase):
  def test_kv_cache_reuse(self):
    """Test that generate reuses the KV cache when tokens extend the cached prefix."""
    from tinygrad.apps.llm import Transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=32)

    captured_inputs = []
    def mock_call(self, tokens, start_pos):
      captured_inputs.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.val))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation: prefill 5 tokens + 1 decode
      tokens = [1, 2, 3, 4, 5]
      gen = model.generate(tokens)
      next(gen)  # prefill
      next(gen)  # decode

      # second call extends the conversation — cached prefix should be reused
      captured_inputs.clear()
      tokens = [1, 2, 3, 4, 5, 42, 42, 10, 11, 12]
      gen = model.generate(tokens)
      next(gen)

    # should only process tokens[7:] = [10, 11, 12] since first 7 are cached
    toks_shape = captured_inputs[0][0][-1]
    self.assertEqual(toks_shape.val if isinstance(toks_shape, UOp) else toks_shape, 3)
    self.assertEqual(captured_inputs[0][1], 7)

  def test_kv_cache_invalidation(self):
    """Test that generate invalidates the KV cache when tokens diverge from the cached prefix."""
    from tinygrad.apps.llm import Transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=32)

    captured_inputs = []
    def mock_call(self, tokens, start_pos):
      captured_inputs.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.val))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation
      gen = model.generate([1, 2, 3, 4, 5])
      next(gen)

      # completely different prompt — KV cache should be invalidated
      captured_inputs.clear()
      gen = model.generate([10, 20, 30])
      next(gen)

    # should process all 3 tokens from start
    toks_shape = captured_inputs[0][0][-1]
    self.assertEqual(toks_shape.val if isinstance(toks_shape, UOp) else toks_shape, 3)
    self.assertEqual(captured_inputs[0][1], 0)

  def test_two_prompts_schedule_cache(self):
    """Third prompt should hit the schedule cache, not miss (first two warm up both jits: prefill + decode)."""
    from tinygrad.apps.llm import Transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=64)

    # first two prompts warm up both jits (prefill + decode)
    ids = list(range(1, 6))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    ids += list(range(10, 15))
    gen = model.generate(ids)
    for _ in range(3): next(gen)
    cache_size_after_warmup = len(schedule_cache)

    # third prompt should reuse the same schedule cache entries, not create new ones
    ids += list(range(20, 25))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    self.assertEqual(cache_size_after_warmup, len(schedule_cache),
      f"third prompt added {len(schedule_cache) - cache_size_after_warmup} new schedule cache entries (expected 0)")

  def test_chunked_prefill(self):
    """When prompt > chunk_size, all chunks should be prefill"""
    from tinygrad.apps.llm import Transformer
    from tinygrad.uop.ops import resolve
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=64)

    def get_prefill_flags(tokens, chunk_size):
      is_prefill = []
      def mock_call(self, tokens, start_pos):
        is_prefill.append(resolve(tokens.shape[1] != 1))
        return Tensor([[42]])
      with patch.object(Transformer, '__call__', mock_call):
        gen = model.generate(tokens, chunk_size=chunk_size)
        for _ in range(3): next(gen)
      model._cached_tokens = []
      return is_prefill

    # 8 tokens, chunk_size=4 -> 2 prefill chunks
    self.assertEqual(get_prefill_flags(list(range(8)), 4), [True, True, False, False])
    # 9 tokens, chunk_size=4 -> 3 prefill chunks (4+4+1)
    self.assertEqual(get_prefill_flags(list(range(9)), 4), [True, True, True, False, False])
    # 4 tokens, chunk_size=4 -> 1 prefill chunk
    self.assertEqual(get_prefill_flags(list(range(4)), 4), [True, False, False])

  def test_qwen35_two_prompts_schedule_cache(self):
    """Qwen3.5 hybrid model (SSM + attention blocks) should be fully symbolic — no schedule cache growth."""
    from tinygrad.apps.llm import Transformer, TransformerBlock
    dim, hidden_dim, norm_eps, max_context = 64, 128, 1e-5, 64
    n_heads, n_kv_heads, head_dim, rope_theta = 2, 2, 32, 10000.0
    n_v_heads, n_k_heads, ssm_head_dim, conv_kernel = 4, 4, 16, 4
    # pattern: 3 GatedDeltaNet + 1 TransformerBlock (full_attention_interval=4)
    blk = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context,
                            n_v_heads=n_v_heads, ssm_n_k_heads=n_k_heads, ssm_head_dim=ssm_head_dim, conv_kernel=conv_kernel) for _ in range(3)]
    blk.append(TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context, attn_gate=True))
    model = Transformer(num_blocks=4, dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                        norm_eps=norm_eps, vocab_size=100, head_dim=head_dim, rope_theta=rope_theta, max_context=max_context, blk=blk)

    # first two prompts warm up the JIT
    gen = model.generate(list(range(1, 6)))
    for _ in range(3): next(gen)
    gen = model.generate(list(range(10, 20)))
    for _ in range(3): next(gen)
    cache_size_after_warmup = len(schedule_cache)

    # third prompt should reuse — no new schedule cache entries
    gen = model.generate(list(range(20, 30)))
    for _ in range(3): next(gen)
    self.assertEqual(cache_size_after_warmup, len(schedule_cache),
      f"third prompt added {len(schedule_cache) - cache_size_after_warmup} new schedule cache entries (expected 0)")

class TestApplyRope(unittest.TestCase):
  def test_partial_rope(self):
    """With rope_dim < head_dim, the non-rotated suffix should pass through unchanged."""
    from tinygrad.apps.llm import apply_rope, precompute_freqs_cis
    freqs = precompute_freqs_cis(16, 10, 10000.0)  # dim=16, only rotate first 16 of 32
    x = Tensor.randn(1, 1, 2, 32)
    result = apply_rope(x, freqs[:2], rope_dim=16)
    # last 16 dims unchanged
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result[..., 16:].tolist(), x[..., 16:].tolist())
    # first 16 dims are rotated (should differ from input)
    self.assertNotEqual(result[..., :16].tolist(), x[..., :16].tolist())

class TestSSMGatedDeltaNet(unittest.TestCase):
  def test_grouped_query_forward(self):
    """GatedDeltaNet block with n_v_heads != n_k_heads should produce correct output shape."""
    from tinygrad.apps.llm import TransformerBlock
    blk = TransformerBlock(dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2, norm_eps=1e-5, head_dim=32, rope_theta=10000.0,
                           n_v_heads=4, ssm_n_k_heads=2, ssm_head_dim=16, conv_kernel=4)
    x = Tensor.randn(1, 1, 64)  # (B=1, T=1, D=64)
    out = blk(x, start_pos=0)
    self.assertEqual(out.shape, (1, 1, 64))

  def test_generate_determinism(self):
    """Two identical generate() calls on a qwen35 model should produce the same tokens (state reset works)."""
    from tinygrad.apps.llm import Transformer, TransformerBlock
    dim, hidden_dim, norm_eps, max_context = 64, 128, 1e-5, 64
    blk = [TransformerBlock(dim, hidden_dim, n_heads=2, n_kv_heads=2, norm_eps=norm_eps, head_dim=32, rope_theta=10000.0, max_context=max_context,
                            n_v_heads=4, ssm_n_k_heads=4, ssm_head_dim=16, conv_kernel=4) for _ in range(3)]
    blk.append(TransformerBlock(dim, hidden_dim, n_heads=2, n_kv_heads=2, norm_eps=norm_eps, head_dim=32,
                                rope_theta=10000.0, max_context=max_context, attn_gate=True))
    model = Transformer(num_blocks=4, dim=dim, hidden_dim=hidden_dim, n_heads=2, n_kv_heads=2,
                        norm_eps=norm_eps, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=max_context, blk=blk)
    prompt = list(range(1, 6))
    out1 = [next(model.generate(list(prompt))) for _ in range(3)]
    out2 = [next(model.generate(list(prompt))) for _ in range(3)]
    self.assertEqual(out1, out2, f"generate() not deterministic across calls: {out1} != {out2}")

if __name__ == '__main__':
  unittest.main()
