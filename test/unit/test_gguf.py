import os, struct, unittest
from tinygrad import dtypes, Tensor, fetch, Device
from tinygrad.nn.state import ggml_data_to_tensor, gguf_load
from tinygrad.device import is_dtype_supported
import numpy as np
from gguf import GGUFReader, GGUFValueType, GGMLQuantizationType, GGML_QUANT_SIZES, dequantize, quantize

ggml_test_block_count = 4

@unittest.skipIf(any(not is_dtype_supported(t) for t in [ dtypes.uint8, dtypes.half ]), "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def test_load_tinyllama_q8_0(self): self._test_gguf_load("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q8_0.gguf?download=true")
  def test_load_tinyllama_q4_0(self): self._test_gguf_load("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf?download=true")
  def test_load_gpt2_q4_1(self): self._test_gguf_load("https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q4_1.gguf?download=true")
  def test_load_sample_q6_k(self): self._test_gguf_load("https://huggingface.co/Isotr0py/test-gguf-sample/resolve/main/Quant_Q6_K_1024.gguf?download=true")

  def test_dequantization_q8_0_hardcoded(self):
    # Q8_0: 2 bytes float16 scale + 32 bytes int8 values, dequant = scale * values
    block = np.frombuffer(np.float16(2.0).tobytes() + np.arange(1, 33, dtype=np.int8).tobytes(), dtype=np.uint8).copy()
    expected = np.arange(1, 33, dtype=np.float32) * 2.0
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 32, GGMLQuantizationType.Q8_0.value).numpy().flatten(), expected)

  def test_dequantization_mxfp4_hardcoded(self):
    # MXFP4: 1 byte shared exponent E + 16 packed bytes (32 x 4-bit values)
    # nibble: bit3=sign, bit2:1=exp, bit0=mant; E=128 gives scale=1.0
    # codes 0-7 = [0, 1, 2, 3, 4, 6, 8, 12], codes 8-15 are their negatives
    block = np.array([0x80] + list(range(16)), dtype=np.uint8)  # E=128, nibbles 0-15 in low, zeros in high
    expected = np.array([0., 1., 2., 3., 4., 6., 8., 12., -0., -1., -2., -3., -4., -6., -8., -12.] + [0.]*16, dtype=np.float32)
    np.testing.assert_equal(ggml_data_to_tensor(Tensor(block), 32, GGMLQuantizationType.MXFP4.value).numpy().flatten(), expected)

  def test_dequantization_q4_0(self): self._test_dequantization(GGMLQuantizationType.Q4_0)
  def test_dequantization_q4_1(self): self._test_dequantization(GGMLQuantizationType.Q4_1)
  def test_dequantization_q8_0(self): self._test_dequantization(GGMLQuantizationType.Q8_0)
  def test_dequantization_q4_k(self): self._test_dequantization(GGMLQuantizationType.Q4_K)
  def test_dequantization_q5_k(self): self._test_dequantization(GGMLQuantizationType.Q5_K)
  def test_dequantization_q6_k(self): self._test_dequantization(GGMLQuantizationType.Q6_K)
  def test_dequantization_mxfp4(self): self._test_dequantization(GGMLQuantizationType.MXFP4)
  def test_dequantization_mxfp4_old(self):
    def encode(nibbles, E):
      packed = [(low & 0xF) | ((high & 0xF) << 4) for low, high in zip(nibbles[:16], nibbles[16:])]
      return np.array([E] + packed, dtype=np.uint8)

    def decode(code, E):
      sign = -1.0 if (code & 0b1000) else 1.0
      exp = (code >> 1) & 0b11
      mant = code & 0b1
      val = 2 * ((1.0 + 0.5 * mant) * np.exp2(exp - 1) if exp else 0.5 * mant)
      scale = np.exp2(E - 128) if E >= 2 else np.exp2(-127 if E == 1 else -128)
      return sign * val * scale

    blocks, expected = [], []
    rng = np.random.default_rng(42)
    for _ in range(4):
      E = rng.integers(0, 256)
      codes = rng.integers(0, 16, size=32, dtype=np.uint8)
      blocks.append(encode(codes, E))
      expected.extend(decode(c, E) for c in codes)
    tensor = Tensor(np.concatenate(blocks))
    out = ggml_data_to_tensor(tensor, len(expected), GGMLQuantizationType.MXFP4.value)
    np.testing.assert_equal(out.numpy(), expected)

  def test_dequantization_mxfp4_block(self):
    # https://gist.github.com/Ananta-Ranganathan/3317b6ed51a3b033e9c2564fafb4e043
    # used the above script to download the first block of blk.0.attn_k_b.weight from
    # https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/blob/main/GLM-4.7-Flash-MXFP4_MOE.gguf
    # and compute the canonical expected dequantized output with the GGUF PY implementation
    block = np.array([0x7a, 0x29, 0xab, 0x61, 0x10, 0x21, 0x02, 0x4a,
                    0x15, 0xca, 0x05, 0x01, 0x9b, 0x39, 0x0b, 0x0b, 0x1c], dtype=np.uint8)
    expected = np.array([-0.01562500, -0.04687500, 0.01562500, 0.00000000,
                        0.01562500,  0.03125000, -0.03125000, 0.09375000,
                        -0.03125000,  0.09375000, 0.01562500, -0.04687500,
                        -0.01562500, -0.04687500, -0.04687500, -0.06250000,
                        0.03125000, -0.03125000, 0.12500000,  0.01562500,
                        0.03125000,  0.00000000, 0.06250000,  0.01562500,
                        -0.06250000,  0.00000000, 0.00000000, -0.01562500,
                        0.04687500,  0.00000000, 0.00000000,  0.01562500], dtype=np.float32)
    out = ggml_data_to_tensor(Tensor(block), 32, GGMLQuantizationType.MXFP4.value)
    np.testing.assert_equal(out.numpy(), expected)

  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

  def _test_dequantization(self, qtype: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    n_el, n_bytes = ggml_test_block_count * block_size, ggml_test_block_count * type_size

    try:
      q_data = quantize((np.random.random((n_el,)).astype(np.float32) * 100 - 50), qtype)
    except NotImplementedError:
      q_data = np.random.default_rng(42).integers(0, 256, size=n_bytes, dtype=np.uint8)
    ref = dequantize(q_data, qtype)

    q_tensor = Tensor(q_data)
    dq_tensor = ggml_data_to_tensor(q_tensor, n_el, qtype.value).reshape(n_el)

    np.testing.assert_equal(dq_tensor.numpy(), ref)

  def _test_gguf_load(self, url: str):
    fp = fetch(url)
    model_size = os.stat(fp).st_size
    gguf_tensor = Tensor.empty(model_size, dtype=dtypes.uint8, device=f"disk:{fp}").to(Device.DEFAULT)
    kv_data, tensors = gguf_load(gguf_tensor)

    reader = GGUFReader(fp)

    for rt in reader.tensors:
      ref = dequantize(rt.data, rt.tensor_type)
      t = tensors[rt.name]
      if t.dtype.name == 'unsigned char' and len(t.shape) == 2:
        t = ggml_data_to_tensor(t.flatten(), ref.size, rt.tensor_type.value)
      np.testing.assert_equal(t.numpy(), ref.reshape(t.shape))

    for k, f in reader.fields.items():
      if k.startswith("GGUF."): continue  # skip file header keys (version, tensor_count, kv_count)
      def read_val(i, parts=f.parts, is_str=(f.types[-1] == GGUFValueType.STRING)):
        return bytes(parts[i]).decode("utf-8") if is_str else parts[i][0].item()
      if f.types[0] == GGUFValueType.ARRAY:
        self.assertEqual(kv_data[k], [read_val(i) for i in f.data])
      else:
        self.assertEqual(kv_data[k], read_val(-1))

class TestGGUFGEMV(unittest.TestCase):
  def _test_gguf_gemv(self, qtype: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    rows, cols = 8192, 2048
    n_blocks = rows * cols // block_size
    rng = np.random.default_rng(42)
    # generate random quantized blocks with valid fp16 scale fields (random bytes can produce NaN scales)
    q_data = rng.integers(0, 256, size=n_blocks * type_size, dtype=np.uint8).reshape(n_blocks, type_size)
    scales = np.float16(rng.standard_normal(n_blocks * 4)).view(np.uint8).reshape(n_blocks, -1)
    if qtype == GGMLQuantizationType.Q8_0: q_data[:, :2] = scales[:, :2]                    # d at offset 0
    elif qtype in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K): q_data[:, :4] = scales[:, :4] # d, dmin at offset 0
    elif qtype == GGMLQuantizationType.Q6_K: q_data[:, -2:] = scales[:, :2]                 # d at end
    elif qtype == GGMLQuantizationType.MXFP4: q_data[:, 0] = rng.integers(120, 136, size=n_blocks, dtype=np.uint8) # constrain byte0
    q_data = q_data.flatten()
    ref = dequantize(q_data, qtype).reshape(rows, cols)

    # build a minimal gguf in memory: header + 1 tensor info + aligned data
    buf = bytearray()
    buf += struct.pack("<4siqq", b"GGUF", 3, 1, 0)              # magic, version, n_tensors, n_kv
    buf += struct.pack("<Q", 6) + b"weight"                      # tensor name
    buf += struct.pack("<I", 2)                                  # ndims
    buf += struct.pack("<QQ", cols, rows)                        # dims (gguf stores reversed)
    buf += struct.pack("<i", qtype.value)
    buf += struct.pack("<Q", 0)                                  # offset
    buf += b"\x00" * ((32 - len(buf) % 32) % 32)                # pad to alignment=32
    buf += q_data.tobytes()

    _, tensors = gguf_load(Tensor(np.frombuffer(buf, dtype=np.uint8)).to(None))
    w = tensors["weight"]
    if w.dtype.name == 'unsigned char' and len(w.shape) == 2:
      w = ggml_data_to_tensor(w.flatten(), rows * cols, qtype.value).reshape(rows, cols)

    x = rng.standard_normal(cols).astype(np.float32)
    np.testing.assert_allclose((w @ Tensor(x)).numpy(), ref @ x, atol=1e-2, rtol=1e-2)
    np.testing.assert_equal(w.numpy(), ref)
    assert np.isfinite(ref).all() and np.isfinite(w.numpy()).all(), f"{qtype.name} has NaN/Inf"

  def test_gguf_gemv_q8_0(self): self._test_gguf_gemv(GGMLQuantizationType.Q8_0)
  def test_gguf_gemv_q4_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q4_K)
  def test_gguf_gemv_q5_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q5_K)
  def test_gguf_gemv_q6_k(self): self._test_gguf_gemv(GGMLQuantizationType.Q6_K)
  def test_gguf_gemv_mxfp4(self): self._test_gguf_gemv(GGMLQuantizationType.MXFP4)

@unittest.skipIf(any(not is_dtype_supported(t) for t in [ dtypes.uint8, dtypes.half ]), "Backend must support uint8 and half")
class TestQuantizedLinear(unittest.TestCase):
  # (qtype, block_elements, block_bytes, scale_bytes, scale_offset)
  _qtypes = [(GGMLQuantizationType.Q8_0, 32, 34, 2, 0), (GGMLQuantizationType.Q4_K, 256, 144, 4, 0),
             (GGMLQuantizationType.Q5_K, 256, 176, 4, 0), (GGMLQuantizationType.Q6_K, 256, 210, 2, 208)]

  def _make_blocks(self, qtype, block_elements, block_bytes, scale_bytes, scale_offset, rows=512, cols=256):
    rng = np.random.default_rng(42)
    n_blocks = rows * cols // block_elements
    q_data = rng.integers(0, 256, size=n_blocks * block_bytes, dtype=np.uint8).reshape(n_blocks, block_bytes)
    # ensure valid fp16 scales (random bytes can produce NaN)
    scales = np.float16(rng.standard_normal(n_blocks * (scale_bytes // 2))).view(np.uint8).reshape(n_blocks, scale_bytes)
    q_data[:, scale_offset:scale_offset + scale_bytes] = scales
    return q_data, rows, cols

  def test_dequantize_matches_ggml(self):
    """QuantizedLinear._dequantize() must match ggml reference for all supported quant types."""
    from tinygrad.apps.llm import QuantizedLinear
    for qtype, bel, bby, sb, so in self._qtypes:
      with self.subTest(qtype=qtype.name):
        q_data, rows, cols = self._make_blocks(qtype, bel, bby, sb, so)
        ref = dequantize(q_data.flatten(), qtype).reshape(rows, cols).astype(np.float16)
        ql = QuantizedLinear(Tensor(q_data), rows, cols, qtype.value)
        np.testing.assert_equal(ql._dequantize().numpy(), ref)

  def test_gemv_matches_linear(self):
    """QuantizedLinear matmul must match dequantized reference for all supported quant types."""
    from tinygrad.apps.llm import QuantizedLinear
    for qtype, bel, bby, sb, so in self._qtypes:
      with self.subTest(qtype=qtype.name):
        q_data, rows, cols = self._make_blocks(qtype, bel, bby, sb, so)
        ref_weights = dequantize(q_data.flatten(), qtype).reshape(rows, cols).astype(np.float16)
        x = np.random.default_rng(42).standard_normal(cols).astype(np.float32)
        ql = QuantizedLinear(Tensor(q_data), rows, cols, qtype.value)
        np.testing.assert_allclose(ql(Tensor(x)).numpy(), (Tensor(x) @ Tensor(ref_weights).transpose()).numpy(), atol=1e-1, rtol=1e-1)

  def test_auto_detect_type(self):
    """QuantizedLinear infers ggml_type from block size when not specified."""
    from tinygrad.apps.llm import QuantizedLinear
    for qtype, bel, bby, sb, so in self._qtypes:
      with self.subTest(qtype=qtype.name):
        q_data, rows, cols = self._make_blocks(qtype, bel, bby, sb, so)
        ql = QuantizedLinear(Tensor(q_data), rows, cols)
        self.assertEqual(ql.ggml_type, qtype.value)

  def test_keep_quantized_gguf_load(self):
    """gguf_load keeps supported quant types as raw uint8 blocks."""
    for qtype, bel, bby, sb, so in self._qtypes:
      with self.subTest(qtype=qtype.name):
        q_data, rows, cols = self._make_blocks(qtype, bel, bby, sb, so)
        n_blocks = rows * cols // bel
        buf = bytearray()
        buf += struct.pack("<4siqq", b"GGUF", 3, 1, 0)
        buf += struct.pack("<Q", 6) + b"weight"
        buf += struct.pack("<I", 2)
        buf += struct.pack("<QQ", cols, rows)
        buf += struct.pack("<i", qtype.value)
        buf += struct.pack("<Q", 0)
        buf += b"\x00" * ((32 - len(buf) % 32) % 32)
        buf += q_data.tobytes()
        _, tensors = gguf_load(Tensor(np.frombuffer(buf, dtype=np.uint8)).to(None))
        self.assertEqual(tensors["weight"].shape, (n_blocks, bby))
        self.assertEqual(tensors["weight"].dtype.name, "unsigned char")

if __name__ == '__main__':
  unittest.main()
