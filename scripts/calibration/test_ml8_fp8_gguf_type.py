import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "gguf-py"))
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
def test_ml8_fp8_registered():
    t = GGMLQuantizationType.ML8_FP8
    assert t.value == 51
    block, size = GGML_QUANT_SIZES[t]
    assert block == 32                      # group_size
    assert size == 32 * 1 + 2               # 32 e4m3 bytes + one fp16 scale = 34
if __name__ == "__main__":
    test_ml8_fp8_registered(); print("ML8_FP8 GGUF TYPE OK")
