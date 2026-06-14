# ggml/src/ggml-cuda/aiter-integration/tools/test_build_triton_pinned.py
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_pin_file_is_a_40hex_sha():
    sha = (HERE / "TRITON_PIN.txt").read_text().split()[0]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_preflight_passes_on_current_driverless_patch_anchors():
    # The driverless patch script is the source of truth for anchors; running it
    # in --check mode against the installed Triton must succeed (anchors present
    # or already patched). Exit 0 = ok; non-zero = a bump broke the patch.
    import triton
    triton_root = Path(triton.__file__).resolve().parent.parent  # .../python
    r = subprocess.run([sys.executable, str(HERE.parent / "cmake" / "patch_triton_driverless_aot.py"),
                        "--check", str(triton_root)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
