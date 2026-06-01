# scripts/calibration/tests/test_gauntlet_args.py
import sys
from pathlib import Path
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
import method_gauntlet as mg


def test_recipe_args_forwards_store_true():
    out = mg.recipe_args({"--faithful-acts": True, "--n-samples": "128",
                          "--faithful-weights": False})
    assert "--faithful-acts" in out
    assert "--faithful-weights" not in out
    i = out.index("--n-samples"); assert out[i + 1] == "128"


def test_qat_stage_present_and_paired():
    # Stage 6 is the W4A8 paired-toggle stage. q1_off must be recipe-identical to
    # stage-5 c_wiki (the 19.2678 zero-point) so Gate C holds by construction.
    assert 6 in mg.STAGES
    names = [n for (n, _) in mg.STAGES[6]]
    assert names == ["q1_off", "q2_acts", "q3_actswt", "q4_heavy"]
    q1 = dict(mg.STAGES[6])["q1_off"]
    c_wiki = dict(mg.STAGES[5])["c_wiki"]
    assert q1 == c_wiki, "q1_off must match c_wiki exactly (Gate C zero-point)"
    # toggles escalate monotonically
    assert dict(mg.STAGES[6])["q2_acts"].get("--faithful-acts") is True
    q3 = dict(mg.STAGES[6])["q3_actswt"]
    assert q3.get("--faithful-acts") is True and q3.get("--faithful-weights") is True
    q4 = dict(mg.STAGES[6])["q4_heavy"]
    assert q4.get("--heavy-rounds") == "4" and q4.get("--act-order") is True
