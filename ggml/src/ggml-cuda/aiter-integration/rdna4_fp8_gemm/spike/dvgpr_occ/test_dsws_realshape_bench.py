#!/usr/bin/env python3
"""Offline-only unit tests for dsws_realshape_bench.py."""

from pathlib import Path
import contextlib
import io
import subprocess
import tempfile
import unittest
from unittest import mock

import dsws_realshape_bench as bench


VALID_LOG = """\
  GPU clock-counter freq ~= 100.00 MHz (Dmax cycles -> seconds)
  G=6 SEGK=256 FM=1 FN=4  NCOMP=4 NAFEED=2 NBFEED=2
  oracle shape 96x512x2048  (super-tile 96x64, KT=128, NTL=8, MTLsuper=1)
  [dsws2 bounds] A last=1/2 OK  B last=1/2 OK  C last=1/2 OK
  [dsws2] 96x512x2048  super-tile=96x64 (G=6 FM=1 FN=4)  TOTAL=8 TOTAL_super=64 n_kseg=8
  [dsws2] *** WARNING: SINGLE CHUNK (8 tiles) -- the 0.75s cap CANNOT fire mid-chunk
  [dsws2 completion] occ[0](live)=0 (0=clean)  occ[20](claim)=8
  [dsws2 WORK-EXACT] computed == G*TOTAL_super == 384  (no work dropped)
  [dsws2 THROUGHPUT] 96x512x2048  TF=0.2  (0.1% of 307 TF fp8 peak)  span=100000 ticks / 1 chunk(s) @ 100 MHz
  [dsws2 oracle] ok=24 bad=0 max_rel=0 tier=LOOSE
  dsws2 oracle CLEAN
"""


class HarnessTest(unittest.TestCase):
    def parse_text(self, text: str) -> bench.ParsedRun:
        with tempfile.TemporaryDirectory(dir=bench.SCRIPT_DIR) as tmp:
            path = Path(tmp) / "ml8_moe_ffn_gate_up_M64.log"
            path.write_text(text, encoding="utf-8")
            return bench.parse_log(path)

    def reject_text(self, text: str, reason: str) -> None:
        with self.assertRaisesRegex(bench.LogRejected, reason):
            self.parse_text(text)

    def test_derives_padded_and_real_throughput(self) -> None:
        run = self.parse_text(VALID_LOG)
        self.assertAlmostEqual(run.padded_tflops, 0.201326592)
        self.assertAlmostEqual(run.real_flop_corrected_tflops, 0.134217728)
        self.assertEqual(run.reps, 1)
        self.assertIsNone(run.spread_percent_reported_1dp)

    def test_corrupted_ticks_are_rejected(self) -> None:
        self.reject_text(VALID_LOG.replace("span=100000", "span=10000"), "SELF_VALIDATION_MISMATCH")

    def test_bad_oracle_is_rejected(self) -> None:
        self.reject_text(VALID_LOG.replace("ok=24 bad=0", "ok=23 bad=1"), "ORACLE_FAILED")

    def test_incomplete_is_rejected_before_timing(self) -> None:
        self.reject_text(VALID_LOG + "dsws2 INCOMPLETE\n", "INVALID_RUN_MARKER")

    def test_cannot_evaluate_is_refused_by_name(self) -> None:
        # A STAGINSTR=0 bin emits no occ[71], so the host prints CANNOT-EVALUATE instead of a verdict.
        # Such a run must be refused under its OWN name -- not as a generic "WORK_EXACT: found 0",
        # which reads like a regex bug and invites someone to loosen the pattern. Absence of a
        # correctness gate is not a pass.
        text = VALID_LOG.replace(
            "  [dsws2 WORK-EXACT] computed == G*TOTAL_super == 384  (no work dropped)\n",
            "*** DSWS2 WORK-EXACT: CANNOT-EVALUATE -- NO CORRECTNESS VERDICT ***\n",
        )
        self.assertNotIn("[dsws2 WORK-EXACT] computed ==", text)  # guard is non-vacuous
        self.reject_text(text, "WORK_EXACT_CANNOT_EVALUATE")

    def test_exact_recovered_shape_count(self) -> None:
        self.assertEqual(len(bench.SHAPES), 33)

    def test_failure_record_cannot_carry_throughput(self) -> None:
        self.assertIsNone(bench.failure_record("bad.log", "rejected")["throughput_tflops"])

    def test_live_nonzero_halts_without_another_invocation(self) -> None:
        # g/acc_n/sswin/waves/tag are the dispatch-config args; g=6 keeps the 96-row super-tile
        # that the UNSUPPORTED/NOT_RUN counts below are computed against.
        args = mock.Mock(target_secs=1.5, stride=8, g=6, acc_n=3, sswin=8, waves=30, tag="rs", chunk=0)
        refused = subprocess.CompletedProcess(args=[], returncode=4, stdout="host refused\n")
        with mock.patch.object(bench.subprocess, "run", return_value=refused) as run_mock:
            with contextlib.redirect_stdout(io.StringIO()):
                records = bench.run_live(args)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(len(records), len(bench.SHAPES))
        self.assertEqual(sum(record["status"] == "FAIL" for record in records), 1)
        # 6 UNSUPPORTED, not 5: mlmf_router_MLP (K=256, SEGK=256 -> n_kseg=1) is now screened out
        # by the ZLOCK fail-safe check instead of being dispatched into a WORK-INEXACT latch.
        self.assertEqual(sum(record["status"] == "NOT_RUN" for record in records), 26)
        self.assertEqual(sum(record["status"] == "UNSUPPORTED" for record in records), 6)


if __name__ == "__main__":
    unittest.main()
