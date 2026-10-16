"""
Tests for OCRead read validation (code_validator.py) and the consensus loop
in PaddleOCR.read(). The rule being protected: the system may reject a
read, but must never invent or "repair" information into a pass.

Run with:  python -m unittest tests.test_code_validator
"""
import unittest
from unittest import mock

import numpy as np

from backend.detection.ocr.code_validator import CodeTemplate

PATTERN = r"^\d{4} ([01]\d|2[0-3]):[0-5]\d BB (0[1-9]|1[0-2])/\d{2}$"


def T(**kw):
    args = dict(value_pattern=PATTERN, min_confidence=0.8, max_corrections=1, max_ignored_tokens=2)
    args.update(kw)
    return CodeTemplate(["#### ##:##", "BB ##/##"], **args)


def toks(*texts, conf=0.99):
    return [{"text": t, "confidence": conf} for t in texts]


class TemplateTests(unittest.TestCase):
    def test_tokens_assemble_into_the_canonical_code(self):
        r = T().validate(toks("4113", "08:26", "BB", "05/26"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")

    def test_spacing_from_the_model_does_not_matter(self):
        r = T().validate(toks("411308:26", "BB05/26"))
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")

    def test_lookalikes_in_literal_positions_are_normalised(self):
        r = T().validate(toks("4113 08.26", "B8 05126"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(r["corrections"], 0)

    def test_fullwidth_colon_from_multilingual_model(self):
        self.assertEqual(T().validate(toks("4113 08：26", "BB 05/26"))["text"], "4113 08:26 BB 05/26")

    def test_digit_is_never_changed_into_another_digit(self):
        # 08:25 fits the format - the validator alone cannot know it should
        # be 08:26 (that is what consensus is for), but it must not alter it.
        r = T().validate(toks("4113 08:25", "BB 05/26"))
        self.assertEqual(r["text"], "4113 08:25 BB 05/26")

    def test_one_letter_lookalike_in_a_digit_position_is_allowed(self):
        r = T().validate(toks("4I13 08:26", "BB 05/26"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(r["corrections"], 1)

    def test_more_corrections_than_allowed_is_rejected(self):
        r = T().validate(toks("4II3 08:26", "BB 05/26"))
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reason"], "format_mismatch")

    def test_truncated_read_is_rejected(self):
        r = T().validate(toks("4113 08:2", "BB 05/26"))
        self.assertEqual(r["reason"], "format_mismatch")

    def test_text_is_never_cut_out_of_a_token(self):
        # "08:250" must not become "08:25".
        r = T().validate(toks("4113 08:250", "BB 05/26"))
        self.assertEqual(r["reason"], "format_mismatch")

    def test_unknown_character_marker_is_rejected(self):
        r = T().validate(toks("4113 08:2?", "BB 05/26"))
        self.assertEqual(r["reason"], "format_mismatch")

    def test_separate_noise_tokens_are_ignored_up_to_the_limit(self):
        r = T().validate(toks("PE/PA", "5203", "21:28", "BB08/27"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "5203 21:28 BB 08/27")
        self.assertEqual(r["ignored_tokens"], ["PE/PA"])
        r = T(max_ignored_tokens=0).validate(toks("PE/PA", "5203", "21:28", "BB08/27"))
        self.assertEqual(r["reason"], "format_mismatch")

    def test_upside_down_cap_token_order_is_recovered(self):
        r = T().validate(toks("05/26", "BB", "08:26", "4113"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(r["orientation"], "reversed")

    def test_scrambled_token_order_still_fields(self):
        for order in (("BB", "4113", "05/26", "08:26"), ("4113", "05/26", "08:26", "BB")):
            r = T().validate(toks(*order))
            self.assertTrue(r["accepted"], order)
            self.assertEqual(r["text"], "4113 08:26 BB 05/26")

    def test_punctuation_speck_on_a_token_edge_is_stripped(self):
        self.assertEqual(T().validate(toks("411308:26", "BB", "05/26."))["text"], "4113 08:26 BB 05/26")
        # ...but a glued digit is information and is never stripped
        self.assertEqual(T().validate(toks("411308:26", "BB", "05/264"))["reason"], "format_mismatch")

    def test_ambiguous_assignment_is_rejected(self):
        # Two tokens that both fit the BB date field with different values.
        r = T().validate(toks("4113 08:26", "BB", "05/26", "07/26"))
        self.assertFalse(r["accepted"])
        self.assertTrue(r.get("ambiguous"))

    def test_low_confidence_is_rejected(self):
        r = T().validate([{"text": "4113 08:26", "confidence": 0.99}, {"text": "BB 05/26", "confidence": 0.5}])
        self.assertEqual(r["reason"], "low_confidence")

    def test_impossible_value_is_rejected(self):
        self.assertEqual(T().validate(toks("4113 27:26", "BB 05/26"))["reason"], "invalid_value")
        self.assertEqual(T().validate(toks("4113 08:26", "BB 13/26"))["reason"], "invalid_value")

    def test_nothing_read(self):
        self.assertEqual(T().validate([])["reason"], "no_text_detected")


class _ScriptedWorker:
    """Returns the next scripted answer on each call (one per variant read)."""
    script = []

    def __init__(self, **kwargs):
        self.calls = 0

    def ocr(self, frame, timeout=None):
        answer = _ScriptedWorker.script[min(self.calls, len(_ScriptedWorker.script) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return [{"text": t, "confidence": 0.99} for t in answer]


class ConsensusTests(unittest.TestCase):
    def _engine(self, *script, variants=("dot_divide", "dot_subtract", "classic"), min_agree=2):
        from backend.detection.ocr import paddle_ocr
        _ScriptedWorker.script = list(script)
        with mock.patch.object(paddle_ocr, "PaddleOCRWorker", _ScriptedWorker):
            engine = paddle_ocr.PaddleOCR(app_state=None)
        engine.dynamic_roi_debug_dir = None
        engine.orientation_retry_enabled = False
        engine.validator = T()
        engine.consensus_variants = list(variants)
        engine.consensus_min_agree = min_agree
        # These tests cover the exact/early-stop mode; tests of read-all set it explicitly.
        engine.consensus_fusion = "exact"
        engine.consensus_early_stop = True
        engine.font_reader = None   # these tests cover the PaddleOCR consensus path
        return engine

    def _frame(self):
        return np.full((480, 744, 3), 190, np.uint8)

    def test_two_agreeing_reads_stop_early(self):
        e = self._engine(["4113 08:26", "BB 05/26"], ["4113", "08:26", "BB05/26"])
        r = e.read(self._frame())
        self.assertTrue(r["validation"]["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(e._paddle.calls, 2)
        self.assertNotIn("failure_reason", r)

    def test_confident_misread_is_outvoted(self):
        e = self._engine(["4113 08:25", "BB 05/26"], ["4113 08:26", "BB 05/26"], ["4113 08:26", "BB 05/26"])
        r = e.read(self._frame())
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(e._paddle.calls, 3)

    def test_disagreement_is_a_reject_not_a_guess(self):
        e = self._engine(["4113 08:25", "BB 05/26"], ["4113 08:26", "BB 05/26"], ["4113 08:28", "BB 05/26"])
        r = e.read(self._frame())
        self.assertFalse(r["validation"]["accepted"])
        self.assertEqual(r["failure_reason"], "no_consensus")

    def test_stops_once_agreement_is_impossible(self):
        e = self._engine(["4113 08:2", "BB 05/26"], ["4113 08:2", "BB 05/26"], ["4113 08:26", "BB 05/26"])
        r = e.read(self._frame())
        self.assertEqual(e._paddle.calls, 2)
        self.assertEqual(r["failure_reason"], "format_mismatch")
        self.assertEqual(r["text"], "4113 08:2 BB 05/26")  # raw read kept for the HMI

    def test_engine_failure_is_still_an_engine_error(self):
        from backend.detection.ocr.paddle_worker import PaddleWorkerError
        e = self._engine(["4113 08:26", "BB 05/26"], PaddleWorkerError("worker died"))
        r = e.read(self._frame())
        self.assertIn("worker died", r["error"])

    def test_full_read_requires_the_winner_to_beat_every_other_code(self):
        good, bad = ["4113 08:26", "BB 05/26"], ["4113 08:25", "BB 05/26"]
        e = self._engine(good, good, bad, bad, variants=("dot_divide", "dot_subtract", "dot_divide_x3", "dot_subtract_x3"),
                         min_agree=2)
        e.consensus_early_stop = False
        r = e.read(self._frame())
        self.assertEqual(e._paddle.calls, 4, "every variant must be read")
        self.assertFalse(r["validation"]["accepted"], "2 vs 2 is a tie, not a pass")
        self.assertEqual(r["failure_reason"], "no_consensus")

    def test_full_read_accepts_a_clear_winner(self):
        good, bad = ["4113 08:26", "BB 05/26"], ["4113 08:25", "BB 05/26"]
        e = self._engine(good, bad, good, good, variants=("dot_divide", "dot_subtract", "dot_divide_x3", "dot_subtract_x3"),
                         min_agree=3)
        e.consensus_early_stop = False
        r = e.read(self._frame())
        self.assertEqual((r["validation"]["accepted"], r["text"]), (True, "4113 08:26 BB 05/26"))

    def test_single_read_mode(self):
        e = self._engine(["4113 08:26", "BB 05/26"], variants=("dot_divide",), min_agree=1)
        self.assertEqual(e.read(self._frame())["text"], "4113 08:26 BB 05/26")
        self.assertEqual(e._paddle.calls, 1)

    def test_tokens_are_put_in_reading_order_by_their_boxes(self):
        from backend.detection.ocr.paddle_ocr import PaddleOCR
        box = lambda x, y: [[x, y], [x + 40, y], [x + 40, y + 20], [x, y + 20]]
        cands = [{"text": "BB", "box": box(10, 52)}, {"text": "08:26", "box": box(80, 12)},
                 {"text": "05/26", "box": box(70, 50)}, {"text": "4113", "box": box(12, 10)}]
        ordered = [c["text"] for c in PaddleOCR._order_candidates(cands)]
        self.assertEqual(ordered, ["4113", "08:26", "BB", "05/26"])


class FusionTests(unittest.TestCase):
    def _r(self, text, conf=0.95, reason=None):
        return {"text": text, "min_token_confidence": conf, "reason": reason, "accepted": reason is None}

    def test_misread_digit_is_outvoted_per_position(self):
        reads = [self._r("4113 08:25 BB 05/26"), self._r("4113 08:26 BB 05/26"),
                 self._r("4113 08:26 BB 05/26"), self._r("4113 08:26 BB 05/26")]
        f = T().fuse(reads, min_reads=3, min_share=0.7)
        self.assertTrue(f["accepted"])
        self.assertEqual(f["text"], "4113 08:26 BB 05/26")
        self.assertAlmostEqual(f["min_share"], 0.75, places=2)

    def test_errors_at_different_positions_still_fuse_to_the_right_code(self):
        # No single read is fully right, but every position has a clear majority.
        reads = [self._r("4118 08:26 BB 05/26"), self._r("4113 08:25 BB 05/26"),
                 self._r("4113 08:26 BB 06/26"), self._r("4113 08:26 BB 05/26"),
                 self._r("4113 08:26 BB 05/26")]
        f = T().fuse(reads, min_reads=3, min_share=0.7)
        self.assertEqual((f["accepted"], f["text"]), (True, "4113 08:26 BB 05/26"))

    def test_genuine_disagreement_rejects(self):
        reads = [self._r("4113 08:25 BB 05/26"), self._r("4113 08:26 BB 05/26"), self._r("4113 08:28 BB 05/26")]
        f = T().fuse(reads, min_reads=3, min_share=0.7)
        self.assertFalse(f["accepted"])
        self.assertEqual(f["reason"], "no_consensus")

    def test_only_correctly_fielded_reads_vote(self):
        reads = [self._r("4113 08:26 BB 05/26"), self._r("4113 08:26 BB 05/26", conf=0.5, reason="low_confidence"),
                 {"text": None, "reason": "format_mismatch", "accepted": False, "min_token_confidence": None}]
        f = T().fuse(reads, min_reads=2, min_share=0.7)
        self.assertEqual((f["accepted"], f["n_reads"]), (True, 2))
        self.assertFalse(T().fuse(reads, min_reads=3, min_share=0.7)["accepted"])

    def test_confidence_weights_the_vote(self):
        reads = [self._r("4113 08:25 BB 05/26", conf=0.30), self._r("4113 08:26 BB 05/26", conf=0.99)]
        f = T().fuse(reads, min_reads=2, min_share=0.7)
        self.assertEqual((f["accepted"], f["text"]), (True, "4113 08:26 BB 05/26"))

    def test_fused_value_is_still_checked(self):
        reads = [self._r("4113 27:26 BB 05/26", reason="invalid_value")] * 3
        self.assertEqual(T().fuse(reads, min_reads=3, min_share=0.7)["reason"], "invalid_value")


class PositionVoteReadTests(ConsensusTests):
    VARIANTS = ("dot_divide", "dot_subtract", "classic", "region", "region_dot", "full_frame", "dot_divide_x3")

    def _vote_engine(self, *script, min_reads=3, min_share=0.7):
        e = self._engine(*script, variants=self.VARIANTS, min_agree=2)
        e.consensus_fusion, e.consensus_min_reads, e.consensus_min_share = "position_vote", min_reads, min_share
        return e

    def test_reads_every_variant_and_fuses(self):
        good, bad = ["4113 08:26", "BB 05/26"], ["4113 08:25", "BB 05/26"]
        e = self._vote_engine(bad, good, ["4113 08:2", "BB 05/26"], good, good, good, good)
        r = e.read(self._frame())
        self.assertEqual(e._paddle.calls, len(self.VARIANTS))
        self.assertTrue(r["validation"]["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(r["validation"]["fusion"]["n_reads"], 6)
        self.assertAlmostEqual(r["validation"]["fusion"]["min_share"], 5 / 6, places=2)

    def test_rejects_when_too_few_reads_fit(self):
        junk = ["4113 08:2", "BB 05/26"]
        e = self._vote_engine(["4113 08:26", "BB 05/26"], junk, junk, junk, junk, junk, ["4113 08:26", "BB 05/26"])
        r = e.read(self._frame())
        self.assertFalse(r["validation"]["accepted"])
        self.assertEqual(r["failure_reason"], "no_consensus")

    def test_frame_region_inputs(self):
        e = self._vote_engine(["x"])
        frame = np.zeros((480, 744, 3), np.uint8)
        located = {"bbox": [300, 250, 460, 330]}
        region = e._frame_variant_input(frame, located, "region")
        self.assertGreater(region.shape[1], 160 + 2 * 40)       # box widened on both sides
        self.assertEqual(e._frame_variant_input(frame, None, "region").shape[:2], e._downscale_roi(frame).shape[:2])


class VisionStatusTests(unittest.TestCase):
    def test_rejected_read_is_nok_with_its_reason(self):
        try:
            from backend.core import vision
        except Exception as exc:  # heavy ML deps not installed here
            self.skipTest(f"backend.core.vision not importable: {exc}")
        n = vision._normalize_result({"detections": [{"text": "4113 08:2", "confidence": 0.9}],
                                      "text": "4113 08:2", "mode": "ocread",
                                      "validation": {"accepted": False, "reason": "format_mismatch"},
                                      "failure_reason": "format_mismatch"})
        self.assertEqual((n["status"], n["failure_reason"]), ("NOK", "format_mismatch"))


if __name__ == "__main__":
    unittest.main()
