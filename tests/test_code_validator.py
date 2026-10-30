"""
Tests for OCRead read validation (code_validator.py). The rule being
protected: the system may reject a read, but must never invent or "repair"
information into a pass.

Run with:  python -m unittest tests.test_code_validator
"""
import datetime
import unittest

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
        # be 08:26 (the PLC's comparison catches that), but it must not alter it.
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


class ExpiryYearTests(unittest.TestCase):
    """max_expiry_years: the YY of a BB / EXP date within this year .. this
    year + max_expiry_years. "BB 05/38" read for 05/28 fit the value
    pattern at digit confidence 0.999."""
    ANY = r"^\d{4} \d{2}:\d{2}( (BB|EXP|PROD) \d{2}/\d{2})?$"

    def _t(self, label, year=2026, max_years=5):
        return CodeTemplate(["#### ##:##", f"{label} ##/##"], value_pattern=self.ANY,
                            max_expiry_years=max_years, today=lambda: datetime.date(year, 10, 29))

    def _reason(self, t, line2):
        return t.validate(toks("6288 15:21", line2))["reason"]

    def test_year_inside_the_window_is_accepted(self):
        for yy in ("26", "28", "31"):
            self.assertIsNone(self._reason(self._t("BB"), f"BB 05/{yy}"), yy)
            self.assertIsNone(self._reason(self._t("EXP"), f"EXP 05/{yy}"), yy)

    def test_year_outside_the_window_is_invalid_value(self):
        for yy in ("25", "32", "38", "00"):
            self.assertEqual(self._reason(self._t("BB"), f"BB 05/{yy}"), "invalid_value", yy)
            self.assertEqual(self._reason(self._t("EXP"), f"EXP 05/{yy}"), "invalid_value", yy)

    def test_window_wraps_at_the_century(self):
        t = self._t("BB", year=2098)
        self.assertIsNone(self._reason(t, "BB 05/01"))
        self.assertEqual(self._reason(t, "BB 05/97"), "invalid_value")

    def test_prod_date_is_not_checked(self):
        self.assertIsNone(self._reason(self._t("PROD"), "PROD 10/20"))

    def test_off_by_default_in_the_constructor(self):
        t = CodeTemplate(["#### ##:##", "BB ##/##"], value_pattern=self.ANY)
        self.assertIsNone(t.validate(toks("6288 15:21", "BB 05/38"))["reason"])

    def test_config_default_is_five_years(self):
        self.assertEqual(CodeTemplate.from_config({"template": ["#### ##:##", "BB ##/##"]}).max_expiry_years, 5)


class VisionStatusTests(unittest.TestCase):
    def test_rejected_read_is_nok_with_its_reason(self):
        from backend.core import vision
        n = vision._normalize_result({"detections": [{"text": "4113 08:2", "confidence": 0.9}],
                                      "text": "4113 08:2", "mode": "ocread",
                                      "validation": {"accepted": False, "reason": "format_mismatch"},
                                      "failure_reason": "format_mismatch"})
        self.assertEqual((n["status"], n["failure_reason"]), ("NOK", "format_mismatch"))


if __name__ == "__main__":
    unittest.main()
