from __future__ import annotations

from scripts.ocr_warwick_dulk_riddle_and_match_rae2 import parse_year_from_texts, repair_time_pair


def test_repair_time_pair_fixes_common_leading_digit_ocr_errors() -> None:
    start, end, status, cost = repair_time_pair("4112", "4207", obs_start="0940", obs_end="1230")
    assert (start, end) == ("1112", "1207")
    assert status == "repaired"
    assert cost == 2


def test_repair_time_pair_recovers_zero_from_six_or_nine_confusion() -> None:
    start, end, status, _ = repair_time_pair("0845", "6990", obs_start="0400", obs_end="1200")
    assert (start, end) == ("0845", "0900")
    assert status == "repaired"


def test_parse_year_from_texts_detects_mid_page_year_rollover() -> None:
    assert parse_year_from_texts(["DATE", "1972"]) == 1972
    assert parse_year_from_texts(["1973"]) == 1973
    assert parse_year_from_texts(["1", "4"]) is None
