"""Regression tests for Vietnamese text normalisation (src/text_normalization.py).

Run with:
    pytest tests/test_text_normalization_vi.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from text_normalization import (  # noqa: E402
    decimal_to_vi,
    detect_language,
    int_to_vi,
    normalize_text,
    protect_spans,
    restore_spans,
)


# ---------------------------------------------------------------------------
# int_to_vi
# ---------------------------------------------------------------------------

def test_int_to_vi_zero():
    assert int_to_vi(0) == "không"


def test_int_to_vi_single_digits():
    assert int_to_vi(1) == "một"
    assert int_to_vi(5) == "năm"
    assert int_to_vi(9) == "chín"


def test_int_to_vi_teens():
    assert int_to_vi(10) == "mười"
    assert int_to_vi(11) == "mười một"
    assert int_to_vi(15) == "mười lăm"
    assert int_to_vi(19) == "mười chín"


def test_int_to_vi_tens():
    assert int_to_vi(20) == "hai mươi"
    assert int_to_vi(21) == "hai mươi mốt"
    assert int_to_vi(24) == "hai mươi tư"
    assert int_to_vi(25) == "hai mươi lăm"
    assert int_to_vi(30) == "ba mươi"
    assert int_to_vi(99) == "chín mươi chín"


def test_int_to_vi_hundreds():
    assert int_to_vi(100) == "một trăm"
    assert int_to_vi(101) == "một trăm lẻ một"
    assert int_to_vi(105) == "một trăm lẻ năm"
    assert int_to_vi(123) == "một trăm hai mươi ba"
    assert int_to_vi(200) == "hai trăm"


def test_int_to_vi_thousands():
    assert int_to_vi(1000) == "một nghìn"
    assert int_to_vi(1001) == "một nghìn không trăm lẻ một"
    assert int_to_vi(2026) == "hai nghìn không trăm hai mươi sáu"
    assert int_to_vi(10000) == "mười nghìn"
    assert int_to_vi(150000) == "một trăm năm mươi nghìn"


def test_int_to_vi_large():
    assert int_to_vi(1_000_000) == "một triệu"
    assert int_to_vi(1_500_000) == "một triệu năm trăm nghìn"


def test_int_to_vi_negative():
    assert int_to_vi(-5) == "âm năm"


# ---------------------------------------------------------------------------
# decimal_to_vi
# ---------------------------------------------------------------------------

def test_decimal_to_vi_simple():
    assert decimal_to_vi("1.5") == "một phẩy năm"


def test_decimal_to_vi_two_frac_digits():
    assert decimal_to_vi("3.14") == "ba phẩy một bốn"


def test_decimal_to_vi_comma_separator():
    assert decimal_to_vi("1,5") == "một phẩy năm"


# ---------------------------------------------------------------------------
# normalize_text — numbers
# ---------------------------------------------------------------------------

def test_normalize_simple_number():
    result = normalize_text("Số 123", lang="vi")
    assert "một trăm hai mươi ba" in result


def test_normalize_number_zero():
    result = normalize_text("Kết quả là 0.", lang="vi")
    assert "không" in result


# ---------------------------------------------------------------------------
# normalize_text — decimal + unit  (1.5kg → "một phẩy năm ki-lô-gam")
# ---------------------------------------------------------------------------

def test_normalize_decimal_with_unit_kg():
    result = normalize_text("1.5kg", lang="vi")
    assert "một phẩy năm ki-lô-gam" in result


def test_normalize_unit_km():
    result = normalize_text("100km", lang="vi")
    assert "một trăm ki-lô-mét" in result


def test_normalize_unit_percent():
    result = normalize_text("50%", lang="vi")
    assert "năm mươi phần trăm" in result


def test_normalize_unit_celsius():
    result = normalize_text("37°C", lang="vi")
    assert "ba mươi bảy độ C" in result


# ---------------------------------------------------------------------------
# normalize_text — date  (12/04/2026 → "ngày mười hai tháng tư năm …")
# ---------------------------------------------------------------------------

def test_normalize_date():
    result = normalize_text("12/04/2026", lang="vi")
    assert "ngày mười hai tháng tư năm hai nghìn không trăm hai mươi sáu" in result


def test_normalize_date_dash():
    result = normalize_text("01-01-2000", lang="vi")
    assert "ngày một tháng một năm" in result
    assert "hai nghìn" in result


# ---------------------------------------------------------------------------
# normalize_text — currency  (₫150,000 → "một trăm năm mươi nghìn đồng")
# ---------------------------------------------------------------------------

def test_normalize_currency_dong_prefix():
    result = normalize_text("₫150,000", lang="vi")
    assert "một trăm năm mươi nghìn đồng" in result


def test_normalize_currency_vnd_suffix():
    result = normalize_text("150,000 VND", lang="vi")
    assert "một trăm năm mươi nghìn đồng" in result


# ---------------------------------------------------------------------------
# normalize_text — phone number  (0901 234 567 → digit-by-digit Vietnamese)
# ---------------------------------------------------------------------------

def test_normalize_phone_number():
    result = normalize_text("0901 234 567", lang="vi")
    # First digit 0 → "không"
    assert "không" in result
    # Second digit 9 → "chín"
    assert "chín" in result
    # Raw digit sequence must not survive
    assert "901" not in result
    assert "234" not in result
    assert "567" not in result


# ---------------------------------------------------------------------------
# protect_spans / restore_spans
# ---------------------------------------------------------------------------

def test_protect_code_block():
    text = "Xem code ```x = 123``` nhé"
    masked, ph = protect_spans(text)
    assert "```" not in masked
    assert "x = 123" not in masked
    restored = restore_spans(masked, ph)
    assert "```x = 123```" in restored


def test_protect_inline_code():
    text = "Dùng hàm `AB123` để xử lý"
    masked, ph = protect_spans(text)
    assert "`AB123`" not in masked
    restored = restore_spans(masked, ph)
    assert "`AB123`" in restored


def test_protect_url():
    text = "Truy cập https://example.com để biết thêm"
    masked, ph = protect_spans(text)
    assert "https://example.com" not in masked
    restored = restore_spans(masked, ph)
    assert "https://example.com" in restored


def test_protect_email():
    text = "Liên hệ support@example.com nhé"
    masked, ph = protect_spans(text)
    assert "support@example.com" not in masked
    restored = restore_spans(masked, ph)
    assert "support@example.com" in restored


def test_protect_id():
    text = "Mã hồ sơ AB123 đã được xử lý"
    masked, ph = protect_spans(text)
    assert "AB123" not in masked
    restored = restore_spans(masked, ph)
    assert "AB123" in restored


def test_normalize_preserves_id_in_inline_code():
    """`AB123` inside backtick code must survive normalization unchanged."""
    text = "`ID AB123` và ```x=123```"
    result = normalize_text(text, lang="vi")
    assert "AB123" in result
    assert "x=123" in result


def test_normalize_preserves_url():
    """URLs must not be rewritten."""
    text = "Xem tại https://example.com/page?id=42"
    result = normalize_text(text, lang="vi")
    assert "https://example.com/page?id=42" in result


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

def test_detect_vi_by_diacritics():
    text = "Đây là văn bản tiếng Việt với các dấu thanh điệu."
    assert detect_language(text) == "vi"


def test_detect_en_by_stopwords():
    text = "This is a complete English sentence with common words and phrases."
    assert detect_language(text) == "en"


def test_detect_vi_by_function_words():
    text = "Sản phẩm được và các cơ quan"
    assert detect_language(text) == "vi"


def test_detect_defaults_to_vi_for_digits_only():
    # Pure numeric strings default to 'vi'
    assert detect_language("12345") == "vi"


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------

def test_normalize_abbrev_ai():
    result = normalize_text("AI đang phát triển", lang="vi")
    assert "Ây Ai" in result
    assert "AI" not in result


def test_normalize_abbrev_tphcm():
    result = normalize_text("Tại TP.HCM hôm nay", lang="vi")
    assert "thành phố Hồ Chí Minh" in result


# ---------------------------------------------------------------------------
# Non-Vietnamese passthrough
# ---------------------------------------------------------------------------

def test_normalize_en_passthrough():
    """English text should pass through with only light cleanup."""
    text = "Hello world  this is a test."
    result = normalize_text(text, lang="en")
    # Extra spaces collapsed
    assert "  " not in result
    # Numbers NOT expanded to Vietnamese words
    assert "one" not in result
    assert "hai" not in result
