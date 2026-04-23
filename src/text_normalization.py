"""Vietnamese text normalization for TTS.

Handles:
- Integer / decimal / cardinal number → Vietnamese words
- Date (dd/mm/yyyy), time (hh:mm)
- Currency (₫ prefix, VND/VNĐ suffix)
- Measurement units (kg, km, %, °C, …)
- Vietnamese phone numbers (0xxx xxx xxxx)
- Common abbreviation expansion
- Heuristic language detection (vi / en)
- Protected-span masking so that code blocks, inline code, URLs, e-mails
  and alphanumeric IDs are never rewritten
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Vietnamese number-to-words
# ---------------------------------------------------------------------------

_UNITS_VI: List[str] = [
    "không", "một", "hai", "ba", "bốn",
    "năm", "sáu", "bảy", "tám", "chín",
]

_VI_MONTHS: List[str] = [
    "",          # index-0 unused
    "một", "hai", "ba", "tư", "năm", "sáu",
    "bảy", "tám", "chín", "mười", "mười một", "mười hai",
]


def _two_digits(n: int) -> str:
    """Convert 0-99 to Vietnamese words."""
    if n < 10:
        return _UNITS_VI[n]
    if n == 10:
        return "mười"
    if n < 20:
        ones = n - 10
        return "mười lăm" if ones == 5 else "mười " + _UNITS_VI[ones]
    tens, ones = divmod(n, 10)
    base = _UNITS_VI[tens] + " mươi"
    if ones == 0:
        return base
    if ones == 1:
        return base + " mốt"
    if ones == 4:
        return base + " tư"
    if ones == 5:
        return base + " lăm"
    return base + " " + _UNITS_VI[ones]


def _three_digits(n: int, is_tail: bool = False) -> str:
    """Convert 0-999 to Vietnamese words.

    Args:
        is_tail: When *True* (this chunk comes after a higher denomination
                 such as thousands / millions), a zero-hundreds value is
                 rendered as "không trăm …" rather than being skipped.
    """
    hundreds, remainder = divmod(n, 100)
    if hundreds == 0:
        if is_tail:
            if remainder < 10:
                return "không trăm lẻ " + _UNITS_VI[remainder]
            return "không trăm " + _two_digits(remainder)
        return _two_digits(remainder)
    base = _UNITS_VI[hundreds] + " trăm"
    if remainder == 0:
        return base
    if remainder < 10:
        return base + " lẻ " + _UNITS_VI[remainder]
    return base + " " + _two_digits(remainder)


def int_to_vi(n: int) -> str:
    """Convert an integer to Vietnamese words.

    Examples::

        int_to_vi(0)      == "không"
        int_to_vi(123)    == "một trăm hai mươi ba"
        int_to_vi(2026)   == "hai nghìn không trăm hai mươi sáu"
        int_to_vi(150000) == "một trăm năm mươi nghìn"
    """
    if n < 0:
        return "âm " + int_to_vi(-n)
    if n == 0:
        return "không"

    parts: List[str] = []

    billions, n = divmod(n, 1_000_000_000)
    if billions:
        parts.append(_three_digits(billions) + " tỷ")

    millions, n = divmod(n, 1_000_000)
    if millions:
        parts.append(_three_digits(millions, is_tail=bool(parts)) + " triệu")

    thousands, n = divmod(n, 1_000)
    if thousands:
        parts.append(_three_digits(thousands, is_tail=bool(parts)) + " nghìn")

    if n > 0:
        parts.append(_three_digits(n, is_tail=bool(parts)))

    return " ".join(parts)


def decimal_to_vi(s: str) -> str:
    """Convert a decimal-number string to Vietnamese words.

    The fractional digits are read one by one after "phẩy".

    Examples::

        decimal_to_vi("1.5")  == "một phẩy năm"
        decimal_to_vi("3.14") == "ba phẩy một bốn"
    """
    s = s.replace(",", ".")
    if "." not in s:
        return int_to_vi(int(s))
    int_str, frac_str = s.split(".", 1)
    int_part = int(int_str) if int_str else 0
    result = int_to_vi(int_part)
    if frac_str:
        frac_words = " ".join(
            _UNITS_VI[int(d)] for d in frac_str if d.isdigit()
        )
        result += " phẩy " + frac_words
    return result


# ---------------------------------------------------------------------------
# Abbreviation table
# ---------------------------------------------------------------------------

_ABBREVS: Dict[str, str] = {
    "TP.HCM":  "thành phố Hồ Chí Minh",
    "TP HCM":  "thành phố Hồ Chí Minh",
    "TPHCM":   "thành phố Hồ Chí Minh",
    "A.I":     "Ây Ai",
    "AI":      "Ây Ai",
    "VN":      "Việt Nam",
    "USD":     "đô-la Mỹ",
    "EUR":     "ơ-rô",
    "VND":     "đồng",
    "VNĐ":     "đồng",
    "GHz":     "giga héc",
    "MHz":     "mê-ga héc",
    "kHz":     "ki-lô héc",
    "TB":      "ti-ra-bai",
    "GB":      "ghi-ga-bai",
    "MB":      "mê-ga-bai",
    "KB":      "ki-lô-bai",
    "CNTT":    "Công nghệ Thông tin",
}

# Pre-compiled patterns sorted by key length (longest first) to avoid
# shorter abbreviations clobbering longer ones.
_ABBREV_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(abbr) + r"\b"), expansion)
    for abbr, expansion in sorted(_ABBREVS.items(), key=lambda kv: -len(kv[0]))
]


# ---------------------------------------------------------------------------
# Unit table
# ---------------------------------------------------------------------------

_UNIT_MAP: Dict[str, str] = {
    "km/h":  "ki-lô-mét trên giờ",
    "km":    "ki-lô-mét",
    "m²":    "mét vuông",
    "m2":    "mét vuông",
    "ha":    "héc-ta",
    "cm":    "xen-ti-mét",
    "mm":    "mi-li-mét",
    "mg":    "mi-li-gam",
    "ml":    "mi-li-lít",
    "kg":    "ki-lô-gam",
    "°c":    "độ C",
    "°f":    "độ F",
    "g":     "gam",
    "l":     "lít",
    "m":     "mét",
    "%":     "phần trăm",
}


# ---------------------------------------------------------------------------
# Regex patterns used by _normalize_vi_rules (applied in order)
# ---------------------------------------------------------------------------

# Date: dd/mm/yyyy or dd-mm-yyyy
_DATE_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b")

# Time: hh:mm or hh:mm:ss
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")

# Currency with ₫ prefix: ₫150,000
_CURRENCY_PREFIX_RE = re.compile(r"₫\s*([\d,\.]+)")

# Currency with VND/VNĐ suffix: 150,000 VND
_CURRENCY_SUFFIX_RE = re.compile(r"([\d,\.]+)\s*(?:VNĐ|VND)\b", re.IGNORECASE)

# Number followed by a measurement unit.
# Order of alternatives matters: longer units must come first.
_NUM_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(km/h|km|m²|m2|ha|cm|mm|mg|ml|kg|°[CF]|[glm%])(?!\w)",
    re.IGNORECASE,
)

# Vietnamese mobile / landline phone: 0xxx xxx xxxx (total 10 digits)
_PHONE_RE = re.compile(r"\b(0\d{2,3})[\s\-]?(\d{3})[\s\-]?(\d{3,4})\b")

# Decimal: digits SEPARATOR digits  where fractional part is 1 or 2 digits
# (3-digit fractions are treated as thousands separators elsewhere)
_DECIMAL_RE = re.compile(r"\b(\d+)[.,](\d{1,2})\b")

# Large integer with thousands separators: 1,000,000 or 1.000.000
_LARGE_INT_RE = re.compile(r"\b\d{1,3}(?:[,.]\d{3})+\b")

# Plain integer (fallback)
_PLAIN_INT_RE = re.compile(r"\b\d+\b")


def _num_unit_replace(m: re.Match) -> str:
    """Callback for _NUM_UNIT_RE substitutions."""
    num_s = m.group(1)
    unit = m.group(2)
    unit_word = _UNIT_MAP.get(unit.lower(), unit)

    sep_search = re.search(r"[.,]", num_s)
    if sep_search:
        frac = num_s[sep_search.end():]
        if len(frac) == 3:
            # Thousands separator → integer
            clean = re.sub(r"[,.]", "", num_s)
            num_words = int_to_vi(int(clean))
        else:
            # Decimal point / comma
            num_words = decimal_to_vi(num_s)
    else:
        num_words = int_to_vi(int(num_s))

    return num_words + " " + unit_word


def _normalize_vi_rules(text: str) -> str:
    """Apply all Vietnamese number / date / unit normalisation rules."""

    # 1. Dates
    def _date_repl(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        return (
            f"ngày {_two_digits(d)} "
            f"tháng {_VI_MONTHS[mo]} "
            f"năm {int_to_vi(y)}"
        )

    text = _DATE_RE.sub(_date_repl, text)

    # 2. Times
    def _time_repl(m: re.Match) -> str:
        h, mins = int(m.group(1)), int(m.group(2))
        result = f"{_two_digits(h)} giờ {_two_digits(mins)} phút"
        if m.group(3):
            result += f" {_two_digits(int(m.group(3)))} giây"
        return result

    text = _TIME_RE.sub(_time_repl, text)

    # 3. Currency prefix ₫
    def _currency_prefix_repl(m: re.Match) -> str:
        raw = re.sub(r"[,.]", "", m.group(1))
        return int_to_vi(int(raw)) + " đồng"

    text = _CURRENCY_PREFIX_RE.sub(_currency_prefix_repl, text)

    # 4. Currency suffix VND / VNĐ
    def _currency_suffix_repl(m: re.Match) -> str:
        raw = re.sub(r"[,.]", "", m.group(1))
        return int_to_vi(int(raw)) + " đồng"

    text = _CURRENCY_SUFFIX_RE.sub(_currency_suffix_repl, text)

    # 5. Number + unit
    text = _NUM_UNIT_RE.sub(_num_unit_replace, text)

    # 6. Phone numbers
    def _phone_repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return " ".join(_UNITS_VI[int(d)] for d in digits)

    text = _PHONE_RE.sub(_phone_repl, text)

    # 7. Decimal numbers (standalone, 1-2 fractional digits)
    def _decimal_repl(m: re.Match) -> str:
        return decimal_to_vi(m.group(1) + "." + m.group(2))

    text = _DECIMAL_RE.sub(_decimal_repl, text)

    # 8. Large integers with thousands separators
    def _large_int_repl(m: re.Match) -> str:
        raw = re.sub(r"[,.]", "", m.group(0))
        return int_to_vi(int(raw))

    text = _LARGE_INT_RE.sub(_large_int_repl, text)

    # 9. Remaining plain integers
    def _int_repl(m: re.Match) -> str:
        return int_to_vi(int(m.group(0)))

    text = _PLAIN_INT_RE.sub(_int_repl, text)

    return text


# ---------------------------------------------------------------------------
# Protected-span masking
# ---------------------------------------------------------------------------

# Patterns that must not be rewritten, evaluated in this order.
_PROTECT_PATS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"```[\s\S]*?```"),                                    "CODEBLOCK"),
    (re.compile(r"`[^`\n]+`"),                                         "INLINECODE"),
    (re.compile(r"https?://\S+"),                                      "URL"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "EMAIL"),
    (re.compile(r"\b[A-Z]{2,}\d+[A-Z0-9]*\b"),                        "ID"),
]


def protect_spans(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace sensitive spans with unique placeholders.

    Returns:
        (masked_text, placeholder_map)  — pass both to :func:`restore_spans`.
    """
    placeholders: Dict[str, str] = {}
    counter = [0]

    def _factory(tag: str):
        def _sub(m: re.Match) -> str:
            key = f"__PROTECTED_{tag}_{counter[0]}__"
            counter[0] += 1
            placeholders[key] = m.group(0)
            return key
        return _sub

    for pat, tag in _PROTECT_PATS:
        text = pat.sub(_factory(tag), text)

    return text, placeholders


def restore_spans(text: str, placeholders: Dict[str, str]) -> str:
    """Restore placeholders produced by :func:`protect_spans`."""
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------

def _expand_abbrevs(text: str) -> str:
    """Expand well-known Vietnamese abbreviations using word boundaries."""
    for pattern, expansion in _ABBREV_PATTERNS:
        text = pattern.sub(expansion, text)
    return text


# ---------------------------------------------------------------------------
# Light punctuation / whitespace cleanup
# ---------------------------------------------------------------------------

def _light_cleanup(text: str) -> str:
    text = text.replace("\t", " ").replace("\n", " ")
    text = re.sub(r"  +", " ", text)
    text = (
        text.replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace('"', "")
        .replace("'", "")
    )
    return text.strip()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_VI_DIACRITICS = frozenset(
    "àáâãèéêìíòóôõùúýăđơư"
    "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
    "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
)

_VI_FUNCTION_WORDS = frozenset([
    "và", "của", "được", "trong", "là", "có", "không",
    "này", "đó", "một", "các", "để", "với", "cho",
    "từ", "về", "theo", "khi", "nếu", "nhưng", "hay",
])

_EN_STOPWORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "this", "that",
    "these", "those", "it", "he", "she", "they", "we", "you",
    "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as",
])


def detect_language(text: str) -> str:
    """Heuristically detect whether *text* is Vietnamese (``'vi'``) or
    English (``'en'``).

    The function uses two signals:

    1. **Diacritic ratio** — Vietnamese-specific diacritic characters
       relative to total alphabetic characters.  A ratio ≥ 4 % strongly
       suggests Vietnamese.
    2. **Function-word comparison** — count of known Vietnamese function
       words versus English stopwords in the token list.

    Defaults to ``'vi'`` when no strong English signal is found (appropriate
    for a Vietnamese-centric TTS system).
    """
    vi_count = sum(1 for c in text if c in _VI_DIACRITICS)
    alpha_count = sum(1 for c in text if c.isalpha())

    if alpha_count == 0:
        return "vi"

    if vi_count / alpha_count >= 0.04:
        return "vi"

    words = [
        w.lower().strip(".,;:!?\"'()[]{}") for w in text.split()
    ]
    words = [w for w in words if w]

    if not words:
        return "vi"

    vi_score = sum(1 for w in words if w in _VI_FUNCTION_WORDS)
    en_score = sum(1 for w in words if w in _EN_STOPWORDS)

    if vi_score > en_score:
        return "vi"
    if en_score > vi_score:
        return "en"

    # Default: assume Vietnamese
    return "vi"


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def normalize_text(text: str, lang: str = "vi") -> str:
    """Normalise *text* for TTS synthesis.

    Pipeline (Vietnamese):
      1. Protect spans (code blocks, inline code, URLs, e-mails, IDs).
      2. Expand common abbreviations.
      3. Apply number / date / unit rules.
      4. Light punctuation & whitespace cleanup.
      5. Restore protected spans.

    For non-Vietnamese input only step 4 (light cleanup) is applied.

    Args:
        text: Input text string.
        lang: Language code — ``'vi'`` or ``'en'``.

    Returns:
        Normalised text ready for the TTS model.
    """
    if not text or not text.strip():
        return text

    if lang != "vi":
        return _light_cleanup(text)

    masked, placeholders = protect_spans(text)
    masked = _expand_abbrevs(masked)
    masked = _normalize_vi_rules(masked)
    masked = _light_cleanup(masked)
    result = restore_spans(masked, placeholders)
    return result
