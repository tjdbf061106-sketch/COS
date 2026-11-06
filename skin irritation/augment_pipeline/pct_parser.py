# -*- coding: utf-8 -*-
"""함량_원문 텍스트 -> 숫자(%) 파서. 명시적으로 불확실하다고 표시된 텍스트(Inconsistency/OCR-garbled/
International Units 등 비율이 아닌 단위)는 의도적으로 실패(None) 처리 — 억지로 추측하지 않음."""
import re

import pandas as pd

NUMBER = r"\d+(?:\.\d+)?"
OCR_RE = re.compile(rf"\(OCR for\s*({NUMBER})\s*%?\)")
RANGE_RE = re.compile(rf"^({NUMBER})[-~]({NUMBER})%?$")
NUM_RE = re.compile(rf"^({NUMBER})%?$")

UNRELIABLE_MARKERS = ["inconsistency", "garbled", "international units", "per mg of product"]
SUFFIXES = [
    " percent by weight", " percent", "% by weight", "% w/w", " w/w",
    "% a.i.", " a.i.", "%",
]


def parse_pct_text(text):
    if pd.isna(text):
        return None
    text = str(text).strip()
    low = text.lower()
    if any(marker in low for marker in UNRELIABLE_MARKERS):
        return None

    m = OCR_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None

    cleaned = text.replace(" ", "").replace(",", "").strip("*'")
    for suf in SUFFIXES:
        suf_clean = suf.replace(" ", "")
        if cleaned.lower().endswith(suf_clean.lower()):
            cleaned = cleaned[: -len(suf_clean)]
            break
    cleaned = cleaned.lstrip(".")  # ".0.5" 같은 OCR 중복 점 보정
    if cleaned.count(".") > 1:
        # "6.Uw/w" 류 혼합 OCR 오류는 신뢰 불가 -> 포기
        return None

    try:
        m = RANGE_RE.match(cleaned)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2
        m = NUM_RE.match(cleaned + ("%" if not cleaned.endswith("%") else ""))
        if m:
            return float(m.group(1))
        if re.fullmatch(NUMBER, cleaned):
            return float(cleaned)
    except ValueError:
        return None
    return None
