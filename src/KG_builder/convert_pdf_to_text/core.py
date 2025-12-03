"""Core utilities for turning academic-profile PDFs into structured data."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

import fitz  # type: ignore

try:
    from .kb_profile_extractor import (  # pylint: disable=unused-import
        DEFAULT_MODEL as _EXTRACTOR_DEFAULT_MODEL,
        extract_academic_profile,
    )

    DEFAULT_MODEL = _EXTRACTOR_DEFAULT_MODEL
    _HAS_EXTRACTOR = True
except Exception:  # pragma: no cover - runtime convenience
    DEFAULT_MODEL = "gpt-4o-mini"
    extract_academic_profile = None  # type: ignore
    _HAS_EXTRACTOR = False


def get_pdf_pages(path: str) -> List[str]:
    """Extract plain text per page using PyMuPDF."""

    pages: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text("text") or page.get_text() or ""
            if not text:
                for block in page.get_text("blocks") or []:
                    if isinstance(block, Iterable):
                        block_text = block[4] if len(block) >= 5 else ""
                        text += block_text or ""
            pages.append(text)
    if all(len(p.strip()) == 0 for p in pages):
        raise RuntimeError(
            "No extractable text found in PDF (scanned images?). Consider OCR.")
    return pages


def _strip_headers_footers(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        raw = ln.strip()
        if re.fullmatch(r"(?:page|trang)\s*\d+\s*", raw, re.IGNORECASE): # remove page
            continue
        if re.fullmatch(r"\d{1,3}", raw): # remove number only line
            continue
        if re.fullmatch(r"[-–—]{3,}", raw): # remove -- only line
            continue
        out.append(ln)
    return out


def _unhyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _join_softwraps(lines: List[str]) -> List[str]:
    buf: List[str] = []
    cur = ""
    for ln in lines:
        s = ln.strip()
        if not s:
            continue # bỏ qua dòng trống
        # Nếu dòng bắt đầu là mục liệt kê hoặc kết thúc bằng dấu “:”
        # => coi như bắt đầu đoạn mới
        if re.match(r"^(\d+\.|[-*•])\s+", s) or s.endswith(":"):
            if cur:
                buf.append(cur.strip())
                cur = ""
            buf.append(s)
            continue
        # Nếu dòng trước đó không kết thúc bằng dấu câu (., !, ?, ;, :)
        # => ghép dòng hiện tại vào
        if cur and not re.search(r"[.!?;:]$", cur):
            cur += " " + s
        else:
            # Nếu dòng trước đó đã kết thúc -> lưu lại, bắt đầu dòng mới
            if cur:
                buf.append(cur.strip())
            cur = s
    # Sau vòng lặp, nếu còn sót dòng cuối -> thêm vào
    if cur:
        buf.append(cur.strip())
    return buf


def clean_page_text(
    raw: str,
    *,
    strip_headers: bool = True,
    join_lines: bool = True,
    dedup_consecutive: bool = True,
) -> str:
    txt = _unhyphenate(raw.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [ln.strip() for ln in txt.split("\n")]
    lines = [ln for ln in lines if ln.strip()]

    if strip_headers:
        lines = _strip_headers_footers(lines)

    if dedup_consecutive:
        deduped: List[str] = []
        prev: Optional[str] = None
        for ln in lines:
            if prev is None or ln != prev:
                deduped.append(ln)
            prev = ln
        lines = deduped

    if join_lines:
        lines = _join_softwraps(lines)

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in lines]
    return "\n".join(lines).strip()


def build_context_with_provenance(
    pages: List[str],
    *,
    strip_headers: bool = True,
    join_lines: bool = True,
    dedup_consecutive: bool = True,
) -> Dict[str, Any]:
    cleaned_pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(pages, start=1):
        cleaned = clean_page_text(
            page,
            strip_headers=strip_headers,
            join_lines=join_lines,
            dedup_consecutive=dedup_consecutive,
        )
        cleaned_pages.append({"page": idx, "text": cleaned})

    context_parts = []
    for chunk in cleaned_pages:
        if chunk["text"]:
            context_parts.append(f"[PAGE {chunk['page']}]\n{chunk['text']}")

    return {
        "context": "\n\n".join(context_parts),
        "pages": cleaned_pages,
    }


def extract_context_from_pdf(
    pdf_path: str,
    *,
    strip_headers: bool = True,
    join_lines: bool = True,
    dedup_consecutive: bool = True,
    max_chars: int = 0,
) -> str:
    pages = get_pdf_pages(pdf_path)
    built = build_context_with_provenance(
        pages,
        strip_headers=strip_headers,
        join_lines=join_lines,
        dedup_consecutive=dedup_consecutive,
    )
    context = built["context"]
    if max_chars and max_chars > 0:
        context = context[:max_chars]
    return context


def _run_extractor(
    context: str,
    *,
    model: str = DEFAULT_MODEL,
    validate_schema: bool = True,
) -> Dict[str, Any]:
    if not _HAS_EXTRACTOR:
        raise RuntimeError("kb_profile_extractor is not available. Install optional deps.")

    return extract_academic_profile(  # type: ignore[misc]
        context,
        model=model,
        validate_schema=validate_schema,
    )


def _patch_doc_id(profile: Dict[str, Any], doc_id: str) -> None:
    def _patch(item: Any) -> None:
        if isinstance(item, dict):
            prov = item.get("provenance")
            if isinstance(prov, dict) and prov.get("doc_id") is None:
                prov["doc_id"] = doc_id
            for value in item.values():
                _patch(value)
        elif isinstance(item, list):
            for value in item:
                _patch(value)

    _patch(profile)


def extract_profile_from_context(
    context: str,
    *,
    model: Optional[str] = None,
    validate_schema: bool = True,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload = context
    if doc_id:
        payload = f"DOC_ID={doc_id}\n" + payload

    profile = _run_extractor(
        payload,
        model=model or DEFAULT_MODEL,
        validate_schema=validate_schema,
    )

    if doc_id:
        _patch_doc_id(profile, doc_id)
    return profile


def extract_profile_from_pdf(
    pdf_path: str,
    *,
    model: Optional[str] = None,
    validate_schema: bool = True,
    strip_headers: bool = True,
    join_lines: bool = True,
    dedup_consecutive: bool = True,
    max_chars: int = 0,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    context = extract_context_from_pdf(
        pdf_path,
        strip_headers=strip_headers,
        join_lines=join_lines,
        dedup_consecutive=dedup_consecutive,
        max_chars=max_chars,
    )
    return extract_profile_from_context(
        context,
        model=model,
        validate_schema=validate_schema,
        doc_id=doc_id,
    )


def has_structured_extractor() -> bool:
    return _HAS_EXTRACTOR


__all__ = [
    "DEFAULT_MODEL",
    "build_context_with_provenance",
    "clean_page_text",
    "extract_context_from_pdf",
    "extract_profile_from_context",
    "extract_profile_from_pdf",
    "has_structured_extractor",
    "get_pdf_pages",
]