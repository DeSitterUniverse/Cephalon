from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
import statistics
from typing import Any

import pdfplumber


PARSER_VERSION = "cephalon-pdf-layout-2026-07"
TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 4,
}


@dataclass
class DocumentBlock:
    text: str
    page_number: int
    block_type: str = "paragraph"
    heading_path: list[str] = field(default_factory=list)
    bounding_box: tuple[float, float, float, float] | None = None
    block_index: int = 0
    heading_level: int | None = None
    font_size: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def section_heading(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None


@dataclass
class ParsedPdf:
    text: str
    blocks: list[DocumentBlock]
    page_count: int
    warnings: list[str] = field(default_factory=list)
    parser_version: str = PARSER_VERSION


@dataclass
class _Line:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    font_size: float
    bold: bool
    column: int = 0


def parse_pdf(path: str) -> ParsedPdf:
    warnings: list[str] = []
    blocks: list[DocumentBlock] = []
    with pdfplumber.open(path, unicode_norm="NFKC") as pdf:
        page_count = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                page_blocks = _parse_page(page, page_number, warnings)
            except Exception as exc:
                warnings.append(f"Page {page_number}: layout parsing failed ({exc}).")
                fallback = (page.extract_text(layout=True) or page.extract_text() or "").strip()
                page_blocks = [
                    DocumentBlock(
                        text=fallback,
                        page_number=page_number,
                        bounding_box=(0.0, 0.0, float(page.width), float(page.height)),
                        provenance={"fallback": True},
                    )
                ] if fallback else []
            if not page_blocks:
                warnings.append(
                    f"Page {page_number}: no native text was found; OCR is disabled."
                )
            blocks.extend(page_blocks)
            page.close()

    _mark_repeated_marginalia(blocks, page_count)
    _assign_heading_paths(blocks)
    searchable = [
        block
        for block in blocks
        if block.block_type not in {"header", "footer"} and block.text.strip()
    ]
    for index, block in enumerate(searchable):
        block.block_index = index
    text = "\n\n".join(block.text.strip() for block in searchable)
    return ParsedPdf(text=text, blocks=searchable, page_count=page_count, warnings=warnings)


def _parse_page(page, page_number: int, warnings: list[str]) -> list[DocumentBlock]:
    table_blocks, table_boxes = _extract_tables(page, page_number, warnings)
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
        extra_attrs=["fontname", "size"],
    )
    words = [
        word
        for word in words
        if str(word.get("text") or "").strip()
        and not any(_inside_bbox(word, bbox) for bbox in table_boxes)
    ]
    lines = _words_to_lines(words)
    body_size = _body_font_size(lines)
    ordered_lines = _reading_order(lines, float(page.width))
    text_blocks = _lines_to_blocks(
        ordered_lines,
        page_number=page_number,
        page_width=float(page.width),
        page_height=float(page.height),
        body_size=body_size,
    )
    return _reading_order_blocks(text_blocks + table_blocks, float(page.width))


def _extract_tables(page, page_number: int, warnings: list[str]) -> tuple[list[DocumentBlock], list[tuple[float, float, float, float]]]:
    blocks: list[DocumentBlock] = []
    boxes: list[tuple[float, float, float, float]] = []
    try:
        tables = page.find_tables(TABLE_SETTINGS)
    except Exception as exc:
        warnings.append(f"Page {page_number}: table detection failed ({exc}).")
        return blocks, boxes

    for table_index, table in enumerate(tables):
        rows = table.extract() or []
        normalized_rows = [
            [_clean_cell(cell) for cell in row]
            for row in rows
            if row and any(_clean_cell(cell) for cell in row)
        ]
        if len(normalized_rows) < 2:
            continue
        width = max(len(row) for row in normalized_rows)
        normalized_rows = [row + [""] * (width - len(row)) for row in normalized_rows]
        text = "\n".join(" | ".join(row).rstrip() for row in normalized_rows).strip()
        if not text:
            continue
        bbox = tuple(round(float(value), 3) for value in table.bbox)
        boxes.append(bbox)
        blocks.append(
            DocumentBlock(
                text=text,
                page_number=page_number,
                block_type="table",
                bounding_box=bbox,
                provenance={
                    "table_index": table_index,
                    "row_count": len(normalized_rows),
                    "column_count": width,
                    "table_settings": "lines",
                },
            )
        )
    return blocks, boxes


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _inside_bbox(word: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    center_x = (float(word["x0"]) + float(word["x1"])) / 2
    center_y = (float(word["top"]) + float(word["bottom"])) / 2
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _words_to_lines(words: list[dict[str, Any]], tolerance: float = 3.0) -> list[_Line]:
    rows: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        target = next(
            (
                row
                for row in reversed(rows[-4:])
                if abs(statistics.median(float(item["top"]) for item in row) - top) <= tolerance
            ),
            None,
        )
        if target is None:
            target = []
            rows.append(target)
        target.append(word)

    lines: list[_Line] = []
    for row in rows:
        row.sort(key=lambda item: float(item["x0"]))
        text = _join_words([str(item["text"]) for item in row])
        if not text:
            continue
        sizes = [float(item.get("size") or 0) for item in row if float(item.get("size") or 0) > 0]
        fonts = " ".join(str(item.get("fontname") or "") for item in row).lower()
        lines.append(
            _Line(
                text=text,
                x0=min(float(item["x0"]) for item in row),
                top=min(float(item["top"]) for item in row),
                x1=max(float(item["x1"]) for item in row),
                bottom=max(float(item["bottom"]) for item in row),
                font_size=statistics.median(sizes) if sizes else 0.0,
                bold="bold" in fonts or "black" in fonts or "semibold" in fonts,
            )
        )
    return lines


def _join_words(words: list[str]) -> str:
    text = " ".join(word.strip() for word in words if word.strip())
    return re.sub(r"\s+([,.;:!?%)\]])", r"\1", text).strip()


def _body_font_size(lines: list[_Line]) -> float:
    sizes = [
        round(line.font_size, 1)
        for line in lines
        if line.font_size > 0 and len(line.text) >= 20
    ]
    return Counter(sizes).most_common(1)[0][0] if sizes else 10.0


def _reading_order(lines: list[_Line], page_width: float) -> list[_Line]:
    if not lines:
        return []
    midpoint = page_width / 2
    spanning = lambda line: line.x0 < midpoint * 0.55 and line.x1 > midpoint * 1.45
    ordered: list[_Line] = []
    segment: list[_Line] = []
    for line in sorted(lines, key=lambda item: (item.top, item.x0)):
        if spanning(line):
            ordered.extend(_order_column_segment(segment, midpoint))
            segment = []
            line.column = 0
            ordered.append(line)
        else:
            segment.append(line)
    ordered.extend(_order_column_segment(segment, midpoint))
    return ordered


def _order_column_segment(lines: list[_Line], midpoint: float) -> list[_Line]:
    left = [line for line in lines if line.x0 < midpoint]
    right = [line for line in lines if line.x0 >= midpoint]
    if len(left) >= 2 and len(right) >= 2:
        for line in left:
            line.column = 1
        for line in right:
            line.column = 2
        return sorted(left, key=lambda item: (item.top, item.x0)) + sorted(
            right, key=lambda item: (item.top, item.x0)
        )
    for line in lines:
        line.column = 0
    return sorted(lines, key=lambda item: (item.top, item.x0))


def _lines_to_blocks(
    lines: list[_Line],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    body_size: float,
) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    current: list[_Line] = []
    current_type = "paragraph"

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = _join_lines([line.text for line in current])
        bbox = (
            min(line.x0 for line in current),
            min(line.top for line in current),
            max(line.x1 for line in current),
            max(line.bottom for line in current),
        )
        font_size = statistics.median(
            [line.font_size for line in current if line.font_size > 0]
        ) if any(line.font_size > 0 for line in current) else None
        heading_level = _heading_level(current[0], body_size) if current_type in {"title", "heading"} else None
        blocks.append(
            DocumentBlock(
                text=text,
                page_number=page_number,
                block_type=current_type,
                bounding_box=tuple(round(value, 3) for value in bbox),
                heading_level=heading_level,
                font_size=font_size,
                provenance={"column": current[0].column, "line_count": len(current)},
            )
        )
        current = []

    previous: _Line | None = None
    for line in lines:
        block_type = _line_type(
            line,
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            body_size=body_size,
        )
        gap = line.top - previous.bottom if previous is not None else 0
        incompatible = (
            current
            and (
                block_type != current_type
                or line.column != current[-1].column
                or gap > max(7.0, body_size * 0.9)
                or block_type in {"title", "heading", "list_item", "caption", "footnote"}
            )
        )
        if incompatible:
            flush()
        if not current:
            current_type = block_type
        current.append(line)
        if block_type in {"title", "heading", "list_item", "caption", "footnote"}:
            flush()
        previous = line
    flush()
    return blocks


def _line_type(
    line: _Line,
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    body_size: float,
) -> str:
    clean = line.text.strip()
    if re.match(r"^(?:figure|fig\.|table)\s+\d+[.:]?", clean, flags=re.IGNORECASE):
        return "caption"
    if re.match(r"^(?:[-*•‣▪]|\(?\d+[.)]|[A-Za-z][.)])\s+", clean):
        return "list_item"
    if line.top >= page_height * 0.88 and line.font_size and line.font_size < body_size * 0.9:
        return "footnote"
    level = _heading_level(line, body_size)
    if level is not None:
        if page_number == 1 and level == 1 and line.x1 - line.x0 >= page_width * 0.35:
            return "title"
        return "heading"
    return "paragraph"


def _heading_level(line: _Line, body_size: float) -> int | None:
    clean = line.text.strip()
    if len(clean) > 180 or clean.endswith((".", ";", ",")):
        return None
    numbered = re.match(r"^(\d+(?:\.\d+){0,3})\s+\S", clean)
    if numbered:
        return min(4, numbered.group(1).count(".") + 1)
    ratio = line.font_size / body_size if body_size and line.font_size else 1.0
    heading_shape = (
        clean.isupper()
        or clean.istitle()
        or bool(re.match(r"^(?:abstract|introduction|methods?|results?|discussion|conclusions?|references)$", clean, re.IGNORECASE))
    )
    if ratio >= 1.45:
        return 1
    if ratio >= 1.22 and (line.bold or heading_shape):
        return 2
    if ratio >= 1.08 and line.bold and heading_shape:
        return 3
    return None


def _join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        if text.endswith("-") and clean[:1].islower():
            text = text[:-1] + clean
        else:
            text = f"{text} {clean}".strip()
    return text


def _reading_order_blocks(blocks: list[DocumentBlock], page_width: float) -> list[DocumentBlock]:
    midpoint = page_width / 2
    ordered: list[DocumentBlock] = []
    segment: list[DocumentBlock] = []

    def flush_segment() -> None:
        nonlocal segment
        left = [
            block
            for block in segment
            if not block.bounding_box or block.bounding_box[0] < midpoint
        ]
        right = [
            block
            for block in segment
            if block.bounding_box and block.bounding_box[0] >= midpoint
        ]
        key = lambda block: (
            block.bounding_box[1] if block.bounding_box else 0,
            block.bounding_box[0] if block.bounding_box else 0,
        )
        if left and right:
            ordered.extend(sorted(left, key=key))
            ordered.extend(sorted(right, key=key))
        else:
            ordered.extend(sorted(segment, key=key))
        segment = []

    for block in sorted(
        blocks,
        key=lambda item: (
            item.bounding_box[1] if item.bounding_box else 0,
            item.bounding_box[0] if item.bounding_box else 0,
        ),
    ):
        spans_columns = (
            block.block_type == "table"
            or (
                block.bounding_box is not None
                and block.bounding_box[0] < midpoint * 0.55
                and block.bounding_box[2] > midpoint * 1.45
            )
        )
        if spans_columns:
            flush_segment()
            ordered.append(block)
        else:
            segment.append(block)
    flush_segment()
    return ordered


def _mark_repeated_marginalia(blocks: list[DocumentBlock], page_count: int) -> None:
    if page_count < 2:
        return
    candidates: dict[str, list[DocumentBlock]] = {}
    page_bottoms: dict[int, float] = {}
    for block in blocks:
        if block.bounding_box:
            page_bottoms[block.page_number] = max(
                page_bottoms.get(block.page_number, 0.0),
                block.bounding_box[3],
            )
    for block in blocks:
        if not block.bounding_box or len(block.text) > 180:
            continue
        estimated_height = max(page_bottoms.get(block.page_number, 0.0), block.bounding_box[3], 1.0)
        top_ratio = block.bounding_box[1] / estimated_height
        bottom_ratio = block.bounding_box[3] / estimated_height
        if top_ratio <= 0.12 or bottom_ratio >= 0.9:
            key = re.sub(r"\d+", "#", re.sub(r"\s+", " ", block.text.lower())).strip()
            if key:
                candidates.setdefault(key, []).append(block)
    threshold = max(2, (page_count + 1) // 2)
    for repeated in candidates.values():
        if len({block.page_number for block in repeated}) < threshold:
            continue
        for block in repeated:
            if block.bounding_box and block.bounding_box[1] <= page_bottoms.get(block.page_number, 1.0) * 0.12:
                block.block_type = "header"
            else:
                block.block_type = "footer"


def _assign_heading_paths(blocks: list[DocumentBlock]) -> None:
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.block_type in {"title", "heading"}:
            level = block.heading_level or (1 if block.block_type == "title" else 2)
            stack = [(existing_level, text) for existing_level, text in stack if existing_level < level]
            stack.append((level, block.text.strip()))
            block.heading_path = [text for _, text in stack]
        elif block.block_type not in {"header", "footer"}:
            block.heading_path = [text for _, text in stack]
