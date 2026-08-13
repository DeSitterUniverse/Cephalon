from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import os
import re
import statistics
from typing import Any

import pdfplumber
from pypdf import PdfReader

from .table_models import StructuredTable, build_table


PARSER_VERSION = "cephalon-pdf-layout-tables-2026-08"
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
    element_id: str | None = None
    asset_ids: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    structured_table: StructuredTable | None = None

    @property
    def section_heading(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None


@dataclass
class PdfAsset:
    asset_id: str
    page_number: int
    bounding_box: tuple[float, float, float, float] | None
    data: bytes = field(repr=False)
    extension: str = ".bin"
    mime_type: str = "application/octet-stream"
    sha256: str = ""
    caption: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class ParsedPdf:
    text: str
    blocks: list[DocumentBlock]
    page_count: int
    assets: list[PdfAsset] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parser_version: str = PARSER_VERSION
    tables: list[StructuredTable] = field(default_factory=list)


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
    assets_by_page, asset_warnings = _extract_embedded_images(path)
    warnings.extend(asset_warnings)
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
            _position_page_assets(assets_by_page.get(page_number, []), page)
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
        block.element_id = _element_id(block)
        block.provenance["element_id"] = block.element_id
    assets = [asset for page_assets in assets_by_page.values() for asset in page_assets]
    _associate_asset_captions(searchable, assets)
    text = "\n\n".join(block.text.strip() for block in searchable)
    return ParsedPdf(
        text=text,
        blocks=searchable,
        page_count=page_count,
        assets=assets,
        warnings=warnings,
        tables=[block.structured_table for block in searchable if block.structured_table is not None],
    )


def _extract_embedded_images(path: str) -> tuple[dict[int, list[PdfAsset]], list[str]]:
    assets_by_page: dict[int, list[PdfAsset]] = {}
    warnings: list[str] = []
    try:
        reader = PdfReader(path)
    except Exception as exc:
        return assets_by_page, [f"Embedded image extraction failed ({exc})."]

    for page_number, page in enumerate(reader.pages, start=1):
        page_assets: list[PdfAsset] = []
        try:
            images = list(page.images)
        except Exception as exc:
            warnings.append(f"Page {page_number}: embedded image enumeration failed ({exc}).")
            continue
        for image_index, image in enumerate(images):
            try:
                data = bytes(image.data)
                digest = hashlib.sha256(data).hexdigest()
                extension = os.path.splitext(str(getattr(image, "name", "")))[1].lower()
                if not re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
                    extension = ".bin"
                pil_image = getattr(image, "image", None)
                width, height = getattr(pil_image, "size", (None, None))
                page_assets.append(PdfAsset(
                    asset_id=f"p{page_number}-img-{digest[:20]}",
                    page_number=page_number,
                    bounding_box=None,
                    data=data,
                    extension=extension,
                    mime_type=_image_mime_type(extension),
                    sha256=digest,
                    width=int(width) if width is not None else None,
                    height=int(height) if height is not None else None,
                ))
            except Exception as exc:
                warnings.append(
                    f"Page {page_number}: embedded image {image_index + 1} could not be extracted ({exc})."
                )
        if page_assets:
            assets_by_page[page_number] = page_assets
    return assets_by_page, warnings


def _position_page_assets(assets: list[PdfAsset], page) -> None:
    """Attach layout coordinates when pdfplumber and pypdf enumerate the same images."""
    layout_images = list(getattr(page, "images", []) or [])
    for asset, layout in zip(assets, layout_images):
        try:
            asset.bounding_box = tuple(
                round(float(layout[key]), 3)
                for key in ("x0", "top", "x1", "bottom")
            )
        except (KeyError, TypeError, ValueError):
            continue


def _associate_asset_captions(blocks: list[DocumentBlock], assets: list[PdfAsset]) -> None:
    captions = [block for block in blocks if block.block_type == "caption" and block.bounding_box]
    for asset in assets:
        candidates = [block for block in captions if block.page_number == asset.page_number]
        if asset.bounding_box:
            candidates.sort(key=lambda block: (
                abs(block.bounding_box[1] - asset.bounding_box[3]),
                abs(block.bounding_box[0] - asset.bounding_box[0]),
            ))
        if candidates:
            caption = candidates[0]
            asset.caption = caption.text
            caption.asset_ids.append(asset.asset_id)
            caption.provenance["asset_ids"] = list(caption.asset_ids)
        for block in blocks:
            if (
                block.page_number == asset.page_number
                and block.bounding_box
                and asset.bounding_box
                and _boxes_near(block.bounding_box, asset.bounding_box)
                and asset.asset_id not in block.asset_ids
            ):
                block.asset_ids.append(asset.asset_id)
                block.provenance["asset_ids"] = list(block.asset_ids)


def _boxes_near(
    block_box: tuple[float, float, float, float],
    asset_box: tuple[float, float, float, float],
) -> bool:
    horizontal_overlap = min(block_box[2], asset_box[2]) - max(block_box[0], asset_box[0])
    vertical_gap = min(abs(block_box[1] - asset_box[3]), abs(asset_box[1] - block_box[3]))
    return horizontal_overlap > 0 and vertical_gap <= 72


def _element_id(block: DocumentBlock) -> str:
    payload = "|".join([
        str(block.page_number),
        block.block_type,
        ",".join(str(value) for value in block.bounding_box or ()),
        block.text.strip(),
    ])
    return f"el-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _image_mime_type(extension: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".jp2": "image/jp2",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(extension, "application/octet-stream")


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
    borderless_blocks, borderless_boxes = _extract_borderless_tables(words, page_number)
    table_blocks.extend(borderless_blocks)
    table_boxes.extend(borderless_boxes)
    words = [
        word
        for word in words
        if not any(_inside_bbox(word, bbox) for bbox in borderless_boxes)
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
                structured_table=build_table(
                    normalized_rows,
                    source_type="pdf",
                    table_index=table_index,
                    page_number=page_number,
                    bounding_box=bbox,
                    cell_boxes=_line_table_cell_boxes(table, len(normalized_rows), width),
                    provenance={"table_settings": "lines", "coordinate_system": "pdf_page"},
                ),
            )
        )
    return blocks, boxes


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _inside_bbox(word: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    center_x = (float(word["x0"]) + float(word["x1"])) / 2
    center_y = (float(word["top"]) + float(word["bottom"])) / 2
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _word_rows(words: list[dict[str, Any]], tolerance: float = 3.0) -> list[list[dict[str, Any]]]:
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
    for row in rows:
        row.sort(key=lambda item: float(item["x0"]))
    return rows


def _split_word_row(row: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    sizes = [float(item.get("size") or 0) for item in row if float(item.get("size") or 0) > 0]
    gap_threshold = max(24.0, (statistics.median(sizes) if sizes else 10.0) * 2.4)
    cells: list[list[dict[str, Any]]] = []
    for word in row:
        if cells and float(word["x0"]) - float(cells[-1][-1]["x1"]) > gap_threshold:
            cells.append([])
        if not cells:
            cells.append([])
        cells[-1].append(word)
    return cells


def _extract_borderless_tables(
    words: list[dict[str, Any]],
    page_number: int,
) -> tuple[list[DocumentBlock], list[tuple[float, float, float, float]]]:
    candidates: list[tuple[list[list[dict[str, Any]]], float]] = []
    for row in _word_rows(words):
        cells = _split_word_row(row)
        if 2 <= len(cells) <= 12:
            candidates.append((cells, min(float(word["top"]) for word in row)))

    groups: list[list[list[list[dict[str, Any]]]]] = []
    current: list[list[list[dict[str, Any]]]] = []
    previous_top: float | None = None
    for cells, top in candidates:
        aligned = (
            current
            and len(cells) == len(current[-1])
            and all(
                abs(float(cell[0]["x0"]) - float(previous[0]["x0"])) <= 16
                for cell, previous in zip(cells, current[-1], strict=True)
            )
            and previous_top is not None
            and top - previous_top <= 32
        )
        if current and not aligned:
            groups.append(current)
            current = []
        current.append(cells)
        previous_top = top
    if current:
        groups.append(current)

    blocks: list[DocumentBlock] = []
    boxes: list[tuple[float, float, float, float]] = []
    for table_index, group in enumerate(groups):
        if len(group) < 2:
            continue
        text_rows = [
            [_join_words([str(word["text"]) for word in cell]) for cell in row]
            for row in group
        ]
        cell_lengths = [len(cell) for row in text_rows for cell in row if cell]
        numeric_data_rows = sum(
            any(re.search(r"\d", cell) for cell in row)
            for row in text_rows[1:]
        )
        if (
            not cell_lengths
            or statistics.median(cell_lengths) > 32
            or (
                statistics.median(cell_lengths) > 20
                and numeric_data_rows != len(text_rows) - 1
            )
        ):
            continue
        flat_words = [word for row in group for cell in row for word in cell]
        bbox = (
            round(min(float(word["x0"]) for word in flat_words), 3),
            round(min(float(word["top"]) for word in flat_words), 3),
            round(max(float(word["x1"]) for word in flat_words), 3),
            round(max(float(word["bottom"]) for word in flat_words), 3),
        )
        boxes.append(bbox)
        blocks.append(DocumentBlock(
            text="\n".join(" | ".join(row) for row in text_rows),
            page_number=page_number,
            block_type="table",
            bounding_box=bbox,
            provenance={
                "table_index": table_index,
                "row_count": len(text_rows),
                "column_count": len(text_rows[0]),
                "table_settings": "text_alignment",
            },
            structured_table=build_table(
                text_rows,
                source_type="pdf",
                table_index=table_index,
                page_number=page_number,
                bounding_box=bbox,
                provenance={"table_settings": "text_alignment", "coordinate_system": "pdf_page"},
                cell_boxes=[
                    [
                        (
                            round(min(float(word["x0"]) for word in cell), 3),
                            round(min(float(word["top"]) for word in cell), 3),
                            round(max(float(word["x1"]) for word in cell), 3),
                            round(max(float(word["bottom"]) for word in cell), 3),
                        ) if cell else None
                        for cell in row
                    ]
                    for row in group
                ],
            ),
        ))
    return blocks, boxes


def _line_table_cell_boxes(table, row_count: int, column_count: int):
    """Return pdfplumber cell boxes when its table geometry exposes them."""
    table_rows = getattr(table, "rows", None)
    if not table_rows:
        return None
    result = []
    for row in list(table_rows)[:row_count]:
        cells = list(getattr(row, "cells", []) or [])[:column_count]
        result.append([
            tuple(round(float(value), 3) for value in cell) if cell is not None else None
            for cell in cells
        ] + [None] * max(0, column_count - len(cells)))
    return result + [[None] * column_count for _ in range(max(0, row_count - len(result)))]


def _words_to_lines(words: list[dict[str, Any]], tolerance: float = 3.0) -> list[_Line]:
    rows = _word_rows(words, tolerance)

    lines: list[_Line] = []
    for row in rows:
        for segment in _split_word_row(row):
            text = _join_words([str(item["text"]) for item in segment])
            if not text:
                continue
            sizes = [float(item.get("size") or 0) for item in segment if float(item.get("size") or 0) > 0]
            fonts = " ".join(str(item.get("fontname") or "") for item in segment).lower()
            lines.append(
                _Line(
                    text=text,
                    x0=min(float(item["x0"]) for item in segment),
                    top=min(float(item["top"]) for item in segment),
                    x1=max(float(item["x1"]) for item in segment),
                    bottom=max(float(item["bottom"]) for item in segment),
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
