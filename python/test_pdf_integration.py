from pathlib import Path

from cephalon_core.services import documents, pdf_parser


FIXTURES = Path(__file__).resolve().parents[1] / "test-fixtures" / "pdfs"


def test_real_rich_pdf_preserves_layout_tables_caption_and_embedded_asset():
    path = FIXTURES / "rich-layout.pdf"

    parsed = pdf_parser.parse_pdf(str(path))
    wrapper_text, wrapper_mode = documents.extract_text(str(path))

    assert parsed.page_count == 2
    assert wrapper_mode == "native_structured"
    assert wrapper_text == parsed.text
    assert "Cephalon Research Fixture" not in parsed.text
    assert "Proceedings Fixture" not in parsed.text
    assert parsed.text.index("left column") < parsed.text.index("right column")
    assert any(
        block.block_type == "table"
        and "Method | Recall" in block.text
        and "RATE | 81.7" in block.text
        for block in parsed.blocks
    )
    caption = next(block for block in parsed.blocks if block.block_type == "caption")
    assert caption.text == "Figure 1: Retrieval pipeline blocks"
    assert caption.element_id and caption.element_id.startswith("el-")
    assert len(parsed.assets) == 1
    assert parsed.assets[0].caption == caption.text
    assert parsed.assets[0].asset_id in caption.asset_ids
    assert parsed.assets[0].width == 560
    assert parsed.assets[0].height == 260


def test_scan_only_pdf_extracts_asset_and_reports_ocr_disabled_without_fake_text():
    parsed = pdf_parser.parse_pdf(str(FIXTURES / "scan-only.pdf"))

    assert parsed.page_count == 1
    assert parsed.text == ""
    assert parsed.blocks == []
    assert len(parsed.assets) == 1
    assert any("OCR is disabled" in warning for warning in parsed.warnings)
