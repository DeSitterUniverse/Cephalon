"""Generate deterministic layout-heavy PDFs used by parser integration tests."""

from pathlib import Path
import tempfile

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-fixtures" / "pdfs"


def _fixture_image(path: Path, *, scan: bool = False) -> None:
    size = (1200, 1500) if scan else (560, 260)
    image = Image.new("RGB", size, "#f7f1e8")
    draw = ImageDraw.Draw(image)
    if scan:
        draw.rectangle((80, 80, 1120, 1420), outline="#222222", width=5)
        draw.text((130, 150), "SCANNED PAGE - OCR INTENTIONALLY DISABLED", fill="#111111")
        draw.text((130, 240), "This text exists only in raster pixels.", fill="#111111")
    else:
        draw.rounded_rectangle((20, 20, 540, 240), radius=24, fill="#243447", outline="#d9822b", width=8)
        draw.rectangle((80, 80, 180, 180), fill="#d9822b")
        draw.rectangle((230, 65, 330, 195), fill="#66a182")
        draw.rectangle((380, 95, 480, 165), fill="#6d8cc4")
    image.save(path, format="PNG")


def _header_footer(pdf: canvas.Canvas, page_number: int) -> None:
    pdf.setFont("Helvetica", 8)
    pdf.drawString(54, 762, "Cephalon Research Fixture")
    pdf.drawCentredString(306, 28, f"Proceedings Fixture · {page_number}")


def rich_layout_pdf(path: Path, image_path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter, pageCompression=0, invariant=1)
    for page_number in (1, 2):
        _header_footer(pdf, page_number)
        if page_number == 1:
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(54, 724, "1 Rich PDF Evidence")
            pdf.setFont("Helvetica", 10)
            left = [
                "The left column describes retrieval quality.",
                "RATE improved recall to 81.7 percent.",
                "Reading order must finish this column first.",
            ]
            right = [
                "The right column reports validation behavior.",
                "Citations remain attached to model-visible evidence.",
                "No language model is needed to parse this page.",
            ]
            for index, text in enumerate(left):
                pdf.drawString(54, 690 - index * 18, text)
            for index, text in enumerate(right):
                pdf.drawString(320, 690 - index * 18, text)

            # Bordered table.
            x0, y0, width, row_height = 54, 540, 310, 24
            rows = [
                ("Method", "Recall"),
                ("Baseline", "72.4"),
                ("RATE", "81.7"),
            ]
            for row in range(len(rows) + 1):
                pdf.line(x0, y0 + row * row_height, x0 + width, y0 + row * row_height)
            for x in (x0, x0 + 210, x0 + width):
                pdf.line(x, y0, x, y0 + len(rows) * row_height)
            for index, (method, recall) in enumerate(reversed(rows)):
                baseline = y0 + 8 + index * row_height
                pdf.drawString(x0 + 8, baseline, method)
                pdf.drawString(x0 + 220, baseline, recall)

            # Borderless table-like block.
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(390, 600, "Stage")
            pdf.drawString(500, 600, "ms")
            pdf.setFont("Helvetica", 10)
            pdf.drawString(390, 578, "Parse")
            pdf.drawString(500, 578, "14")
            pdf.drawString(390, 556, "Validate")
            pdf.drawString(500, 556, "3")

            pdf.drawImage(str(image_path), 80, 260, width=280, height=130, mask="auto")
            pdf.setFont("Helvetica-Oblique", 9)
            pdf.drawString(80, 244, "Figure 1: Retrieval pipeline blocks")
        else:
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(54, 714, "2 Repeated Margins and Continuation")
            pdf.setFont("Helvetica", 10)
            pdf.drawString(54, 680, "Repeated headers and footers must not enter searchable evidence.")
            pdf.drawString(54, 660, "Page-contained parent chunks preserve this page boundary.")
        pdf.showPage()
    pdf.save()


def scan_only_pdf(path: Path, image_path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter, pageCompression=0, invariant=1)
    pdf.drawImage(str(image_path), 36, 36, width=540, height=720, mask="auto")
    pdf.showPage()
    pdf.save()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cephalon-pdf-fixtures-") as temp_dir:
        temp = Path(temp_dir)
        figure = temp / "figure.png"
        scan = temp / "scan.png"
        _fixture_image(figure)
        _fixture_image(scan, scan=True)
        rich_layout_pdf(OUTPUT / "rich-layout.pdf", figure)
        scan_only_pdf(OUTPUT / "scan-only.pdf", scan)


if __name__ == "__main__":
    main()
