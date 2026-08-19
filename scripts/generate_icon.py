"""Generate the native Cephalon application icon from the SVG mark.

The Windows resource compiler needs an ICO file, while the Linux desktop
entry uses the checked-in SVG. Keep the small raster derivative reproducible
and visually aligned with ``assets/cephalon.svg``.
"""

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "cephalon.ico"
PNG_OUTPUT = ROOT / "assets" / "cephalon.png"
SCALE = 4


def points(values: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(round(x * SCALE), round(y * SCALE)) for x, y in values]


def main() -> None:
    size = 64 * SCALE
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    top = points([(32, 6), (54, 20), (32, 34), (10, 20)])
    left = points([(10, 20), (32, 34), (32, 58), (10, 44)])
    right = points([(54, 20), (32, 34), (32, 58), (54, 44)])
    stroke = 4 * SCALE

    draw.polygon(top, fill=(38, 30, 20, 255))
    draw.polygon(left, fill=(201, 94, 18, 255))
    draw.polygon(right, fill=(138, 59, 11, 255))
    draw.line(top + [top[0]], fill=(255, 154, 46, 255), width=stroke, joint="curve")
    draw.line(left + [left[0]], fill=(201, 94, 18, 255), width=stroke, joint="curve")
    draw.line(right + [right[0]], fill=(138, 59, 11, 255), width=stroke, joint="curve")
    draw.line(points([(32, 34), (32, 58)]), fill=(255, 208, 138, 255), width=2 * SCALE)

    image = image.resize((64, 64), Image.Resampling.LANCZOS)
    image.save(PNG_OUTPUT, format="PNG")
    image.save(OUTPUT, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64)])


if __name__ == "__main__":
    main()
