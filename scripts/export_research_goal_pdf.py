#!/usr/bin/env python3
"""Export 科研目标.md to a readable PDF with Chinese font support."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "科研目标.md"
OUT_DIR = ROOT / "output" / "pdf"
OUT_PDF = OUT_DIR / "科研目标.pdf"

FONT_DIR = Path("/mnt/c/Windows/Fonts")
FONT_REGULAR = FONT_DIR / "Noto Sans SC (TrueType).otf"
FONT_BOLD = FONT_DIR / "Noto Sans SC Bold (TrueType).otf"
FONT_FALLBACK = FONT_DIR / "msyh.ttc"

PAGE_W, PAGE_H = 1654, 2339  # A4 at about 200 DPI
MARGIN_X = 130
MARGIN_TOP = 120
MARGIN_BOTTOM = 110
BODY_SIZE = 34
SMALL_SIZE = 30
H1_SIZE = 58
H2_SIZE = 42
LINE_GAP = 14

INK = (32, 36, 40)
MUTED = (92, 100, 112)
BLUE = (30, 80, 160)
RULE = (216, 222, 230)
SOFT = (246, 248, 251)
FORMULA_BG = (247, 249, 252)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.truetype(str(FONT_FALLBACK), size=size)


F_BODY = font(FONT_REGULAR, BODY_SIZE)
F_SMALL = font(FONT_REGULAR, SMALL_SIZE)
F_BOLD = font(FONT_BOLD, BODY_SIZE)
F_H1 = font(FONT_BOLD, H1_SIZE)
F_H2 = font(FONT_BOLD, H2_SIZE)
F_FORMULA = font(FONT_REGULAR, 33)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def line_height(fnt: ImageFont.FreeTypeFont, extra: int = LINE_GAP) -> int:
    ascent, descent = fnt.getmetrics()
    return ascent + descent + extra


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    text = text.strip()
    if not text:
        return [""]

    parts = re.split(r"(\s+)", text)
    lines: list[str] = []
    current = ""

    for part in parts:
        if not part:
            continue
        candidate = current + part
        if text_width(draw, candidate.strip(), fnt) <= width:
            current = candidate
            continue

        if current.strip():
            lines.append(current.strip())
            current = ""

        token = part.strip()
        if not token:
            continue
        buf = ""
        for ch in token:
            cand = buf + ch
            if text_width(draw, cand, fnt) <= width:
                buf = cand
            else:
                if buf:
                    lines.append(buf)
                buf = ch
        current = buf

    if current.strip():
        lines.append(current.strip())
    return lines or [""]


def clean_inline(text: str) -> str:
    text = text.replace("**", "")
    text = text.replace("`", "")
    text = text.replace("$", "")
    text = clean_formula(text)
    return text.strip()


LATEX_REPL = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\Delta": "Δ",
    r"\sum": "Σ",
    r"\omega": "ω",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\theta": "θ",
    r"\epsilon": "ε",
    r"\in": "∈",
    r"\cdot": "·",
    r"\propto": "∝",
    r"\frac": "frac",
    r"\lVert": "||",
    r"\rVert": "||",
    r"\{": "{",
    r"\}": "}",
}


def clean_formula(text: str) -> str:
    text = text.strip()
    for src, dst in LATEX_REPL.items():
        text = text.replace(src, dst)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("^", "^").replace("_", "_")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("fracA_00 - A_LMA_00", "(A_00 - A_LM) / A_00")
    text = text.replace("fracn_k / div_kΣ_j n_j / div_j", "(n_k / div_k) / Σ_j(n_j / div_j)")
    return text


class PdfBuilder:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page = self.new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP
        self.page_no = 1

    def new_page(self) -> Image.Image:
        return Image.new("RGB", (PAGE_W, PAGE_H), "white")

    def finish_page(self) -> None:
        footer = f"科研目标：FedCHI    {self.page_no}"
        self.draw.line((MARGIN_X, PAGE_H - 82, PAGE_W - MARGIN_X, PAGE_H - 82), fill=RULE, width=2)
        self.draw.text((MARGIN_X, PAGE_H - 64), footer, font=F_SMALL, fill=MUTED)
        self.pages.append(self.page)
        self.page_no += 1
        self.page = self.new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP

    def ensure_space(self, height: int) -> None:
        if self.y + height > PAGE_H - MARGIN_BOTTOM:
            self.finish_page()

    def add_gap(self, h: int) -> None:
        self.y += h

    def add_paragraph(self, text: str, fnt: ImageFont.FreeTypeFont = F_BODY, color=INK, indent: int = 0) -> None:
        text = clean_inline(text)
        if not text:
            self.add_gap(14)
            return
        width = PAGE_W - 2 * MARGIN_X - indent
        lines = wrap_text(self.draw, text, fnt, width)
        h = len(lines) * line_height(fnt)
        self.ensure_space(h)
        x = MARGIN_X + indent
        for line in lines:
            self.draw.text((x, self.y), line, font=fnt, fill=color)
            self.y += line_height(fnt)

    def add_heading(self, text: str, level: int) -> None:
        fnt = F_H1 if level == 1 else F_H2
        color = BLUE if level == 1 else INK
        gap_before = 0 if self.y == MARGIN_TOP else 34
        lines = wrap_text(self.draw, clean_inline(text), fnt, PAGE_W - 2 * MARGIN_X)
        h = gap_before + len(lines) * line_height(fnt, 18) + (24 if level == 1 else 14)
        self.ensure_space(h)
        self.y += gap_before
        for line in lines:
            self.draw.text((MARGIN_X, self.y), line, font=fnt, fill=color)
            self.y += line_height(fnt, 18)
        if level == 1:
            self.draw.line((MARGIN_X, self.y + 6, PAGE_W - MARGIN_X, self.y + 6), fill=RULE, width=3)
            self.y += 32
        else:
            self.y += 14

    def add_formula(self, text: str) -> None:
        text = clean_formula(text)
        lines = wrap_text(self.draw, text, F_FORMULA, PAGE_W - 2 * MARGIN_X - 90)
        h = len(lines) * line_height(F_FORMULA, 16) + 42
        self.ensure_space(h)
        x0, x1 = MARGIN_X, PAGE_W - MARGIN_X
        y0, y1 = self.y, self.y + h
        self.draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=FORMULA_BG, outline=RULE, width=2)
        yy = y0 + 20
        for line in lines:
            tw = text_width(self.draw, line, F_FORMULA)
            self.draw.text(((PAGE_W - tw) // 2, yy), line, font=F_FORMULA, fill=INK)
            yy += line_height(F_FORMULA, 16)
        self.y = y1 + 18

    def add_quote(self, text: str) -> None:
        lines = wrap_text(self.draw, clean_inline(text), F_BODY, PAGE_W - 2 * MARGIN_X - 70)
        h = len(lines) * line_height(F_BODY) + 36
        self.ensure_space(h)
        x = MARGIN_X + 34
        self.draw.line((MARGIN_X, self.y + 4, MARGIN_X, self.y + h - 4), fill=BLUE, width=6)
        yy = self.y + 18
        for line in lines:
            self.draw.text((x, yy), line, font=F_BODY, fill=INK)
            yy += line_height(F_BODY)
        self.y += h + 10

    def add_table(self, rows: list[list[str]]) -> None:
        rows = [[clean_inline(cell) for cell in row] for row in rows if row]
        if not rows:
            return
        cols = max(len(row) for row in rows)
        rows = [row + [""] * (cols - len(row)) for row in rows]
        table_w = PAGE_W - 2 * MARGIN_X
        col_w = table_w // cols

        wrapped_rows: list[list[list[str]]] = []
        row_heights: list[int] = []
        for ridx, row in enumerate(rows):
            fnt = F_BOLD if ridx == 0 else F_SMALL
            wrapped = [wrap_text(self.draw, cell, fnt, col_w - 24) for cell in row]
            wrapped_rows.append(wrapped)
            row_heights.append(max(len(cell_lines) for cell_lines in wrapped) * line_height(fnt, 8) + 26)

        total_h = sum(row_heights)
        self.ensure_space(total_h + 24)
        y = self.y
        for ridx, wrapped in enumerate(wrapped_rows):
            rh = row_heights[ridx]
            fill = SOFT if ridx == 0 else "white"
            self.draw.rectangle((MARGIN_X, y, MARGIN_X + table_w, y + rh), fill=fill, outline=RULE)
            x = MARGIN_X
            fnt = F_BOLD if ridx == 0 else F_SMALL
            for cell_lines in wrapped:
                self.draw.line((x, y, x, y + rh), fill=RULE, width=1)
                yy = y + 13
                for line in cell_lines:
                    self.draw.text((x + 12, yy), line, font=fnt, fill=INK)
                    yy += line_height(fnt, 8)
                x += col_w
            self.draw.line((MARGIN_X + table_w, y, MARGIN_X + table_w, y + rh), fill=RULE, width=1)
            y += rh
        self.y = y + 24


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip()
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        if not all(set(cell) <= {"-", ":", " "} for cell in cells):
            rows.append(cells)
        i += 1
    return rows, i


def export_pdf() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    builder = PdfBuilder()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            builder.add_gap(12)
            i += 1
            continue
        if stripped.startswith("$$"):
            formula_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                formula_lines.append(lines[i].strip())
                i += 1
            builder.add_formula(" ".join(formula_lines))
            i += 1
            continue
        if stripped.startswith("# "):
            builder.add_heading(stripped[2:], 1)
            i += 1
            continue
        if stripped.startswith("## "):
            builder.add_heading(stripped[3:], 2)
            i += 1
            continue
        if stripped.startswith(">"):
            builder.add_quote(stripped.lstrip(">").strip())
            i += 1
            continue
        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            builder.add_table(rows)
            continue
        if re.match(r"^\d+\.\s+", stripped):
            builder.add_paragraph(stripped, F_BODY, indent=28)
            i += 1
            continue
        if stripped.startswith("- "):
            builder.add_paragraph("• " + stripped[2:], F_BODY, indent=24)
            i += 1
            continue
        builder.add_paragraph(stripped)
        i += 1

    builder.finish_page()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first, rest = builder.pages[0], builder.pages[1:]
    first.save(OUT_PDF, save_all=True, append_images=rest, resolution=200.0)
    print(OUT_PDF.relative_to(ROOT))
    print(f"pages={len(builder.pages)}")


if __name__ == "__main__":
    export_pdf()
