"""Local web UI for PaddleOCR: upload an image or PDF and view OCR results in the browser."""

import base64
import gc
import glob
import io
import json
import os
import re
import time
import traceback
import uuid

# The model cache lives beside the project on E:. Windows puts it under the user profile
# on C: by default, and this machine's C: is full — a download that runs out of room
# leaves a half-written model that then fails to load. Set before paddle is imported.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", r"E:\paddlex_cache")

import cv2
import fitz  # PyMuPDF
import gradio as gr
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsmap
from docx.shared import Cm, Pt, RGBColor
from paddleocr import (
    LayoutDetection,
    PaddleOCR,
    TableStructureRecognition,
    TextDetection,
    TextRecognition,
)

# Some PDF exporters (e.g. certain note-taking apps) emit a broken ToUnicode
# mapping for the lam + hamza-alef ligature (لأ/لإ/لآ) used after the Arabic
# definite article, storing it reversed as alef + hamza-alef + lam
# (e.g. "األساسية" instead of the correct "الأساسية"). This is a defect baked
# into the source PDF's font, not a PyMuPDF/OCR issue, but it is safe and
# deterministic to reverse: it only ever fires at a word boundary.
_LAM_HAMZA_SWAP_RE = re.compile(r"(?<!\w)ا([أإآ])ل")

# PDF/OCR extraction frequently loses the space at an Arabic <-> Latin boundary,
# gluing tokens together (e.g. ":Tagوسم" instead of ":Tag وسم"). Arabic letters are
# matched as ء-ي only, so Arabic punctuation (،؛؟) keeps hugging its word.
_AR = "ء-ي"
_LAT = "A-Za-z0-9"
_AR_THEN_LAT_RE = re.compile(rf"([{_AR}])([{_LAT}])")
_LAT_THEN_AR_RE = re.compile(rf"([{_LAT}])([{_AR}])")


def normalize_mixed_text(text):
    """Insert the missing space at Arabic/Latin boundaries."""
    text = _AR_THEN_LAT_RE.sub(r"\1 \2", text)
    text = _LAT_THEN_AR_RE.sub(r"\1 \2", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# In this class of PDF the punctuation that visually ends an Arabic run is stored
# in front of the next word (" .متوفرة" instead of " متوفرة."). Moving it back is only
# applied when the following word starts with an Arabic letter, so Latin/technical
# tokens ("CI/CD:", "e.g.", URLs, decimals) are never touched.
_MISPLACED_PUNCT_RE = re.compile(r"(^|\s)([.،:؛!؟])([ء-ي][^\s]*)")


def fix_misplaced_punctuation(text):
    return _MISPLACED_PUNCT_RE.sub(r"\1\3\2", text)


def _clean_line(text):
    """Apply the PDF-glyph and spacing fixes to a single line/paragraph."""
    text = _LAM_HAMZA_SWAP_RE.sub(r"ال\1", text)
    text = re.sub(r"([•●])​?\s*", r"\1 ", text)
    text = fix_misplaced_punctuation(text)
    return normalize_mixed_text(text)


def _merge_split_headers(items):
    """A numbered header often lands as two blocks: '2' then ') فهم …'."""
    merged = []
    i = 0
    while i < len(items):
        if items[i].isdigit() and i + 1 < len(items) and items[i + 1].startswith(")"):
            merged.append(f"{items[i]}{items[i + 1]}")
            i += 2
        else:
            merged.append(items[i])
            i += 1
    return merged


# Some PDFs carry a text layer whose ToUnicode mapping transposes letters, so the page
# renders perfectly but the extracted text is scrambled ("البرمجة" comes out "الربمجة").
# These tokens are the corrupted forms of the most common Arabic words; none of them is a
# real word, so even a couple of them is strong evidence the layer cannot be trusted.
_SCRAMBLED_MARKERS = ("يف", "ىلع", "ىلإ", "يتلا", "يذلا", "اذه", "نأ", "لىع", "نم")
_SCRAMBLE_THRESHOLD = 3


def looks_scrambled(text):
    """True when a PDF's embedded Arabic text shows the letter-transposition defect."""
    words = re.findall(r"[ء-ي]+", text)
    if len(words) < 20:  # too little text to judge
        return False
    hits = sum(1 for w in words if w in _SCRAMBLED_MARKERS)
    return hits >= _SCRAMBLE_THRESHOLD


# Two closed, hand-picked word lists — not a general ي/ى or ه/ة rule — because both
# letter pairs are also completely legitimate spellings in most other words (many verbs
# and pronoun suffixes genuinely end in ه; many names genuinely end in ي). A general
# rule would flag more correct text than actual OCR errors. These specific words were
# picked because the two glyphs in each pair are near-identical (alef maqsura/ya differ
# only by two dots; ha/ta-marbuta differ only by two dots and a stroke), which is exactly
# the class of confusion a recognizer makes, and because they are common enough in the
# administrative documents this tool is aimed at to be worth flagging on sight.
_ALEF_MAQSURA_WORDS = {
    "علي": "على", "الي": "إلى", "الى": "إلى", "حتي": "حتى", "متي": "متى",
    "لدي": "لدى", "سوي": "سوى",
}
_TA_MARBUTA_WORDS = {
    "مدرسه": "مدرسة", "جامعه": "جامعة", "لغه": "لغة", "وثيقه": "وثيقة",
    "وزاره": "وزارة", "مديريه": "مديرية", "مؤسسه": "مؤسسة", "حكومه": "حكومة",
    "لجنه": "لجنة", "خطه": "خطة", "مرحله": "مرحلة", "مساحه": "مساحة", "منطقه": "منطقة",
}
# Arabic nouns appear with the definite article far more often than bare ("الجامعة", not
# "جامعة"), so each noun above is also indexed with "ال" glued on the front — the same
# word, still unambiguous, just in its far more common running-text form.
_CONFUSION_WORDS = {**_ALEF_MAQSURA_WORDS, **_TA_MARBUTA_WORDS}
for _wrong, _right in _TA_MARBUTA_WORDS.items():
    _CONFUSION_WORDS.setdefault("ال" + _wrong, "ال" + _right)
del _wrong, _right
_CONFUSION_RE = re.compile(
    r"(?<!\w)(" + "|".join(sorted(_CONFUSION_WORDS, key=len, reverse=True)) + r")(?!\w)"
)


def suggest_correction(text):
    """A read-only correction suggestion for a line matching a known OCR confusion,
    or None when nothing in it does. Never applied automatically — see `_CONFUSION_WORDS`."""
    if not _CONFUSION_RE.search(text):
        return None
    return _CONFUSION_RE.sub(lambda m: _CONFUSION_WORDS[m.group(1)], text)


def _line_style(line):
    """The size and colour that cover most of one line — what the eye reads it as."""
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return None
    widest = max(spans, key=lambda s: len(s.get("text", "")))
    return round(widest.get("size", 0), 1), widest.get("color", 0)


def _typographic_runs(block):
    """The block's lines, cut wherever the typography changes.

    MuPDF puts a small grey caption and the large figure beneath it in one block. Joined,
    they read as one nonsense line and get redrawn as one; cut at the change of size or
    colour, each keeps its own place, size and colour on the page."""
    runs, current, style = [], [], None
    for line in block.get("lines", []):
        found = _line_style(line)
        if found is None:
            continue
        if style is not None and found != style and current:
            runs.append(current)
            current = []
        style = found
        current.append(line)
    if current:
        runs.append(current)
    return _merge_markers(runs)


def _is_marker(run):
    """A run holding nothing but a bullet or a dash — it belongs to the item beside it."""
    text = "".join(
        span.get("text", "") for line in run for span in line.get("spans", [])
    ).strip()
    return len(text) <= 2 and not any(ch.isalnum() for ch in text)


def _merge_markers(runs):
    """Fold a lone bullet back into its list item.

    Bullets are usually set in another font, and often another colour, so on their own
    they would be cut off from the text they introduce."""
    merged, pending = [], []
    for run in runs:
        if _is_marker(run):
            pending.extend(run)
            continue
        merged.append(pending + run)
        pending = []
    if pending:
        if merged:
            merged[-1].extend(pending)
        else:
            merged.append(pending)
    return merged


def _describe_block(block):
    """Every typographic run in one dict block, described on its own."""
    return [run for run in map(_describe_lines, _typographic_runs(block)) if run]


def _describe_lines(block_lines):
    """Text, box and typography of one run of lines, all from the same spans.

    Everything is read from a single source: `get_text("blocks")` and `get_text("dict")`
    number their blocks differently (the dict includes image blocks), so pairing them by
    index silently attaches the neighbouring block's box, size and colour to every
    paragraph. Reading text and style from the same spans removes that whole class of bug.
    The box also comes from the spans rather than the block, because a block's own bounds
    include its blank lines and would push the text upwards when redrawn."""
    lines = []
    spans = []
    for line in block_lines:
        parts = [s.get("text", "") for s in line.get("spans", [])]
        text = "".join(parts).strip()
        if not text:
            continue
        lines.append(text)
        spans.extend(s for s in line.get("spans", []) if s.get("text", "").strip())
    if not lines or not spans:
        return None

    # the size and colour covering the most characters win, so a stray superscript
    # cannot redefine a whole paragraph
    sizes, colours = {}, {}
    bold_chars = 0
    for span in spans:
        length = len(span.get("text", ""))
        sizes[round(span["size"], 1)] = sizes.get(round(span["size"], 1), 0) + length
        colours[span.get("color", 0)] = colours.get(span.get("color", 0), 0) + length
        if span.get("flags", 0) & 16:
            bold_chars += length
    total = sum(sizes.values()) or 1
    boxes = [s["bbox"] for s in spans if s.get("bbox")]

    return {
        "text": " ".join(lines),
        "box": (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ),
        "size": max(sizes, key=sizes.get),
        "bold": bold_chars / total > 0.6,
        "colour": f"#{max(colours, key=colours.get) & 0xFFFFFF:06x}",
        "lines": len(lines),
    }


def _rgb255(triple):
    """PyMuPDF gives colours as 0–1 floats; keep them raw so each exporter can format
    them its own way (CSS for the page view, hex + alpha for Word)."""
    if not triple:
        return None
    return [max(0, min(255, int(round(c * 255)))) for c in triple[:3]]


def _css_colour(rgb, alpha=1.0):
    if not rgb:
        return None
    r, g, b = rgb[:3]
    if alpha is not None and alpha < 0.99:
        return f"rgba({r},{g},{b},{alpha:.2f})"
    return f"#{r:02x}{g:02x}{b:02x}"


def _corner_radius(drawing):
    """Recover a rounded rectangle's corner radius from its straight edges.

    A rounded rect is drawn as four lines joined by corner curves; each straight edge
    starts one radius away from the corner, which is what this measures."""
    rect = drawing["rect"]
    best = 0.0
    for item in drawing["items"]:
        if item[0] != "l":
            continue
        p1, p2 = item[1], item[2]
        if abs(p1.y - p2.y) < 0.5:  # horizontal edge
            best = max(best, min(abs(p1.x - rect.x0), abs(p2.x - rect.x0)))
        elif abs(p1.x - p2.x) < 0.5:  # vertical edge
            best = max(best, min(abs(p1.y - rect.y0), abs(p2.y - rect.y0)))
    limit = min(rect.width, rect.height) / 2
    return round(min(best, limit), 2)


def extract_pdf_drawings(page):
    """Vector shapes (panels, boxes, rules, connectors) with their colours.

    On a chart-heavy page the boxes carry as much meaning as the text, so redrawing the
    page without them leaves labels floating in space."""
    shapes = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.width <= 0 or rect.height <= 0:
            continue
        fill = _rgb255(drawing.get("fill"))
        stroke = _rgb255(drawing.get("color"))
        if not fill and not stroke:
            continue
        shapes.append(
            {
                "box": (rect.x0, rect.y0, rect.x1, rect.y1),
                "fill": fill,
                "fill_alpha": float(drawing.get("fill_opacity") or 1.0),
                "stroke": stroke,
                "stroke_alpha": float(drawing.get("stroke_opacity") or 1.0),
                "width": float(drawing.get("width") or 0),
                "radius": _corner_radius(drawing),
            }
        )
    return shapes


# Below this, in PDF points, a placed image is almost always a tracking pixel or a
# spacer GIF left over from whatever authoring tool produced the file, not real content.
MIN_PDF_IMAGE_SIZE = 20
# Below this, in the image's own *native* pixel dimensions (not its placed size on the
# page), it is not a photo at all but a 1x1 or 2x2 solid-colour swatch — a common trick
# for a gradient or flat-colour fill, stretched by the page to whatever size is needed.
# Measured directly on a real 115-page document: one such swatch was reused, stretched to
# a different rectangle, in 8 different spots on a single page alone, and would otherwise
# have been extracted as 8 separate blurry "figures" burying the two genuine photos on it.
MIN_NATIVE_IMAGE_PIXELS = 8


def extract_pdf_images(page):
    """Raster images placed directly on a PDF page — photos, logos, screenshots.

    `extract_pdf_drawings` only ever sees vector drawings (`page.get_drawings()`); a real
    embedded photo never appears there at all, so a page extracted through the "embedded
    text" path (real digital text, no OCR) silently lost every picture on it until now —
    confirmed on this project's own BRD test file, which places the same watermark image
    on all five of its pages and had every one of them go missing. `page.get_images()` and
    `extract_image()` are the actual PyMuPDF entry points for this; a soft-mask xref that
    sometimes rides along with an image is not itself directly placed on the page, so
    `get_image_rects()` naturally returns nothing for it and it is skipped without any
    special-casing."""
    doc = page.parent
    page_area = max(1.0, page.rect.width * page.rect.height)
    figures = []
    seen_xrefs = set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        native_width, native_height = img[2], img[3]
        if native_width < MIN_NATIVE_IMAGE_PIXELS or native_height < MIN_NATIVE_IMAGE_PIXELS:
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        try:
            raw = doc.extract_image(xref)
        except Exception:  # noqa: BLE001 - one bad image must not break the whole page
            continue
        image_bytes = raw.get("image")
        if not image_bytes:
            continue
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            continue
        # re-encoded to PNG so both this and the OCR-detected figure path (`figure_shapes`)
        # store the same format, and every downstream reader only has to handle one
        ok, buf = cv2.imencode(".png", decoded)
        if not ok:
            continue
        png_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        for rect in rects:
            if rect.width < MIN_PDF_IMAGE_SIZE or rect.height < MIN_PDF_IMAGE_SIZE:
                continue
            if rect.width * rect.height > FIGURE_MAX_PAGE_SHARE * page_area:
                # a scanned page saved as PDF places its entire content as a single large
                # image with no separate text layer at all — this is that image, not a
                # distinct photo on the page, and must never be treated as one: OCR still
                # has to read text out of the exact same pixels, and if this were kept as
                # a "figure" the exclusion step would delete every line it finds as if it
                # were a caption sitting on top of an unrelated picture
                continue
            figures.append({"box": (rect.x0, rect.y0, rect.x1, rect.y1), "png_b64": png_b64})
    return figures


def _edge_counts(values, step):
    """How many blocks end (or start) on each band of the page, `step` units wide."""
    counts = {}
    for value in values:
        key = round(value / step)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _shares_edge(value, counts, step):
    """True when another block lines up on this edge, within one band either side.

    The band is a fraction of the text column rather than a fixed distance: a PDF aligns
    its margins to the point, but a photographed page never repeats an edge exactly."""
    key = round(value / step)
    return sum(counts.get(key + offset, 0) for offset in (-1, 0, 1)) > 1


def _is_rtl(text):
    """Whether a block reads right-to-left, by which script most of its letters are in."""
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return arabic >= latin


COLUMN_MARGIN = 2.0        # points of air kept between two side-by-side blocks
EDGE_BAND = 0.012          # of the text column: how close two edges must be to agree
CENTRE_TOLERANCE = 0.04    # of the text column, before a block counts as centred


def relax_block_widths(blocks):
    """Give every paragraph the free space beside it, and record how it was aligned.

    A block's box is only as wide as its glyphs were in the source font. Redrawn in a web
    or Word font that runs a few percent wider, a heading that exactly filled its box no
    longer fits, wraps onto a second line, and that line lands on top of the block below.
    Widening each box into the empty space around it — stopping at any block that shares
    its rows — removes the wrap without moving any text, because every block stays anchored
    to the side it was aligned to: right for Arabic, left for Latin, both for a title."""
    if not blocks:
        return blocks
    column_left = min(b["box"][0] for b in blocks)
    column_right = max(b["box"][2] for b in blocks)
    column_mid = (column_left + column_right) / 2
    tolerance = max(4.0, (column_right - column_left) * CENTRE_TOLERANCE)
    # blocks that share an edge were aligned to the same guide: a margin, or the indent of
    # a bullet list. That agreement is far more reliable than measuring one block's gaps,
    # which makes every short indented line look as if it had been centred.
    step = max(1.0, (column_right - column_left) * EDGE_BAND)
    right_edges = _edge_counts((b["box"][2] for b in blocks), step)
    left_edges = _edge_counts((b["box"][0] for b in blocks), step)

    for block in blocks:
        # the extent the glyphs actually cover, kept before the box grows: it is the only
        # way to tell afterwards whether a line filled its column or merely started one
        block["ink"] = block["box"]
        if block.get("fixed"):
            continue
        x0, y0, x1, y1 = block["box"]
        left_gap, right_gap = x0 - column_left, column_right - x1
        on_right = _shares_edge(x1, right_edges, step)
        on_left = _shares_edge(x0, left_edges, step)
        if on_right and on_left:
            align = "right" if _is_rtl(block.get("text", "")) else "left"
        elif on_right:
            align = "right"
        elif on_left:
            align = "left"
        elif (abs((x0 + x1) / 2 - column_mid) <= tolerance
              and (x1 - x0) < 0.9 * (column_right - column_left)):
            align = "center"
        elif right_gap <= left_gap:
            align = "right"
        else:
            align = "left"

        # a block on the same rows is a real neighbour: never grow across it
        limit_left, limit_right = column_left, column_right
        for other in blocks:
            if other is block:
                continue
            ox0, oy0, ox1, oy1 = other["box"]
            shared = min(y1, oy1) - max(y0, oy0)
            if shared <= 0.4 * min(y1 - y0, oy1 - oy0):
                continue
            if ox1 <= x0:
                limit_left = max(limit_left, ox1 + COLUMN_MARGIN)
            elif ox0 >= x1:
                limit_right = min(limit_right, ox0 - COLUMN_MARGIN)

        if align == "center":
            room = min(x0 - limit_left, limit_right - x1)
            if room > 0:
                x0, x1 = x0 - room, x1 + room
        elif align == "right":
            x0 = min(x0, limit_left)
        else:
            x1 = max(x1, limit_right)
        block["box"] = (x0, y0, x1, y1)
        block["align"] = align
    return blocks


CELL_PADDING = 2.0   # points kept clear of a cell's ruling line


def _page_tables(page):
    """The tables MuPDF can find on the page, as (bounding box, cell blocks) pairs.

    A table row is a single text block to MuPDF, so its cells are joined into one long
    line that reads straight across the ruling between them. Rebuilding the row cell by
    cell keeps each one inside its own box, which is both how the page looks and how the
    text should read."""
    try:
        found = list(page.find_tables().tables)
    except Exception:
        return []

    tables = []
    for table in found:
        rtl = _is_rtl(" ".join(cell or "" for row in table.extract() for cell in row))
        # read the cells the way the table reads: row by row, and inside a row from the
        # side the language starts on. Their own rectangles say that; the index order in
        # `table.cells` does not.
        cells = sorted(
            (c for c in table.cells if c),
            key=lambda c: (round(c[1]), -c[0] if rtl else c[0]),
        )
        blocks = [b for cell in cells for b in _cell_blocks(page, cell)]
        if blocks:
            tables.append((fitz.Rect(table.bbox), blocks))
    return tables


def _cell_blocks(page, cell):
    """A table cell as blocks: one per typographic style inside it.

    A cell usually reads in one style and becomes a single block filling it, however many
    lines it wraps to. Only when the styles differ - a small grey caption over a large
    figure, as in a statistics tile - is it split, and then each part keeps its own line
    and only borrows the cell's width."""
    rect = fitz.Rect(cell)
    if rect.is_empty:
        return []
    parts = [
        info
        for raw in page.get_text("dict", clip=rect).get("blocks", [])
        for info in _describe_block(raw)
    ]
    if not parts:
        return []

    groups = []
    for part in parts:
        style = (round(part["size"], 1), part.get("colour"), bool(part["bold"]))
        if groups and groups[-1][0] == style:
            groups[-1][1].append(part)
        else:
            groups.append((style, [part]))

    left, right = rect.x0 + CELL_PADDING, rect.x1 - CELL_PADDING
    blocks = []
    for _, group in groups:
        text = _clean_line(" ".join(part["text"] for part in group))
        if not text:
            continue
        if len(groups) > 1:
            top = min(part["box"][1] for part in group)
            bottom = max(part["box"][3] for part in group)
        else:
            top, bottom = rect.y0 + CELL_PADDING, rect.y1 - CELL_PADDING
        largest = max(group, key=lambda part: part["size"])
        blocks.append(
            {
                "text": text,
                "box": (left, top, right, bottom),
                "size": largest["size"],
                "bold": any(part["bold"] for part in group),
                "colour": largest.get("colour"),
                "lines": sum(part["lines"] for part in group),
                "align": "right" if _is_rtl(text) else "left",
                "fixed": True,   # a cell already has its own width: never widen it
            }
        )
    return blocks


def extract_pdf_blocks(page):
    """Extract a PDF page as flowing paragraphs that keep their position on the page.

    PyMuPDF's text blocks correspond to logical paragraphs (one bullet, one heading,
    one body paragraph each), so joining the lines inside a block reconstructs text that
    reads like the original document instead of a stack of line fragments. The block's
    bounding box, font size and weight are carried along so the page can also be redrawn
    with the original layout."""
    tables = _page_tables(page)
    blocks = []
    rebuilt = set()
    for raw_block in page.get_text("dict").get("blocks", []):
        for described in _describe_block(raw_block):
            x0, y0, x1, y1 = described["box"]
            centre = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
            inside = next(
                (i for i, (area, _) in enumerate(tables) if area.contains(centre)), None
            )
            if inside is not None:
                # the whole table is emitted once, in place of its first block
                if inside not in rebuilt:
                    blocks.extend(tables[inside][1])
                    rebuilt.add(inside)
                continue
            text = _clean_line(described["text"])
            if not text:
                continue
            described["text"] = text
            blocks.append(described)

    # a numbered header often lands as two blocks: "2" then ") فهم …"
    merged = []
    i = 0
    while i < len(blocks):
        current = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        if current["text"].isdigit() and nxt and nxt["text"].startswith(")"):
            x0 = min(current["box"][0], nxt["box"][0])
            y0 = min(current["box"][1], nxt["box"][1])
            x1 = max(current["box"][2], nxt["box"][2])
            y1 = max(current["box"][3], nxt["box"][3])
            merged.append(
                {
                    "text": f"{current['text']}{nxt['text']}",
                    "box": (x0, y0, x1, y1),
                    "size": max(current["size"], nxt["size"]),
                    "bold": current["bold"] or nxt["bold"],
                    "colour": current.get("colour") or nxt.get("colour"),
                    "lines": max(current.get("lines", 1), nxt.get("lines", 1)),
                }
            )
            i += 2
        else:
            merged.append(current)
            i += 1
    return relax_block_widths(merged)

_HEADING_RE = re.compile(r"^\d+\)")

CONFIDENCE_LOW = 0.8
CONFIDENCE_VERY_LOW = 0.5


def _html_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def alert_html(level, message):
    """One semantic alert banner. `message` may use **bold** markdown-style emphasis."""
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _html_escape(message))
    return f'<div class="alert alert-{level}">{escaped}</div>'


def build_status_html(tiles=None, alerts=()):
    """Compose the status panel: an optional KPI tile row plus stacked alert banners."""
    parts = []
    if tiles:
        elapsed, pages, items, avg_score = tiles
        if avg_score >= 0.85:
            conf_accent = "accent-good"
        elif avg_score >= 0.6:
            conf_accent = "accent-warn"
        else:
            conf_accent = "accent-bad"
        parts.append(
            '<div class="stat-grid">'
            f'<div class="stat-tile"><span class="stat-icon">⏱️</span>'
            f'<span class="stat-value">{elapsed:.1f}<small> ث</small></span>'
            '<span class="stat-label">زمن المعالجة</span></div>'
            f'<div class="stat-tile"><span class="stat-icon">📄</span>'
            f'<span class="stat-value">{pages}</span>'
            '<span class="stat-label">الصفحات</span></div>'
            f'<div class="stat-tile"><span class="stat-icon">🔤</span>'
            f'<span class="stat-value">{items}</span>'
            '<span class="stat-label">عنصر نصي</span></div>'
            f'<div class="stat-tile {conf_accent}"><span class="stat-icon">📊</span>'
            f'<span class="stat-value">{avg_score * 100:.0f}<small> %</small></span>'
            '<span class="stat-label">متوسط الثقة</span></div>'
            "</div>"
        )
    for level, message in alerts:
        parts.append(alert_html(level, message))
    return "".join(parts)


def build_review_rows(table_rows):
    """Editable review queue: only the lines that need a human look, worst first.

    Returns (rows, summary_html). Each row keeps the master `#` id so an edit here can be
    merged back into the full result set regardless of filtering or sort order."""
    flagged = []
    for row_id, text, score, page_no, _ in table_rows:
        if score < CONFIDENCE_VERY_LOW:
            flag = "🔴"
        elif score < CONFIDENCE_LOW:
            flag = "🟠"
        else:
            continue
        flagged.append([row_id, flag, text, round(score, 3), page_no, suggest_correction(text) or ""])

    total = len(table_rows)
    if not total:
        return [], alert_html("info", "لا توجد نتائج بعد.")
    if not flagged:
        return [], alert_html(
            "info",
            f"✅ **لا شيء يحتاج مراجعة.** جميع العناصر الـ {total} تجاوزت حد الثقة "
            f"{int(CONFIDENCE_LOW * 100)}%.",
        )

    flagged.sort(key=lambda item: item[3])
    very_low = sum(1 for item in flagged if item[1] == "🔴")
    summary = alert_html(
        "warning",
        f"⚠️ **{len(flagged)} عنصر يحتاج مراجعة** من أصل {total} "
        f"({len(flagged) / total:.0%}) — منها {very_low} بثقة منخفضة جداً. "
        "الأسوأ أولاً · 🔴 أقل من 50% · 🟠 50–80% · عدّل عمود «النص» مباشرة. "
        "عمود «اقتراح تصحيح» يظهر فقط لأنماط خلط حروف معروفة (علي/على، جامعه/جامعة وأمثالها) "
        "— اقتراح للمراجعة، انسخه إلى عمود النص إن كان صحيحاً.",
    )
    return flagged, summary


def _grid_rows(edited):
    """Normalise whatever a Dataframe hands back into a plain list of rows.

    Gradio passes the grid as a pandas DataFrame (iterating one yields column names, not
    rows, and truth-testing one raises), and as a {"headers", "data"} dict over the API.
    """
    if edited is None:
        return []
    if hasattr(edited, "values") and hasattr(edited, "columns"):  # pandas DataFrame
        return edited.values.tolist()
    if isinstance(edited, dict):
        return edited.get("data") or []
    return list(edited)


def merge_edits(edited, master_rows, text_column, allow_structural=False):
    """Merge an edited grid back into the master rows, keyed on the `#` id in column 0.

    Matching by id rather than by position keeps this correct even when the grid is
    searched, filtered or re-sorted, and when the edit came from the review subset.

    `allow_structural` controls whether the grid's own row count is trusted. The full
    results grid supports it: a row Gradio's dynamic Dataframe lets the user delete really
    is gone, and a row it lets them add really is new. The confidence-review grid is a
    *filtered subset* of the same rows, so a row missing from it only means it cleared the
    confidence threshold — treating that as a deletion would silently erase good text the
    moment it stopped needing review."""
    grid = _grid_rows(edited)
    seen_ids, appended, edits = set(), [], {}
    for row in grid:
        if len(row) <= text_column:
            continue
        try:
            row_id = int(row[0])
        except (TypeError, ValueError):
            if allow_structural:
                appended.append(row)  # a new row added via the grid's own "+" button
            continue
        seen_ids.add(row_id)
        edits[row_id] = "" if row[text_column] is None else str(row[text_column])

    merged, changed = [], 0
    for row in master_rows:
        row_id = row[0]
        if allow_structural and row_id not in seen_ids:
            changed += 1  # removed via the grid's own "-" button
            continue
        row = list(row)
        new_text = edits.get(row_id)
        if new_text is not None and new_text != row[1]:
            row[1] = new_text
            changed += 1
        merged.append(row)

    if allow_structural and appended:
        next_id = max((row[0] for row in master_rows), default=0) + 1
        for row in appended:
            text = "" if len(row) <= text_column or row[text_column] is None else str(row[text_column])
            if not text.strip():
                continue
            # a brand-new line has no position on the source page, so it is plain text —
            # it appears in the flowing Word export but not in the faithful-layout view
            page = merged[-1][3] if merged else 1
            source = merged[-1][4] if merged else "OCR"
            merged.append([next_id, text, 1.0, page, source])
            next_id += 1
            changed += 1
    return merged, changed


def write_exports(table_rows, page_sources, page_sizes, run_dir, layout=None):
    """(Re)write the txt/json/docx exports so downloads always match what is on screen.

    Two Word files are written, because no single one can be both: the editable document
    flows like any other file but does not reproduce positions, while the faithful one
    pins every block to its exact spot and can only do that with floating frames."""
    pages = {}
    for _, text, _, page_no, _ in table_rows:
        pages.setdefault(page_no, []).append(text)

    text_lines = []
    payload = []
    multi = len(pages) > 1
    for page_no in sorted(pages):
        source = page_sources.get(page_no, "OCR")
        if multi:
            text_lines.append(f"--- صفحة {page_no} ({source}) ---")
        text_lines.extend(pages[page_no])
        payload.append({"page": page_no, "source": source, "lines": pages[page_no]})

    text_output = "\n".join(text_lines)
    json_output = json.dumps(payload, ensure_ascii=False, indent=2)

    txt_path = os.path.join(run_dir, "extracted_text.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_output)
    json_path = os.path.join(run_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_output)
    docx_path = layout_docx_path = None
    if table_rows:
        docx_path = build_docx(table_rows, page_sizes, run_dir, layout)
        if layout:
            layout_docx_path = build_layout_docx(table_rows, layout, page_sizes, run_dir)
    return json_output, txt_path, json_path, docx_path, layout_docx_path


def build_layout_html(table_rows, layout, only_page=None):
    """Redraw the pages with every block at its original position on the page.

    Positions, sizes and font sizes are emitted as percentages / container units rather
    than pixels, so one markup fits any pane width: the page keeps the exact aspect ratio
    of the source (Letter, A4, or an image's own shape) and everything scales with it.
    """
    if not table_rows or not layout:
        return ""

    texts = {row[0]: row[1] for row in table_rows}
    page_of = {row[0]: row[3] for row in table_rows}

    chunks = ['<div class="doc-view">']
    for page_no in sorted(layout):
        if only_page is not None and int(page_no) != int(only_page):
            continue
        page = layout[page_no]
        width = float(page.get("width") or 1)
        height = float(page.get("height") or 1)
        # width is capped both by the pane and by the height budget, so a landscape page
        # can never be squeezed into a portrait box (which silently breaks the ratio)
        max_px = max(80, round(SHEET_MAX_HEIGHT * width / height))
        chunks.append(
            f'<div class="doc-sheet" style="aspect-ratio:{width:.3f}/{height:.3f};'
            f'width:min(100%,{max_px}px)">'
        )
        # shapes first: they sit behind the text, exactly as on the source page
        for shape in page.get("shapes", []):
            x0, y0, x1, y1 = shape["box"]
            style = [
                f"right:{max(0.0, (width - x1) / width * 100):.3f}%",
                f"top:{max(0.0, y0 / height * 100):.3f}%",
                f"width:{max(0.0, (x1 - x0) / width * 100):.3f}%",
                f"height:{max(0.0, (y1 - y0) / height * 100):.3f}%",
            ]
            fill = _css_colour(shape.get("fill"), shape.get("fill_alpha", 1.0))
            stroke = _css_colour(shape.get("stroke"), shape.get("stroke_alpha", 1.0))
            if fill:
                style.append(f"background:{fill}")
            if stroke and shape.get("width"):
                thickness = max(shape["width"] / width * 100, 0.05)
                style.append(f"border:{thickness:.3f}cqw solid {stroke}")
            if shape.get("radius"):
                style.append(f"border-radius:{shape['radius'] / width * 100:.3f}cqw")
            chunks.append(f'<div class="doc-shape" style="{";".join(style)}"></div>')

        # figures next: a diagram sits above panel fills but still under any leftover
        # text (there should be none inside it, but a caption beside it must stay legible)
        for figure in page.get("figures", []):
            x0, y0, x1, y1 = figure["box"]
            style = (
                f"right:{max(0.0, (width - x1) / width * 100):.3f}%;"
                f"top:{max(0.0, y0 / height * 100):.3f}%;"
                f"width:{max(0.0, (x1 - x0) / width * 100):.3f}%;"
                f"height:{max(0.0, (y1 - y0) / height * 100):.3f}%"
            )
            src = figure.get("png_b64")
            if src:
                chunks.append(
                    f'<img class="doc-figure" style="{style}" '
                    f'src="data:image/png;base64,{src}" alt="">'
                )

        for block in page.get("blocks", []):
            row_id = block.get("id")
            text = texts.get(row_id)
            if text is None or int(page_of.get(row_id, page_no)) != int(page_no):
                continue
            x0, y0, x1, y1 = block["box"]
            right = max(0.0, (width - x1) / width * 100)
            top = max(0.0, y0 / height * 100)
            box_width = max(1.0, (x1 - x0) / width * 100)
            # 1cqw == 1% of the sheet's width, so font size tracks the page as it scales
            font = block.get("size", 11.0) / width * 100
            weight = "700" if block.get("bold") else "400"
            colour = block.get("colour")
            tint = f";color:{colour}" if colour else ""
            align = block.get("align") or "start"
            # a block the source drew on one line must stay on one line: a wider fallback
            # font would otherwise wrap it, and the extra line would land on the block below
            wrap = ";white-space:nowrap" if block.get("lines", 1) <= 1 else ""
            chunks.append(
                f'<div class="doc-block" style="right:{right:.3f}%;top:{top:.3f}%;'
                f'width:{box_width:.3f}%;font-size:{font:.3f}cqw;font-weight:{weight};'
                f'text-align:{align}{wrap}{tint}">'
                f"{_html_escape(text)}</div>"
            )
        chunks.append("</div>")
    chunks.append("</div>")
    return "".join(chunks)


def build_pages_html(table_rows, only_page=None):
    """Render the extracted text as Word-like pages.

    Each line uses `unicode-bidi: plaintext`, so a line that starts with Arabic reads
    right-to-left and one that starts with Latin reads left-to-right — the correct
    behaviour for mixed Arabic/English documents. `only_page` limits the output to a
    single page for the paired image/text review view."""
    if not table_rows:
        return ""

    pages = {}
    for _, text, _, page_no, _source in table_rows:
        if only_page is not None and page_no != only_page:
            continue
        pages.setdefault(page_no, []).append(text)

    chunks = ['<div class="doc-view">']
    for page_no in sorted(pages):
        chunks.append('<article class="doc-page">')
        chunks.append(f'<header class="doc-page-head"><span>صفحة {page_no}</span></header>')
        for line in pages[page_no]:
            css_class = "doc-line doc-heading" if _HEADING_RE.match(line) else "doc-line"
            chunks.append(f'<p class="{css_class}">{_html_escape(line)}</p>')
        chunks.append("</article>")
    chunks.append("</div>")
    return "".join(chunks)


def _set_rtl(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    p_pr.append(bidi)


EMU_PER_POINT = 12700
_WPS_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _anchor_shell(shape_id, name, left_pt, top_pt, width_pt, height_pt, graphic_xml, behind_doc=False):
    """The floating-frame wrapper every positioned object shares.

    python-docx has no API for absolute positioning at all, so a text box, a shape and a
    picture are all built by hand as this same `<wp:anchor>`; only the `<a:graphic>`
    payload inside it differs between them."""
    cx, cy = int(width_pt * EMU_PER_POINT), int(height_pt * EMU_PER_POINT)
    return (
        f'<w:r xmlns:w="{nsmap["w"]}" xmlns:wp="{_WP_NS}" xmlns:a="{_A_NS}" '
        f'xmlns:wps="{_WPS_NS}" xmlns:pic="{_PIC_NS}" xmlns:r="{_R_NS}">'
        "<w:drawing>"
        '<wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0" '
        f'relativeHeight="{shape_id}" behindDoc="{1 if behind_doc else 0}" locked="0" '
        'layoutInCell="1" allowOverlap="1">'
        '<wp:simplePos x="0" y="0"/>'
        '<wp:positionH relativeFrom="page">'
        f"<wp:posOffset>{int(left_pt * EMU_PER_POINT)}</wp:posOffset></wp:positionH>"
        '<wp:positionV relativeFrom="page">'
        f"<wp:posOffset>{int(top_pt * EMU_PER_POINT)}</wp:posOffset></wp:positionV>"
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:wrapNone/>'
        f'<wp:docPr id="{shape_id}" name="{name} {shape_id}"/>'
        f"{graphic_xml}"
        "</wp:anchor></w:drawing></w:r>"
    )


def _textbox_anchor_xml(
    shape_id, text, left_pt, top_pt, width_pt, height_pt, size_pt, bold, colour=None,
    align="right", wrap=True,
):
    """A floating, transparent Word text box placed at an absolute spot on the page.

    python-docx has no API for positioned frames, so the DrawingML anchor is built by
    hand; it is what Word itself writes for a free-floating text box."""
    # Word reads w:jc as leading/trailing, not left/right: inside a right-to-left
    # paragraph "left" means the leading edge, which is the right of the page. So the
    # side the block was aligned to is translated through the paragraph's own direction.
    rtl = _is_rtl(text)
    if align == "center":
        justify = "center"
    else:
        justify = "left" if (align == "right") == rtl else "right"
    body = (
        f'<w:p xmlns:w="{nsmap["w"]}">'
        f'<w:pPr>{"<w:bidi/>" if rtl else ""}'
        '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
        f'<w:jc w:val="{justify}"/></w:pPr>'
        "<w:r><w:rPr>"
        '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>'
        f'<w:sz w:val="{max(2, int(round(size_pt * 2)))}"/>'
        f'<w:szCs w:val="{max(2, int(round(size_pt * 2)))}"/>'
        f'{"<w:b/><w:bCs/>" if bold else ""}'
        f'{f"<w:color w:val={chr(34)}{colour.lstrip(chr(35)).upper()}{chr(34)}/>" if colour else ""}'
        f'{"<w:rtl/>" if rtl else ""}'
        "</w:rPr>"
        f'<w:t xml:space="preserve">{_html_escape(text)}</w:t>'
        "</w:r></w:p>"
    )
    cx, cy = int(width_pt * EMU_PER_POINT), int(height_pt * EMU_PER_POINT)
    graphic = (
        "<a:graphic><a:graphicData "
        'uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
        '<wps:wsp><wps:cNvSpPr txBox="1"/><wps:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
        "<a:ln><a:noFill/></a:ln></wps:spPr>"
        f"<wps:txbx><w:txbxContent>{body}</w:txbxContent></wps:txbx>"
        '<wps:bodyPr rot="0" spcFirstLastPara="0" vertOverflow="overflow" '
        'horzOverflow="overflow" vert="horz" '
        f'wrap="{"square" if wrap else "none"}" lIns="0" tIns="0" rIns="0" '
        # the box grows or shrinks with its own text now, instead of a fixed size frozen
        # at export time — editing a line (typing more, or less) no longer clips it or
        # leaves dead space; a taller box may now overlap whatever sits below it on the
        # page, which is the same trade every text box in Word makes when it grows
        'bIns="0" numCol="1" anchor="t" upright="0"><a:spAutoFit/></wps:bodyPr>'
        "</wps:wsp></a:graphicData></a:graphic>"
    )
    return _anchor_shell(shape_id, "TextBox", left_pt, top_pt, width_pt, height_pt, graphic)


def _ooxml_fill(rgb, alpha):
    """A DrawingML solid fill, with alpha only when the source was actually translucent."""
    if not rgb:
        return "<a:noFill/>"
    r, g, b = rgb[:3]
    body = ""
    if alpha is not None and alpha < 0.99:
        body = f'<a:alpha val="{int(max(0.0, min(1.0, alpha)) * 100000)}"/>'
    return f'<a:solidFill><a:srgbClr val="{r:02X}{g:02X}{b:02X}">{body}</a:srgbClr></a:solidFill>'


def _shape_anchor_xml(shape_id, shape, left_pt, top_pt, width_pt, height_pt, radius_pt):
    """A floating rectangle behind the text — the panels, cards and rules of the page.

    `behindDoc` keeps it under the text boxes; a rounded corner becomes a roundRect whose
    adjustment is expressed as a fraction of half the shorter side, as OOXML expects."""
    if radius_pt and min(width_pt, height_pt) > 0:
        adj = int(min(0.5, radius_pt / (min(width_pt, height_pt) / 2)) * 50000)
        geometry = f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val {adj}"/></a:avLst></a:prstGeom>'
    else:
        geometry = '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'

    stroke = shape.get("stroke")
    if stroke and shape.get("width"):
        line = (
            f'<a:ln w="{max(1, int(shape["width"] * EMU_PER_POINT))}">'
            f'{_ooxml_fill(stroke, shape.get("stroke_alpha", 1.0))}</a:ln>'
        )
    else:
        line = "<a:ln><a:noFill/></a:ln>"

    cx, cy = int(width_pt * EMU_PER_POINT), int(height_pt * EMU_PER_POINT)
    graphic = (
        "<a:graphic><a:graphicData "
        'uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
        "<wps:wsp><wps:cNvSpPr/><wps:spPr>"
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f"{geometry}"
        f'{_ooxml_fill(shape.get("fill"), shape.get("fill_alpha", 1.0))}'
        f"{line}</wps:spPr>"
        '<wps:bodyPr rot="0" vert="horz" wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" '
        'anchor="t"><a:noAutofit/></wps:bodyPr>'
        "</wps:wsp></a:graphicData></a:graphic>"
    )
    return _anchor_shell(shape_id, "Shape", left_pt, top_pt, width_pt, height_pt, graphic, behind_doc=True)


def _picture_anchor_xml(document, shape_id, png_bytes, left_pt, top_pt, width_pt, height_pt):
    """A floating picture at an absolute spot — a cropped diagram/figure from a scan.

    python-docx's own `add_picture` only ever produces an inline `<wp:inline>` picture
    (`StoryPart.new_pic_inline`); there is no positioned counterpart anywhere in the
    library. The image still has to become a real part of the .docx package for Word to
    find it, so `document.part.get_or_add_image()` is used for that (it embeds the bytes
    and returns a relationship id), and the `<pic:pic>` that references that id is then
    built by hand into the same anchor shell as the text boxes and shapes above."""
    rid, _ = document.part.get_or_add_image(io.BytesIO(png_bytes))
    cx, cy = int(width_pt * EMU_PER_POINT), int(height_pt * EMU_PER_POINT)
    graphic = (
        f'<a:graphic><a:graphicData uri="{_PIC_NS}">'
        '<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="Figure"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic>"
    )
    return _anchor_shell(shape_id, "Figure", left_pt, top_pt, width_pt, height_pt, graphic)


def build_layout_docx(table_rows, layout, page_sizes, run_dir):
    """Export a Word file that reproduces the source layout, block by block.

    Every block becomes a floating text box at its original spot. Coordinates arrive in
    whatever unit the page was measured in (PDF points, or image pixels for OCR), so each
    page is scaled by its own factor onto a real paper size."""
    if not layout:
        return None

    texts = {row[0]: row[1] for row in table_rows}
    order = {}
    for row in table_rows:
        order.setdefault(row[3], []).append(row[0])
    document = Document()
    section = document.sections[0]
    section.left_margin = section.right_margin = Pt(0)
    section.top_margin = section.bottom_margin = Pt(0)

    pages = sorted(layout)
    first = layout[pages[0]]
    # keep the source page size when it is real paper (PDF points); otherwise fit the
    # image's aspect ratio onto Letter width so the result is still a printable page
    if page_sizes:
        page_w, page_h = page_sizes[0]
    else:
        ratio = float(first.get("height") or 1) / float(first.get("width") or 1)
        page_w, page_h = 612.0, 612.0 * ratio
    section.page_width = Pt(page_w)
    section.page_height = Pt(page_h)

    shape_id = 1000
    for index, page_no in enumerate(pages):
        page = layout[page_no]
        unit_w = float(page.get("width") or 1)
        unit_h = float(page.get("height") or 1)
        scale_x = page_w / unit_w
        scale_y = page_h / unit_h

        holder = document.add_paragraph()
        holder.paragraph_format.space_before = Pt(0)
        holder.paragraph_format.space_after = Pt(0)

        # shapes first so they land behind the text, matching the source page
        for shape in page.get("shapes", []):
            x0, y0, x1, y1 = shape["box"]
            shape_id += 1
            holder._p.append(
                parse_xml(
                    _shape_anchor_xml(
                        shape_id,
                        shape,
                        left_pt=x0 * scale_x,
                        top_pt=y0 * scale_y,
                        width_pt=max(0.5, (x1 - x0) * scale_x),
                        height_pt=max(0.5, (y1 - y0) * scale_y),
                        radius_pt=shape.get("radius", 0) * scale_x,
                    )
                )
            )

        # figures sit above the panel fills, at the same spot they were cropped from
        for figure in page.get("figures", []):
            x0, y0, x1, y1 = figure["box"]
            png_b64 = figure.get("png_b64")
            if not png_b64:
                continue
            shape_id += 1
            holder._p.append(
                parse_xml(
                    _picture_anchor_xml(
                        document,
                        shape_id,
                        base64.b64decode(png_b64),
                        left_pt=x0 * scale_x,
                        top_pt=y0 * scale_y,
                        width_pt=max(0.5, (x1 - x0) * scale_x),
                        height_pt=max(0.5, (y1 - y0) * scale_y),
                    )
                )
            )

        # joined the same way as the editable export: a line that filled its column and
        # is followed immediately by another in the same style is one paragraph that
        # wrapped in the source, not two independent lines — merging them here means
        # editing one no longer leaves a second, orphaned box sitting where it used to be
        for block in _page_blocks(layout, page_no, order.get(page_no, []), texts):
            text = " ".join(t for t in (texts.get(i) for i in block["ids"]) if t)
            if not text.strip():
                continue
            x0, y0, x1, y1 = block["box"]
            shape_id += 1
            holder._p.append(
                parse_xml(
                    _textbox_anchor_xml(
                        shape_id,
                        text,
                        left_pt=x0 * scale_x,
                        top_pt=y0 * scale_y,
                        # the box already carries the free space beside the block, so no
                        # extra slack here: it would drag the anchored edge off position
                        width_pt=max(8.0, (x1 - x0) * scale_x),
                        height_pt=max(6.0, (y1 - y0) * scale_y * 1.25),
                        size_pt=max(4.0, block.get("size", 11.0) * scale_y),
                        bold=bool(block.get("bold")),
                        colour=block.get("colour"),
                        align=block.get("align") or "right",
                        wrap=block.get("lines", 1) > 1,
                    )
                )
            )
        if index < len(pages) - 1:
            document.add_page_break()

    path = os.path.join(run_dir, "extracted_layout.docx")
    document.save(path)
    return path


# Two consecutive blocks belong to the same paragraph when the first one filled its line
# and the second follows immediately underneath in the same style.
PARA_FILL_RATIO = 0.8
PARA_MAX_GAP = 0.9      # x line height


def _set_rtl_table(table):
    """Mirror a table so its first column is the rightmost one, as Arabic reads."""
    table._tbl.tblPr.append(OxmlElement("w:bidiVisual"))


def _cluster(values, tolerance):
    """Group nearby coordinates and return the centre of each group, in order."""
    groups = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _nearest(centres, value):
    return min(range(len(centres)), key=lambda i: abs(centres[i] - value))


def _style_run(run, size_pt, bold, colour, rtl):
    run.font.size = Pt(max(4.0, size_pt))
    run.font.bold = bool(bold)
    run.font.name = "Arial"
    if colour:
        try:
            run.font.color.rgb = RGBColor.from_string(colour.lstrip("#").upper())
        except ValueError:
            pass
    if rtl:
        run._r.get_or_add_rPr().append(OxmlElement("w:rtl"))


INDENT_STEP = 9.0        # points: half a default tab, the finest indent worth keeping


def _snap_indent(value, limit):
    """An indent rounded to a usable step and never wider than the text it introduces."""
    return min(max(0.0, round(value / INDENT_STEP) * INDENT_STEP), limit)


def _write_paragraph(document, block, texts, scale, margin_right, margin_left, indent_limit):
    """One block as an ordinary Word paragraph, keeping its typography and its indent."""
    text = " ".join(t for t in (texts.get(i) for i in block["ids"]) if t)
    if not text.strip():
        return
    rtl = _is_rtl(text)
    paragraph = document.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(3)
    align = block.get("align") or ("right" if rtl else "left")
    paragraph.alignment = {
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
    }[align]
    if rtl:
        _set_rtl(paragraph)
    # the indent a bullet or a quoted block had on the page, measured from the margin.
    # It is snapped to a step and capped: on a photographed page every line starts a few
    # pixels off its neighbour, and carrying that noise through gives a ragged document.
    x0, _, x1, _ = block["box"]
    if align != "center":
        fmt.right_indent = Pt(_snap_indent((margin_right - x1) * scale, indent_limit))
        fmt.left_indent = Pt(_snap_indent((x0 - margin_left) * scale, indent_limit))
    _style_run(
        paragraph.add_run(text),
        block["size"] * scale,
        block.get("bold"),
        block.get("colour"),
        rtl,
    )


def _write_table(document, cells, texts, scale):
    """A run of table cells as a real Word table, with its own rows, columns and spans.

    When every cell carries the structure model's own row/column/span (see
    `parse_table_grid`), that is trusted outright. Otherwise — a PDF-embedded table, or
    a scan where the token sequence could not be parsed reliably — rows and columns are
    found from the edges cells share, exactly as before; a two-line cell is taller than
    its neighbour, so clustering on centres would not line them up."""
    has_grid = all(c.get("row") is not None and c.get("col") is not None for c in cells)
    if has_grid:
        row_count = max(c["row"] + (c.get("rowspan") or 1) for c in cells)
        col_count = max(c["col"] + (c.get("colspan") or 1) for c in cells)
    else:
        heights = sorted(c["box"][3] - c["box"][1] for c in cells) or [10.0]
        widths = sorted(c["box"][2] - c["box"][0] for c in cells) or [10.0]
        row_centres = _cluster(
            [c["box"][1] for c in cells], max(2.0, heights[len(heights) // 2] * 0.4)
        )
        col_centres = _cluster(
            [c["box"][0] for c in cells], max(2.0, widths[len(widths) // 2] * 0.4)
        )
        if not row_centres or not col_centres:
            return
        row_count, col_count = len(row_centres), len(col_centres)

    body = " ".join(texts.get(i, "") for c in cells for i in c["ids"])
    rtl = _is_rtl(body)
    table = document.add_table(rows=row_count, cols=col_count)
    table.style = "Table Grid"
    table.autofit = False
    if rtl:
        _set_rtl_table(table)

    widths = [0.0] * col_count
    spans = []  # (top_left_row, top_left_col, bottom_right_row, bottom_right_col)
    for cell in cells:
        if has_grid:
            row, col = cell["row"], cell["col"]
            rowspan, colspan = cell.get("rowspan") or 1, cell.get("colspan") or 1
        else:
            row = _nearest(row_centres, cell["box"][1])
            col = _nearest(col_centres, cell["box"][0])
            rowspan = colspan = 1
        # a mirrored table numbers its columns from the right; a span's near and far
        # edges swap which one lands at the numerically smaller (Word "top-left") index
        if rtl:
            col0, col1 = col_count - colspan - col, col_count - 1 - col
        else:
            col0, col1 = col, col + colspan - 1
        if rowspan > 1 or colspan > 1:
            spans.append((row, col0, row + rowspan - 1, col1))
        width_share = (cell["box"][2] - cell["box"][0]) * scale / max(1, colspan)
        for column in range(col0, col1 + 1):
            widths[column] = max(widths[column], width_share)
        target = table.cell(row, col0)
        text = " ".join(t for t in (texts.get(i) for i in cell["ids"]) if t)
        if not text.strip():
            continue
        paragraph = target.paragraphs[0]
        if paragraph.text:
            paragraph = target.add_paragraph()
        cell_rtl = _is_rtl(text)
        paragraph.alignment = (
            WD_ALIGN_PARAGRAPH.RIGHT if cell_rtl else WD_ALIGN_PARAGRAPH.LEFT
        )
        if cell_rtl:
            _set_rtl(paragraph)
        _style_run(
            paragraph.add_run(text),
            cell["size"] * scale,
            cell.get("bold"),
            cell.get("colour"),
            cell_rtl,
        )

    for row0, col0, row1, col1 in spans:
        try:
            table.cell(row0, col0).merge(table.cell(row1, col1))
        except Exception:  # noqa: BLE001 - a malformed span must not abort the whole export
            traceback.print_exc()

    # column widths are set last: python-docx only sums a merge's width when both sides
    # already have one, so setting them before merging would silently lose the total
    for index, width in enumerate(widths):
        if width > 0:
            for row in table.rows:
                row.cells[index].width = Pt(width)


# A bullet, a dash or a numbered heading always opens a new paragraph, however well the
# line above it filled its column.
_STARTS_ITEM_RE = re.compile(r"^\s*(?:[\u2022\u25aa\u25e6\-\u2013\u2014*]|\d+[.)]|\d+\.\d)")


def _page_blocks(layout, page_no, order, texts=None):
    """The page's blocks in reading order, with the lines of a paragraph joined.

    Each visual line of a paragraph arrives as its own block. A Word document that keeps
    them apart cannot reflow when edited, so a line that filled its column is joined to
    the one directly beneath it in the same style - which is where the paragraph broke."""
    texts = texts or {}

    def texts_of(block):
        return texts.get(block.get("id"), "")

    page = layout.get(page_no) or layout.get(str(page_no)) or {}
    by_id = {b.get("id"): b for b in page.get("blocks", [])}
    blocks = [by_id[i] for i in order if i in by_id]
    if not blocks:
        return []

    column = max(b["box"][2] for b in blocks) - min(b["box"][0] for b in blocks)
    joined = []
    for block in blocks:
        merged = dict(block, ids=[block["id"]])
        previous = joined[-1] if joined else None
        starts_item = bool(_STARTS_ITEM_RE.match(texts_of(block)))
        if (
            previous is not None
            and not previous.get("fixed")
            and not block.get("fixed")
            and not starts_item
        ):
            ink = previous.get("ink") or previous["box"]
            width = ink[2] - ink[0]
            gap = block["box"][1] - previous["box"][3]
            height = max(1.0, previous["box"][3] - previous["box"][1])
            # a PDF's own font size is exact, so two lines 0.6pt apart are already a
            # different size; an OCR line's size is only ever a guess from its box
            # (further narrowed by a width cap that varies with how long the line's own
            # text happens to be) - measured directly on two lines of one paragraph that
            # obviously belong together, that guess alone swung 25% between them. The
            # width-fill and gap checks below are what actually guards against merging
            # unrelated short lines (a bullet word never fills 80% of the column,
            # whatever tolerance is used here), so size only has to rule out a real jump
            # in size - a heading followed by body text - not confirm an exact match.
            size_tolerance = max(0.6, previous["size"] * 0.3)
            same_style = (
                abs(previous["size"] - block["size"]) <= size_tolerance
                and bool(previous.get("bold")) == bool(block.get("bold"))
                and previous.get("colour") == block.get("colour")
                and previous.get("align") == block.get("align")
            )
            if same_style and width >= column * PARA_FILL_RATIO and gap <= height * PARA_MAX_GAP:
                previous["ids"].extend(merged["ids"])
                previous["box"] = (
                    min(previous["box"][0], block["box"][0]),
                    previous["box"][1],
                    max(previous["box"][2], block["box"][2]),
                    block["box"][3],
                )
                # a caller deciding whether to let this box wrap needs the true line
                # count, not just the first line's — otherwise a merged, multi-line
                # paragraph would still be marked `wrap=False` and clip on re-edit
                previous["lines"] = previous.get("lines", 1) + block.get("lines", 1)
                continue
        joined.append(merged)
    return joined


def build_docx(table_rows, page_sizes, run_dir, layout=None):
    """Export an ordinary, editable Word document that keeps the source's typography.

    Unlike the layout export nothing floats: paragraphs flow and tables are real Word
    tables, so the file can be edited like any document. Positions are not reproduced,
    only what carries meaning: reading order, size, weight, colour, alignment, indent."""
    document = Document()
    section = document.sections[0]
    section.left_margin = section.right_margin = Cm(2)
    section.top_margin = section.bottom_margin = Cm(1.5)
    layout = layout or {}
    if page_sizes:
        section.page_width = Pt(page_sizes[0][0])
        section.page_height = Pt(page_sizes[0][1])
    elif layout:
        # an image has no paper size of its own: fit its shape onto Letter width, so the
        # pixel measurements below turn into points a printer can use
        first = layout[sorted(layout)[0]]
        ratio = float(first.get("height") or 1) / float(first.get("width") or 1)
        section.page_width = Pt(612.0)
        section.page_height = Pt(612.0 * ratio)
    pages, order, texts = {}, {}, {}
    for row_id, text, _, page_no, _ in table_rows:
        pages.setdefault(page_no, []).append(text)
        order.setdefault(page_no, []).append(row_id)
        texts[row_id] = text

    for index, page_no in enumerate(sorted(pages)):
        if index > 0:
            document.add_page_break()

        blocks = _page_blocks(layout, page_no, order[page_no], texts)
        if not blocks:
            # no layout for this page: fall back to plain right-aligned lines
            for line in pages[page_no]:
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                _set_rtl(paragraph)
                run = paragraph.add_run(line)
                if _HEADING_RE.match(line):
                    run.bold = True
                    run.font.size = Pt(14)
            continue

        # the page's own text column becomes the document's margins, so every indent
        # measured against it lands where it did on the page
        page_width = float((layout.get(page_no) or layout.get(str(page_no)))["width"])
        paper_width = (
            page_sizes[index][0] if index < len(page_sizes) else section.page_width.pt
        )
        scale = paper_width / page_width
        margin_left = min(b["box"][0] for b in blocks)
        margin_right = max(b["box"][2] for b in blocks)
        indent_limit = (margin_right - margin_left) * scale * 0.25
        if index == 0:
            # a block's own box can land a point or two outside the page itself — OCR's
            # detected edge is an estimate, not a guarantee — and Word's margin attribute
            # is unsigned XML: a negative value here does not clip to zero, it throws
            section.left_margin = Pt(max(0.0, margin_left * scale))
            section.right_margin = Pt(max(0.0, (page_width - margin_right) * scale))

        cells = []
        for block in blocks:
            if block.get("fixed"):
                cells.append(block)
                continue
            if cells:
                _write_table(document, cells, texts, scale)
                cells = []
            _write_paragraph(
                document, block, texts, scale, margin_right, margin_left, indent_limit
            )
        if cells:
            _write_table(document, cells, texts, scale)

    path = os.path.join(run_dir, "extracted.docx")
    document.save(path)
    return path


LANG_CHOICES = [
    ("العربية + الإنجليزية معاً (مستندات مختلطة)", "ar+en"),
    (" (صيني/إنجليزي/ياباني ولغات لاتينية)", "ch"),
    ("العربية", "ar"),
    ("الإنجليزية", "en"),
    ("الفرنسية", "fr"),
    ("الروسية", "ru"),
    ("اليابانية", "japan"),
    ("الكورية", "korean"),
]

PDF_MODE_CHOICES = [
    ("تلقائي: استخدم النص المضمّن إن توفر، وإلا OCR (موصى به)", "auto"),
    ("استخراج النص المضمّن فقط (بدون OCR إطلاقاً)", "extract_only"),
    ("OCR فقط (تجاهل النص المضمّن حتى لو وُجد)", "ocr_only"),
]

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"]

MIN_EMBEDDED_CHARS = 10  # non-whitespace chars needed to treat a PDF page as "digital text"
PDF_RENDER_SCALE = 200 / 72  # ~200 DPI

# Measured on this setup: Arabic recognition holds at ~100% down to ~18px line height,
# drops to ~93% at 12px and collapses to ~67% at 8px, because the dots that distinguish
# ن/ت/ب/ي stop being resolvable in the pixels.
MIN_RELIABLE_TEXT_HEIGHT = 18

# Height budget for a reconstructed sheet, matched to the source-image pane beside it.
SHEET_MAX_HEIGHT = 600

OUTPUT_DIR = "ocr_webui_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_ocr_instances = {}
_table_instances = {}
_figure_instances = {}

# Two small models (about 15 MB together) rather than the full PP-StructureV3 stack:
# one finds the table on the page, the other reads its grid. Everything else — the text
# itself — we already have from OCR, so there is nothing to run twice.
TABLE_LAYOUT_MODEL = "PicoDet_layout_1x_table"
TABLE_STRUCTURE_MODEL = "SLANet_plus"
TABLE_MIN_SCORE = 0.5      # below this a "table" is usually a framed paragraph
TABLE_CELL_MARGIN = 3      # pixels of air kept inside a cell's ruling

# The "L" tier of the same general-layout family, 125 MB against the "S" tier's 5 MB.
# Upgraded after the "S" model demonstrably missed a real diagram on a noisy phone-photo
# exam page (it labelled the region "text"), and a tiled multi-pass workaround for "S"
# still only produced an imprecise, oversized box — see FIGURE_MAX_PAGE_SHARE below,
# which stayed in place as a safety net regardless of which tier is loaded here.
FIGURE_LAYOUT_MODEL = "PP-DocLayout-L"
FIGURE_LABELS = ("image", "chart")
FIGURE_MIN_SCORE = 0.5

# --- الهوية البصرية: أخضر/أبيض/أسود العلم السوري، مع الأحمر لعناصر التنبيه/النجوم ---
BRAND_GREEN = "#148A4B"
BRAND_GREEN_DARK = "#0B5A2C"
BRAND_BLACK = "#111412"
BRAND_RED = "#CE1126"

SYRIAN_GREEN = gr.themes.Color(
    name="syrian_green",
    c50="#EAF7EF",
    c100="#CBEBD8",
    c200="#9DD8B6",
    c300="#6CC393",
    c400="#3DAD72",
    c500=BRAND_GREEN,
    c600="#0F7238",
    c700=BRAND_GREEN_DARK,
    c800="#083F1F",
    c900="#062C16",
    c950="#03170B",
)

THEME = gr.themes.Soft(
    primary_hue=SYRIAN_GREEN,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Cairo"), "Segoe UI", "Tahoma", "sans-serif"],
)

# Structural + brand CSS. Surface colors stay on Gradio's theme variables (so the page
# still adapts to light/dark); the flag greens/red are used only as accents (header,
# primary actions, semantic alerts) rather than repainting every surface.
CUSTOM_CSS = f"""
.gradio-container {{ direction: rtl; max-width: 1500px !important; }}

/* ===================== الترويسة والهوية ===================== */
#app-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    padding: 14px 4px 16px;
    border-bottom: 1px solid var(--border-color-primary);
}}
.brand {{ display: flex; align-items: center; gap: 12px; }}
.brand-mark {{ flex-shrink: 0; border-radius: 7px; box-shadow: 0 0 0 1px var(--border-color-primary); }}
.brand-text .title {{ font-size: 1.35rem; font-weight: 800; line-height: 1.3; }}
.brand-text .subtitle {{ margin-top: 2px; font-size: .85rem; color: var(--body-text-color-subdued); }}
.header-badge {{
    font-size: .78rem;
    font-weight: 600;
    padding: 7px 14px;
    border-radius: 999px;
    white-space: nowrap;
    background: rgba(20, 138, 75, .12);
    color: {BRAND_GREEN};
    border: 1px solid rgba(20, 138, 75, .35);
}}
.flag-stripe {{
    height: 3px;
    width: 100%;
    margin-bottom: 18px;
    border-radius: 999px;
    background: linear-gradient(90deg,
        {BRAND_GREEN} 0 33.33%, #ffffff 33.33% 66.66%, {BRAND_BLACK} 66.66% 100%);
    /* keeps the white middle band visible on a light page background */
    box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .12);
}}

/* ===================== تسميات الحقول (Gradio) ===================== */
/* Gradio's Soft theme renders every field label as a solid primary-colored chip and
   hardcodes dir="ltr" on it — harmless with the default indigo theme tucked in a closed
   accordion, but glaring once the primary hue is brand green and the label is Arabic and
   visible by default. Restore a plain, RTL-correct label instead. */
[data-testid="block-info"] {{
    background: transparent !important;
    border: none !important;
    color: var(--body-text-color-subdued) !important;
    padding: 0 2px !important;
    font-weight: 600 !important;
    direction: rtl !important;
    text-align: right !important;
    white-space: normal !important;
}}

/* ===================== لوحة التحكم ===================== */
.section-title {{
    font-weight: 700;
    font-size: .82rem;
    color: var(--body-text-color-subdued);
    letter-spacing: .02em;
    padding: 2px 4px;
}}
.section-title p {{ margin: 0; }}
.section-title.with-gap {{ margin-top: 14px; }}

/* ===================== لوحة الإحصاءات (بطاقات KPI) ===================== */
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}}
.stat-tile {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    padding: 14px 8px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-top: 3px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    text-align: center;
}}
.stat-tile.accent-good {{ border-top-color: {BRAND_GREEN}; }}
.stat-tile.accent-warn {{ border-top-color: #f59e0b; }}
.stat-tile.accent-bad {{ border-top-color: {BRAND_RED}; }}
.stat-icon {{ font-size: 1.15rem; }}
.stat-value {{ font-size: 1.4rem; font-weight: 800; color: var(--body-text-color); }}
.stat-value small {{ font-size: .8rem; font-weight: 600; opacity: .75; }}
.stat-label {{ font-size: .74rem; color: var(--body-text-color-subdued); }}

/* ===================== شرائط التنبيه الدلالية ===================== */
.alert {{
    padding: 12px 16px;
    border-radius: var(--radius-lg);
    border: 1px solid;
    font-size: .9rem;
    line-height: 1.75;
    margin-top: 10px;
}}
.alert:first-child {{ margin-top: 0; }}
.alert strong {{ font-weight: 700; }}
.alert p {{ margin: 0; }}
.alert-info {{ background: rgba(20, 138, 75, .08); border-color: rgba(20, 138, 75, .3); }}
.alert-warning {{ background: rgba(245, 158, 11, .1); border-color: rgba(245, 158, 11, .38); }}
.alert-error {{ background: rgba(206, 17, 38, .1); border-color: rgba(206, 17, 38, .4); }}

/* --- results table: RTL text, comfortable rows --- */
#results-table table {{ direction: rtl; }}
#results-table th, #results-table td {{ text-align: right !important; }}
#results-table td {{ line-height: 1.7; }}

/* --- شريط التنقّل بين الصفحات --- */
#page-nav {{
    align-items: center;
    gap: 8px;
    padding: 6px 2px 10px;
    margin-bottom: 6px;
    border-bottom: 1px solid var(--border-color-primary);
}}

/* --- Word-like document view --- */
.doc-view {{
    max-height: 620px;
    overflow-y: auto;
    padding: 6px 4px;
    background: var(--background-fill-secondary);
    border-radius: var(--radius-lg);
}}
.doc-page {{
    max-width: 820px;
    margin: 0 auto 22px;
    /* lighter than a real page margin: the pane sits beside the source image, so the
       room is better spent on text than on white space */
    padding: 26px 30px 32px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-md);
    box-shadow: 0 2px 10px rgba(0, 0, 0, .16);
}}
.doc-page-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: -18px 0 22px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color-primary);
    font-size: .78rem;
    color: var(--body-text-color-subdued);
}}
/* plaintext bidi: each line takes its direction from its own first strong character,
   which is what mixed Arabic/English documents need */
.doc-line {{
    unicode-bidi: plaintext;
    text-align: start;
    margin: 0 0 .85em;
    line-height: 1.85;
    font-size: 1rem;
    color: var(--body-text-color);
}}
.doc-heading {{
    font-weight: 700;
    font-size: 1.1rem;
    margin: 1.4em 0 .6em;
    border-right: 3px solid {BRAND_GREEN};
    padding-right: 10px;
}}

/* --- الصفحة بتخطيطها الأصلي: كل كتلة في موضعها من الورقة --- */
.doc-sheet {{
    position: relative;
    container-type: inline-size;   /* يجعل 1cqw = 1% من عرض الورقة */
    /* العرض يأتي محسوباً في الـ inline style: min(عرض اللوحة، ما يوافق ارتفاع 600px)،
       فتبقى نسبة الورقة صحيحة سواء كانت طولية أو عرضية */
    margin: 0 auto 22px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-md);
    box-shadow: 0 2px 10px rgba(0, 0, 0, .16);
    overflow: hidden;
}}
.doc-shape {{
    position: absolute;
    box-sizing: border-box;
    pointer-events: none;
}}
.doc-figure {{
    position: absolute;
    box-sizing: border-box;
    object-fit: contain;
    pointer-events: none;
}}
.doc-block {{
    position: absolute;
    z-index: 1;   /* always above the shapes it sits on */
    unicode-bidi: plaintext;
    text-align: start;
    line-height: 1.35;
    /* الخط نفسه المستخدم في المصدر يبقي عرض النص قريباً من الأصل */
    font-family: Arial, "Segoe UI", Tahoma, sans-serif;
    color: var(--body-text-color);
}}

/* --- keep machine-readable output left-to-right --- */
#json-view, #json-view * {{ direction: ltr; text-align: left; }}

/* --- تبويبات على شكل حبوب --- */
.tabs > .tab-nav {{ border-bottom: 1px solid var(--border-color-primary); gap: 4px; }}
.tabs > .tab-nav button {{
    border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
    font-weight: 600;
}}
.tabs > .tab-nav button.selected {{
    color: {BRAND_GREEN} !important;
    border-color: {BRAND_GREEN} !important;
}}
"""


def _free_gpu():
    """Drop cached pipelines and release their GPU memory."""
    _ocr_instances.clear()
    _bilingual_instances.clear()
    gc.collect()
    try:
        import paddle

        paddle.device.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort only
        pass


def get_ocr(lang, device="gpu"):
    """Return the pipeline for a language, keeping only one resident at a time.

    Each language loads its own detection + recognition models; caching them all would
    exhaust an 8 GB card after a couple of language switches."""
    key = (lang, device)
    if key not in _ocr_instances:
        if _ocr_instances:
            _free_gpu()
        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
        if lang != "ch":
            kwargs["lang"] = lang
        _ocr_instances[key] = PaddleOCR(**kwargs)
    return _ocr_instances[key]


_STRONG_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9/\-\.]{2,}")
_ARABIC_RE = re.compile(r"[ء-ي]")

_bilingual_instances = {}


def imread_unicode(path):
    """Read an image whose path may contain non-ASCII characters.

    cv2.imread goes through the ANSI filesystem API on Windows and silently returns None
    for any path with Arabic (or other non-Latin) characters — e.g. "لقطة شاشة.png".
    Reading the bytes ourselves and decoding from memory avoids that entirely.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """Write an image to a path that may contain non-ASCII characters."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


_skew_detector_instances = {}

# Below this a page is judged straight enough to leave alone; measured against a real
# photographed exam page, whose lines varied between -7.3 deg and +2.5 deg — well above it.
DESKEW_MIN_ANGLE = 0.8


def get_skew_detector(device="gpu"):
    """The lightweight text detector used only to measure a page's tilt, built once."""
    if device not in _skew_detector_instances:
        _skew_detector_instances[device] = TextDetection(model_name="PP-OCRv5_server_det", device=device)
    return _skew_detector_instances[device]


def _dominant_skew(polys):
    """A page's dominant text-line tilt in degrees, weighted by each line's own width.

    A single long, confidently detected line should outweigh a short one or a stray
    mark, so the estimate is a width-weighted median angle, not a plain average — one
    mis-angled short fragment cannot then drag the whole page's estimate off course."""
    angles, weights = [], []
    for poly in polys:
        quad = np.asarray(poly, dtype=float)
        top_edge = quad[1] - quad[0]
        width = float(np.linalg.norm(top_edge))
        if width < 1:
            continue
        angles.append(float(np.degrees(np.arctan2(top_edge[1], top_edge[0]))))
        weights.append(width)
    if not angles:
        return 0.0
    order = np.argsort(angles)
    angles_sorted = np.array(angles)[order]
    cum_weight = np.cumsum(np.array(weights)[order])
    midpoint = cum_weight[-1] / 2
    return float(angles_sorted[np.searchsorted(cum_weight, midpoint)])


def _rotate_image(image, angle):
    """Rotate a whole page level. Detection runs again afterwards, from scratch, on the
    result — it is itself more accurate on straight text, not just the boxes it finds."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _enhance_contrast(image):
    """Lift local contrast on a washed-out photographed page via CLAHE on lightness only.

    Colour is left alone — CLAHE runs on the L channel of LAB and the a/b channels pass
    through untouched — so this cannot introduce a colour cast; it only pulls faint grey
    text closer to true black."""
    l_channel, a_channel, b_channel = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def preprocess_page_image(image_path, run_dir, device="gpu"):
    """Straighten and, if needed, brighten a photographed page before anything reads it.

    This runs once, at the very top of the page pipeline, rather than inside table
    detection, figure detection and OCR separately — those three all measure boxes
    against whatever pixels they are handed, and if each corrected the page on its own
    they could each land on a slightly different rotation, leaving table cells, figure
    boxes and text boxes measured in three different coordinate spaces on the same page.

    Whether to also lift contrast is decided by the very same measurement, not a second,
    independent threshold: every clean page tried here — PDF pages rendered at 200 DPI,
    a flatbed-quality scan — measured an exact 0.0 deg tilt, while the one genuine phone
    photo on hand measured 1.0 deg. A global contrast statistic (plain std, the p95-p5
    range, even Otsu-foreground darkness) turned out to swing on how much of a page is
    ink rather than on how crisp that ink is, and flagged a hand-made clean synthetic page
    as worse than the real photo — not a reliable trigger. A page skewed enough to
    straighten is, in practice, the same reliable evidence that it was photographed, not
    scanned or rendered, so contrast is only lifted alongside a real skew correction."""
    image = imread_unicode(image_path)
    if image is None:
        return image_path, image

    try:
        polys = next(iter(get_skew_detector(device).predict(image)))["dt_polys"]
    except Exception:  # noqa: BLE001 - skew correction is best-effort, OCR must still run
        polys = []
    angle = _dominant_skew(polys)
    if abs(angle) <= DESKEW_MIN_ANGLE:
        return image_path, image

    image = _enhance_contrast(image)
    image = _rotate_image(image, angle)
    new_path = os.path.join(run_dir, f"preprocessed_{uuid.uuid4().hex[:8]}.png")
    imwrite_unicode(new_path, image)
    return new_path, image


def _crop_quad(img, poly):
    """Perspective-crop one detected text quadrilateral."""
    pts = np.array(poly, dtype="float32")
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    crop = cv2.warpPerspective(
        img, cv2.getPerspectiveTransform(pts, dst), (w, h), borderMode=cv2.BORDER_REPLICATE
    )
    if h / max(w, 1) >= 1.5:  # vertical text line
        crop = np.rot90(crop)
    return crop


def sort_polys_reading_order(polys, rtl=True):
    """Order detected text boxes the way a human reads the page.

    The standalone detector emits boxes in an arbitrary order (in practice roughly
    bottom-to-top), while the full PaddleOCR pipeline sorts them internally. Boxes are
    grouped into rows by vertical overlap, rows run top-to-bottom, and within a row the
    order is right-to-left for Arabic pages.
    """
    if len(polys) == 0:
        return []

    items = []
    for poly in polys:
        pts = np.array(poly, dtype=float)
        items.append(
            {
                "poly": poly,
                "top": float(pts[:, 1].min()),
                "bottom": float(pts[:, 1].max()),
                "left": float(pts[:, 0].min()),
                "right": float(pts[:, 0].max()),
            }
        )

    heights = sorted(it["bottom"] - it["top"] for it in items)
    line_tol = max(heights[len(heights) // 2] * 0.6, 5.0)

    items.sort(key=lambda it: it["top"])
    rows, current = [], [items[0]]
    for it in items[1:]:
        row_center = sum(x["top"] + x["bottom"] for x in current) / (2 * len(current))
        if abs((it["top"] + it["bottom"]) / 2 - row_center) <= line_tol:
            current.append(it)
        else:
            rows.append(current)
            current = [it]
    rows.append(current)

    ordered = []
    for row in rows:
        row.sort(key=lambda it: it["right"] if rtl else it["left"], reverse=rtl)
        ordered.extend(it["poly"] for it in row)
    return ordered


def _merge_bilingual(arabic, latin):
    """Combine the two recognisers' readings of the same text line.

    A line with no real Arabic is taken from the Latin model. A line that is mainly Arabic
    keeps the Arabic reading, but any Latin token the Arabic model failed to read is spliced
    back in — measured on a mixed page this recovers 11/13 technical terms versus 3/13 with
    the Arabic model alone, without hurting the Arabic text.
    """
    texts, scores, kept = [], [], []
    for index, ((ar_text, ar_score), (la_text, la_score)) in enumerate(zip(arabic, latin)):
        has_arabic = len(_ARABIC_RE.findall(ar_text)) >= 2
        latin_tokens = _STRONG_LATIN_RE.findall(la_text)

        if not has_arabic:
            if latin_tokens or la_score >= ar_score:
                text, score = la_text, la_score
            else:
                text, score = ar_text, ar_score
        else:
            text, score = ar_text, ar_score
            missing = [t for t in latin_tokens if t.lower() not in ar_text.lower()]
            if missing and la_score > 0.5:
                text = f"{ar_text} {' '.join(missing)}".strip()

        if text.strip():
            texts.append(text)
            scores.append(score)
            # the index of the detection this line came from, so its box can follow it
            kept.append(index)
    return texts, scores, kept


class BilingualOcr:
    """Detect each text line once, read it with both the Arabic and Latin recognisers."""

    def __init__(self, device):
        self.detector = TextDetection(model_name="PP-OCRv5_server_det", device=device)
        self.rec_arabic = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec", device=device)
        self.rec_latin = TextRecognition(model_name="en_PP-OCRv5_mobile_rec", device=device)

    def predict(self, image_path, run_dir, thresh, walls=()):
        img = imread_unicode(image_path)
        if img is None:
            raise ValueError(f"تعذّرت قراءة الصورة (ملف تالف أو صيغة غير مدعومة): {image_path}")

        height, width = img.shape[0], img.shape[1]
        polys = next(iter(self.detector.predict(image_path)))["dt_polys"]
        if len(polys) == 0:
            return {
                "image": image_path,
                "texts": [],
                "scores": [],
                "median_height": None,
                "boxes": [],
                "dims": (width, height),
            }
        polys = sort_polys_reading_order(polys, rtl=True)

        crops = [_crop_quad(img, p) for p in polys]
        arabic = [(r["rec_text"], float(r["rec_score"])) for r in self.rec_arabic.predict(crops)]
        latin = [(r["rec_text"], float(r["rec_score"])) for r in self.rec_latin.predict(crops)]

        texts, scores, kept = _merge_bilingual(arabic, latin)
        # keep the source detection index alongside each line so the box survives filtering
        keep = [(t, s, k) for t, s, k in zip(texts, scores, kept) if s >= thresh]
        texts = [normalize_mixed_text(t) for t, _, _ in keep]
        scores = [s for _, s, _ in keep]
        boxes = []
        for _, _, index in keep:
            pts = np.array(polys[index], dtype=float)
            boxes.append(
                (
                    float(pts[:, 0].min()),
                    float(pts[:, 1].min()),
                    float(pts[:, 0].max()),
                    float(pts[:, 1].max()),
                )
            )

        annotated = os.path.join(run_dir, "bilingual_ocr_res_img.png")
        marked = img.copy()
        cv2.polylines(marked, [np.array(p, dtype=np.int32) for p in polys], True, (0, 200, 0), 2)
        if not imwrite_unicode(annotated, marked):
            annotated = image_path

        heights = sorted(
            float(np.max(p[:, 1]) - np.min(p[:, 1])) for p in (np.array(x) for x in polys)
        )
        if len(boxes) == len(texts):
            texts, scores, boxes = merge_line_fragments(texts, scores, boxes, walls)
        return {
            "image": annotated,
            "texts": texts,
            "scores": scores,
            "median_height": heights[len(heights) // 2] if heights else None,
            "boxes": boxes,
            "dims": (width, height),
        }


def get_bilingual(device="gpu"):
    if device not in _bilingual_instances:
        _bilingual_instances.clear()
        _bilingual_instances[device] = BilingualOcr(device)
    return _bilingual_instances[device]


class OcrRunner:
    """Runs OCR on the GPU and transparently falls back to CPU if the GPU cannot cope.

    A shared GPU (browser, other apps) can leave too little VRAM for the detector, which
    surfaces as a CUDNN allocation failure. Falling back keeps the job completing —
    slower, but with identical output — instead of failing the whole run."""

    def __init__(self, lang):
        self.lang = lang
        self.device = "gpu"
        self.fell_back = False

    def _once(self, image_path, run_dir, thresh, walls):
        if self.lang == "ar+en":
            return get_bilingual(self.device).predict(image_path, run_dir, thresh, walls)
        return _ocr_single_image(
            get_ocr(self.lang, self.device), image_path, run_dir, thresh, walls
        )

    def run(self, image_path, run_dir, thresh, walls=()):
        try:
            return self._once(image_path, run_dir, thresh, walls)
        except (OSError, RuntimeError, MemoryError, SystemError):
            if self.device == "cpu":
                raise
            self.device = "cpu"
            self.fell_back = True
            _free_gpu()
            return self._once(image_path, run_dir, thresh, walls)


def _median_text_height(res):
    """Median pixel height of the detected text lines, or None if unavailable.

    Arabic letters are distinguished mainly by their dots; below roughly 18px those dots
    are destroyed by the image resolution itself, so recognition degrades badly no matter
    which model is used. Measuring this lets the UI warn before the user trusts the output.
    """
    try:
        boxes = res["rec_boxes"]
    except (KeyError, TypeError):
        return None
    heights = [float(b[3]) - float(b[1]) for b in boxes if len(b) >= 4]
    heights = [h for h in heights if h > 0]
    if not heights:
        return None
    heights.sort()
    return heights[len(heights) // 2]


def _image_size(path):
    """(width, height) of an image, without decoding the whole file where possible."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:  # noqa: BLE001 - layout is optional, never fail the run for it
        data = imread_unicode(path)
        return (data.shape[1], data.shape[0]) if data is not None else (0, 0)


# On design-heavy pages the detector returns one box per word instead of per line, which
# shreds headings and paragraphs. Measured on such a page, the gap between words on one
# line runs 0.04–0.36 of the line height while genuinely separate columns sit at 3.6 and
# above — an order of magnitude apart, so a threshold in between is safe.
MERGE_MAX_GAP = 1.2  # × line height
MERGE_MAX_OVERLAP = -0.4


def merge_line_fragments(texts, scores, boxes, walls=()):
    """Join word-level detections back into lines, without welding separate columns.

    `walls` are vertical edges — the sides of table cells — that a line may not cross.
    Two cells of the same row sit a few pixels apart, far closer than the gap that
    normally separates columns, so without them a row's cells weld into one line."""
    items = [
        (t, s, tuple(float(v) for v in b))
        for t, s, b in zip(texts, scores, boxes)
        if str(t).strip()
    ]
    if len(items) < 2:
        return texts, scores, boxes

    items.sort(key=lambda it: it[2][1])
    rows, current = [], [items[0]]
    for item in items[1:]:
        ref = current[-1][2]
        ref_h = max(1.0, ref[3] - ref[1])
        box = item[2]
        if abs((box[1] + box[3]) / 2 - (ref[1] + ref[3]) / 2) <= ref_h * 0.6:
            current.append(item)
        else:
            rows.append(current)
            current = [item]
    rows.append(current)

    out_texts, out_scores, out_boxes = [], [], []
    for row in rows:
        row.sort(key=lambda it: -it[2][2])  # right to left
        group = [row[0]]
        for item in row[1:]:
            prev = group[-1][2]
            height = max(1.0, prev[3] - prev[1])
            # right-to-left: the next box sits to the LEFT, so the gap runs from its
            # right edge to the previous box's left edge
            gap = (prev[0] - item[2][2]) / height
            blocked = any(item[2][2] < wall < prev[0] for wall in walls)
            if not blocked and MERGE_MAX_OVERLAP <= gap <= MERGE_MAX_GAP:
                group.append(item)
                continue
            _flush_group(group, out_texts, out_scores, out_boxes)
            group = [item]
        _flush_group(group, out_texts, out_scores, out_boxes)
    return out_texts, out_scores, out_boxes


def _flush_group(group, out_texts, out_scores, out_boxes):
    out_texts.append(" ".join(g[0].strip() for g in group if g[0].strip()))
    out_scores.append(min(g[1] for g in group))  # a line is only as good as its worst word
    out_boxes.append(
        (
            min(g[2][0] for g in group),
            min(g[2][1] for g in group),
            max(g[2][2] for g in group),
            max(g[2][3] for g in group),
        )
    )


def sample_text_colour(image, box):
    """Read a line's ink colour straight off the page image.

    OCR reports no colours, so light text on a dark panel would be redrawn in the default
    dark ink and vanish. Within a text box the background is the majority colour, so the
    pixels furthest from the median are the glyphs themselves."""
    if image is None:
        return None
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = image[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    background = np.median(crop, axis=0)
    distance = np.linalg.norm(crop - background, axis=1)
    ink = crop[distance >= np.percentile(distance, 85)]
    if ink.size == 0:
        return None
    b, g, r = (int(round(v)) for v in ink.mean(axis=0))  # OpenCV order
    # A photograph never yields a clean black: the ink comes back as a muddy grey-brown
    # that looks washed out when redrawn on white. Dark and unsaturated means black.
    if max(r, g, b) < 130 and max(r, g, b) - min(r, g, b) < 60:
        return None
    return f"#{r:02x}{g:02x}{b:02x}"


def get_table_reader(device="gpu"):
    """The table detector and structure reader, built once and kept."""
    if device not in _table_instances:
        _table_instances[device] = (
            LayoutDetection(model_name=TABLE_LAYOUT_MODEL, device=device),
            TableStructureRecognition(model_name=TABLE_STRUCTURE_MODEL, device=device),
        )
    return _table_instances[device]


def _quad_to_box(quad):
    """A cell's four corners as an upright rectangle."""
    points = np.array(quad, dtype=float).reshape(-1, 2)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def parse_table_grid(structure, bbox_count):
    """Map each cell SLANet detected, in the order its boxes were emitted, to the row,
    column and span it occupies in the table's own HTML-like token sequence.

    Confirmed directly against the model's own decode logic (`TableLabelDecode.decode` in
    the installed `paddlex`) and its dictionary file: a box is emitted exactly at each
    `<td`/`<td>`/`<td></td>` token, in order, and a merged cell's span sits in a separate
    ` colspan="N"`/` rowspan="N"` token that follows it — describing the cell that was
    just opened, not the next one. So a cell cannot be placed on the grid (which needs its
    span to know how many positions to occupy) the moment its `<td` token is seen; it has
    to stay "open" until the following token proves it holds no more span information —
    the next `<td`, the next `<tr>`, or the end of the table, whichever comes first."""
    grid = []
    occupied = set()
    row = -1
    col = 0
    bbox_index = 0
    pending = None  # the cell currently open for a possible colspan/rowspan token

    def close_pending():
        nonlocal col, pending
        if pending is None:
            return
        r, c, rowspan, colspan = pending["row"], pending["col"], pending["rowspan"], pending["colspan"]
        grid.append({"row": r, "col": c, "rowspan": rowspan, "colspan": colspan})
        for dr in range(rowspan):
            for dc in range(colspan):
                occupied.add((r + dr, c + dc))
        col = c + colspan
        pending = None

    for token in structure:
        if token == "<tr>":
            close_pending()
            row += 1
            col = 0
        elif token.startswith(" colspan="):
            if pending is not None:
                pending["colspan"] = int(token.split('"')[1])
        elif token.startswith(" rowspan="):
            if pending is not None:
                pending["rowspan"] = int(token.split('"')[1])
        elif token in ("<td>", "<td", "<td></td>"):
            if bbox_index >= bbox_count:
                break
            close_pending()
            while (row, col) in occupied:
                col += 1
            pending = {"row": row, "col": col, "rowspan": 1, "colspan": 1}
            bbox_index += 1
    close_pending()
    return grid


def detect_tables(image, device="gpu"):
    """Find the tables on a scanned page and read the grid of each one.

    Returns one entry per table with the cell rectangles in page pixels, each carrying
    its row/column/span when the structure model's own token sequence could be parsed
    (silently omitted otherwise, so a mismatched or unexpected sequence just falls back to
    the existing geometry-only reconstruction downstream rather than mis-mapping cells).
    Nothing is recognised here: the text already came from OCR, and each line only has to
    be put in the cell it falls inside."""
    try:
        detector, reader = get_table_reader(device)
    except Exception:  # noqa: BLE001 - table reading is optional, OCR must still run
        traceback.print_exc()
        return []

    tables = []
    for page in detector.predict(image):
        for found in page.get("boxes", []):
            if float(found.get("score", 0)) < TABLE_MIN_SCORE:
                continue
            x0, y0, x1, y1 = (int(round(float(v))) for v in found["coordinate"])
            x0, y0 = max(0, x0), max(0, y0)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            cells = []
            for grid_result in reader.predict(crop):
                bboxes = grid_result.get("bbox", [])
                spans = parse_table_grid(grid_result.get("structure", []), len(bboxes))
                reliable = len(spans) == len(bboxes)
                for i, quad in enumerate(bboxes):
                    cx0, cy0, cx1, cy1 = _quad_to_box(quad)
                    cell = {"box": (cx0 + x0, cy0 + y0, cx1 + x0, cy1 + y0)}
                    if reliable:
                        cell.update(spans[i])
                    cells.append(cell)
            if cells:
                tables.append({"box": (x0, y0, x1, y1), "cells": cells})
    return tables


def cell_walls(tables):
    """The vertical edges of every detected cell — the lines OCR must not read across."""
    return sorted(
        {edge for table in tables for cell in table["cells"] for edge in (cell["box"][0], cell["box"][2])}
    )


def table_cell_shapes(tables):
    """The ruling of every detected cell, so a scanned table is drawn as a table."""
    return [
        {
            "box": cell["box"],
            "fill": None,
            "fill_alpha": 1.0,
            "stroke": (90, 90, 90),
            "stroke_alpha": 1.0,
            "width": 1.0,
            "radius": 0,
        }
        for table in tables
        for cell in table["cells"]
    ]


def merge_table_cells(texts, scores, boxes, tables):
    """Fold the OCR lines that fall inside a table cell into one block for that cell.

    A cell holding two lines is one entry in the document, not two floating lines, and
    giving it the cell's own rectangle is what stops its text from spilling over the
    ruling into the cell beside it. When the structure model's own row/column/span for a
    cell is available (see `parse_table_grid`), it rides along on the merged block so the
    Word export can draw a real merged cell instead of guessing the grid from geometry."""
    if not tables or not boxes:
        return texts, scores, boxes

    cells = [cell for table in tables for cell in table["cells"]]
    owner = {}
    for index, block in enumerate(boxes):
        x0, y0, x1, y1 = block["box"]
        centre = ((x0 + x1) / 2, (y0 + y1) / 2)
        for cell_index, cell in enumerate(cells):
            cx0, cy0, cx1, cy1 = cell["box"]
            if cx0 <= centre[0] <= cx1 and cy0 <= centre[1] <= cy1:
                owner.setdefault(cell_index, []).append(index)
                break

    if not owner:
        return texts, scores, boxes

    merged_texts, merged_scores, merged_boxes = [], [], []
    done = set()
    for index, block in enumerate(boxes):
        if index in done:
            continue
        cell_index = next((c for c, members in owner.items() if index in members), None)
        if cell_index is None:
            merged_texts.append(texts[index])
            merged_scores.append(scores[index])
            merged_boxes.append(block)
            continue
        cell = cells[cell_index]
        cx0, cy0, cx1, cy1 = cell["box"]
        members = owner[cell_index]
        done.update(members)
        rtl = _is_rtl(" ".join(texts[i] for i in members))
        ordered = sorted(
            members,
            key=lambda i: (
                round(boxes[i]["box"][1] / 8),
                -boxes[i]["box"][2] if rtl else boxes[i]["box"][0],
            ),
        )
        merged_texts.append(" ".join(texts[i].strip() for i in ordered if texts[i].strip()))
        merged_scores.append(min(scores[i] for i in ordered))
        merged_boxes.append(
            {
                "text": merged_texts[-1],
                "box": (
                    cx0 + TABLE_CELL_MARGIN,
                    cy0 + TABLE_CELL_MARGIN,
                    cx1 - TABLE_CELL_MARGIN,
                    cy1 - TABLE_CELL_MARGIN,
                ),
                "size": max(b["size"] for b in (boxes[i] for i in ordered)),
                "bold": False,
                "colour": boxes[ordered[0]].get("colour"),
                "lines": len(ordered),
                "align": "right" if rtl else "left",
                "fixed": True,
                "row": cell.get("row"),
                "col": cell.get("col"),
                "rowspan": cell.get("rowspan"),
                "colspan": cell.get("colspan"),
            }
        )
    return merged_texts, merged_scores, merged_boxes


def get_figure_detector(device="gpu"):
    """The general-layout detector used to find diagrams/figures, built once and kept."""
    if device not in _figure_instances:
        _figure_instances[device] = LayoutDetection(model_name=FIGURE_LAYOUT_MODEL, device=device)
    return _figure_instances[device]


FIGURE_MAX_PAGE_SHARE = 0.4  # a "figure" bigger than this is more likely a bad crop than real


def detect_figures(image, device="gpu"):
    """Find diagram/chart regions on a scanned page — never text, never a table.

    A diagram's meaning lives in its drawing, not in text PaddleOCR could read off it line
    by line, so nothing is recognised inside these boxes: they are kept exactly as scanned
    and dropped back onto the page (and into Word) as a picture.

    Measured directly against a real, noisy phone-photo exam page with a small embedded
    diagram: a single whole-page pass missed it (mislabelled the region "text"), and
    feeding the detector a grid of large overlapping crops did catch it — but only with a
    box that swallowed most of the surrounding paragraph too, imprecise enough that
    filtering OCR text under it would have deleted real answers. Tiling to chase a smaller
    figure is a genuine, disclosed limitation of this 5 MB model on hard scans, not
    something worth trading for a slower and unreliable crop; `FIGURE_MAX_PAGE_SHARE`
    below is what rejects that kind of oversized, low-precision box outright rather than
    risk it silently."""
    try:
        detector = get_figure_detector(device)
    except Exception:  # noqa: BLE001 - figure detection is optional, OCR must still run
        traceback.print_exc()
        return []

    height, width = image.shape[:2]
    page_area = max(1.0, float(width * height))
    figures = []
    for page in detector.predict(image):
        for box in page.get("boxes", []):
            if box.get("label") not in FIGURE_LABELS or float(box.get("score", 0)) < FIGURE_MIN_SCORE:
                continue
            x0, y0, x1, y1 = (int(round(float(v))) for v in box["coordinate"])
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            if (x1 - x0) * (y1 - y0) > FIGURE_MAX_PAGE_SHARE * page_area:
                continue  # likely an imprecise crop dominated by surrounding text, not a real figure
            figures.append({"box": (x0, y0, x1, y1)})
    return figures


def figure_shapes(figures, image):
    """Each figure as a drawable block, its crop already encoded for storage.

    The crop has to survive a JSON round trip inside Gradio's `State`, like the rest of
    `layout` — a raw pixel array cannot, so it is kept as base64-encoded PNG bytes."""
    shapes = []
    for figure in figures:
        x0, y0, x1, y1 = figure["box"]
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            continue
        shapes.append({"box": figure["box"], "png_b64": base64.b64encode(buf.tobytes()).decode("ascii")})
    return shapes


def _centre_in_box(box, outer):
    """True when a box's own centre falls inside another box."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def exclude_figures_over_tables(figures, tables):
    """Drop any candidate figure that lands on a region already confirmed as a table.

    A table and a figure are mutually exclusive: `detect_tables` has already read an
    actual grid of cells there, so a "figure" covering the same area is the layout
    detector mistaking a bordered table for a picture — confirmed directly on a plain
    black-and-white table with a merged header, misclassified as `image` at 0.65
    confidence. Left unfiltered, that single bad label would reach
    `drop_boxes_inside_figures` and erase every cell's text as if it sat inside a photo."""
    if not tables:
        return figures
    return [f for f in figures if not any(_centre_in_box(f["box"], t["box"]) for t in tables)]


def drop_boxes_inside_figures(texts, scores, boxes, figures):
    """Remove OCR lines whose centre falls inside a detected figure.

    A label drawn on a diagram (a tank's name in an engineering sketch, say) is part of
    the picture now; reading it out separately would duplicate it as a floating line on
    top of the same picture."""
    if not figures or not boxes:
        return texts, scores, boxes
    kept_t, kept_s, kept_b = [], [], []
    for t, s, b in zip(texts, scores, boxes):
        if not any(_centre_in_box(b["box"], f["box"]) for f in figures):
            kept_t.append(t)
            kept_s.append(s)
            kept_b.append(b)
    return kept_t, kept_s, kept_b


# A detected line's box is only a rough guide to its font size: it swells around a boxed
# label, a slanted line or a stray mark. These bound the damage.
OCR_LINE_SPACING = 1.3      # box height / font size for a clean line
# Measured across these scans, a line's width divides by its character count and size to
# about 0.39 for Arabic. The cap uses the narrow end of that range, so it only bites when
# a box is genuinely too tall for the text in it - never on an ordinary long line.
OCR_CHAR_WIDTH = 0.34
# A page is set in a handful of sizes, not a continuum. Estimates are snapped to this
# ladder of multiples of the page's median so body text comes out at one single size.
OCR_SIZE_LADDER = (0.8, 1.0, 1.45, 1.9, 2.4)


def _width_cap(text, box):
    """The largest size a line can be and still fit the width it was detected in."""
    letters = len((text or "").strip()) or 1
    return (box[2] - box[0]) / (letters * OCR_CHAR_WIDTH)


def _ocr_blocks(result, image=None):
    """Turn an OCR result into layout blocks shaped like the PDF ones.

    OCR gives no font metrics, so each line's size is estimated from its own box and then
    held near the page's median: a scanned page is set in one or two sizes, and a line
    that measures three times its neighbours is a bad box, not a heading."""
    texts = list(result.get("texts", []))
    boxes = list(result.get("boxes", []))
    heights = [max(1.0, (b[3] - b[1]) / OCR_LINE_SPACING) for b in boxes]
    if not heights:
        return []
    median = sorted(heights)[len(heights) // 2]

    blocks = []
    for text, box, height in zip(texts, boxes, heights):
        x0, y0, x1, y1 = box
        step = min(OCR_SIZE_LADDER, key=lambda rung: abs(rung - height / median))
        blocks.append(
            {
                "text": text,
                "box": (x0, y0, x1, y1),
                "size": max(1.0, min(median * step, _width_cap(text, box))),
                "bold": False,
                "lines": 1,   # OCR detects one line at a time
                "colour": sample_text_colour(image, box),
            }
        )
    return relax_block_widths(blocks)


def _boxes_from(res):
    """Per-line bounding boxes as (x0, y0, x1, y1), index-aligned with the texts."""
    try:
        boxes = res["rec_boxes"]
    except (KeyError, TypeError):
        return []
    out = []
    for box in boxes:
        if len(box) >= 4:
            out.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
    return out


def _ocr_single_image(ocr, image_path, run_dir, thresh, walls=()):
    """Run OCR on one image file and return everything the UI needs, layout included."""
    res = next(iter(ocr.predict(image_path, text_rec_score_thresh=thresh)))
    res.save_to_img(run_dir)
    # PaddleOCR names the annotated file after the input's extension
    # (foo.jpg -> foo_ocr_res_img.jpg), so match any extension, not just .png.
    saved = glob.glob(os.path.join(run_dir, "*_ocr_res_img.*"))
    annotated = max(saved, key=os.path.getmtime) if saved else image_path
    texts = [normalize_mixed_text(t) for t in res["rec_texts"]]
    scores = [float(s) for s in res["rec_scores"]]
    boxes = _boxes_from(res)
    if len(boxes) == len(texts):
        texts, scores, boxes = merge_line_fragments(texts, scores, boxes, walls)
    return {
        "image": annotated,
        "texts": texts,
        "scores": scores,
        "median_height": _median_text_height(res),
        "boxes": boxes,
        "dims": _image_size(image_path),
    }


def _render_pdf_page(pdf_page, run_dir, page_index):
    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE))
    path = os.path.join(run_dir, f"page_{page_index + 1:03d}.png")
    pix.save(path)
    return path


def run_ocr(file_path, lang, thresh, hide_empty, pdf_mode, read_tables=True,
            progress=gr.Progress()):
    """Entry point for the UI: never raises, so a failure shows a readable message."""
    try:
        return _run_ocr(file_path, lang, thresh, hide_empty, pdf_mode, read_tables, progress)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user instead of a bare error badge
        traceback.print_exc()
        detail = str(exc).splitlines()[0][:200] if str(exc).strip() else ""
        blob = f"{type(exc).__name__} {detail}".lower()
        if "cudnn" in blob or "memory" in blob or "alloc" in blob:
            hint = "ذاكرة كرت الرسوم غير كافية — أغلق المتصفح أو البرامج الثقيلة ثم أعد المحاولة."
        elif "قراءة الصورة" in detail or "decode" in blob:
            hint = "تعذّرت قراءة الملف — تأكد أنه صورة سليمة بإحدى الصيغ المدعومة."
        else:
            hint = "أعد المحاولة، وإن تكرر الخطأ أعد تشغيل الخادم."
        message = build_status_html(
            alerts=[("error", f"❌ **تعذّرت المعالجة**  \n`{type(exc).__name__}: {detail}`  \n{hint}")]
        )
        return _blank_outputs(message)


def _blank_outputs(status_html):
    """The full output tuple for a run that produced nothing (error / no file)."""
    return (
        [],  # gallery
        [],  # master table
        "",  # document html
        None,  # json view
        status_html,
        None,  # txt download
        None,  # json download
        None,  # editable Word download
        None,  # layout Word download
        [],  # review table
        "",  # review summary
        gr.update(visible=False),  # results panel
        [],  # rows state
        {},  # meta state
        gr.update(choices=[], value=None),  # page selector
        None,  # page image
    )


def _run_ocr(file_path, lang, thresh, hide_empty, pdf_mode, read_tables, progress):
    if file_path is None:
        empty_status = build_status_html(
            alerts=[("warning", "**لم يتم اختيار ملف** — ارفع صورة أو ملف PDF من لوحة التحكم ثم اضغط تشغيل.")]
        )
        return _blank_outputs(empty_status)

    start = time.time()
    run_dir = os.path.join(OUTPUT_DIR, uuid.uuid4().hex)
    os.makedirs(run_dir, exist_ok=True)

    is_pdf = file_path.lower().endswith(".pdf")

    annotated_images = []
    table_rows = []
    text_lines = []
    all_pages_json = []
    all_scores = []
    page_sizes = []
    text_heights = []
    page_sources = {}  # page number -> "نص مضمّن" / "OCR"
    page_images = {}  # page number -> annotated image path, for the paired review view
    layout = {}  # page number -> {width, height, blocks:[{id, box, size, bold}]}
    scrambled_layer = False  # set for PDFs whose embedded text layer transposes letters
    row_num = 0
    table_count = 0
    figure_count = 0
    runner = OcrRunner(lang)  # models load lazily, only if OCR is actually needed

    if is_pdf:
        doc = fitz.open(file_path)
        num_pages = len(doc)
        # The defect lives in the file's font encoding, so it is judged once for the whole
        # document: a layer that scrambles letters on one page cannot be trusted on any.
        scrambled_layer = any(looks_scrambled(page.get_text()) for page in doc)

        for page_index in progress.tqdm(range(num_pages), desc="معالجة الصفحات"):
            page = doc[page_index]
            page_sizes.append((page.rect.width, page.rect.height))
            embedded_text = page.get_text().strip()
            non_ws_chars = len("".join(embedded_text.split()))

            use_embedded = pdf_mode == "extract_only" or (
                pdf_mode == "auto"
                and non_ws_chars >= MIN_EMBEDDED_CHARS
                and not scrambled_layer
            )

            page_no = page_index + 1
            boxes = None
            if use_embedded:
                source = "نص مضمّن"
                page_image = _render_pdf_page(page, run_dir, page_index)
                blocks = extract_pdf_blocks(page)
                texts = [b["text"] for b in blocks]
                scores = [1.0] * len(texts)
                boxes = blocks
                pdf_images = extract_pdf_images(page)
                figure_count += len(pdf_images)
                layout[page_no] = {
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": [],
                    "shapes": extract_pdf_drawings(page),
                    "figures": pdf_images,
                }
            else:
                source = "OCR"
                raw_page_image = _render_pdf_page(page, run_dir, page_index)
                # straightened and, if that fired, contrast-lifted once here, so every
                # reader below — tables, figures, OCR itself — sees the same corrected
                # pixels in the same coordinate space instead of each guessing separately
                raw_page_image, image = preprocess_page_image(raw_page_image, run_dir, runner.device)
                # the grid is read before the text, so recognition already knows where the
                # cell walls are and never welds two cells of a row into one line
                tables = detect_tables(image, runner.device) if read_tables else []
                # a page needing OCR for its *text* (a scanned page, or one whose text
                # layer is corrupted) still has a completely intact PDF object model for
                # its *images* — extracting those directly is exact, unlike the pixel-based
                # detector below, so it is tried first and always wins on overlap
                pdf_images = extract_pdf_images(page)
                for pdf_image in pdf_images:
                    pdf_image["box"] = tuple(v * PDF_RENDER_SCALE for v in pdf_image["box"])
                model_figures = exclude_figures_over_tables(
                    detect_figures(image, runner.device), tables
                )
                model_figures = [
                    f
                    for f in model_figures
                    if not any(_centre_in_box(f["box"], pf["box"]) for pf in pdf_images)
                ]
                figures = pdf_images + figure_shapes(model_figures, image)
                result = runner.run(raw_page_image, run_dir, thresh, cell_walls(tables))
                page_image = result["image"]
                texts, scores = result["texts"], result["scores"]
                if result["median_height"]:
                    text_heights.append(result["median_height"])
                boxes = _ocr_blocks(result, image)
                texts, scores, boxes = merge_table_cells(texts, scores, boxes, tables)
                texts, scores, boxes = drop_boxes_inside_figures(texts, scores, boxes, figures)
                table_count += len(tables)
                figure_count += len(figures)
                if boxes or figures:
                    # the page's vector art does not depend on the text layer, so it is
                    # still worth drawing when the text itself had to come from OCR;
                    # its coordinates are in points and must follow the render scale
                    shapes = extract_pdf_drawings(page)
                    for shape in shapes:
                        shape["box"] = tuple(v * PDF_RENDER_SCALE for v in shape["box"])
                        shape["width"] *= PDF_RENDER_SCALE
                        shape["radius"] *= PDF_RENDER_SCALE
                    layout[page_no] = {
                        "width": result["dims"][0],
                        "height": result["dims"][1],
                        "blocks": [],
                        "shapes": shapes + table_cell_shapes(tables),
                        "figures": figures,
                    }

            annotated_images.append((page_image, f"صفحة {page_no} ({source})"))
            page_sources[page_no] = source
            page_images[page_no] = page_image

            page_lines = []
            for index, (t, s) in enumerate(zip(texts, scores)):
                if hide_empty and not t.strip():
                    continue
                row_num += 1
                table_rows.append([row_num, t, round(s, 3), page_no, source])
                if boxes and index < len(boxes):
                    layout[page_no]["blocks"].append(
                        {
                            "id": row_num,
                            "box": boxes[index]["box"],
                            "size": boxes[index]["size"],
                            "bold": boxes[index]["bold"],
                            "colour": boxes[index].get("colour"),
                            "align": boxes[index].get("align"),
                            "lines": boxes[index].get("lines", 1),
                            "fixed": boxes[index].get("fixed", False),
                            "ink": boxes[index].get("ink"),
                            "row": boxes[index].get("row"),
                            "col": boxes[index].get("col"),
                            "rowspan": boxes[index].get("rowspan"),
                            "colspan": boxes[index].get("colspan"),
                        }
                    )
                page_lines.append(t)
                all_scores.append(s)

            if num_pages > 1 and page_lines:
                text_lines.append(f"--- صفحة {page_index + 1} ({source}) ---")
            text_lines.extend(page_lines)

            all_pages_json.append(
                {"page": page_index + 1, "source": source, "lines": page_lines}
            )
        doc.close()
    else:
        file_path, image = preprocess_page_image(file_path, run_dir, runner.device)
        tables = detect_tables(image, runner.device) if read_tables else []
        figures = exclude_figures_over_tables(detect_figures(image, runner.device), tables)
        result = runner.run(file_path, run_dir, thresh, cell_walls(tables))
        page_image = result["image"]
        texts, scores = result["texts"], result["scores"]
        if result["median_height"]:
            text_heights.append(result["median_height"])
        annotated_images.append((page_image, "OCR"))
        page_sources[1] = "OCR"
        page_images[1] = page_image
        boxes = _ocr_blocks(result, image)
        texts, scores, boxes = merge_table_cells(texts, scores, boxes, tables)
        texts, scores, boxes = drop_boxes_inside_figures(texts, scores, boxes, figures)
        table_count += len(tables)
        figure_count += len(figures)
        if boxes or figures:
            layout[1] = {
                "width": result["dims"][0],
                "height": result["dims"][1],
                "blocks": [],
                "shapes": table_cell_shapes(tables),
                "figures": figure_shapes(figures, image),
            }
        page_lines = []
        for index, (t, s) in enumerate(zip(texts, scores)):
            if hide_empty and not t.strip():
                continue
            row_num += 1
            table_rows.append([row_num, t, round(s, 3), 1, "OCR"])
            if boxes and index < len(boxes):
                layout[1]["blocks"].append(
                    {
                        "id": row_num,
                        "box": boxes[index]["box"],
                        "size": boxes[index]["size"],
                        "bold": boxes[index]["bold"],
                        "colour": boxes[index].get("colour"),
                        "align": boxes[index].get("align"),
                        "lines": boxes[index].get("lines", 1),
                        "fixed": boxes[index].get("fixed", False),
                        "ink": boxes[index].get("ink"),
                        "row": boxes[index].get("row"),
                        "col": boxes[index].get("col"),
                        "rowspan": boxes[index].get("rowspan"),
                        "colspan": boxes[index].get("colspan"),
                    }
                )
            page_lines.append(t)
            all_scores.append(s)
        text_lines.extend(page_lines)
        all_pages_json.append({"page": 1, "source": "OCR", "lines": page_lines})

    elapsed = time.time() - start
    json_output, txt_path, json_path, docx_path, layout_docx_path = write_exports(
        table_rows, page_sources, page_sizes, run_dir, layout=layout
    )

    avg_score = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
    page_count = len(all_pages_json)
    embedded_pages = sum(1 for p in all_pages_json if p["source"] == "نص مضمّن")
    ocr_pages = page_count - embedded_pages

    alerts = []
    if is_pdf and scrambled_layer and pdf_mode == "auto":
        alerts.append(
            (
                "warning",
                "🔤 **طبقة النص في هذا الملف تالفة** — ترميز الخط يقلب ترتيب الحروف "
                "(«البرمجة» تُقرأ «الربمجة») رغم أن الصفحة تُعرض سليمة. "
                "لذلك استُخدم OCR تلقائياً على كل الصفحات لأنه يعطي نصاً أصح هنا.",
            )
        )
    if is_pdf and embedded_pages and ocr_pages:
        alerts.append(
            ("info", f"🧩 {embedded_pages} صفحة من نص مضمّن (دقة كاملة) و {ocr_pages} صفحة بالـ OCR.")
        )
    elif is_pdf and embedded_pages:
        alerts.append(("info", "🧩 كل الصفحات من نص مضمّن في الملف (دقة كاملة، لم تُستخدم OCR إطلاقاً)."))
    if runner.fell_back:
        alerts.append(
            (
                "warning",
                "🖥️ **تم التحويل إلى المعالج (CPU) تلقائياً** — ذاكرة كرت الرسوم لم تكفِ "
                "(مشغولة ببرامج أخرى كالمتصفح). النتيجة مطابقة لكن أبطأ. "
                "لإعادة استخدام الكرت: أغلق البرامج الثقيلة ثم أعد التشغيل.",
            )
        )
    if text_heights:
        median_h = sorted(text_heights)[len(text_heights) // 2]
        if median_h < MIN_RELIABLE_TEXT_HEIGHT:
            alerts.append(
                (
                    "warning",
                    f"🔍 **تنبيه دقة:** متوسط ارتفاع سطر النص في الصورة {median_h:.0f}px فقط "
                    f"(المطلوب {MIN_RELIABLE_TEXT_HEIGHT:.0f}px فأكثر). عند هذا الحجم تختفي نقاط "
                    "الحروف العربية (ن/ت/ب/ي) من البكسلات نفسها فتكثر الأخطاء. "
                    "الحل هو **إعادة التقاط الصورة بدقة أعلى** — تكبير الصورة الحالية لا يفيد.",
                )
            )
    if table_count:
        alerts.append(
            (
                "info",
                f"▦ تم التعرّف على {table_count} جدول في الصفحات الممسوحة، ووُضع نص كل خلية داخل خليتها.",
            )
        )
    if figure_count:
        alerts.append(
            (
                "info",
                f"🖼️ تم استخراج {figure_count} صورة/شكل من المستند، وأُدرج كل منها في مكانه بدل تجاهله.",
            )
        )
    if not table_rows:
        alerts.append(("error", "لم يُستخرج أي نص. جرّب تغيير اللغة أو خفض حد الثقة."))

    status = build_status_html(tiles=(elapsed, page_count, len(table_rows), avg_score), alerts=alerts)

    review_rows, review_summary = build_review_rows(table_rows)
    page_numbers = sorted(page_sources)
    first_page = page_numbers[0] if page_numbers else None
    pages_html = build_layout_html(table_rows, layout, only_page=first_page) or build_pages_html(
        table_rows, only_page=first_page
    )
    meta = {
        "run_dir": run_dir,
        "page_sizes": page_sizes,
        "page_sources": page_sources,
        "page_images": page_images,
        "layout": layout,
    }

    return (
        annotated_images,
        table_rows,
        pages_html,
        json_output,
        status,
        txt_path,
        json_path,
        docx_path,
        layout_docx_path,
        review_rows,
        review_summary,
        gr.update(visible=bool(table_rows)),
        table_rows,
        meta,
        gr.update(choices=[(f"صفحة {n}", n) for n in page_numbers], value=first_page),
        page_images.get(first_page),
    )


# ------------------------------ التحرير والتنقّل ------------------------------


def _page_of(meta, page_no):
    images = (meta or {}).get("page_images", {})
    # State survives a round trip through JSON, which turns the int keys into strings.
    return images.get(page_no) or images.get(str(page_no))


def _layout_of(meta):
    """Layout keyed by page number; State round-trips through JSON and stringifies keys."""
    raw = (meta or {}).get("layout") or {}
    return {int(k): v for k, v in raw.items()}


def render_document(rows, meta, page_no, continuous, faithful):
    """Render the document pane in the requested mode."""
    layout = _layout_of(meta) if faithful else None
    only = None if continuous or page_no is None else int(page_no)
    if layout:
        html = build_layout_html(rows, layout, only_page=only)
        if html:
            return html
    return build_pages_html(rows, only_page=only)


def show_page(rows, meta, page_no, continuous, faithful):
    """Render one page (or the whole document) plus its matching source image."""
    html = render_document(rows, meta, page_no, continuous, faithful)
    if continuous or page_no is None:
        return html, gr.update(value=None, visible=False)
    return html, gr.update(value=_page_of(meta, int(page_no)), visible=True)


def step_page(meta, page_no, delta):
    pages = sorted(int(p) for p in (meta or {}).get("page_sources", {}))
    if not pages:
        return gr.update()
    try:
        idx = pages.index(int(page_no))
    except (ValueError, TypeError):
        idx = 0
    return gr.update(value=pages[max(0, min(len(pages) - 1, idx + delta))])


def apply_edits(edited, rows, meta, page_no, continuous, faithful, text_column, from_review):
    """Merge an edit made in either grid, then rebuild every derived view and export.

    Only the full results grid (`text_column == 1`) may add or delete rows — see
    `merge_edits` for why the review grid must not."""
    before_ids = {row[0] for row in rows}
    rows, changed = merge_edits(edited, rows, text_column, allow_structural=(text_column == 1))
    if not changed:
        return (gr.update(),) * 10

    meta = dict(meta or {})
    removed_ids = before_ids - {row[0] for row in rows}
    if removed_ids:
        # a deleted row's block would otherwise survive as an empty box in the layout
        # view and in the faithful Word export
        layout = _layout_of(meta)
        for page in layout.values():
            page["blocks"] = [b for b in page.get("blocks", []) if b.get("id") not in removed_ids]
        meta["layout"] = layout

    review_rows, review_summary = build_review_rows(rows)
    json_output, txt_path, json_path, docx_path, layout_docx_path = write_exports(
        rows,
        meta.get("page_sources", {}),
        meta.get("page_sizes", []),
        meta.get("run_dir", OUTPUT_DIR),
        layout=_layout_of(meta),
    )
    doc_html = render_document(rows, meta, page_no, continuous, faithful)
    return (
        rows,
        doc_html,
        json_output,
        txt_path,
        json_path,
        docx_path,
        layout_docx_path,
        review_summary,
        meta,
        # refresh the *other* grid only, so the one being typed in is not yanked away
        rows if from_review else review_rows,
    )


BRAND_LOGO_SVG = f"""
<svg class="brand-mark" width="34" height="34" viewBox="0 0 30 30" xmlns="http://www.w3.org/2000/svg">
    <rect width="30" height="10" y="0" fill="{BRAND_GREEN}"/>
    <rect width="30" height="10" y="10" fill="#ffffff"/>
    <rect width="30" height="10" y="20" fill="{BRAND_BLACK}"/>
    <circle cx="10.5" cy="15" r="1.7" fill="{BRAND_RED}"/>
    <circle cx="15" cy="15" r="1.7" fill="{BRAND_RED}"/>
    <circle cx="19.5" cy="15" r="1.7" fill="{BRAND_RED}"/>
</svg>
"""

WELCOME_STATUS = build_status_html(
    alerts=[
        (
            "info",
            "👋 **جاهز للعمل.** ارفع صورة أو ملف PDF من لوحة التحكم، ثم اضغط **تشغيل الاستخراج**.  \n"
            "ملفات PDF التي تحتوي نصاً مضمّناً تُستخرج مباشرة بدقة كاملة دون الحاجة إلى OCR.",
        )
    ]
)

with gr.Blocks(title="محرك استخراج النصوص — PaddleOCR") as demo:
    gr.HTML(
        f"""
        <div id="app-header">
            <div class="brand">
                {BRAND_LOGO_SVG}
                <div class="brand-text">
                    <div class="title">محرك استخراج النصوص</div>
                    <div class="subtitle">
                        PaddleOCR — تعرّف ضوئي محلي على الصور وملفات PDF، مع كشف تلقائي للنص المضمّن ودعم كامل للعربية
                    </div>
                </div>
            </div>
            <div class="header-badge">🔒 يعمل بالكامل محلياً على جهازك</div>
        </div>
        <div class="flag-stripe"></div>
        """
    )

    with gr.Row(equal_height=False):
        # ---------------- لوحة التحكم ----------------
        with gr.Column(scale=4, min_width=320):
            gr.Markdown("📂 المصدر", elem_classes=["section-title"])
            with gr.Group():
                file_input = gr.File(
                    label="صورة أو ملف PDF",
                    file_types=SUPPORTED_EXTENSIONS,
                    height=170,
                )
                gr.Markdown(
                    "الصيغ المدعومة: PNG · JPG · JPEG · BMP · TIF · TIFF · WEBP · PDF",
                    elem_classes=["section-title"],
                )

            gr.Markdown("🌐 اللغة", elem_classes=["section-title", "with-gap"])
            with gr.Group():
                lang_input = gr.Dropdown(
                    label="لغة النص — تُستخدم عند اللجوء إلى OCR فقط",
                    choices=LANG_CHOICES,
                    value="ar+en",
                    info="اختر «العربية + الإنجليزية معاً» للمستندات المختلطة",
                )

            run_button = gr.Button("▶️  تشغيل الاستخراج", variant="primary", size="lg")

            with gr.Accordion("⚙️  إعدادات متقدمة", open=False):
                pdf_mode_input = gr.Radio(
                    label="طريقة معالجة ملفات PDF",
                    choices=PDF_MODE_CHOICES,
                    value="auto",
                )
                thresh_input = gr.Slider(
                    label="الحد الأدنى لثقة التعرف",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    info="النتائج الأقل من هذه الثقة تُستبعد من المخرجات",
                )
                hide_empty_input = gr.Checkbox(
                    label="إخفاء النتائج الفارغة وغير القابلة للقراءة",
                    value=True,
                )
                tables_input = gr.Checkbox(
                    label="التعرّف على الجداول في الصفحات الممسوحة",
                    value=True,
                    info="يقرأ شبكة الجدول ويضع كل نص في خليته — لا يلزم للصفحات ذات النص المضمّن",
                )

        # ---------------- النتائج ----------------
        with gr.Column(scale=8):
            status_output = gr.HTML(WELCOME_STATUS)

            rows_state = gr.State([])
            meta_state = gr.State({})

            with gr.Column(visible=False) as results_panel:
                with gr.Tabs():
                    with gr.Tab("📝  المستند"):
                        with gr.Row(elem_id="page-nav"):
                            prev_button = gr.Button("السابق ▶", size="sm", scale=0)
                            page_selector = gr.Dropdown(
                                label=None, show_label=False, choices=[], scale=1, container=False
                            )
                            next_button = gr.Button("◀ التالي", size="sm", scale=0)
                            faithful_toggle = gr.Checkbox(
                                label="تخطيط الصفحة الأصلي",
                                value=True,
                                scale=0,
                                container=False,
                            )
                            continuous_toggle = gr.Checkbox(
                                label="عرض متصل", value=False, scale=0, container=False
                            )
                        with gr.Row(equal_height=False):
                            page_image = gr.Image(
                                label=None,
                                show_label=False,
                                height=620,
                                scale=1,
                                interactive=False,
                            )
                            text_output = gr.HTML(value="", padding=False)
                    with gr.Tab("📋  الجدول"):
                        gr.Markdown(
                            "✏️ عمود «النص» قابل للتحرير — أي تعديل يُحدِّث المستند وملفات التصدير فوراً.",
                            elem_classes=["section-title"],
                        )
                        table_output = gr.Dataframe(
                            headers=["#", "النص", "الثقة", "الصفحة", "المصدر"],
                            datatype=["number", "str", "number", "number", "str"],
                            elem_id="results-table",
                            column_widths=["56px", "auto", "84px", "84px", "110px"],
                            max_height=560,
                            show_search="search",
                            wrap=True,
                            interactive=True,
                            static_columns=[0, 2, 3, 4],
                        )
                    with gr.Tab("🎯  مراجعة الثقة"):
                        review_summary_output = gr.HTML("", elem_id="review-summary")
                        review_table = gr.Dataframe(
                            headers=["#", "", "النص", "الثقة", "الصفحة", "⚠️ اقتراح تصحيح"],
                            datatype=["number", "str", "str", "number", "number", "str"],
                            elem_id="results-table",
                            column_widths=["56px", "44px", "auto", "84px", "84px", "auto"],
                            max_height=520,
                            wrap=True,
                            interactive=True,
                            static_columns=[0, 1, 3, 4, 5],
                        )
                    with gr.Tab("🖼️  المعاينة"):
                        gallery_output = gr.Gallery(
                            label=None,
                            show_label=False,
                            columns=1,
                            height=430,
                            object_fit="contain",
                        )
                    with gr.Tab("🧩  JSON"):
                        json_output = gr.Code(
                            label=None,
                            language="json",
                            elem_id="json-view",
                            max_lines=26,
                        )

                gr.Markdown("تصدير النتائج", elem_classes=["section-title"])
                with gr.Row():
                    download_docx = gr.DownloadButton(
                        "📄  Word قابل للتحرير", variant="primary"
                    )
                    download_docx_layout = gr.DownloadButton("🖨️  Word مطابق للتخطيط")
                    download_txt = gr.DownloadButton("📝  نص عادي (.txt)")
                    download_json = gr.DownloadButton("🧩  بيانات (.json)")

    run_button.click(
        fn=run_ocr,
        inputs=[
            file_input,
            lang_input,
            thresh_input,
            hide_empty_input,
            pdf_mode_input,
            tables_input,
        ],
        outputs=[
            gallery_output,
            table_output,
            text_output,
            json_output,
            status_output,
            download_txt,
            download_json,
            download_docx,
            download_docx_layout,
            review_table,
            review_summary_output,
            results_panel,
            rows_state,
            meta_state,
            page_selector,
            page_image,
        ],
    )

    # --- التنقّل بين الصفحات في عرض المستند ---
    nav_inputs = [rows_state, meta_state, page_selector, continuous_toggle, faithful_toggle]
    nav_outputs = [text_output, page_image]
    page_selector.change(fn=show_page, inputs=nav_inputs, outputs=nav_outputs)
    continuous_toggle.change(fn=show_page, inputs=nav_inputs, outputs=nav_outputs)
    faithful_toggle.change(fn=show_page, inputs=nav_inputs, outputs=nav_outputs)
    prev_button.click(
        fn=lambda meta, page_no: step_page(meta, page_no, -1),
        inputs=[meta_state, page_selector],
        outputs=page_selector,
    )
    next_button.click(
        fn=lambda meta, page_no: step_page(meta, page_no, 1),
        inputs=[meta_state, page_selector],
        outputs=page_selector,
    )

    # --- التحرير المباشر: كل تعديل يعيد بناء المستند وملفات التصدير ---
    edit_outputs_common = [
        rows_state,
        text_output,
        json_output,
        download_txt,
        download_json,
        download_docx,
        download_docx_layout,
        review_summary_output,
        meta_state,
    ]
    # `.input()` fires only on real user typing, so these two handlers refreshing each
    # other's grid cannot bounce back and forth.
    table_output.input(
        fn=lambda edited, rows, meta, page_no, cont, faith: apply_edits(
            edited, rows, meta, page_no, cont, faith, text_column=1, from_review=False
        ),
        inputs=[
            table_output,
            rows_state,
            meta_state,
            page_selector,
            continuous_toggle,
            faithful_toggle,
        ],
        outputs=edit_outputs_common + [review_table],
    )
    review_table.input(
        fn=lambda edited, rows, meta, page_no, cont, faith: apply_edits(
            edited, rows, meta, page_no, cont, faith, text_column=2, from_review=True
        ),
        inputs=[
            review_table,
            rows_state,
            meta_state,
            page_selector,
            continuous_toggle,
            faithful_toggle,
        ],
        outputs=edit_outputs_common + [table_output],
    )

# Light theme only. Gradio resolves light/dark from the OS preference unless the page URL
# carries ?__theme=light, and it does that *before* injecting custom head/js — so rewriting
# the URL in place is too late and the only reliable hook is a one-time redirect.
# Injected as raw script text, so it must invoke itself — a bare arrow function is never called.
FORCE_LIGHT_THEME_JS = """
(function () {
    var url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.toString());
    }
})();
"""

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=THEME,
        css=CUSTOM_CSS,
        js=FORCE_LIGHT_THEME_JS,
        show_error=True,
    )
