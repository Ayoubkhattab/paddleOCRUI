"""Local web UI for PaddleOCR: upload an image or PDF and view OCR results in the browser."""

import gc
import glob
import json
import os
import re
import time
import traceback
import uuid

import cv2
import fitz  # PyMuPDF
import gradio as gr
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt
from paddleocr import PaddleOCR, TextDetection, TextRecognition

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


def extract_pdf_paragraphs(page):
    """Extract a PDF page as flowing paragraphs rather than raw visual lines.

    PyMuPDF's text blocks correspond to logical paragraphs (one bullet, one heading,
    one body paragraph each), so joining the lines inside a block reconstructs text
    that reads like the original document instead of a stack of line fragments."""
    paragraphs = []
    for block in page.get_text("blocks"):
        lines = [ln.strip() for ln in block[4].splitlines()]
        lines = [ln for ln in lines if ln.strip("​ \t")]
        if not lines:
            continue
        text = _clean_line(" ".join(lines))
        if text:
            paragraphs.append(text)
    return _merge_split_headers(paragraphs)

_HEADING_RE = re.compile(r"^\d+\)")

CONFIDENCE_LOW = 0.8
CONFIDENCE_VERY_LOW = 0.5


def build_highlighted(table_rows):
    """Review queue: only the lines that actually need a human look, newest problems first.

    Returns (segments, summary_markdown)."""
    flagged = []
    for _, text, score, page_no, _ in table_rows:
        if score < CONFIDENCE_VERY_LOW:
            flagged.append((text, "ثقة منخفضة جداً", score, page_no))
        elif score < CONFIDENCE_LOW:
            flagged.append((text, "ثقة منخفضة", score, page_no))

    total = len(table_rows)
    if not total:
        return [], "لا توجد نتائج بعد."
    if not flagged:
        return [], (
            f"✅ **لا شيء يحتاج مراجعة.** جميع العناصر الـ {total} تجاوزت حد الثقة "
            f"{int(CONFIDENCE_LOW * 100)}%."
        )

    flagged.sort(key=lambda item: item[2])
    segments = []
    for text, label, score, page_no in flagged:
        segments.append((f"صفحة {page_no} · {score:.0%}  ", None))
        segments.append((text, label))
        segments.append(("\n", None))

    very_low = sum(1 for item in flagged if item[1] == "ثقة منخفضة جداً")
    summary = (
        f"⚠️ **{len(flagged)} عنصر يحتاج مراجعة** من أصل {total} "
        f"({len(flagged) / total:.0%}) — منها {very_low} بثقة منخفضة جداً. "
        "العناصر مرتّبة من الأسوأ إلى الأفضل."
    )
    return segments, summary


def _html_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pages_html(table_rows):
    """Render the extracted text as Word-like pages.

    Each line uses `unicode-bidi: plaintext`, so a line that starts with Arabic reads
    right-to-left and one that starts with Latin reads left-to-right — the correct
    behaviour for mixed Arabic/English documents."""
    if not table_rows:
        return ""

    pages = {}
    for _, text, _, page_no, source in table_rows:
        pages.setdefault(page_no, {"source": source, "lines": []})["lines"].append(text)

    chunks = ['<div class="doc-view">']
    for page_no in sorted(pages):
        page = pages[page_no]
        chunks.append('<article class="doc-page">')
        chunks.append(
            f'<header class="doc-page-head"><span>صفحة {page_no}</span>'
            f'<span class="doc-page-src">{_html_escape(page["source"])}</span></header>'
        )
        for line in page["lines"]:
            css_class = "doc-line doc-heading" if _HEADING_RE.match(line) else "doc-line"
            chunks.append(f'<p class="{css_class}">{_html_escape(line)}</p>')
        chunks.append("</article>")
    chunks.append("</div>")
    return "".join(chunks)


def _set_rtl(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    p_pr.append(bidi)


def build_docx(table_rows, page_sizes, run_dir):
    """Export the (filtered) results table to a .docx: one section per page, RTL paragraphs,
    numbered-section headers (e.g. '1) ...') rendered bold. Plain text only — this does not
    reconstruct original tables/colors/shading, which would require a layout-aware pipeline."""
    document = Document()
    section = document.sections[0]
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    if page_sizes:
        width_pt, height_pt = page_sizes[0]
        section.page_width = Pt(width_pt)
        section.page_height = Pt(height_pt)

    pages = {}
    for _, text, _, page_no, _ in table_rows:
        pages.setdefault(page_no, []).append(text)

    for i, page_no in enumerate(sorted(pages)):
        if i > 0:
            document.add_page_break()
        for line in pages[page_no]:
            p = document.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            _set_rtl(p)
            run = p.add_run(line)
            if _HEADING_RE.match(line):
                run.bold = True
                run.font.size = Pt(14)

    path = os.path.join(run_dir, "extracted.docx")
    document.save(path)
    return path


LANG_CHOICES = [
    ("تلقائي (صيني/إنجليزي/ياباني ولغات لاتينية)", "ch"),
    ("العربية + الإنجليزية معاً (مستندات مختلطة)", "ar+en"),
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

OUTPUT_DIR = "ocr_webui_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_ocr_instances = {}

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Cairo"), "Segoe UI", "Tahoma", "sans-serif"],
)

# All colors come from Gradio theme variables so light/dark mode both stay correct.
CUSTOM_CSS = """
.gradio-container { direction: rtl; max-width: 1500px !important; }

/* --- header --- */
#app-header {
    padding: 18px 22px;
    margin-bottom: 4px;
    border: 1px solid var(--border-color-primary);
    border-right: 4px solid var(--primary-500);
    border-radius: var(--radius-lg);
    background: var(--background-fill-secondary);
}
#app-header .title { font-size: 1.45rem; font-weight: 700; line-height: 1.4; }
#app-header .subtitle {
    margin-top: 4px;
    font-size: .92rem;
    color: var(--body-text-color-subdued);
}

/* --- section headings above cards --- */
.section-title {
    font-weight: 600;
    font-size: .9rem;
    color: var(--body-text-color-subdued);
    letter-spacing: .01em;
    padding: 6px 4px 2px;
}
.section-title p { margin: 0; }

/* --- status / result summary bar --- */
.status-box {
    direction: rtl;
    text-align: right;
    padding: 14px 18px;
    border: 1px solid var(--border-color-primary);
    border-right: 4px solid var(--primary-500);
    border-radius: var(--radius-lg);
    background: var(--background-fill-secondary);
}
.status-box p { margin: .25em 0; line-height: 1.7; }

/* --- results table: RTL text, comfortable rows --- */
#results-table table { direction: rtl; }
#results-table th, #results-table td { text-align: right !important; }
#results-table td { line-height: 1.7; }

/* --- Word-like document view --- */
.doc-view {
    max-height: 620px;
    overflow-y: auto;
    padding: 6px 4px;
    background: var(--background-fill-secondary);
    border-radius: var(--radius-lg);
}
.doc-page {
    max-width: 820px;
    margin: 0 auto 22px;
    padding: 46px 54px 52px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-md);
    box-shadow: 0 2px 10px rgba(0, 0, 0, .16);
}
.doc-page-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: -18px 0 22px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color-primary);
    font-size: .78rem;
    color: var(--body-text-color-subdued);
}
.doc-page-src {
    padding: 2px 10px;
    border-radius: 999px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
}
/* plaintext bidi: each line takes its direction from its own first strong character,
   which is what mixed Arabic/English documents need */
.doc-line {
    unicode-bidi: plaintext;
    text-align: start;
    margin: 0 0 .85em;
    line-height: 1.85;
    font-size: 1rem;
    color: var(--body-text-color);
}
.doc-heading {
    font-weight: 700;
    font-size: 1.1rem;
    margin: 1.4em 0 .6em;
}

/* --- confidence review --- */
#review-summary { padding: 4px 4px 8px; }

/* --- keep machine-readable output left-to-right --- */
#json-view, #json-view * { direction: ltr; text-align: left; }

/* --- export bar --- */
#export-bar { border-top: 1px solid var(--border-color-primary); padding-top: 10px; }
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
    texts, scores = [], []
    for (ar_text, ar_score), (la_text, la_score) in zip(arabic, latin):
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
    return texts, scores


class BilingualOcr:
    """Detect each text line once, read it with both the Arabic and Latin recognisers."""

    def __init__(self, device):
        self.detector = TextDetection(model_name="PP-OCRv5_server_det", device=device)
        self.rec_arabic = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec", device=device)
        self.rec_latin = TextRecognition(model_name="en_PP-OCRv5_mobile_rec", device=device)

    def predict(self, image_path, run_dir, thresh):
        img = imread_unicode(image_path)
        if img is None:
            raise ValueError(f"تعذّرت قراءة الصورة (ملف تالف أو صيغة غير مدعومة): {image_path}")

        polys = next(iter(self.detector.predict(image_path)))["dt_polys"]
        if len(polys) == 0:
            return image_path, [], [], None
        polys = sort_polys_reading_order(polys, rtl=True)

        crops = [_crop_quad(img, p) for p in polys]
        arabic = [(r["rec_text"], float(r["rec_score"])) for r in self.rec_arabic.predict(crops)]
        latin = [(r["rec_text"], float(r["rec_score"])) for r in self.rec_latin.predict(crops)]

        texts, scores = _merge_bilingual(arabic, latin)
        keep = [(t, s) for t, s in zip(texts, scores) if s >= thresh]
        texts = [normalize_mixed_text(t) for t, _ in keep]
        scores = [s for _, s in keep]

        annotated = os.path.join(run_dir, "bilingual_ocr_res_img.png")
        marked = img.copy()
        cv2.polylines(marked, [np.array(p, dtype=np.int32) for p in polys], True, (0, 200, 0), 2)
        if not imwrite_unicode(annotated, marked):
            annotated = image_path

        heights = sorted(
            float(np.max(p[:, 1]) - np.min(p[:, 1])) for p in (np.array(x) for x in polys)
        )
        median_h = heights[len(heights) // 2] if heights else None
        return annotated, texts, scores, median_h


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

    def _once(self, image_path, run_dir, thresh):
        if self.lang == "ar+en":
            return get_bilingual(self.device).predict(image_path, run_dir, thresh)
        return _ocr_single_image(get_ocr(self.lang, self.device), image_path, run_dir, thresh)

    def run(self, image_path, run_dir, thresh):
        try:
            return self._once(image_path, run_dir, thresh)
        except (OSError, RuntimeError, MemoryError, SystemError):
            if self.device == "cpu":
                raise
            self.device = "cpu"
            self.fell_back = True
            _free_gpu()
            return self._once(image_path, run_dir, thresh)


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


def _ocr_single_image(ocr, image_path, run_dir, thresh):
    """Run OCR on one image file, return (annotated_path, texts, scores, median_height)."""
    res = next(iter(ocr.predict(image_path, text_rec_score_thresh=thresh)))
    res.save_to_img(run_dir)
    # PaddleOCR names the annotated file after the input's extension
    # (foo.jpg -> foo_ocr_res_img.jpg), so match any extension, not just .png.
    saved = glob.glob(os.path.join(run_dir, "*_ocr_res_img.*"))
    annotated = max(saved, key=os.path.getmtime) if saved else image_path
    texts = [normalize_mixed_text(t) for t in res["rec_texts"]]
    scores = [float(s) for s in res["rec_scores"]]
    return annotated, texts, scores, _median_text_height(res)


def _render_pdf_page(pdf_page, run_dir, page_index):
    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE))
    path = os.path.join(run_dir, f"page_{page_index + 1:03d}.png")
    pix.save(path)
    return path


def run_ocr(file_path, lang, thresh, hide_empty, pdf_mode, progress=gr.Progress()):
    """Entry point for the UI: never raises, so a failure shows a readable message."""
    try:
        return _run_ocr(file_path, lang, thresh, hide_empty, pdf_mode, progress)
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
        message = f"❌ **تعذّرت المعالجة**  \n`{type(exc).__name__}: {detail}`  \n{hint}"
        return [], [], "", None, message, None, None, None, [], "", gr.update(visible=False)


def _run_ocr(file_path, lang, thresh, hide_empty, pdf_mode, progress):
    if file_path is None:
        empty_status = "⚠️ **لم يتم اختيار ملف** — ارفع صورة أو ملف PDF من لوحة التحكم ثم اضغط تشغيل."
        return [], [], "", None, empty_status, None, None, None, [], "", gr.update(visible=False)

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
    row_num = 0
    runner = OcrRunner(lang)  # models load lazily, only if OCR is actually needed

    if is_pdf:
        doc = fitz.open(file_path)
        num_pages = len(doc)
        for page_index in progress.tqdm(range(num_pages), desc="معالجة الصفحات"):
            page = doc[page_index]
            page_sizes.append((page.rect.width, page.rect.height))
            embedded_text = page.get_text().strip()
            non_ws_chars = len("".join(embedded_text.split()))

            use_embedded = pdf_mode == "extract_only" or (
                pdf_mode == "auto" and non_ws_chars >= MIN_EMBEDDED_CHARS
            )

            if use_embedded:
                source = "نص مضمّن"
                page_image = _render_pdf_page(page, run_dir, page_index)
                texts = extract_pdf_paragraphs(page)
                scores = [1.0] * len(texts)
            else:
                source = "OCR"
                raw_page_image = _render_pdf_page(page, run_dir, page_index)
                page_image, texts, scores, line_h = runner.run(raw_page_image, run_dir, thresh)
                if line_h:
                    text_heights.append(line_h)

            annotated_images.append((page_image, f"صفحة {page_index + 1} ({source})"))

            page_lines = []
            for t, s in zip(texts, scores):
                if hide_empty and not t.strip():
                    continue
                row_num += 1
                table_rows.append([row_num, t, round(s, 3), page_index + 1, source])
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
        page_image, texts, scores, line_h = runner.run(file_path, run_dir, thresh)
        if line_h:
            text_heights.append(line_h)
        annotated_images.append((page_image, "OCR"))
        page_lines = []
        for t, s in zip(texts, scores):
            if hide_empty and not t.strip():
                continue
            row_num += 1
            table_rows.append([row_num, t, round(s, 3), 1, "OCR"])
            page_lines.append(t)
            all_scores.append(s)
        text_lines.extend(page_lines)
        all_pages_json.append({"page": 1, "source": "OCR", "lines": page_lines})

    elapsed = time.time() - start
    text_output = "\n".join(text_lines)
    json_output = json.dumps(all_pages_json, ensure_ascii=False, indent=2)

    txt_path = os.path.join(run_dir, "extracted_text.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_output)

    json_path = os.path.join(run_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_output)

    avg_score = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
    page_count = len(all_pages_json)
    embedded_pages = sum(1 for p in all_pages_json if p["source"] == "نص مضمّن")
    ocr_pages = page_count - embedded_pages

    status = (
        f"✅ **تمت المعالجة في {elapsed:.1f} ثانية**  \n"
        f"📄 عدد الصفحات: {page_count} — 🔤 عدد العناصر النصية: {len(table_rows)} "
        f"— 📊 متوسط الثقة: {avg_score * 100:.0f}%"
    )
    if is_pdf and embedded_pages and ocr_pages:
        status += f"  \n🧩 {embedded_pages} صفحة من نص مضمّن (دقة كاملة) و {ocr_pages} صفحة بالـ OCR."
    elif is_pdf and embedded_pages:
        status += "  \n🧩 كل الصفحات من نص مضمّن في الملف (دقة كاملة، لم تُستخدم OCR إطلاقاً)."
    if runner.fell_back:
        status += (
            "  \n🖥️ **تم التحويل إلى المعالج (CPU) تلقائياً** — ذاكرة كرت الرسوم لم تكفِ "
            "(مشغولة ببرامج أخرى كالمتصفح). النتيجة مطابقة لكن أبطأ. "
            "لإعادة استخدام الكرت: أغلق البرامج الثقيلة ثم أعد التشغيل."
        )
    if text_heights:
        median_h = sorted(text_heights)[len(text_heights) // 2]
        if median_h < MIN_RELIABLE_TEXT_HEIGHT:
            status += (
                f"  \n🔍 **تنبيه دقة:** متوسط ارتفاع سطر النص في الصورة {median_h:.0f}px فقط "
                f"(المطلوب {MIN_RELIABLE_TEXT_HEIGHT:.0f}px فأكثر). عند هذا الحجم تختفي نقاط "
                "الحروف العربية (ن/ت/ب/ي) من البكسلات نفسها فتكثر الأخطاء. "
                "الحل هو **إعادة التقاط الصورة بدقة أعلى** — تكبير الصورة الحالية لا يفيد."
            )
    if not table_rows:
        status += "  \n⚠️ لم يُستخرج أي نص. جرّب تغيير اللغة أو خفض حد الثقة."

    docx_path = build_docx(table_rows, page_sizes, run_dir) if table_rows else None
    highlighted, review_summary = build_highlighted(table_rows)
    pages_html = build_pages_html(table_rows)

    return (
        annotated_images,
        table_rows,
        pages_html,
        json_output,
        status,
        txt_path,
        json_path,
        docx_path,
        highlighted,
        review_summary,
        gr.update(visible=bool(table_rows)),
    )


WELCOME_STATUS = (
    "👋 **جاهز للعمل.** ارفع صورة أو ملف PDF من لوحة التحكم، ثم اضغط **تشغيل الاستخراج**.  \n"
    "ملفات PDF التي تحتوي نصاً مضمّناً تُستخرج مباشرة بدقة كاملة دون الحاجة إلى OCR."
)

with gr.Blocks(title="محرك استخراج النصوص — PaddleOCR") as demo:
    gr.HTML(
        """
        <div id="app-header">
            <div class="title">🔎 محرك استخراج النصوص</div>
            <div class="subtitle">
                PaddleOCR — تعرّف ضوئي محلي على الصور وملفات PDF، مع كشف تلقائي للنص المضمّن ودعم كامل للعربية
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # ---------------- لوحة التحكم ----------------
        with gr.Column(scale=4, min_width=320):
            gr.Markdown("المصدر", elem_classes=["section-title"])
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
                run_button = gr.Button("▶️  تشغيل الاستخراج", variant="primary", size="lg")

            with gr.Accordion("⚙️  خيارات متقدمة", open=False):
                pdf_mode_input = gr.Radio(
                    label="طريقة معالجة ملفات PDF",
                    choices=PDF_MODE_CHOICES,
                    value="auto",
                )
                lang_input = gr.Dropdown(
                    label="لغة النص — تُستخدم عند اللجوء إلى OCR فقط",
                    choices=LANG_CHOICES,
                    value="ch",
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

        # ---------------- النتائج ----------------
        with gr.Column(scale=8):
            status_output = gr.Markdown(WELCOME_STATUS, elem_classes=["status-box"])

            with gr.Column(visible=False) as results_panel:
                with gr.Tabs():
                    with gr.Tab("📝  المستند"):
                        text_output = gr.HTML(
                            value="",
                            padding=False,
                        )
                    with gr.Tab("📋  الجدول"):
                        table_output = gr.Dataframe(
                            headers=["#", "النص", "الثقة", "الصفحة", "المصدر"],
                            datatype=["number", "str", "number", "number", "str"],
                            elem_id="results-table",
                            column_widths=["56px", "auto", "84px", "84px", "110px"],
                            max_height=560,
                            show_search="search",
                            wrap=True,
                        )
                    with gr.Tab("🎯  مراجعة الثقة"):
                        review_summary_output = gr.Markdown(
                            "", elem_id="review-summary", elem_classes=["status-box"]
                        )
                        highlight_output = gr.HighlightedText(
                            label=None,
                            color_map={"ثقة منخفضة": "#f59e0b", "ثقة منخفضة جداً": "#ef4444"},
                            show_legend=True,
                            rtl=True,
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
                    download_docx = gr.DownloadButton("📄  Word (.docx)", variant="primary")
                    download_txt = gr.DownloadButton("📝  نص عادي (.txt)")
                    download_json = gr.DownloadButton("🧩  بيانات (.json)")

    run_button.click(
        fn=run_ocr,
        inputs=[file_input, lang_input, thresh_input, hide_empty_input, pdf_mode_input],
        outputs=[
            gallery_output,
            table_output,
            text_output,
            json_output,
            status_output,
            download_txt,
            download_json,
            download_docx,
            highlight_output,
            review_summary_output,
            results_panel,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=THEME,
        css=CUSTOM_CSS,
    )
