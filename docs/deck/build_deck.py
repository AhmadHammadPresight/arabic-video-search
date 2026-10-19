# /// script
# requires-python = ">=3.10"
# dependencies = ["python-pptx==1.0.2"]
# ///
"""Build the Arabic AI Video Search use-case deck.

Run from the repository root:

    uv run docs/deck/build_deck.py

The deck is written next to this script.
Every fact and number comes from the PoC technical plan (report.md and its board);
each slide's speaker notes name the section and the upstream links.
Illustrations are drawn from shapes: no photos of real people are used.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("arabic-video-search-use-cases.pptx")

# Fonts: Aptos for Latin text and Dubai for Arabic.
# Both ship with Microsoft 365 on Windows and macOS, and Dubai shapes Arabic correctly.
LATIN_FONT = "Aptos"
ARABIC_FONT = "Dubai"

NAVY = "0F2A44"
INK = "1F2937"
MUTED = "5B6573"
TEAL = "0E7C7B"
TEAL_LIGHT = "E3F2F1"
AMBER = "B45F06"
AMBER_LIGHT = "FDF1E1"
PANEL = "F3F5F7"
LINE = "D5DBE1"
WHITE = "FFFFFF"
FRAME = "2B3A4A"
FRAME_FLOOR = "3A4B5C"
SHIRT_RED = "C0392B"
MAROON = "7B2D3B"
BAG_BLACK = "111111"
JACKET_BLUE = "2E6FD8"
TROUSER = "4A5563"
SKIN = "C9A27E"
GREY_TOP = "8A96A3"
GREEN_TOP = "3F8F5B"

REPORT_PATH = "/Users/ahmad.hammad/Desktop/Repos/firstmate/data/avs-technical-plan/report.md"
BOARD_PATH = "/Users/ahmad.hammad/Desktop/Repos/firstmate/data/avs-technical-plan/board.html"
SOURCE_LINE = (
    "Source: Arabic AI Video Search PoC technical plan, "
    f"report file://{REPORT_PATH} and board file://{BOARD_PATH}"
)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = 0.6

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


# ---------------------------------------------------------------- text


@dataclass
class R:
    """A run of text with optional overrides of its paragraph's style."""

    text: str
    bold: bool | None = None
    color: str | None = None
    size: float | None = None
    italic: bool | None = None


@dataclass
class P:
    """A paragraph: plain text or a list of runs."""

    runs: str | list[R]
    size: float = 16
    color: str = INK
    bold: bool = False
    align: str = "auto"  # auto: right for right-to-left paragraphs, else left
    rtl: bool | None = None  # None: right-to-left when the text is mostly Arabic
    space_after: float = 4
    space_before: float = 0
    line_spacing: float | None = None
    italic: bool = False

    def run_list(self) -> list[R]:
        return [R(self.runs)] if isinstance(self.runs, str) else self.runs

    def is_rtl(self) -> bool:
        if self.rtl is not None:
            return self.rtl
        text = "".join(r.text for r in self.run_list())
        letters = [c for c in text if c.isalpha()]
        arabic = sum(1 for c in letters if ARABIC_RE.match(c))
        return bool(letters) and arabic / len(letters) > 0.5


ARABIC_SPAN_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+"
    r"(?:\s+[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+)*"
)
LATIN_SPAN_RE = re.compile(r"[A-Za-z0-9]+(?:[\s\-.:'\"]+[A-Za-z0-9]+)*")


def split_by_script(text_: str, rtl: bool) -> list[str]:
    """Split a run into Arabic and Latin pieces.

    Each piece gets its own language tag, so PowerPoint orders mixed lines correctly.
    Neutral characters such as spaces and dots stay with the paragraph's own direction.
    """
    pattern = LATIN_SPAN_RE if rtl else ARABIC_SPAN_RE
    pieces, pos = [], 0
    for m in pattern.finditer(text_):
        if m.start() > pos:
            pieces.append(text_[pos : m.start()])
        pieces.append(m.group())
        pos = m.end()
    if pos < len(text_):
        pieces.append(text_[pos:])
    return pieces or [text_]


def style_run(run, *, size, color, bold, italic, text):
    font = run.font
    font.size = Pt(size)
    font.bold = bold
    font.italic = italic
    font.color.rgb = rgb(color)
    font.name = LATIN_FONT
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", "ar-JO" if ARABIC_RE.search(text) else "en-GB")
    latin = rpr.find(qn("a:latin"))
    for tag, face in (("a:cs", ARABIC_FONT), ("a:ea", LATIN_FONT)):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
            latin.addnext(el)
        el.set("typeface", face)


def fill_text_frame(tf, paras: list[P], *, anchor="top", margins=(0, 0, 0, 0)):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (
        Inches(m) for m in margins
    )
    tf.vertical_anchor = {
        "top": MSO_ANCHOR.TOP,
        "middle": MSO_ANCHOR.MIDDLE,
        "bottom": MSO_ANCHOR.BOTTOM,
    }[anchor]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        rtl = p.is_rtl()
        align = p.align if p.align != "auto" else ("r" if rtl else "l")
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if rtl:
            para._p.get_or_add_pPr().set("rtl", "1")
        para.space_after = Pt(p.space_after)
        para.space_before = Pt(p.space_before)
        if p.line_spacing:
            para.line_spacing = p.line_spacing
        for r in p.run_list():
            for piece in split_by_script(r.text, rtl):
                run = para.add_run()
                run.text = piece
                style_run(
                    run,
                    size=r.size or p.size,
                    color=r.color or p.color,
                    bold=p.bold if r.bold is None else r.bold,
                    italic=p.italic if r.italic is None else r.italic,
                    text=piece,
                )


def text(slide, x, y, w, h, paras, *, anchor="top", margins=(0, 0, 0, 0)):
    if isinstance(paras, P):
        paras = [paras]
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_text_frame(shape.text_frame, paras, anchor=anchor, margins=margins)
    return shape


# ---------------------------------------------------------------- shapes


def shape(slide, kind, x, y, w, h, *, fill=None, line=None, line_w=1.0, radius=None, dash=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
        if dash:
            s.line.dash_style = dash
    else:
        s.line.fill.background()
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def box(slide, x, y, w, h, *, fill=PANEL, line=None, radius=0.08, paras=None, anchor="top", pad=0.18, line_w=1.0):
    s = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, line=line, radius=radius, line_w=line_w)
    if paras is not None:
        fill_text_frame(
            s.text_frame,
            paras if isinstance(paras, list) else [paras],
            anchor=anchor,
            margins=(pad, pad * 0.8, pad, pad * 0.8),
        )
    return s


def circle_label(slide, x, y, d, label, *, fill=TEAL, color=WHITE, size=14):
    s = shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill=fill)
    fill_text_frame(s.text_frame, [P(label, size=size, color=color, bold=True, align="c", space_after=0)], anchor="middle")
    return s


def connector(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    style = c._element.find(qn("p:style"))
    if style is not None:
        c._element.remove(style)  # the default style adds a theme shadow
    return c


def plain_line(slide, x1, y1, x2, y2, *, color=LINE, width=1.0, dash=None):
    c = connector(slide, x1, y1, x2, y2)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = dash
    return c


def chevron(slide, x, y, w=0.32, h=0.42, color=LINE):
    return shape(slide, MSO_SHAPE.CHEVRON, x, y, w, h, fill=color)


def person(slide, x, y, h, *, top=GREY_TOP, bottom=TROUSER, bag=None, bag_side="r", head=SKIN, outline=None):
    """A simple standing figure, x/y is its top-left corner, h its height (inches)."""
    w = 0.42 * h
    cx = x + w / 2
    hd = 0.19 * h
    shape(slide, MSO_SHAPE.OVAL, cx - hd / 2, y, hd, hd, fill=head, line=outline, line_w=0.75)
    tw, th = 0.40 * h, 0.40 * h
    torso = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, cx - tw / 2, y + 0.21 * h, tw, th, fill=top, radius=0.3, line=outline, line_w=0.75)
    lw = 0.30 * h
    shape(slide, MSO_SHAPE.RECTANGLE, cx - lw / 2, y + 0.58 * h, lw, 0.42 * h, fill=bottom, line=outline, line_w=0.75)
    shape(slide, MSO_SHAPE.RECTANGLE, cx - 0.012 * h, y + 0.68 * h, 0.024 * h, 0.32 * h, fill=FRAME if outline else "222B35")
    if bag:
        bw, bh = 0.17 * h, 0.2 * h
        bx = cx + tw / 2 - 0.02 * h if bag_side == "r" else cx - tw / 2 - bw + 0.02 * h
        shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, bx, y + 0.5 * h, bw, bh, fill=bag, radius=0.2)
    return torso


def suitcase(slide, x, y, h, color=BAG_BLACK):
    w = 0.62 * h
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x + w * 0.32, y, w * 0.36, h * 0.18, fill=None, line=color, line_w=1.5, radius=0.3)
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y + h * 0.14, w, h * 0.86, fill=color, radius=0.12)


def camera_icon(slide, x, y, s, color=NAVY):
    shape(slide, MSO_SHAPE.RECTANGLE, x + 0.08 * s, y, 0.1 * s, 0.32 * s, fill=color)
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y + 0.26 * s, 0.78 * s, 0.38 * s, fill=color, radius=0.25)
    shape(slide, MSO_SHAPE.TRAPEZOID, x + 0.74 * s, y + 0.28 * s, 0.26 * s, 0.34 * s, fill=color).rotation = 90


def frame(slide, x, y, w, h, *, label=None, people=(), floor=True):
    """A CCTV-style video frame with simple figures inside.

    people: list of (rel_x, rel_h, kwargs) with rel_x as a fraction of the width
    and rel_h as the figure height relative to the frame height.
    """
    shape(slide, MSO_SHAPE.RECTANGLE, x, y, w, h, fill=FRAME)
    if floor:
        shape(slide, MSO_SHAPE.RECTANGLE, x, y + h * 0.62, w, h * 0.38, fill=FRAME_FLOOR)
    for rel_x, rel_h, kw in people:
        ph = h * rel_h
        if kw.pop("_suitcase", False):
            suitcase(slide, x + w * rel_x, y + h * 0.9 - ph, ph, kw.get("color", BAG_BLACK))
        else:
            person(slide, x + w * rel_x, y + h * 0.92 - ph, ph, outline=None, **kw)
    if label:
        text(
            slide, x + 0.06, y + 0.04, w - 0.12, 0.26,
            P(label, size=9, color=WHITE, bold=True, space_after=0),
        )


# ---------------------------------------------------------------- slide chrome


@dataclass
class Deck:
    prs: Presentation
    total: int
    count: int = 0
    slides: list = field(default_factory=list)

    def new(self, *, kicker=None, title=None, arabic=None, notes=None, dark=False):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.count += 1
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(NAVY if dark else WHITE)
        if not dark:
            shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, 13.333, 0.12, fill=TEAL)
            plain_line(slide, MARGIN, 7.02, 13.333 - MARGIN, 7.02, color=LINE)
            text(
                slide, MARGIN, 7.08, 8, 0.3,
                P("Arabic AI Video Search  ·  PoC use cases", size=10, color=MUTED, space_after=0),
            )
            text(
                slide, 13.333 - MARGIN - 2, 7.08, 2, 0.3,
                P(f"{self.count} / {self.total}", size=10, color=MUTED, align="r", space_after=0),
            )
        if kicker:
            text(slide, MARGIN, 0.38, 8.4, 0.32, P(kicker.upper(), size=12, color=TEAL, bold=True, space_after=0))
        if arabic:
            text(
                slide, 13.333 - MARGIN - 4.2, 0.3, 4.2, 0.45,
                P(arabic, size=18, color=TEAL, bold=True, space_after=0, rtl=True),
            )
        if title:
            text(slide, MARGIN, 0.72, 12.1, 1.0, P(title, size=28, color=NAVY, bold=True, space_after=0, line_spacing=0.95))
        if notes:
            slide.notes_slide.notes_text_frame.text = notes.strip() + "\n\n" + SOURCE_LINE
        self.slides.append(slide)
        return slide


def numbered_steps(slide, x, y, w, steps, *, gap=0.16, size=15, lead_size=16, num_d=0.42, heights=None):
    """Vertical list of numbered steps. steps: list of (lead, body)."""
    cy = y
    for i, (lead, body) in enumerate(steps):
        h = heights[i] if heights else 0.9
        circle_label(slide, x, cy + 0.02, num_d, str(i + 1), size=14)
        paras = [P(lead, size=lead_size, bold=True, color=NAVY, space_after=2)]
        if body:
            paras += body if isinstance(body, list) else [P(body, size=size, color=INK, space_after=0)]
        text(slide, x + num_d + 0.18, cy, w - num_d - 0.18, h, paras)
        cy += h + gap
    return cy


def query_box(slide, x, y, w, arabic, english, *, label="The officer types", h=1.25, ar_size=24):
    text(slide, x, y, w, 0.3, P(label.upper(), size=11, color=MUTED, bold=True, space_after=0))
    box(
        slide, x, y + 0.32, w, h, fill=WHITE, line=TEAL, line_w=1.75, radius=0.12,
        paras=[
            P(arabic, size=ar_size, color=NAVY, bold=True, rtl=True, space_after=2),
            P(english, size=13, color=MUTED, italic=True, align="r", space_after=0, rtl=False),
        ],
        anchor="middle", pad=0.2,
    )
    return y + 0.32 + h


def result_row(slide, x, y, w, *, cam, place, time, score, checks, people, warn=False, h=0.86):
    fw = h * 1.45
    frame(slide, x, y, fw, h, people=people)
    color = AMBER if warn else TEAL
    paras = [
        P([R(f"{place}  ", bold=True, color=NAVY), R(f"{time}", color=INK)], size=13, space_after=1),
        P([R(f"match {score}   ", color=MUTED), R(checks, color=color, bold=True)], size=12, space_after=0),
    ]
    box(slide, x + fw + 0.08, y, w - fw - 0.08, h, fill=AMBER_LIGHT if warn else PANEL, paras=paras, anchor="middle", pad=0.14, radius=0.06)


def time_panel(slide, x, y, w, h, by_hand, with_search, note=None):
    box(slide, x, y, w, h, fill=TEAL_LIGHT, radius=0.06)
    text(slide, x + 0.2, y + 0.14, w - 0.4, 0.3, P("TIME SAVED", size=11, color=TEAL, bold=True, space_after=0))
    paras = [
        P([R("By hand: ", bold=True, color=NAVY), R(by_hand)], size=13, space_after=4),
        P([R("With search: ", bold=True, color=TEAL), R(with_search)], size=13, space_after=0),
    ]
    if note:
        paras.append(P(note, size=10.5, color=MUTED, italic=True, space_before=4, space_after=0))
    text(slide, x + 0.2, y + 0.44, w - 0.4, h - 0.5, paras)


# ---------------------------------------------------------------- slides


def slide_title(deck: Deck):
    s = deck.new(
        dark=True,
        notes="""
Opening.
This deck shows how security teams would use Arabic video search, as three search stories: a suspect, a left bag, and a lost child.
The technical choices behind it are settled in the PoC technical plan (decisions D1 to D7).
The query on the right is the plan's headline example: "a man wearing a red shirt".
""",
    )
    shape(s, MSO_SHAPE.RECTANGLE, 0, 0, 0.18, 7.5, fill=TEAL)
    text(s, 0.9, 1.2, 7.4, 0.4, P("PROOF OF CONCEPT  ·  USE CASES", size=14, color="7FD1CC", bold=True, space_after=0))
    text(s, 0.9, 1.7, 7.6, 1.1, P("Arabic AI Video Search", size=48, color=WHITE, bold=True, space_after=0))
    text(
        s, 0.9, 2.75, 7.4, 0.7,
        P("البحث الذكي في تسجيلات كاميرات المراقبة باللغة العربية", size=24, color="BFE6E3", rtl=True, align="l", space_after=0),
    )
    text(
        s, 0.9, 3.75, 7.0, 1.4,
        [
            P("Search CCTV recordings by describing a person in Arabic.", size=20, color=WHITE, space_after=6),
            P("How a security team would use it: a suspect, a left bag and a lost child.", size=20, color="C9D3DD", space_after=0),
        ],
    )
    text(s, 0.9, 6.45, 7, 0.4, P("October 2026  ·  Discussion draft", size=13, color="9FB0C0", space_after=0))

    # Search bar and two result frames, as a simple illustration.
    x0, y0, w0 = 8.45, 1.55, 4.25
    box(
        s, x0, y0, w0, 0.95, fill=WHITE, radius=0.18,
        paras=[P("رجل يرتدي قميص أحمر", size=24, color=NAVY, bold=True, rtl=True, space_after=0)],
        anchor="middle", pad=0.25,
    )
    text(s, x0, y0 + 1.02, w0, 0.32, P("\"a man wearing a red shirt\"", size=13, color="C9D3DD", italic=True, align="r", space_after=0))
    fy = y0 + 1.6
    for i, (cam, t) in enumerate((("CAM-03  14:02:17", "Main entrance"), ("CAM-07  14:05:40", "Food court"))):
        fx = x0 + i * 2.2
        frame(s, fx, fy, 2.05, 1.45, label=cam, people=[(0.12, 0.62, dict(top=GREY_TOP)), (0.55, 0.78, dict(top=SHIRT_RED, bag=BAG_BLACK))])
        text(s, fx, fy + 1.52, 2.05, 0.3, P(t, size=12, color="C9D3DD", space_after=0))
    box(
        s, x0, fy + 2.0, w0, 0.6, fill="153A5C", radius=0.2,
        paras=[P([R("A person confirms every match", bold=True, color=WHITE)], size=14, align="c", space_after=0)],
        anchor="middle",
    )


def slide_problem(deck: Deck):
    s = deck.new(
        kicker="The problem",
        arabic="المشكلة",
        title="After an incident, the answer sits in hours of footage from many cameras, searched by hand",
        notes="""
The point: the search is manual and grows with every camera and every hour.
The 100 hours figure is our arithmetic from the plan's own example: a missing-child case covering 50 nearby cameras for 2 hours (report section 4, "City scale").
English-only: NVIDIA's open Video Search and Summarization blueprint uses the same search shape but works in English only (report section 1; https://docs.nvidia.com/vss/3.2.1/agent-workflow-search.html).
No public benchmark covers Jordanian Arabic, and no Arabic text-to-person dataset exists (report section 2).
""",
    )
    # Wall of camera tiles.
    gx, gy, tw, th, gap = 0.6, 2.05, 1.38, 0.95, 0.12
    people_sets = [
        [(0.2, 0.6, dict(top=GREY_TOP))],
        [(0.5, 0.7, dict(top=GREEN_TOP))],
        [],
        [(0.15, 0.55, dict(top=MAROON)), (0.6, 0.62, dict(top=GREY_TOP))],
        [(0.4, 0.66, dict(top=JACKET_BLUE))],
        [(0.65, 0.5, dict(top=GREY_TOP))],
        [(0.3, 0.7, dict(top=SHIRT_RED, bag=BAG_BLACK))],
        [],
        [(0.1, 0.6, dict(top=GREEN_TOP)), (0.55, 0.5, dict(top=GREY_TOP))],
        [(0.45, 0.68, dict(top=GREY_TOP))],
        [],
        [(0.25, 0.62, dict(top=JACKET_BLUE))],
    ]
    for i in range(12):
        r, c = divmod(i, 4)
        frame(s, gx + c * (tw + gap), gy + r * (th + gap), tw, th, label=f"CAM-{i + 1:02d}", people=people_sets[i])
    text(
        s, gx, gy + 3 * (th + gap) + 0.05, 4 * tw + 3 * gap, 0.6,
        P("Each camera's recording is scrubbed in turn, for the whole time window.", size=13, color=MUTED, italic=True, space_after=0),
    )

    x = 6.85
    w = 5.85
    points = [
        ("Many cameras, one description", "A witness or a parent describes a person. Staff then play back camera after camera, looking for that person."),
        ("The work grows with every camera and hour", [
            P([R("A lost-child case covering "), R("50 nearby cameras for 2 hours", bold=True), R(" is "), R("100 hours of video", bold=True, color=TEAL), R(" to watch.")], size=15, space_after=0),
        ]),
        ("Search tools that understand descriptions work in English", "The closest open design, NVIDIA's video-search blueprint, works in English only, and no public test covers Jordanian Arabic."),
    ]
    numbered_steps(s, x, 2.05, w, points, heights=[1.05, 1.05, 1.05], gap=0.3)


def slide_what(deck: Deck):
    s = deck.new(
        kicker="What the system does",
        arabic="ماذا يفعل النظام",
        title="Type a description in Arabic. Get the matching clips, with camera, place and time.",
        notes="""
One picture of the whole system, left to right.
1. The officer types a description in Arabic, formal or dialect. Example from the plan: "رجل يرتدي قميص أحمر", a man wearing a red shirt.
2. A local language model turns it into clear fields (person, clothing colours, bag) plus a clean English description (decision D1, hybrid; D4, local only).
3. The search is narrowed to the chosen cameras and time window, then matched against a catalogue of every person already seen in the footage.
4. Results show a short clip, camera, place, time, and why it matched.
5. A person confirms or rejects every result: a result is a lead, not proof (report section 5).
How it works underneath (report section 1): indexing decodes the footage, detects and tracks people, keeps a few best crops per person, describes each one, and stores it with camera, place, time and a pointer to the clip.
""",
    )
    y = 2.0
    cols = [0.6, 3.15, 5.7, 8.25, 10.8]
    cw = 1.95
    heads = [
        ("1", "Type", "اكتب الوصف"),
        ("2", "Understand", "فهم الوصف"),
        ("3", "Search", "البحث"),
        ("4", "Results", "النتائج"),
        ("5", "Confirm", "التأكيد"),
    ]
    for i, (n, en, ar) in enumerate(heads):
        x = cols[i]
        circle_label(s, x, y, 0.46, n, fill=AMBER if i == 4 else TEAL)
        text(s, x + 0.56, y - 0.02, cw, 0.32, P(en, size=17, bold=True, color=NAVY, space_after=0))
        text(s, x + 0.56, y + 0.27, cw - 0.56, 0.3, P(ar, size=12, color=MUTED, rtl=True, align="l", space_after=0))
        if i < 4:
            chevron(s, cols[i + 1] - 0.42, y + 1.55, 0.26, 0.42)

    py = 2.75
    # 1. Type
    box(
        s, cols[0], py, cw + 0.15, 1.1, fill=WHITE, line=TEAL, line_w=1.75, radius=0.15,
        paras=[P("رجل يرتدي قميص أحمر", size=14, bold=True, color=NAVY, rtl=True, align="c", space_after=0)],
        anchor="middle", pad=0.08,
    )
    text(s, cols[0], py + 1.18, cw + 0.15, 0.6, P("\"a man wearing a red shirt\", in formal Arabic or dialect", size=12, color=MUTED, italic=True, space_after=0))
    # 2. Understand
    for j, (ar, en) in enumerate((("رجل", "man"), ("قميص أحمر", "red shirt"))):
        box(
            s, cols[1], py + j * 0.62, cw + 0.15, 0.5, fill=TEAL_LIGHT, radius=0.5,
            paras=[P([R(ar, bold=True, color=NAVY), R("  ·  "), R(en)], size=14, align="c", space_after=0, rtl=False)],
            anchor="middle", pad=0.08,
        )
    text(s, cols[1], py + 1.32, cw + 0.15, 1.2, P("A local language model turns the words, including dialect, into clear fields and plain English.", size=12, color=MUTED, space_after=0))
    # 3. Search
    for j in range(3):
        camera_icon(s, cols[2] + 0.05 + j * 0.68, py + 0.12, 0.55, color=NAVY if j < 2 else LINE)
    box(s, cols[2], py + 0.75, cw + 0.15, 0.48, fill=PANEL, radius=0.5,
        paras=[P("8 cameras · 13:00-15:00", size=12, color=NAVY, align="c", space_after=0)], anchor="middle", pad=0.06)
    text(s, cols[2], py + 1.32, cw + 0.15, 1.2, P("Only the cameras and hours that matter are searched, in a catalogue of every person already seen.", size=12, color=MUTED, space_after=0))
    # 4. Results
    for j, (cam, place, top) in enumerate((("CAM-03", "Entrance · 14:02", SHIRT_RED), ("CAM-07", "Food court · 14:05", SHIRT_RED))):
        frame(s, cols[3], py + j * 0.7, 0.85, 0.6, people=[(0.3, 0.8, dict(top=top))])
        text(s, cols[3] + 0.93, py + j * 0.7 + 0.02, 1.3, 0.6,
             [P(cam, size=12, bold=True, color=NAVY, space_after=0), P(place, size=11, color=MUTED, space_after=0)])
    text(s, cols[3], py + 1.47, cw + 0.15, 1.2, P("Short clips, ranked, each with camera, place, time and why it matched.", size=12, color=MUTED, space_after=0))
    # 5. Confirm
    box(s, cols[4], py, cw, 0.5, fill=TEAL_LIGHT, radius=0.5,
        paras=[P("✓  Confirm", size=14, bold=True, color=TEAL, align="c", space_after=0)], anchor="middle", pad=0.06)
    box(s, cols[4], py + 0.62, cw, 0.5, fill=AMBER_LIGHT, radius=0.5,
        paras=[P("✕  Reject", size=14, bold=True, color=AMBER, align="c", space_after=0)], anchor="middle", pad=0.06)
    text(s, cols[4], py + 1.32, cw, 1.2, P("A person decides. Every result is a lead, not proof, and every search is logged.", size=12, color=MUTED, space_after=0))

    box(
        s, 0.6, 5.35, 12.13, 0.85, fill=PANEL, radius=0.1,
        paras=[P([
            R("Underneath: ", bold=True, color=NAVY),
            R("each recording is first turned into a catalogue of every person seen, with clothing, colours and bags, camera, place and time. "
              "A query searches that catalogue and jumps to the right second of video."),
        ], size=14, space_after=0)],
        anchor="middle", pad=0.25,
    )


def slide_stories(deck: Deck):
    s = deck.new(
        kicker="Three stories",
        arabic="ثلاث حالات استخدام",
        title="One search engine, three ways a security team would use it",
        notes="""
All three stories are appearance searches with the same engine; they differ in what the target looks like and how hard it is to recognise (decision D6, all three as search stories).
Where they apply, from the original brief: finding missing persons or children, assisting investigations, and security in airports, tourism areas and government buildings.
Extra work over the core search (report D6 answer): suspect none; left bag small, since backpack, handbag and suitcase are classes the detector already knows; lost child medium, because age is hard to read from a body.
The automatic left-bag alert is not part of the PoC.
""",
    )
    cards = [
        ("1", "Finding a suspect", "البحث عن مشتبه به", "From a witness description, find the person across a site's cameras.", "زلمة لابس بلوزة حمرا ومعه شنطة سودا", "Investigations · malls · tourist areas",
         [(0.36, 0.86, dict(top=SHIRT_RED, bag=BAG_BLACK))]),
        ("2", "Finding who left a bag", "من ترك الحقيبة؟", "Find the moment an unattended bag appeared, then the person who brought it.", "حقيبة سفر سوداء", "Airports · government buildings",
         [(0.2, 0.84, dict(top=GREY_TOP)), (0.62, 0.34, dict(_suitcase=True))]),
        ("3", "Finding a lost child", "البحث عن طفل مفقود", "From a parent's description, find the most recent sightings near where the child was lost.", "طفل صغير لابس جاكيت أزرق", "Malls · airports · tourist areas",
         [(0.18, 0.86, dict(top=GREY_TOP)), (0.58, 0.52, dict(top=JACKET_BLUE))]),
    ]
    cw, gap, y = 3.85, 0.29, 1.95
    for i, (n, en, ar, desc, q, where, ppl) in enumerate(cards):
        x = 0.6 + i * (cw + gap)
        box(s, x, y, cw, 4.85, fill=PANEL, radius=0.05)
        frame(s, x, y, cw, 1.55, label=f"STORY {n}", people=ppl)
        text(s, x + 0.25, y + 1.72, cw - 0.5, 0.42, P(en, size=19, bold=True, color=NAVY, space_after=0))
        text(s, x + 0.25, y + 2.12, cw - 0.5, 0.4, P(ar, size=16, color=TEAL, bold=True, rtl=True, align="l", space_after=0))
        text(s, x + 0.25, y + 2.6, cw - 0.5, 0.9, P(desc, size=14, space_after=0))
        box(s, x + 0.25, y + 3.5, cw - 0.5, 0.62, fill=WHITE, line=TEAL, radius=0.15,
            paras=[P(q, size=14, bold=True, color=NAVY, rtl=True, space_after=0)], anchor="middle", pad=0.12)
        text(s, x + 0.25, y + 4.27, cw - 0.5, 0.4, P(where, size=12, color=MUTED, space_after=0))


def slide_suspect(deck: Deck):
    s = deck.new(
        kicker="Story 1 · Suspect from a witness description",
        arabic="البحث عن مشتبه به من وصف شاهد",
        title="A witness saw a man in a red top with a black bag. Where is he on our cameras?",
        notes="""
The core search, no extra work beyond the engine itself (report D6 answer).
The query is the plan's Jordanian-dialect example: "زلمة لابس بلوزة حمرا ومعه شنطة سودا", "a guy wearing a red top, with a black bag".
The parser returns fields: man; top red; carrying bag, black; plus English wording, a camera set and a time window (board section 1, "One query, step by step").
Dialect traps the parser must handle: colour forms such as حمرا and سودا (report section 2).
Results shown are an illustration from the plan's demo mock-up; the fourth (amber) result shows why a person reviews every match.
Time saved: "under 2 minutes" is the plan's proposed success measure for the demo stories, not a measured result (report section 6).
Demo footage: MEVA cafe, doors and hall cameras from the same 5-minute window, with a bag theft (report D2 answer; MEVA is CC BY 4.0, https://mevadata.org/resources/MEVA-data-license.txt).
""",
    )
    steps = [
        ("The officer types what the witness said", "In the witness's own words and dialect, with no form to fill."),
        ("The system reads the dialect", "man · red top · black bag. It searches only the site's cameras for the afternoon of the incident."),
        ("Ranked clips appear", "Each shows camera, place, time and why it matched. Near-misses are flagged for a closer look."),
        ("A person confirms, then follows him", "The confirmed clip starts a timeline of where he appeared next, each step confirmed."),
    ]
    numbered_steps(s, 0.6, 1.95, 6.0, steps, heights=[0.78, 0.98, 0.98, 0.98], gap=0.18, size=14)

    rx, rw = 6.95, 5.78
    yq = query_box(s, rx, 1.85, rw, "زلمة لابس بلوزة حمرا ومعه شنطة سودا", "\"a guy wearing a red top, with a black bag\" (Jordanian dialect)", h=1.0, ar_size=22)
    text(s, rx, yq + 0.15, rw, 0.3, P("WHAT THE OFFICER SEES  (ILLUSTRATION)", size=11, color=MUTED, bold=True, space_after=0))
    ry = yq + 0.48
    rows = [
        dict(cam="CAM-03", place="Cam 03 · Main entrance", time="14:02:17", score="0.91", checks="red top ✓  black bag ✓",
             people=[(0.35, 0.82, dict(top=SHIRT_RED, bag=BAG_BLACK))]),
        dict(cam="CAM-07", place="Cam 07 · Food court", time="14:05:40", score="0.88", checks="red top ✓  black bag ✓",
             people=[(0.1, 0.6, dict(top=GREY_TOP)), (0.5, 0.78, dict(top=SHIRT_RED, bag=BAG_BLACK))]),
        dict(cam="CAM-05", place="Cam 05 · Corridor 2", time="13:58:22", score="0.64", checks="maroon, not red?  check",
             people=[(0.4, 0.8, dict(top=MAROON, bag=BAG_BLACK))], warn=True),
    ]
    for i, r in enumerate(rows):
        result_row(s, rx, ry + i * 0.66, rw, h=0.58, **r)
    time_panel(
        s, rx, 5.95, rw, 0.95,
        "every camera for the afternoon, played back one by one.",
        "the target in under 2 minutes (PoC target).",
    )


def slide_bag(deck: Deck):
    s = deck.new(
        kicker="Story 2 · Finding who left a bag",
        arabic="حقيبة متروكة: من تركها؟",
        title="An unattended suitcase is found in a lobby. Who brought it, and where did they go?",
        notes="""
Left bag as a search, small extra work: backpack, handbag and suitcase are object classes the RF-DETR detector already knows (report D6 answer).
Known weakness: small bags far from the camera get missed (board, D6).
An automatic left-bag alert (owner walks away, bag stays) is a separate, optional feature and is not part of the PoC (report D6 answer).
The second query, "a man in a black hoodie", is an illustration; one MEVA clip in the starter set shows a man in a black hoodie wheeling a suitcase away (board D2 clip list).
Demo footage: the MEVA bus-station waiting room clip with the only labelled "person abandons package" in public MEVA data, and a CAVIAR left-bag lobby clip (report D2 answer).
Time saved: "under 2 minutes" is the plan's proposed success measure, not a measured result (report section 6).
""",
    )
    steps = [
        ("Search for the bag itself", "The officer types the bag's description and searches the lobby cameras for the last hour."),
        ("Jump to the moment it appeared", "The clip shows the bag arriving with a person, and being put down."),
        ("Describe that person and search again", "Their clothing becomes the next query, across the site's other cameras."),
        ("Follow them camera to camera", "Each sighting is confirmed by an officer before the next one is used."),
    ]
    numbered_steps(s, 0.6, 1.95, 6.0, steps, heights=[0.78, 0.78, 0.78, 0.78], gap=0.2, size=14)
    box(
        s, 0.6, 5.85, 6.0, 0.85, fill=PANEL, radius=0.08,
        paras=[P([R("A search the officer starts. ", bold=True, color=NAVY),
                  R("An automatic left-bag alarm is a separate feature and is not part of the PoC.")], size=13, space_after=0)],
        anchor="middle", pad=0.2,
    )

    rx, rw = 6.95, 5.78
    yq = query_box(s, rx, 1.85, rw, "حقيبة سفر سوداء", "\"a black suitcase\"", h=0.85, ar_size=22)
    text(s, rx, yq + 0.15, rw, 0.3, P("THE LOBBY CAMERA, REWOUND  (ILLUSTRATION)", size=11, color=MUTED, bold=True, space_after=0))
    fy, fw, fh = yq + 0.5, 1.82, 1.25
    shots = [
        ("13:41", "Lobby empty", [(0.6, 0.55, dict(top=GREY_TOP))]),
        ("13:47", "Arrives with bag", [(0.25, 0.8, dict(top=BAG_BLACK, bottom=TROUSER)), (0.62, 0.3, dict(_suitcase=True))]),
        ("13:49", "Bag left alone", [(0.62, 0.3, dict(_suitcase=True))]),
    ]
    for i, (t, cap, ppl) in enumerate(shots):
        fx = rx + i * (fw + 0.16)
        frame(s, fx, fy, fw, fh, label=f"CAM-02  {t}", people=ppl)
        text(s, fx, fy + fh + 0.05, fw, 0.3, P(cap, size=12, color=NAVY if i == 1 else MUTED, bold=i == 1, space_after=0))
    shape(s, MSO_SHAPE.RECTANGLE, rx + fw + 0.16 - 0.04, fy - 0.04, fw + 0.08, fh + 0.08, line=AMBER, line_w=2.25)
    yq2 = fy + fh + 0.45
    box(
        s, rx, yq2, rw, 0.62, fill=WHITE, line=TEAL, line_w=1.5, radius=0.15,
        paras=[P([R("Next search:  ", color=MUTED, size=12), R("رجل يرتدي هودي أسود", bold=True, color=NAVY, size=17), R("  \"a man in a black hoodie\"", color=MUTED, size=12, italic=True)], align="c", space_after=0, rtl=False)],
        anchor="middle", pad=0.12,
    )
    time_panel(
        s, rx, 5.95, rw, 0.95,
        "rewind every lobby camera until the bag first appears.",
        "straight to that moment; under 2 minutes (PoC target).",
    )


def slide_child(deck: Deck):
    s = deck.new(
        kicker="Story 3 · Finding a lost child",
        arabic="البحث عن طفل مفقود",
        title="A parent has lost a small child in a blue jacket near Gate 3",
        notes="""
This is appearance search: clothing, colours and size. It never uses face recognition (decision D5: "out of scope of course ... we need to search people based on appearance").
Medium extra work over the core search: age is hard to read from a body; even trained models score F1 59-76 for "under 16" (report D6 answer; https://arxiv.org/pdf/2509.02099).
So the system shows candidates, not answers, with a vision-model check and a height cue, and an officer reviews each one (board D6).
The query is from the plan's mixed-Arabic example: "طفل صغير لابس جاكيت أزرق", a small child wearing a blue jacket.
Time saved, our arithmetic from the plan's own example (report section 4): 50 nearby cameras for 2 hours is 100 hours of video; at 5 frames per second it is 1.8M frames, about 22 minutes on one NVIDIA L4 GPU or 6 on four, before the describe step (https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Performance.html).
Those are production estimates, not PoC measurements.
Demo: no usable public CCTV footage contains children, so the demo uses an adult stand-in until a consented child scene exists (decisions D2 and D6).
""",
    )
    box(
        s, 0.6, 1.85, 12.13, 0.55, fill=AMBER_LIGHT, radius=0.5,
        paras=[P([R("Appearance search only: ", bold=True, color=AMBER), R("clothing, colours and size. "), R("Never face recognition.", bold=True, color=NAVY)], size=15, align="c", space_after=0)],
        anchor="middle", pad=0.1,
    )
    steps = [
        ("The officer types the parent's description", "Search starts at the cameras nearest Gate 3, for the last hour, newest first."),
        ("Candidates, not answers", "Telling a child from an adult by body is hard, so a model check and a height cue flag likely children for review."),
        ("Each confirmed sighting moves the search", "It becomes the new starting point; the search widens ring by ring only if nothing is found."),
    ]
    numbered_steps(s, 0.6, 2.7, 6.0, steps, heights=[0.85, 0.98, 0.98], gap=0.2, size=14)
    text(
        s, 0.6, 5.95, 6.0, 0.9,
        P("In the demo an adult stands in for the child: public CCTV footage contains no children, and a consented recording comes later only if needed.",
          size=12, color=MUTED, italic=True, space_after=0),
    )

    rx, rw = 6.95, 5.78
    yq = query_box(s, rx, 2.6, rw, "طفل صغير لابس جاكيت أزرق", "\"a small child wearing a blue jacket\"", h=0.85, ar_size=22)
    ry = yq + 0.2
    frame(s, rx, ry, 2.0, 1.35, label="CAM-14  15:31", people=[(0.12, 0.8, dict(top=GREY_TOP)), (0.55, 0.48, dict(top=JACKET_BLUE))])
    box(
        s, rx + 2.12, ry, rw - 2.12, 1.35, fill=PANEL, radius=0.06,
        paras=[
            P([R("Cam 14 · Gate 3 corridor", bold=True, color=NAVY), R("  15:31")], size=13, space_after=2),
            P([R("blue jacket ✓  small ✓", color=TEAL, bold=True)], size=12, space_after=2),
            P("Likely a child: an officer checks the clip.", size=12, color=AMBER, space_after=2),
            P("Illustration", size=10, color=MUTED, italic=True, space_after=0),
        ],
        anchor="middle", pad=0.16,
    )
    time_panel(
        s, rx, 5.47, rw, 1.43,
        "50 nearby cameras for 2 hours is 100 hours of video.",
        "those cameras indexed in about 22 min on one data-centre GPU, 6 on four.",
        note="Our arithmetic from the plan's example, for a production system.",
    )


def slide_city(deck: Deck):
    s = deck.new(
        kicker="Across many cameras",
        arabic="من عدة كاميرات إلى مدينة كاملة",
        title="A search never starts with \"all of Amman\": it starts from a place and a time",
        notes="""
How search scales (report section 4): process at each site, send only a small index centrally, fetch clips on demand, and always narrow by place and time, widening camera by camera along a camera map (a camera topology graph).
Following a person between cameras is an assisted, after-the-fact search: in the AI City Challenge 2025 the best after-the-fact (offline) HOTA was about 70 and the best live (online) 28.8 (https://arxiv.org/html/2508.13564v1). HOTA is a 0-100 tracking accuracy score.
Index size, our arithmetic in the plan (1080p, 15 fps, H.265, about 2 Mbps, 30-day retention): 1,000 cameras keep about 650 TB of video but only about 2.6 TB of search index, about 8 Mbps to the centre.
Camera names and walking times in the diagram are illustrative.
""",
    )
    # Ring diagram.
    cx, cy, ro, ri = 3.45, 4.2, 2.05, 1.2
    for r, color in ((ro, LINE), (ri, TEAL)):
        shape(s, MSO_SHAPE.OVAL, cx - r, cy - r, 2 * r, 2 * r, fill=None, line=color, line_w=1.5, dash=7)
    inner = [("C9", 35), ("C13", 125), ("C15", 215), ("C7", 305)]
    outer = [("C3", 80), ("C21", 170), ("C30", 260), ("C4", 350)]
    for name, ang in outer:
        a = math.radians(ang)
        px, py = cx + ro * math.cos(a), cy - ro * math.sin(a)
        plain_line(s, cx, cy, px, py, color=LINE, width=1)
    for name, ang in inner:
        a = math.radians(ang)
        px, py = cx + ri * math.cos(a), cy - ri * math.sin(a)
        plain_line(s, cx, cy, px, py, color=TEAL, width=1.25)
    for name, ang in outer:
        a = math.radians(ang)
        circle_label(s, cx + ro * math.cos(a) - 0.28, cy - ro * math.sin(a) - 0.28, 0.56, name, fill="B8C2CC", color=NAVY, size=11)
    for name, ang in inner:
        a = math.radians(ang)
        circle_label(s, cx + ri * math.cos(a) - 0.3, cy - ri * math.sin(a) - 0.3, 0.6, name, fill=TEAL, size=11)
    circle_label(s, cx - 0.42, cy - 0.42, 0.84, "C12", fill=NAVY, size=14)
    text(s, cx - 1.2, cy + 0.45, 2.4, 0.3, P("last seen 14:05", size=11, color=NAVY, bold=True, align="c", space_after=0))
    text(s, 0.6, 6.62, 6.0, 0.3, P([R("●  ", color=TEAL), R("searched first: under 5 min walk      "), R("●  ", color="B8C2CC"), R("next ring: 5-15 min      illustrative")], size=11, color=MUTED, space_after=0))

    x, w = 6.85, 5.88
    steps = [
        ("Start from the last sighting", "The report, or the first confirmed match: one camera, one time."),
        ("Search the nearest cameras first", "A map of which cameras lead to which, with walking times, sets the order."),
        ("Widen ring by ring", "Only if nothing is found. Each confirmed match becomes the new starting point."),
        ("\"Where did they go next?\" is assisted", [P([R("Live tracking across a city is unsolved: the best live system scored "), R("28.8", bold=True), R(", against about "), R("70", bold=True), R(" after the fact (2025 benchmark). So a person confirms each step.")], size=14, space_after=0)]),
    ]
    numbered_steps(s, x, 1.95, w, steps, heights=[0.72, 0.72, 0.72, 1.18], gap=0.16, size=14)
    box(
        s, x, 5.85, w, 0.95, fill=TEAL_LIGHT, radius=0.08,
        paras=[P([R("Video stays where it was recorded. ", bold=True, color=NAVY),
                  R("Only a small index travels: for 1,000 cameras over 30 days, about 2.6 TB of index against about 650 TB of video.")], size=13, space_after=0)],
        anchor="middle", pad=0.2,
    )


def slide_scale(deck: Deck):
    s = deck.new(
        kicker="From one site to a city",
        arabic="ما هو الواقعي",
        title="A site pilot is realistic; a whole city is a government programme we plug into",
        notes="""
What is realistic (report section 4): the PoC and a single-site pilot are realistic; a district needs a public-sector sponsor; city-wide coverage is a government programme, and we should build the search layer that plugs into one.
Table detail from the board, section 4, "What is realistic, and what needs a national programme".
Amman context: no published count of all cameras exists; GAM runs about 5,600 traffic cameras (https://www.jordannews.jo/Section-109/News/GAM-5-600-Traffic-Monitoring-Cameras-in-Operation-Only-25-Dedicated-to-Traffic-Violations-50992); GAM tendered about 3,100 more in Feb 2026, with categories that include facial recognition.
Precedents: Dubai Oyoon has more than 300,000 cameras; Singapore more than 100,000 police cameras.
""",
    )
    rows = [
        ("Scale", "Example", "What it takes", "Realistic for us?"),
        ("PoC", "A few cameras, uploaded recordings", "This plan, on one computer", "Yes: this plan"),
        ("Site pilot", "One mall, airport terminal or ministry", "On-site GPU server, the site's video system, the owner as data controller, an impact assessment", "Yes, with a partner"),
        ("District", "About 1,000 cameras, many owners", "A box at each site, a central index, data-sharing agreements", "Needs a public-sector sponsor"),
        ("Whole city", "Tens of thousands of cameras, thousands of owners", "A legal mandate, camera standards, a secure network, an operations centre, oversight", "Only as a government programme"),
    ]
    widths = [1.45, 2.35, 3.2, 1.6]
    x0, y0 = 0.6, 1.95
    heights = [0.45, 0.62, 0.95, 0.78, 0.95]
    verdict_colors = [None, TEAL, TEAL, AMBER, NAVY]
    y = y0
    for ri, row in enumerate(rows):
        x = x0
        h = heights[ri]
        for ci, cell in enumerate(row):
            w = widths[ci]
            if ri == 0:
                box(s, x, y, w - 0.06, h, fill=NAVY, radius=0.06, paras=[P(cell, size=12, bold=True, color=WHITE, space_after=0)], anchor="middle", pad=0.12)
            else:
                fill = PANEL
                color = INK
                bold = ci == 0
                if ci == 3:
                    fill = {TEAL: TEAL_LIGHT, AMBER: AMBER_LIGHT, NAVY: "E4E9EF"}[verdict_colors[ri]]
                    color = verdict_colors[ri]
                    bold = True
                box(s, x, y, w - 0.06, h, fill=fill, radius=0.06,
                    paras=[P(cell, size=12.5, bold=bold, color=NAVY if ci == 0 else color, space_after=0)], anchor="middle", pad=0.12)
            x += w
        y += h + 0.06

    px, pw = 9.5, 3.23
    box(s, px, 1.95, pw, 4.95, fill=TEAL_LIGHT, radius=0.04)
    text(s, px + 0.22, 2.1, pw - 0.44, 0.35, P("AMMAN TODAY", size=11, color=TEAL, bold=True, space_after=0))
    stats = [
        ("about 5,600", "traffic cameras run by the Greater Amman Municipality"),
        ("about 3,100", "more tendered in February 2026"),
        ("no count", "has been published of all public and private cameras"),
    ]
    sy = 2.45
    for big, small in stats:
        text(s, px + 0.22, sy, pw - 0.44, 0.95,
             [P(big, size=22, bold=True, color=NAVY, space_after=0), P(small, size=12, color=INK, space_after=0)])
        sy += 0.95
    plain_line(s, px + 0.22, sy + 0.02, px + pw - 0.22, sy + 0.02, color="B9DDDA")
    text(s, px + 0.22, sy + 0.12, pw - 0.44, 1.2,
         [P("For comparison", size=11, color=TEAL, bold=True, space_after=2),
          P("Dubai's Oyoon links more than 300,000 cameras; Singapore has more than 100,000 police cameras.", size=12, color=INK, space_after=0)])
    box(
        s, 0.6, 6.05, 8.66, 0.8, fill=WHITE, line=TEAL, line_w=1.25, radius=0.1,
        paras=[P([R("Our part: ", bold=True, color=TEAL), R("the Arabic search layer, built so it can plug into a city programme later.")], size=15, space_after=0)],
        anchor="middle", pad=0.2,
    )


def slide_safeguards(deck: Deck):
    s = deck.new(
        kicker="Safeguards",
        arabic="الضمانات والخصوصية",
        title="Built in from day one: appearance only, people decide, every search on record",
        notes="""
Report section 5, "Legal, privacy and ethics". This is not legal advice: a Jordanian lawyer should review before any pilot with real footage.
No face recognition: decision D5.
Lead, not proof: false matches are expected; the Robert Williams wrongful arrest and NIST's 10-100x demographic differentials in face recognition are the reasons for the design rule. Every result shows its confidence and why it matched, and is confirmed by people, two for any action.
Audit: role-based access, a case number per search, an append-only audit log; the index is deleted with the footage; faces blurred in exports; a data protection impact assessment before any real-footage pilot.
Data in Jordan: the PoC uses local models only, no hosted models even for testing (decision D4).
Jordan PDPL No. 24 of 2023: in force since 17 Mar 2024, full compliance since 17 Mar 2025. CCTV is personal data; biometric data is sensitive; consent must be explicit and documented, or an Art. 6 exception must apply; transfers abroad are restricted (Art. 15); breaches notified within 72 hours. Official translation: https://www.modee.gov.jo/EBV4.0/Root_Storage/EN/1/PDP_Law_-_English_Version-_officail_translation.pdf
""",
    )
    cards = [
        ("No face recognition", "لا للتعرف على الوجوه", "Searches clothing, colours, bags and size. It never identifies who someone is."),
        ("A lead, not proof", "نتيجة للتحقق لا دليل", "Each result shows its confidence and why it matched. A person confirms; two people before any action."),
        ("Every search on record", "سجل تدقيق", "A case number for each search and an append-only log: who searched, the exact query, what they opened."),
        ("Data stays in Jordan", "البيانات داخل الأردن", "Models run locally. The PoC sends nothing to any cloud AI service."),
        ("Jordan's data law (PDPL)", "قانون حماية البيانات الشخصية", "Law No. 24 of 2023: CCTV is personal data; consent or a legal exception; transfers abroad restricted; breaches reported within 72 hours."),
    ]
    cw, gap, y, h = 2.29, 0.17, 1.95, 3.95
    for i, (en, ar, body) in enumerate(cards):
        x = 0.6 + i * (cw + gap)
        box(s, x, y, cw, h, fill=PANEL, radius=0.05)
        shape(s, MSO_SHAPE.RECTANGLE, x, y, cw, 0.08, fill=AMBER if i == 0 else TEAL)
        if i == 0:
            shape(s, MSO_SHAPE.NO_SYMBOL, x + 0.22, y + 0.3, 0.55, 0.55, fill=AMBER)
        else:
            circle_label(s, x + 0.22, y + 0.3, 0.55, str(i + 1), fill=TEAL, size=15)
        text(s, x + 0.22, y + 1.0, cw - 0.4, 0.62, P(en, size=16, bold=True, color=NAVY, space_after=0, line_spacing=0.95))
        text(s, x + 0.22, y + 1.62, cw - 0.4, 0.36, P(ar, size=11.5, color=TEAL, bold=True, rtl=True, align="l", space_after=0))
        text(s, x + 0.22, y + 2.05, cw - 0.4, 1.85, P(body, size=13, space_after=0))
    box(
        s, 0.6, 6.08, 12.13, 0.78, fill=AMBER_LIGHT, radius=0.08,
        paras=[P([R("Also: ", bold=True, color=AMBER),
                  R("the search index is deleted with the footage, other people's faces are blurred in exports, and an impact assessment comes before any pilot on real footage. "
                    "Not legal advice: a Jordanian lawyer reviews first.")], size=13, space_after=0)],
        anchor="middle", pad=0.2,
    )


def slide_demo(deck: Deck):
    s = deck.new(
        kicker="The MVP demo",
        arabic="العرض التجريبي",
        title="Upload recordings, search in Arabic, and get clips with timestamps",
        notes="""
MVP scope (report section 6): upload about 10 short public clips, Arabic search across formal, Jordanian and Gulf phrasing, a results grid with clip, camera, place, time and why it matched, confirm or reject, a simple next-sighting timeline, and an audit log.
Out of scope: live streams (D7), face recognition (D5), city scale, vehicles and voice.
Where it runs (D3): one local computer. App services in Docker; the models run directly on the computer, because Docker containers there cannot use its GPU.
Footage (D2, D7): recordings from public datasets, no live cameras: MEVA (Kitware and IARPA, CC BY 4.0, https://mevadata.org/resources/MEVA-data-license.txt), NVIDIA PhysicalAI-SmartSpaces synthetic store scenes (CC BY 4.0), and a CAVIAR left-bag clip (Creative Commons BY-SA). Clips are trimmed; attribution is shown in the demo.
The screen on the right is an illustration from the plan's demo mock-up, not a screenshot.
""",
    )
    steps = [
        ("Upload", "About 10 short recordings, each tagged with camera, place and start time."),
        ("Search in Arabic", "Formal Arabic, Jordanian or Gulf phrasing."),
        ("See the clips", "Ranked, with camera, place, time and why each matched."),
        ("Confirm", "The timeline shows where the person appeared next. Every search is logged."),
    ]
    numbered_steps(s, 0.6, 1.95, 5.3, steps, heights=[0.75, 0.55, 0.75, 0.75], gap=0.18, size=14)

    # Mock search screen.
    mx, my, mw, mh = 6.3, 1.9, 6.43, 4.0
    box(s, mx, my, mw, mh, fill="EEF1F4", line=LINE, radius=0.03)
    shape(s, MSO_SHAPE.RECTANGLE, mx, my, mw, 0.32, fill=NAVY)
    text(s, mx + 0.15, my + 0.03, mw - 0.3, 0.26, P("Video search  ·  البحث في الفيديو", size=11, color=WHITE, bold=True, space_after=0, rtl=False))
    box(s, mx + 0.2, my + 0.48, mw - 1.45, 0.55, fill=WHITE, line=TEAL, line_w=1.25, radius=0.2,
        paras=[P("زلمة لابس بلوزة حمرا ومعه شنطة سودا", size=16, bold=True, color=NAVY, rtl=True, space_after=0)], anchor="middle", pad=0.12)
    box(s, mx + mw - 1.15, my + 0.48, 0.95, 0.55, fill=TEAL, radius=0.2,
        paras=[P("بحث", size=15, bold=True, color=WHITE, align="c", rtl=True, space_after=0)], anchor="middle", pad=0.05)
    chips = [("رجل · man", TEAL_LIGHT, 0.95), ("بلوزة حمرا · red top", TEAL_LIGHT, 1.4),
             ("شنطة سودا · black bag", TEAL_LIGHT, 1.55), ("8 cameras · 13:00-15:00", WHITE, 1.75)]
    cx = mx + 0.2
    for label, fill, cwid in chips:
        box(s, cx, my + 1.15, cwid, 0.36, fill=fill, line=LINE if fill == WHITE else None, radius=0.5,
            paras=[P(label, size=10, color=NAVY, align="c", space_after=0, rtl=False)], anchor="middle", pad=0.04)
        cx += cwid + 0.08
    rows = [
        dict(cam="CAM-03", place="Cam 03 · Main entrance", time="14:02:17", score="0.91", checks="red top ✓  black bag ✓",
             people=[(0.35, 0.82, dict(top=SHIRT_RED, bag=BAG_BLACK))]),
        dict(cam="CAM-07", place="Cam 07 · Food court", time="14:05:40", score="0.88", checks="red top ✓  black bag ✓",
             people=[(0.5, 0.78, dict(top=SHIRT_RED, bag=BAG_BLACK))]),
        dict(cam="CAM-12", place="Cam 12 · Car park B", time="14:11:03", score="0.81", checks="red top ✓  bag ✓",
             people=[(0.3, 0.7, dict(top=SHIRT_RED, bag=BAG_BLACK))]),
        dict(cam="CAM-05", place="Cam 05 · Corridor 2", time="13:58:22", score="0.64", checks="maroon, not red?  check",
             people=[(0.4, 0.8, dict(top=MAROON, bag=BAG_BLACK))], warn=True),
    ]
    for i, r in enumerate(rows):
        result_row(s, mx + 0.2, my + 1.65 + i * 0.57, mw - 0.4, h=0.5, **r)
    text(s, mx, my + mh + 0.04, mw, 0.28, P("Illustration of the demo's search screen, not a screenshot.", size=10, color=MUTED, italic=True, space_after=0))

    box(
        s, 0.6, 6.25, 12.13, 0.65, fill=TEAL_LIGHT, radius=0.08,
        paras=[P([R("Runs on one computer: ", bold=True, color=NAVY),
                  R("the app in Docker, the AI models on the computer itself. Recorded clips from public datasets, no live cameras, and no cloud AI.")], size=13.5, space_after=0)],
        anchor="middle", pad=0.2,
    )
    text(
        s, 0.6, 5.55, 5.3, 0.6,
        P("Footage: MEVA (Kitware, IARPA; CC BY 4.0), NVIDIA synthetic store scenes (CC BY 4.0), CAVIAR (CC BY-SA). Actors and synthetic people only.",
          size=10.5, color=MUTED, italic=True, space_after=0),
    )


def slide_measures(deck: Deck):
    s = deck.new(
        kicker="How we will know it works",
        arabic="معايير النجاح",
        title="The proposed targets against phase 2: three met, two missed, one not yet measured",
        notes="""
Targets as proposed in the plan (report section 6), still to be agreed. Each is paired with what phase 2 measured, from the project README, section "Evaluation: phase 1 and phase 2 side by side", and its timings of the clean run.
Test set: 113 queries written by us on the demo clips, 110 of them with a target (three match nobody), across formal Arabic, Jordanian, Gulf, mixed Arabic-English and English. No public benchmark covers dialect queries against images (report section 2).
Top 10: the hybrid finds a correct track for 109 of 110 (99%), option A for 100 (91%). The fusion settings were tuned on the first 47 scored queries; on the 63 written afterwards the hybrid scores 98% and option A 95%, the fairer estimate.
Dialect gap, measured on top-10 hit rate: hybrid Jordanian 100%, Gulf 100%, formal 100%; option A Jordanian 80% against formal 97%.
3 of 5: precision at 5 is 0.51 for the hybrid's first ranking, 0.49 after the double-check, 0.44 for option A. It is capped for targets seen in fewer than 5 tracks.
Indexing: the 11 starter clips (486 tracks) took 44 minutes, 36.5 of them for the vision model's attributes, against about 7 minutes without them.
Speed: hybrid first results median 2.1 s, 90th percentile 2.3 s, under 5 s for 112 of 113 searches; the double-check of the top 8 adds a median 17.2 s and fills in after.
Officer time per story: not measured in phase 2.
""",
    )
    status_style = {
        "met": ("MET", TEAL, TEAL_LIGHT),
        "missed": ("MISSED", AMBER, AMBER_LIGHT),
        "none": ("NOT MEASURED", MUTED, "E4E9EF"),
    }
    tiles = [
        ("80%", "of test queries find the right person in the top 10",
         "99% with phase 2, 91% with option A. 98% on the 63 queries written after tuning.", "met"),
        ("10 points", "the largest gap allowed between dialect and formal Arabic",
         "no gap with phase 2: Jordanian, Gulf and formal all 100%. Option A: 17 points.", "met"),
        ("3 of 5", "top results correct, on average",
         "about 2.5 of 5 (0.49 after the double-check). Capped where fewer than 5 are correct.", "missed"),
        ("< 20 min", "to index the demo clips",
         "44 min for the 11 starter clips, 36.5 of them for clothing details; about 7 without.", "missed"),
        ("< 5 s", "until results appear; double-checks fill in after",
         "2.1 s median, under 5 s for 112 of 113 searches. The double-check adds about 17 s.", "met"),
        ("< 2 min", "for an officer to find the target of each story",
         "not yet; phase 2 had no timed run with an officer.", "none"),
    ]
    tw, th, gx, gy = 3.9, 1.88, 0.215, 0.16
    for i, (big, small, measured, status) in enumerate(tiles):
        label, accent, light = status_style[status]
        r, c = divmod(i, 3)
        x = 0.6 + c * (tw + gx)
        y = 1.9 + r * (th + gy)
        box(s, x, y, tw, th, fill=PANEL, radius=0.06)
        shape(s, MSO_SHAPE.RECTANGLE, x, y + 0.2, 0.07, th - 0.4, fill=accent)
        text(s, x + 0.3, y + 0.14, 2.2, 0.5, P(big, size=26, bold=True, color=NAVY, space_after=0))
        cw = 0.95 if status != "none" else 1.45
        box(s, x + tw - 0.2 - cw, y + 0.2, cw, 0.3, fill=light, line=accent, line_w=1.0, radius=0.5,
            paras=[P(label, size=9.5, bold=True, color=accent, align="c", space_after=0)], anchor="middle", pad=0.03)
        text(s, x + 0.3, y + 0.62, tw - 0.5, 0.48, P(small, size=12, color=MUTED, space_after=0, line_spacing=0.95))
        text(s, x + 0.3, y + 1.1, tw - 0.5, 0.7,
             P([R("Measured: ", bold=True, color=accent), R(measured, color=INK)], size=12.5, space_after=0, line_spacing=0.95))
    box(
        s, 0.6, 5.95, 12.13, 0.88, fill=AMBER_LIGHT, radius=0.08,
        paras=[P([R("Main risk: ", bold=True, color=AMBER),
                  R("no public test exists for Jordanian or Gulf Arabic against video, so the 113 test queries are our own. "
                    "Settings were tuned on the first 48; the 63 written later are the fairer check (98% top 10, option A 95%).")], size=13, space_after=0)],
        anchor="middle", pad=0.22,
    )


# ---------------------------------------------------------------- build


def set_theme_fonts(prs: Presentation):
    """Make Aptos and Dubai the theme fonts, so any default text matches the deck."""
    master = prs.slide_masters[0]
    for rel in master.part.rels.values():
        if rel.reltype.endswith("/theme"):
            part = rel.target_part
            root = etree.fromstring(part.blob)
            for group in ("a:majorFont", "a:minorFont"):
                g = root.find(f".//{qn(group)}")
                g.find(qn("a:latin")).set("typeface", LATIN_FONT)
                g.find(qn("a:cs")).set("typeface", ARABIC_FONT)
            part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


SLIDES = [
    slide_title,
    slide_problem,
    slide_what,
    slide_stories,
    slide_suspect,
    slide_bag,
    slide_child,
    slide_city,
    slide_scale,
    slide_safeguards,
    slide_demo,
    slide_measures,
]


def build(out: Path = OUT) -> Path:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    set_theme_fonts(prs)
    props = prs.core_properties
    props.title = "Arabic AI Video Search: PoC use cases"
    props.subject = "How security teams would use Arabic video search"
    props.author = ""
    props.last_modified_by = ""
    props.created = props.modified = dt.datetime(2026, 10, 17)
    props.revision = 1
    deck = Deck(prs, total=len(SLIDES))
    for make in SLIDES:
        make(deck)
    prs.save(out)
    return out


if __name__ == "__main__":
    print(build())
