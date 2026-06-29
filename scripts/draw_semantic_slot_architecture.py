from html import escape
from pathlib import Path


W, H = 1800, 1120
OUT = Path("screenshots/semantic_slot_architecture.svg")

COLORS = {
    "bg": "#EFE7DB",
    "paper": "#FBF8F2",
    "text": "#2F2B28",
    "muted": "#746D66",
    "line": "#8C837A",
    "encoder": "#6F8FB6",
    "encoder_light": "#DCE7F2",
    "decoder": "#8FAE9A",
    "decoder_light": "#E4EEE7",
    "mint": "#9FC2B4",
    "mint_light": "#E5F1ED",
    "coral": "#D08C7A",
    "coral_light": "#F2DCD5",
    "lavender": "#A8A0C8",
    "lavender_light": "#E8E4F3",
    "sand": "#F6EFE5",
    "white": "#FFFDF8",
}


def attrs(**kwargs):
    return " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in kwargs.items())


class SVG:
    def __init__(self):
        self.parts = []

    def add(self, markup):
        self.parts.append(markup)

    def rect(self, x, y, w, h, rx=18, fill="none", stroke=None, sw=2, cls=None, opacity=1, dashed=False, filter_=None):
        extra = []
        if stroke:
            extra.append(f'stroke="{stroke}" stroke-width="{sw}"')
        if dashed:
            extra.append('stroke-dasharray="10 8"')
        if cls:
            extra.append(f'class="{cls}"')
        if filter_:
            extra.append(f'filter="{filter_}"')
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" opacity="{opacity}" {" ".join(extra)}/>'
        )

    def line(self, x1, y1, x2, y2, color=None, sw=3, dashed=False, arrow=True, opacity=1):
        color = color or COLORS["line"]
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" opacity="{opacity}"{dash}{marker}/>'
        )

    def path(self, d, color=None, sw=3, dashed=False, arrow=True, fill="none", opacity=1):
        color = color or COLORS["line"]
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash}{marker}/>'
        )

    def text(self, x, y, lines, size=28, weight=500, fill=None, anchor="middle", line_gap=1.25, italic=False):
        fill = fill or COLORS["text"]
        if isinstance(lines, str):
            lines = [lines]
        style = "font-style:italic;" if italic else ""
        self.add(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" style="{style}">'
        )
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else size * line_gap
            self.add(f'<tspan x="{x}" dy="{dy}">{escape(str(line))}</tspan>')
        self.add("</text>")

    def pill(self, x, y, w, h, label, fill, stroke=None, size=20, weight=700):
        self.rect(x, y, w, h, rx=h / 2, fill=fill, stroke=stroke or fill, sw=1.5)
        self.text(x + w / 2, y + h / 2 + size * 0.36, label, size=size, weight=weight)

    def render(self):
        return "\n".join(self.parts)


def tensor_stack(svg, x, y, w, h, count, fill, stroke, dx=10, dy=-7):
    for i in range(count - 1, -1, -1):
        svg.rect(x + i * dx, y + i * dy, w, h, rx=8, fill=fill, stroke=stroke, sw=1.8, opacity=0.92)


def token_grid(svg, x, y, cols=14, rows=14, cell=9, gap=4, fill="#DCE7F2", stroke="#6F8FB6"):
    for r in range(rows):
        for c in range(cols):
            tone = fill
            if (r + c) % 7 == 0:
                tone = "#C7D8EA"
            svg.rect(x + c * (cell + gap), y + r * (cell + gap), cell, cell, rx=2, fill=tone, stroke=stroke, sw=0.7, opacity=0.95)


def token_row(svg, x, y, labels, fill, stroke, size=17, gap=7):
    cur = x
    for label in labels:
        w = max(46, 12 * len(label) + 18)
        svg.rect(cur, y, w, 34, rx=7, fill=fill, stroke=stroke, sw=1.2)
        svg.text(cur + w / 2, y + 22, label, size=size, weight=650)
        cur += w + gap
    return cur


def metric_card(svg, x, y, title, before, after, delta):
    svg.rect(x, y, 286, 96, rx=16, fill=COLORS["white"], stroke="#E0D5CA", sw=1.5, filter_="url(#softShadow)")
    svg.text(x + 22, y + 31, title, size=21, weight=800, anchor="start")
    svg.text(x + 22, y + 67, f"{before} -> {after}", size=21, weight=750, anchor="start", fill=COLORS["muted"])
    svg.pill(x + 212, y + 47, 62, 28, delta, COLORS["coral_light"], COLORS["coral"], size=15, weight=800)


def build_svg():
    svg = SVG()
    svg.add(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <filter id="softShadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#2F2B28" flood-opacity="0.10"/>
  </filter>
  <filter id="tinyShadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#2F2B28" flood-opacity="0.10"/>
  </filter>
  <marker id="arrow" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto" markerUnits="strokeWidth">
    <path d="M1,1 L11,6.5 L1,12 Z" fill="{COLORS['line']}"/>
  </marker>
  <style>
    text {{ font-family: Helvetica, Arial, "Microsoft YaHei", sans-serif; letter-spacing: 0; }}
  </style>
</defs>''')
    svg.rect(0, 0, W, H, rx=0, fill=COLORS["bg"])
    svg.rect(48, 42, W - 96, H - 84, rx=34, fill=COLORS["paper"], stroke="#E4DBCF", sw=1.5, filter_="url(#softShadow)")

    svg.text(820, 92, "CaptionLens: Semantic Slot Architecture", size=38, weight=850)
    svg.text(820, 130, "ResNet50 Encoder + Semantic Slot Memory Adapter + Transformer Decoder", size=22, weight=560, fill=COLORS["muted"])
    svg.pill(1355, 76, 144, 36, "Our Module", COLORS["coral_light"], COLORS["coral"], size=18)
    svg.pill(1516, 76, 116, 36, "MSCOCO17", COLORS["mint_light"], COLORS["mint"], size=18)

    # Main pipeline cards.
    y = 190
    cards = [
        (90, y, 190, 210, "Input Image", COLORS["sand"]),
        (330, y, 210, 210, "Preprocess", COLORS["white"]),
        (590, y, 230, 210, "CNN Encoder", COLORS["encoder_light"]),
        (875, y, 250, 210, "Visual Tokens", COLORS["white"]),
        (1180, y - 16, 310, 242, "Semantic Slot Memory", COLORS["coral_light"]),
        (1540, y, 200, 210, "Caption", COLORS["decoder_light"]),
    ]
    for x, yy, w, h, title, fill in cards:
        stroke = COLORS["coral"] if "Semantic" in title else "#D8CFC4"
        sw = 3 if "Semantic" in title else 1.5
        svg.rect(x, yy, w, h, rx=22, fill=fill, stroke=stroke, sw=sw, filter_="url(#softShadow)")
        svg.text(x + w / 2, yy + 38, title, size=24, weight=820)

    # Input image.
    svg.rect(122, 255, 126, 92, rx=10, fill="#E9F0F2", stroke="#D5CCC2", sw=1.2)
    svg.path("M132 333 L165 296 L189 322 L207 305 L240 333", color=COLORS["encoder"], sw=3, arrow=False)
    svg.add(f'<circle cx="210" cy="278" r="12" fill="{COLORS["coral"]}" opacity="0.85"/>')
    svg.text(185, 374, "RGB image", size=18, weight=650, fill=COLORS["muted"])

    # Preprocess.
    svg.rect(364, 250, 142, 110, rx=14, fill=COLORS["sand"], stroke="#D8CFC4", sw=1.3)
    svg.text(435, 286, ["Resize", "224 x 224"], size=18, weight=650)
    svg.line(384, 320, 486, 320, color="#B8ADA2", sw=2, arrow=False)
    svg.text(435, 348, ["Normalize", "ImageNet mean/std"], size=16, weight=600, fill=COLORS["muted"])

    # Encoder stack.
    tensor_stack(svg, 627, 267, 72, 92, 5, "#DCE7F2", COLORS["encoder"], dx=13, dy=-8)
    svg.text(705, 374, "ResNet50", size=20, weight=800, fill=COLORS["encoder"])
    svg.text(705, 397, "regional features", size=16, weight=600, fill=COLORS["muted"])

    # Visual tokens.
    token_grid(svg, 915, 245, cell=8, gap=4, fill=COLORS["encoder_light"], stroke=COLORS["encoder"])
    svg.text(1000, 374, ["14 x 14 grid", "196 x 512 visual tokens"], size=17, weight=650, fill=COLORS["muted"])

    # Semantic slot card contents.
    token_row(svg, 1210, 246, ["q1", "q2", "...", "q16"], COLORS["lavender_light"], COLORS["lavender"], size=16)
    svg.text(1336, 230, "learnable slot queries", size=16, weight=700, fill=COLORS["muted"])
    svg.rect(1230, 298, 220, 48, rx=12, fill=COLORS["white"], stroke=COLORS["coral"], sw=1.7)
    svg.text(1340, 322, ["Cross-Attention", "slots attend to visual tokens"], size=16, weight=720)
    svg.rect(1260, 363, 160, 46, rx=12, fill=COLORS["lavender_light"], stroke=COLORS["lavender"], sw=1.7)
    svg.text(1340, 392, "Slot Self-Attention", size=17, weight=740)
    svg.text(1336, 431, "16 semantic slots + 196 visual tokens = 212 memory tokens", size=16, weight=700, fill=COLORS["coral"])
    svg.path("M1266 280 C1248 288 1248 304 1256 320", color=COLORS["lavender"], sw=2.2, arrow=True, opacity=0.78)
    svg.path("M1434 346 C1460 358 1454 382 1422 386", color=COLORS["lavender"], sw=2.2, arrow=True, opacity=0.78)

    # Decoder larger panel.
    svg.rect(590, 500, 860, 365, rx=24, fill=COLORS["white"], stroke="#D8CFC4", sw=1.5, filter_="url(#softShadow)")
    svg.text(1020, 548, "(b) Transformer Decoder with Enhanced Visual Memory", size=27, weight=850)
    svg.text(1020, 582, "Self-Attention models word history; Cross-Attention aligns words with semantic-slot memory.", size=18, weight=560, fill=COLORS["muted"])

    # Token generation branch.
    token_row(svg, 640, 635, ["<START>", "an", "airplane", "is"], COLORS["decoder_light"], COLORS["decoder"], size=15)
    svg.rect(658, 705, 275, 70, rx=15, fill=COLORS["decoder_light"], stroke=COLORS["decoder"], sw=1.8)
    svg.text(795, 733, "Masked Self-Attention", size=19, weight=800)
    svg.text(795, 759, "word-to-word context", size=15, weight=600, fill=COLORS["muted"])

    # Cross attention branch.
    svg.rect(1010, 620, 360, 176, rx=18, fill=COLORS["lavender_light"], stroke=COLORS["lavender"], sw=2)
    svg.text(1190, 652, "Cross-Attention", size=22, weight=850)
    svg.text(1190, 680, "text tokens attend to enhanced memory", size=16, weight=600, fill=COLORS["muted"])
    token_row(svg, 1040, 716, ["slot1", "slot2", "...", "v196"], COLORS["white"], COLORS["lavender"], size=14)
    svg.path("M935 740 L1010 740", color=COLORS["line"], sw=3)
    svg.path("M1190 796 L1190 836", color=COLORS["line"], sw=3)
    svg.rect(1086, 836, 208, 48, rx=12, fill=COLORS["decoder_light"], stroke=COLORS["decoder"], sw=1.6)
    svg.text(1190, 866, "Next-token logits", size=18, weight=800)

    # Baseline dashed route.
    svg.path("M1000 402 C1000 458 1000 475 1000 500", color="#AAA095", sw=2.2, dashed=True, arrow=True, opacity=0.85)
    svg.text(1044, 478, "baseline passes raw 196 tokens", size=15, weight=650, fill=COLORS["muted"], anchor="start")

    # Arrows main pipeline.
    svg.line(280, y + 105, 330, y + 105, sw=3.4)
    svg.line(540, y + 105, 590, y + 105, sw=3.4)
    svg.line(820, y + 105, 875, y + 105, sw=3.4)
    svg.line(1125, y + 105, 1180, y + 105, sw=3.8, color=COLORS["coral"])
    svg.line(1490, y + 105, 1540, y + 105, sw=3.4)

    # Output caption.
    svg.rect(1573, 264, 134, 70, rx=14, fill=COLORS["white"], stroke=COLORS["decoder"], sw=1.8)
    svg.text(1640, 290, ["an airplane is", "parked on a runway"], size=18, weight=780)
    token_row(svg, 1562, 358, ["<END>"], COLORS["decoder_light"], COLORS["decoder"], size=15)

    # Lower contribution and metrics panel.
    svg.rect(90, 910, 1660, 142, rx=24, fill=COLORS["sand"], stroke="#D8CFC4", sw=1.5)
    svg.text(128, 952, "(c) Empirical gain on MSCOCO 2017 validation", size=24, weight=850, anchor="start")
    svg.text(128, 986, ["Semantic slots compress visual regions into high-level memory.", "Same 20-epoch setting as baseline."], size=18, weight=560, fill=COLORS["muted"], anchor="start")
    metric_card(svg, 885, 930, "BLEU-4", "0.2821", "0.2849", "+0.99%")
    metric_card(svg, 1205, 930, "CIDEr", "0.8882", "0.9111", "+2.58%")
    svg.pill(1530, 952, 172, 38, "All metrics up", COLORS["mint_light"], COLORS["mint"], size=18)
    svg.text(1616, 1007, "Baseline: ResNet50 -> Decoder", size=16, weight=600, fill=COLORS["muted"])

    # Labels.
    svg.text(900, 174, "(a) End-to-end image captioning pipeline", size=24, weight=820, fill=COLORS["text"])
    svg.add("</svg>")
    return svg.render()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {OUT}")
