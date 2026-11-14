"""Export the submission section of the single English Markdown report to PDF.

Historical experiment details remain in the same Markdown source. The PDF
contains the self-contained submission narrative between explicit markers.
Requires reportlab; no browser, office installation, or external service.
"""
import argparse
from html import escape
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    value = escape(text)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" color="#1d4e89">\1</a>', value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', value)
    value = re.sub(r"\*\*(.+?)\*\*", r'<b>\1</b>', value)
    return value


def export_pdf(source, destination):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, LongTable, TableStyle, KeepTogether
    text = source.read_text(encoding="utf-8")
    start, end = "<!-- SUBMISSION_START -->", "<!-- SUBMISSION_END -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("Exactly one complete submission section is required")
    text = text.split(start)[1].split(end)[0].strip()
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        raise ValueError("Submission report must be English")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", fontName="Times-Bold", fontSize=18, leading=21, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="PaperBody", fontName="Times-Roman", fontSize=10, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="PaperHeading", fontName="Times-Bold", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name="PaperSubheading", fontName="Times-Bold", fontSize=11, leading=14, spaceBefore=8, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="PaperCell", fontName="Times-Roman", fontSize=8, leading=10, spaceAfter=0))
    styles.add(ParagraphStyle(name="PaperCode", fontName="Courier", fontSize=7, leading=9, spaceAfter=8))
    story, lines, i = [], text.splitlines(), 0
    width = letter[0] - 100
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.extend(textwrap.wrap(lines[i], 105, replace_whitespace=False, drop_whitespace=False) or [""])
                i += 1
            story.append(Preformatted("\n".join(code), styles["PaperCode"]))
            i += 1
            continue
        if line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in cells):
                    table.append([Paragraph(inline(c), styles["PaperCell"]) for c in cells])
                i += 1
            n = len(table[0])
            if any(len(row) != n for row in table):
                raise ValueError("Malformed table")
            widths = ([width] if n == 1 else [width * .24] + [width * .76 / (n - 1)] * (n - 1))
            obj = LongTable(table, colWidths=widths, repeatRows=1, hAlign="LEFT")
            obj.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                                     ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f7f9")]),
                                     ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#b9c2cc")),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                     ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.extend([KeepTogether([obj]) if len(table) <= 4 else obj, Spacer(1, 8)])
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["PaperTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["PaperHeading"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["PaperSubheading"]))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + inline(line[2:]), styles["PaperBody"]))
        elif line.startswith("<!--"):
            pass
        else:
            paragraph = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].lstrip().startswith(("#", "|", "```", "- ", "<!--")):
                i += 1
                paragraph.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(paragraph)), styles["PaperBody"]))
        i += 1
    def footer(canvas, doc):
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(50, 28, "CS598APE | Lightweight Adaptive Top-K Retrieval | Final project report")
        canvas.drawRightString(letter[0] - 50, 28, str(doc.page))
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(destination), pagesize=letter, rightMargin=50, leftMargin=50,
                                 topMargin=45, bottomMargin=45, title="Lightweight Adaptive Top-K Retrieval in FlashRAG",
                                 author="Yidong Zhang and Yuang Cai")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"[PDF] {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/performance_profilling_report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/performance_profilling_report.pdf")
    args = parser.parse_args()
    export_pdf(args.source, args.output)
