"""Build the Word report from a recording and its captured chart results."""
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


MODULES = {
    "trend": "Acceleration trend",
    "sust": "ASTM F2291 sustained acceleration",
    "egg": "ASTM F2291 eggplot",
    "gb": "GB 8408 fitting",
    "acczone": "Acceleration zone",
    "zr": "Z axis reversals",
    "xr": "X and Y axis reversals",
}
INK = "212E3B"
MUTED = "465B6C"
NAVY = "24465D"


def _element(tag, **attrs):
    el = OxmlElement("w:" + tag)
    for key, value in attrs.items():
        el.set(qn("w:" + key), str(value))
    return el


def _style(doc, name, size, color=INK, *, bold=False, before=0, after=4,
           keep_next=False):
    style = (doc.styles[name] if name in doc.styles else
             doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH))
    style.font.name = "Calibri"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor.from_string(color)
    # Built-in Title has a decorative bottom border in the default template.
    props = style.element.get_or_add_pPr()
    border = props.find(qn("w:pBdr"))
    if border is not None:
        props.remove(border)
    fmt = style.paragraph_format
    fmt.space_before, fmt.space_after = Pt(before), Pt(after)
    fmt.line_spacing = 1.08
    fmt.keep_with_next = keep_next
    fmt.widow_control = True


def _result(verdict):
    if not verdict:
        return "Unavailable", "No analysis result was returned.", MUTED
    kind, finding = verdict
    if kind == "info":
        return "Info", finding, NAVY
    return ("Pass", finding, "146346") if kind else ("Fail", finding, "9B352F")


def _table(doc, headers, rows, widths, *, numbers=False, status_colors=()):
    table = doc.add_table(rows=0, cols=len(widths))
    table.autofit = False
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    pr = table._tbl.tblPr
    borders = _element("tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(_element(edge, val="single", sz=4, color="D9D9D9"))
    pr.append(borders)
    margins = _element("tblCellMar")
    for edge, value in (("top", 65), ("bottom", 65), ("left", 95), ("right", 95)):
        margins.append(_element(edge, w=value, type="dxa"))
    pr.append(margins)
    for i, values in enumerate([headers, *rows]):
        row = table.add_row()
        row_pr = row._tr.get_or_add_trPr()
        row_pr.append(_element("cantSplit"))
        if not i:
            row_pr.append(_element("tblHeader", val="true"))
        for j, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            fill = NAVY if i == 0 else ("FFFFFF" if i % 2 else "EFF4F7")
            cell._tc.get_or_add_tcPr().append(_element("shd", fill=fill))
            p = cell.paragraphs[0]
            p.style = "Report table"
            if numbers and j:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif j == 1 and status_colors:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            run.bold = (i == 0 or (i > 0 and j == 1 and bool(status_colors)))
            color = ("FFFFFF" if i == 0 else
                     status_colors[i - 1] if j == 1 and status_colors else INK)
            run.font.color.rgb = RGBColor.from_string(color)


class AccelerationReport:
    def __init__(self, record_count, generated=None):
        self.doc = doc = Document()
        self.record_count = record_count
        self.record_number = 0
        self.figure_number = 0
        self.generated = generated or datetime.now()
        sec = doc.sections[0]
        sec.page_width, sec.page_height = Inches(8.5), Inches(11)
        sec.left_margin = sec.right_margin = Inches(0.85)
        sec.top_margin, sec.bottom_margin = Inches(0.72), Inches(0.72)
        sec.footer_distance = Inches(0.35)
        _style(doc, "Normal", 11)
        _style(doc, "Title", 28, "000000", bold=True, after=9, keep_next=True)
        _style(doc, "Heading 1", 19, "000000", bold=True, after=5, keep_next=True)
        _style(doc, "Heading 2", 14, "000000", bold=True, before=11, after=4, keep_next=True)
        _style(doc, "Report label", 9, MUTED, bold=True, after=3, keep_next=True)
        _style(doc, "Report note", 10, MUTED)
        _style(doc, "Report table", 10, after=0)
        _style(doc, "Report figure", 10, bold=True, before=8, after=3, keep_next=True)
        _style(doc, "Report legend", 9, MUTED, after=5)
        doc.core_properties.title = "Acceleration analysis report"
        doc.core_properties.author = "BERT Reader"
        foot = sec.footer.paragraphs[0]
        foot.style = "Report note"
        foot.paragraph_format.tab_stops.add_tab_stop(Inches(6.8),
                                                      WD_TAB_ALIGNMENT.RIGHT)
        foot.add_run("BERT Reader   /   Acceleration analysis\tPage ")
        foot._p.append(_element("fldSimple", instr=" PAGE "))

    def add_record(self, data, name, verdicts, img_dir, mods, sections, *,
                   plot_type, cutoff, restraint, condition, height, prone):
        doc = self.doc
        selected = [key for key in MODULES if mods.get(key)]
        self.record_number += 1
        label = doc.add_paragraph(
            f"RECORDING {self.record_number} OF {self.record_count}     /     "
            f"{self.generated:%Y-%m-%d %H:%M}", "Report label")
        label.paragraph_format.page_break_before = self.record_number > 1
        if self.record_number == 1:
            doc.add_paragraph("Acceleration analysis report", "Title")
        doc.add_heading(name, level=1)
        times = data.data["Time"]
        duration = times.iloc[-1] - times.iloc[0]
        doc.add_paragraph(
            f"{len(times):,} samples   ·   {duration:,.3f} s duration   ·   "
            f"{data.fs:g} Hz sampling rate", "Report note")
        if any(k in selected for k in ("trend", "sust", "egg", "gb")):
            source = {"Standard": "Calibrated", "Filter": "Filtered"}.get(plot_type, plot_type)
            doc.add_paragraph(f"Plot data: {source}.   Filter setting: {cutoff:g} Hz.",
                              "Report note")
        elif any(k in selected for k in ("zr", "xr")):
            doc.add_paragraph(f"Reversal data: Calibrated.   Filter setting: {cutoff:g} Hz.",
                              "Report note")
        if any(k in selected for k in ("sust", "egg", "xr")):
            doc.add_paragraph(f"Restraint: {restraint}.   Condition: {condition}.   "
                              f"Height setting: {height:g} in.   Prone: {'Yes' if prone else 'No'}.",
                              "Report note")
        checks = [k for k in selected if k != "trend"]
        if checks:
            doc.add_heading("Analysis results", level=2)
            results = [_result(verdicts.get(k)) for k in checks]
            statuses = [status for status, _, _ in results]
            counts = [(statuses.count(s), label) for s, label in
                      (("Fail", "failed"), ("Pass", "passed"), ("Info", "informational"),
                       ("Unavailable", "unavailable"))]
            doc.add_paragraph("Selected analyses: " + ", ".join(
                f"{count} {label}" for count, label in counts if count) + ".", "Report note")
            rows = [(MODULES[k], status, detail)
                    for k, (status, detail, _) in zip(checks, results)]
            _table(doc, ("Analysis", "Result", "Finding"), rows, (2.15, 0.95, 3.7),
                   status_colors=[color for _, _, color in results])
        if "trend" in selected:
            doc.add_heading("Axis statistics", level=2)
            doc.add_paragraph("Peak values and their timestamps use filtered, "
                              "zero-adjusted data.", "Report note")
            stats = data.get_data_stats(data.std_data)
            rows = [(axis, *(f"{v:.6g}" for v in values)) for axis, values in stats.iterrows()]
            _table(doc, ("Axis", "t max (s)", "t min (s)", "Max (g)", "Min (g)", "Zero (g)"),
                   rows, (0.7, 1.25, 1.25, 1.2, 1.2, 1.2), numbers=True)
        doc.add_heading("Recorded charts", level=2)
        for key in selected:
            entries = sections.get(key, [])
            if not entries:
                doc.add_paragraph(f"{MODULES[key]}: chart image unavailable.", "Report note")
            for caption, filename, legend in entries:
                if key == "acczone":
                    caption = "1 Hz filtered and zero-adjusted"
                elif key in ("zr", "xr"):
                    caption = " · ".join(s for s in (
                        caption, f"{cutoff:g} Hz filtered and zero-adjusted",
                        "0.05 g deadband" if key == "xr" else None) if s)
                self._figure(MODULES[key], caption, Path(img_dir) / filename, legend)

    def _figure(self, module, caption, path, legend):
        doc = self.doc
        if not path.is_file():
            doc.add_paragraph(f"{caption or module}: chart image unavailable.", "Report note")
            return
        self.figure_number += 1
        doc.add_paragraph(
            f"Figure {self.figure_number}   {module}" + (f"   /   {caption}" if caption else ""),
            "Report figure")
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = bool(legend)
        para.paragraph_format.space_after = Pt(2)
        picture = para.add_run().add_picture(str(path))
        # Bound both dimensions. Portrait egg plots must not fill an entire page.
        width, height = picture.width, picture.height
        scale = min(Inches(6.75) / width, Inches(3.9) / height)
        picture.width, picture.height = int(width * scale), int(height * scale)
        picture._inline.docPr.set("descr", caption or module)
        if legend:
            p = doc.add_paragraph(style="Report legend")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for color, series in legend:
                mark = p.add_run("■ ")
                mark.font.color.rgb = RGBColor.from_string(color.lstrip("#").upper())
                p.add_run(series + "   ")

    def save(self, path):
        self.doc.save(path)
