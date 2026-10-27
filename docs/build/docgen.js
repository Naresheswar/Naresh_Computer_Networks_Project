// Shared helpers for building the report and the patent draft with docx-js.
// Plain, understated styling: black headings, plain tables, Times New Roman.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, PageBreak, LevelFormat, Footer, PageNumber, TabStopType,
} = require("docx");

const FONT = "Times New Roman";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1300; // A4
const CONTENT_W = PAGE_W - 2 * MARGIN;

function runs(text, opts = {}) {
  // **bold** and _italic_ inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: w,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        borders, width: { size: w[j], type: WidthType.DXA },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          alignment: j === 0 || opts.left ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: runs(String(c), { bold: i === 0, size: 20, font: FONT }),
        })],
      })),
    })),
  });
}

function build(blocks, outPath, { header } = {}) {
  const children = [];
  for (const b of blocks) {
    if (b.h1) children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, children: [new TextRun({ text: b.h1 })] }));
    else if (b.h2) children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [new TextRun({ text: b.h2 })] }));
    else if (b.h3) children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, keepNext: true, children: [new TextRun({ text: b.h3 })] }));
    else if (b.p !== undefined) children.push(new Paragraph({ alignment: b.align || AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 }, children: runs(b.p, b.style || {}) }));
    else if (b.center) children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: b.after ?? 120, before: b.before ?? 0 }, children: runs(b.center, { size: b.size, bold: b.bold, italics: b.italics }) }));
    else if (b.bullets) b.bullets.forEach((t) => children.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 290 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) })));
    else if (b.numbers) b.numbers.forEach((t) => children.push(new Paragraph({ numbering: { reference: b.ref || "num", level: 0 }, spacing: { after: 80, line: 290 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) })));
    else if (b.eq) children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 }, children: [new TextRun({ text: b.eq, font: "Cambria Math", size: 22 })] }));
    else if (b.code) b.code.split("\n").forEach((l) => children.push(new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: l || " ", font: "Consolas", size: 18 })] })));
    else if (b.table) {
      children.push(table(b.table, b.widths || b.table[0].map(() => 1), b));
      if (b.caption) children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 200 }, children: runs(b.caption, { size: 20, italics: true }) }));
    } else if (b.img) {
      const data = fs.readFileSync(b.img);
      const wpx = b.width || 560;
      const ratio = b.ratio || 0.7;
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({ type: "png", data, transformation: { width: wpx, height: Math.round(wpx * ratio) } })] }));
      if (b.caption) children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(b.caption, { size: 20, italics: true }) }));
    } else if (b.pagebreak) children.push(new Paragraph({ children: [new PageBreak()] }));
    else if (b.space) children.push(new Paragraph({ spacing: { after: b.space }, children: [] }));
  }
  const doc = new Document({
    creator: "Naresh Nireenjan",
    styles: {
      default: { document: { run: { font: FONT, size: 24, color: "000000" } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 30, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 26, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, italics: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
        ...["num", "claims", "num2", "num3", "num4"].map((r) => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 400 } } } }] })),
      ],
    },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
      children,
    }],
  });
  return Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath); });
}

module.exports = { build, AlignmentType };
