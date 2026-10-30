// Render manuscript.json to a Word document with docx-js.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, PageBreak, PageOrientation, LineNumberRestartFormat, LevelFormat,
  ShadingType, Footer, PageNumber,
} = require("docx");

const R = "/home/claude/results/";
const blocks = JSON.parse(fs.readFileSync(process.argv[3] || (R + "manuscript.json"), "utf8"));
const NOLINES = process.argv[4] === "nolines";
const FONT = "Times New Roman";
const A4 = { width: 11906, height: 16838 };
const MARGIN = 1440;

function runs(text, base = {}) {
  // markup: *italic*, _{sub}, ^{sup}; single-character subscripts written as x_r are converted too
  text = text.replace(/(?<=[A-Za-zμβησπΣ)𝒳])_([A-Za-z0-9])(?![A-Za-z0-9{])/gu, "_{$1}");
  const out = [];
  const re = /(\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t[0] === "*") out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (t[0] === "_") out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function makeTable(b, textWidthDxa) {
  const n = b.head.length;
  let w = b.w ? b.w.map((x) => x) : Array(n).fill(1);
  const tot = w.reduce((a, c) => a + c, 0);
  const cols = w.map((x) => Math.floor((x / tot) * textWidthDxa));
  const size = b.tiny ? 13 : (b.small ? 15 : 18); // half-points
  const cell = (txt, i, head, lastRow) => new TableCell({
    width: { size: cols[i], type: WidthType.DXA },
    borders: { top: head ? border : none, bottom: head || lastRow ? border : none, left: none, right: none },
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({ spacing: { before: 0, after: 0, line: 240 },
      children: runs(String(txt), { size, bold: head }) })],
  });
  const rows = [new TableRow({ tableHeader: true, children: b.head.map((h, i) => cell(h, i, true, false)) })];
  b.rows.forEach((r, k) => rows.push(new TableRow({ children: r.map((c, i) => cell(c, i, false, k === b.rows.length - 1)) })));
  const out = [];
  out.push(new Paragraph({ spacing: { before: 240, after: 80, line: 276 }, keepNext: true, children: runs(b.cap, { size: 20, bold: false }) }));
  out.push(new Table({ width: { size: textWidthDxa, type: WidthType.DXA }, columnWidths: cols, rows }));
  if (b.note) out.push(new Paragraph({ spacing: { before: 60, after: 240, line: 240 }, children: runs(b.note, { size: 16 }) }));
  return out;
}

const sections = [];
let cur = [];
let landscape = false;
function flush(orient) {
  if (!cur.length) return;
  const isL = orient === "L";
  sections.push({
    properties: {
      page: { size: { width: A4.width, height: A4.height, orientation: isL ? PageOrientation.LANDSCAPE : PageOrientation.PORTRAIT },
              margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
      ...(NOLINES ? {} : { lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS } }),
    },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: cur,
  });
  cur = [];
}

const portraitW = A4.width - 2 * MARGIN, landW = A4.height - 2 * MARGIN;
const P = { spacing: { line: 360, after: 120 }, alignment: AlignmentType.JUSTIFIED };

for (const b of blocks) {
  if (b.t === "title") cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240, line: 300 },
      children: runs(b.x, { size: 30, bold: true }) }));
  else if (b.t === "meta") cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: runs(b.x, { size: 22 }) }));
  else if (b.t === "h1") cur.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 120 }, keepNext: true, children: runs(b.x, { size: 24, bold: true }) }));
  else if (b.t === "h2") cur.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 }, keepNext: true, children: runs(b.x, { size: 22, bold: true, italics: true }) }));
  else if (b.t === "p") cur.push(new Paragraph({ ...P, children: runs(b.x, { size: 22 }) }));
  else if (b.t === "eq") cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120, line: 300 },
      children: [...runs(b.x, { size: 22 }), new TextRun({ text: `\t(${b.n})`, font: FONT, size: 22 })],
      tabStops: [{ type: "right", position: portraitW }] }));
  else if (b.t === "bullets") b.x.forEach((x) => cur.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 300 }, children: runs(x, { size: 22 }) })));
  else if (b.t === "pb") cur.push(new Paragraph({ children: [new PageBreak()] }));
  else if (b.t === "fig") {
    const s = pngSize(b.path); const wpx = Math.round(b.w * 96); const hpx = Math.round(wpx * s.h / s.w);
    cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.path), transformation: { width: wpx, height: hpx } })] }));
    cur.push(new Paragraph({ spacing: { after: 240, line: 276 }, alignment: AlignmentType.JUSTIFIED, children: runs(b.cap, { size: 20 }) }));
  } else if (b.t === "table") {
    if (b.head.length > 6) b.tiny = true;
    cur.push(...makeTable(b, portraitW));
  } else if (b.t === "refs") b.x.forEach((x) => cur.push(new Paragraph({ spacing: { after: 60, line: 276 }, indent: { left: 440, hanging: 440 }, children: runs(x, { size: 20 }) })));
}
flush("P");

const doc = new Document({
  creator: "Truong Hai Bang",
  title: process.argv[5] || "Exact admissible-value bounds for fuzzy additive prescreening",
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] }] },
  sections,
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(process.argv[2] || R + "manuscript.docx", buf); console.log("written"); });
