// Rebuilds the shareable Word summary from the checked-in infographic.
// Run from the repository root:
// node shampoo_demo/docs/create_new_ingredient_protocol_docx.js

const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  HeadingLevel,
  ImageRun,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("C:/Users/DS/.codex/docx-runtime/node_modules/docx");

const root = path.resolve(__dirname, "..");
const imagePath = path.join(__dirname, "assets", "new_ingredient_onboarding_protocol.png");
const outputPath = path.join(__dirname, "신규원료_적용범위_편입_프로토콜.docx");
const image = fs.readFileSync(imagePath);

const navy = "173B63";
const teal = "007C91";
const paleBlue = "EAF3FA";
const paleOrange = "FFF2E8";
const border = { style: BorderStyle.SINGLE, size: 8, color: "B7C9D8" };

function text(value, options = {}) {
  return new TextRun({ text: value, font: "Malgun Gothic", size: 21, ...options });
}

function bullet(value) {
  return new Paragraph({
    text: value,
    bullet: { level: 0 },
    spacing: { after: 90 },
    run: { font: "Malgun Gothic", size: 20 },
  });
}

function cell(value, width, shade) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, color: shade },
    borders: { top: border, bottom: border, left: border, right: border },
    margins: { top: 110, bottom: 110, left: 140, right: 140 },
    children: [new Paragraph({ children: [text(value)] })],
  });
}

const doc = new Document({
  creator: "COS Team",
  title: "신규 원료의 샴푸 물성 예측 모델 적용범위 편입 프로토콜",
  description: "18개 원료 기반 모델의 한계와 신규 원료 편입 절차를 설명하는 공유용 문서",
  sections: [{
    properties: { page: { margin: { top: 720, bottom: 720, left: 760, right: 760 } } },
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [text("신규 원료의 모델 적용범위 편입 프로토콜", { bold: true, size: 34, color: navy })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 240 },
        children: [text("COS 학술제 · 샴푸 물성 설계 연구 공유용 요약", { size: 19, color: "53738C" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 260 },
        children: [new ImageRun({ data: image, type: "png", transformation: { width: 650, height: 325 } })],
      }),
      new Paragraph({ text: "1. 현재 모델의 적용범위", heading: HeadingLevel.HEADING_1, spacing: { before: 70, after: 120 } }),
      bullet("현재 B안 모델은 공개 샴푸 데이터의 18개 상용 원료와 유사한 조성·시험 조건 안에서만 물성 예측을 지원한다."),
      bullet("새 원료, 새 공급사, 새 등급, 활성분 함량 변화, 학습 범위를 크게 벗어난 투입량은 예측 근거가 부족하므로 자동 예측 대상이 아니다."),
      new Paragraph({ text: "2. 신규 원료 편입 절차", heading: HeadingLevel.HEADING_1, spacing: { before: 150, after: 120 } }),
      new Table({
        width: { size: 7000, type: WidthType.DXA },
        columnWidths: [850, 2100, 4050],
        rows: [
          new TableRow({ children: [cell("단계", 850, paleBlue), cell("수행 내용", 2100, paleBlue), cell("확보하려는 근거", 4050, paleBlue)] }),
          new TableRow({ children: [cell("1", 850, "FFFFFF"), cell("원료 정보 확보", 2100, "FFFFFF"), cell("공급사·등급·활성분·기능·권장 농도·규격서 등 비교 가능한 원료 정보를 확보", 4050, "FFFFFF")]}),
          new TableRow({ children: [cell("2", 850, "FFFFFF"), cell("기존 원료와 비교", 2100, "FFFFFF"), cell("기존 18종과의 기능·활성분·농도 범위 차이와 확인 공백을 확인", 4050, "FFFFFF")]}),
          new TableRow({ children: [cell("3", 850, "FFFFFF"), cell("최소 처방 실험", 2100, "FFFFFF"), cell("대표 조합과 농도 수준을 정해 신규 원료를 포함한 최소 실험군 구성", 4050, "FFFFFF")]}),
          new TableRow({ children: [cell("4", 850, "FFFFFF"), cell("물성 측정", 2100, "FFFFFF"), cell("기존 자료와 같은 조건에서 단기 안정성, 탁도, 점도를 측정", 4050, "FFFFFF")]}),
          new TableRow({ children: [cell("5", 850, "FFFFFF"), cell("재학습·검증", 2100, "FFFFFF"), cell("신규 원료 처방을 별도 평가해 예측 오차와 적용범위를 확인", 4050, "FFFFFF")]}),
          new TableRow({ children: [cell("6", 850, "FFFFFF"), cell("적용범위 등록", 2100, "FFFFFF"), cell("기준을 통과한 경우에만 지원 원료와 허용 조건으로 등록", 4050, "FFFFFF")]}),
        ],
      }),
      new Paragraph({ text: "3. 앱이 출력해야 할 결과", heading: HeadingLevel.HEADING_1, spacing: { before: 180, after: 100 } }),
      new Paragraph({
        shading: { type: ShadingType.CLEAR, color: paleOrange },
        borders: { top: border, bottom: border, left: border, right: border },
        spacing: { before: 100, after: 100 },
        indent: { left: 150, right: 150 },
        children: [text("통과: 예측 지원 원료로 등록하고 적용 가능한 농도·시험 조건·예측 오차를 표시한다.  미통과: 수치를 억지로 출력하지 않고 추가 실험과 확인 항목을 제시한다.", { bold: true, color: "8A3B00" })],
      }),
      new Paragraph({ text: "발표용 핵심 문장", heading: HeadingLevel.HEADING_1, spacing: { before: 180, after: 100 } }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        shading: { type: ShadingType.CLEAR, color: "E7F5F3" },
        spacing: { before: 160, after: 160 },
        children: [text("AI가 새 원료를 즉시 판정하는 것이 아니라, 원료 정보와 최소 실험으로 검증한 뒤 적용범위에 편입하는 의사결정 보조 도구를 만든다.", { bold: true, size: 22, color: teal })],
      }),
      new Paragraph({
        spacing: { before: 180 },
        children: [text("주의: 이 도구는 제품 안전성·자극성·출시 가능 여부를 자동 판정하거나 시험을 면제하지 않는다.", { size: 18, color: "7A3A00", italics: true })],
      }),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(outputPath);
});
