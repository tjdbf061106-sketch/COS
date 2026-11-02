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
      new Paragraph({ text: "문제상황: 왜 바로 예측하면 안 되는가", heading: HeadingLevel.HEADING_1, spacing: { before: 150, after: 100 } }),
      new Table({
        width: { size: 7000, type: WidthType.DXA },
        columnWidths: [2200, 2400, 2400],
        rows: [
          new TableRow({ children: [cell("현재 확보한 근거", 2200, paleBlue), cell("새 원료를 즉시 넣을 때의 문제", 2400, paleBlue), cell("잘못하면 생기는 해석 오류", 2400, paleBlue)] }),
          new TableRow({ children: [cell("18개 상용 원료의 실제 투입량과 측정 물성", 2200, "FFFFFF"), cell("새 원료는 기존 입력 열에 존재하지 않음", 2400, "FFFFFF"), cell("모델이 아는 것처럼 임의의 숫자를 반환하는 ‘가짜 정밀도’", 2400, "FFFFFF")] }),
          new TableRow({ children: [cell("같은 원료 조합이 겹치지 않도록 한 평가", 2200, "FFFFFF"), cell("새 원료 조합에서의 오차는 아직 측정되지 않음", 2400, "FFFFFF"), cell("기존 성능을 신규 원료 성능으로 과장", 2400, "FFFFFF")] }),
          new TableRow({ children: [cell("원료별 공급사·활성분·등급 정보 일부", 2200, "FFFFFF"), cell("동일 INCI라도 실제 상용 원료 조성이 달라질 수 있음", 2400, "FFFFFF"), cell("공급사 변경을 단순 동등 교체로 오인", 2400, "FFFFFF")] }),
          new TableRow({ children: [cell("안정성, 탁도, 점도 측정은 정해진 시험 조건의 결과", 2200, "FFFFFF"), cell("pH·혼합·온도·관찰시간이 달라지면 비교가 어려움", 2400, "FFFFFF"), cell("예측값을 실제 제품 성능·안전성으로 오해", 2400, "FFFFFF")] }),
        ],
      }),
      new Paragraph({ text: "이 프로토콜이 필요한 이유", heading: HeadingLevel.HEADING_1, spacing: { before: 180, after: 100 } }),
      bullet("예측 범위를 넓히는 목적은 새 원료를 ‘무조건 맞히는 AI’를 만드는 것이 아니라, 어떤 정보와 실험이 있어야 새 원료를 예측 대상에 포함할 수 있는지 명확히 하는 데 있다."),
      bullet("새 원료 데이터를 기존 조건과 같은 형식으로 쌓아야만, 모델 성능 저하·공급사 차이·시험 조건 차이를 분리해 해석할 수 있다."),
      bullet("앱이 불확실한 상황에서 ‘예측 불가·추가 실험 필요’를 말할 수 있어야 실제 개발 현장에서 위험한 자동화를 줄일 수 있다."),
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
      new Paragraph({ text: "3. 실제 수행 방법: 단계별 산출물과 담당", heading: HeadingLevel.HEADING_1, spacing: { before: 180, after: 100 } }),
      new Table({
        width: { size: 7000, type: WidthType.DXA },
        columnWidths: [1000, 2100, 2300, 1600],
        rows: [
          new TableRow({ children: [cell("구간", 1000, paleBlue), cell("팀이 할 일", 2100, paleBlue), cell("남겨야 할 산출물", 2300, paleBlue), cell("주 담당", 1600, paleBlue)] }),
          new TableRow({ children: [cell("사전 정의", 1000, "FFFFFF"), cell("신규 원료의 역할, 비교 대상, 적용하려는 제품 범위와 ‘통과’ 기준을 실험 전에 합의", 2100, "FFFFFF"), cell("원료 등록 카드, 적용범위 초안, 통과/보류 기준", 2300, "FFFFFF"), cell("제형·기획", 1600, "FFFFFF")] }),
          new TableRow({ children: [cell("정보 수집", 1000, "FFFFFF"), cell("규격서·기술자료·활성분·등급·공급사·권장 사용량·보관 조건을 원료 단위로 기록", 2100, "FFFFFF"), cell("출처 링크/문서 버전, 미확인 항목 목록", 2300, "FFFFFF"), cell("원료 조사", 1600, "FFFFFF")] }),
          new TableRow({ children: [cell("실험 설계", 1000, "FFFFFF"), cell("기존 처방을 기준점으로 정하고, 신규 원료의 농도·대표 조합·반복 측정·대조군을 계획", 2100, "FFFFFF"), cell("처방표, 시험 순서, pH·혼합·온도·관찰시간 기록지", 2300, "FFFFFF"), cell("제형·시험", 1600, "FFFFFF")] }),
          new TableRow({ children: [cell("측정", 1000, "FFFFFF"), cell("기존 데이터와 동일한 정의로 단기 상안정성부터 판정하고, 안정 처방에만 탁도·점도 측정", 2100, "FFFFFF"), cell("원시 측정값, 결측 사유, 단위, 반복값", 2300, "FFFFFF"), cell("시험", 1600, "FFFFFF")] }),
          new TableRow({ children: [cell("분석", 1000, "FFFFFF"), cell("새 원료가 포함된 처방을 별도 검증 집합으로 두고 재학습·오차·적용범위를 평가", 2100, "FFFFFF"), cell("분할 기록, Sens/Spec/MCC/ROC-AUC, 회귀 오차, 오류 사례", 2300, "FFFFFF"), cell("데이터·모델", 1600, "FFFFFF")] }),
          new TableRow({ children: [cell("등록", 1000, "FFFFFF"), cell("통과한 조건만 앱의 지원 원료 목록에 반영하고, 나머지는 ‘추가 실험 필요’로 유지", 2100, "FFFFFF"), cell("원료 라이브러리 버전, 허용 농도/조건, 변경 이력", 2300, "FFFFFF"), cell("앱·검토", 1600, "FFFFFF")] }),
        ],
      }),
      new Paragraph({ text: "4. 실험과 분석에서 지켜야 할 원칙", heading: HeadingLevel.HEADING_1, spacing: { before: 180, after: 100 } }),
      bullet("원료 정보를 모른 채 구조나 활성분을 임의 값으로 채우지 않는다. 모르는 값은 ‘미확인’으로 남긴다."),
      bullet("새 원료 처방을 학습과 평가에 동시에 섞지 않는다. 새 원료가 포함된 처방을 따로 빼고 오차를 확인해야 한다."),
      bullet("안정성 분류는 Sensitivity·Specificity·MCC·ROC-AUC를 함께 보고, 특정 숫자 하나만으로 통과시키지 않는다."),
      bullet("점도·탁도 예측은 안정 처방에서만 해석하고, 오차·유사 조합·시험 조건을 결과 화면에 함께 표시한다."),
      bullet("이 절차는 자극성·안전성·규제 적합성·출시 가능 여부를 판정하거나 시험을 면제하는 절차가 아니다."),
      new Paragraph({ text: "5. 팀이 지금 결정해야 할 질문", heading: HeadingLevel.HEADING_1, spacing: { before: 180, after: 100 } }),
      new Paragraph({
        shading: { type: ShadingType.CLEAR, color: "E7F5F3" },
        borders: { top: border, bottom: border, left: border, right: border },
        spacing: { before: 110, after: 110 },
        indent: { left: 150, right: 150 },
        children: [text("① 첫 번째로 편입할 신규 원료는 무엇인가?  ② 그 원료의 역할과 비교 기준은 무엇인가?  ③ 동일 조건에서 확보 가능한 최소 실험은 무엇인가?  ④ ‘예측 지원 원료’로 등록하기 위한 사전 합의 기준은 무엇인가?", { bold: true, color: teal })],
      }),
      new Paragraph({ text: "6. 앱이 출력해야 할 결과", heading: HeadingLevel.HEADING_1, spacing: { before: 180, after: 100 } }),
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
