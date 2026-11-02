"""Small standard-library renderer for a shared Word/HTML research report."""
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED
import html
import struct

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture"}
for prefix, uri in NS.items(): ET.register_namespace(prefix, uri)


def render(blocks, docx_path, html_path=None, base_dir=None):
    docx_path = Path(docx_path); docx_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = Path(base_dir or docx_path.parent)
    def el(name, parent=None, attrs=None, **wattrs):
        prefix, local = name.split(":") if ":" in name else ("w", name)
        data = dict(attrs or {}); data.update({"{"+NS["w"]+"}"+k: str(v) for k,v in wattrs.items()})
        node = ET.Element("{"+NS[prefix]+"}"+local, data)
        if parent is not None: parent.append(node)
        return node
    doc = el("document"); body = el("body", doc); image_files = []
    def paragraph(parent, text, style="Normal"):
        p = el("p", parent); pr = el("pPr", p); el("pStyle", pr, val=style)
        run = el("r", p); t = el("t", run); t.text = str(text)
        return p
    html_blocks = []
    for kind, value in blocks:
        if kind in ["title", "h", "p", "code"]:
            style = {"title":"Title", "h":"Heading1", "p":"Normal", "code":"Code"}[kind]
            for line in str(value).splitlines(): paragraph(body, line, style)
            tag = {"title":"h1", "h":"h2", "p":"p", "code":"pre"}[kind]
            html_blocks.append(f"<{tag}>{html.escape(str(value))}</{tag}>")
        elif kind == "table":
            headers, rows = value; tbl = el("tbl", body); pr = el("tblPr", tbl); el("tblW", pr, w=9360, type="dxa")
            borders = el("tblBorders", pr)
            for edge in ["top","bottom","left","right","insideH","insideV"]: el(edge,borders,val="single",sz=4,color="CCD9E1")
            grid = el("tblGrid", tbl)
            for _ in headers: el("gridCol",grid,w=9360//len(headers))
            for i, row in enumerate([headers] + list(rows)):
                tr = el("tr", tbl); trpr = el("trPr", tr); el("cantSplit",trpr)
                if i == 0: el("tblHeader",trpr)
                for cell in row:
                    tc = el("tc",tr); tcp = el("tcPr",tc); el("tcW",tcp,w=9360//len(headers),type="dxa")
                    if i == 0: el("shd",tcp,fill="E9F2F6")
                    paragraph(tc,cell,"TableText")
            html_blocks.append("<table><thead><tr>"+"".join("<th>"+html.escape(str(c))+"</th>" for c in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+html.escape(str(c))+"</td>" for c in row)+"</tr>" for row in rows)+"</tbody></table>")
        elif kind == "image":
            relative, caption = value; path = base_dir / relative; image_files.append(path)
            number = len(image_files); width,height = struct.unpack(">II",path.read_bytes()[16:24])
            factor = min(5900000/width,6500000/height); cx,cy = int(width*factor),int(height*factor)
            p = el("p",body); r = el("r",p); drawing = el("drawing",r); inline = el("wp:inline",drawing)
            el("wp:extent",inline,attrs={"cx":str(cx),"cy":str(cy)})
            el("wp:docPr",inline,attrs={"id":str(number),"name":path.name})
            graphic = el("a:graphic",inline); gd = el("a:graphicData",graphic,attrs={"uri":NS["pic"]}); pic = el("pic:pic",gd)
            nonvisual = el("pic:nvPicPr",pic); el("pic:cNvPr",nonvisual,attrs={"id":"0","name":path.name}); el("pic:cNvPicPr",nonvisual)
            fill = el("pic:blipFill",pic); el("a:blip",fill,attrs={"{"+NS["r"]+"}embed":f"image{number}"})
            stretch = el("a:stretch",fill); el("a:fillRect",stretch)
            shape = el("pic:spPr",pic); xfrm = el("a:xfrm",shape); el("a:off",xfrm,attrs={"x":"0","y":"0"}); el("a:ext",xfrm,attrs={"cx":str(cx),"cy":str(cy)})
            geom = el("a:prstGeom",shape,attrs={"prst":"rect"}); el("a:avLst",geom)
            paragraph(body,caption,"TableText")
            html_blocks.append(f"<figure><img src='{html.escape(relative, quote=True)}' alt='{html.escape(caption,quote=True)}'><figcaption>{html.escape(caption)}</figcaption></figure>")
    section = el("sectPr",body); el("pgSz",section,w=11906,h=16838); el("pgMar",section,top=1080,right=1273,bottom=1080,left=1273,header=720,footer=720,gutter=0)
    styles = el("styles")
    for name,size in [("Normal",21),("Title",36),("Heading1",27),("TableText",18),("Code",18)]:
        st = el("style",styles,type="paragraph",styleId=name); el("name",st,val=name); pp = el("pPr",st); el("spacing",pp,after=140,line=290,lineRule="auto")
        if name in ["Title","Heading1"]: el("keepNext",pp)
        rp = el("rPr",st); el("rFonts",rp,ascii="Consolas" if name=="Code" else "Calibri",hAnsi="Calibri",eastAsia="맑은 고딕"); el("sz",rp,val=size)
        if name in ["Title","Heading1"]: el("b",rp); el("color",rp,val="173B55")
    rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="styles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    for i,path in enumerate(image_files,1): rels += f'<Relationship Id="image{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{i}.png"/>'
    rels += '</Relationships>'
    with ZipFile(docx_path,"w",ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        z.writestr("_rels/.rels",'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="doc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml",ET.tostring(doc,encoding="utf-8",xml_declaration=True))
        z.writestr("word/styles.xml",ET.tostring(styles,encoding="utf-8",xml_declaration=True))
        z.writestr("word/_rels/document.xml.rels",rels)
        for i,path in enumerate(image_files,1): z.writestr(f"word/media/image{i}.png",path.read_bytes())
    with ZipFile(docx_path) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith((".xml",".rels")): ET.fromstring(z.read(name))
    if html_path:
        css="body{font:16px/1.7 'Malgun Gothic',sans-serif;color:#193349;max-width:1150px;margin:36px auto;padding:0 24px}h1{font-size:28px}h2{font-size:22px;margin-top:36px;border-bottom:2px solid #237487}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd9e1;padding:9px;vertical-align:top}th{background:#e9f2f6}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:13px;color:#536672}pre{background:#edf2f4;padding:16px;white-space:pre-wrap}@media print{body{font-size:10pt;margin:0}h2{page-break-after:avoid}tr{page-break-inside:avoid}thead{display:table-header-group}}"
        Path(html_path).write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>샴푸 연구 개선 결과</title><style>"+css+"</style><body>"+"\n".join(html_blocks)+"</body></html>",encoding="utf-8")
