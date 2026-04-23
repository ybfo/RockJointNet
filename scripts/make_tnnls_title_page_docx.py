"""Generate the TNNLS title page as a standalone Word document.

The script intentionally uses only Python's standard library. This keeps the
reproducibility package independent from python-docx while still producing a
valid .docx file that can be opened and edited in Microsoft Word.
"""

from __future__ import annotations

import datetime as _dt
import html
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "RockJointNet_TNNLS_manuscript"
OUTPUT_DIR = ROOT / "RockJointNet_paper_figure_table_code" / "outputs"

DOCX_PATHS = [
    MANUSCRIPT_DIR / "tnnls_title_page.docx",
    OUTPUT_DIR / "tnnls_title_page.docx",
]


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def run(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    size: int = 20,
    superscript: bool = False,
    color: str | None = None,
    underline: bool = False,
) -> str:
    props = [
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>',
        f'<w:sz w:val="{size}"/>',
        f'<w:szCs w:val="{size}"/>',
    ]
    if bold:
        props.append("<w:b/>")
        props.append("<w:bCs/>")
    if italic:
        props.append("<w:i/>")
        props.append("<w:iCs/>")
    if superscript:
        props.append('<w:vertAlign w:val="superscript"/>')
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    if underline:
        props.append('<w:u w:val="single"/>')
    return (
        "<w:r>"
        f"<w:rPr>{''.join(props)}</w:rPr>"
        f'<w:t xml:space="preserve">{esc(text)}</w:t>'
        "</w:r>"
    )


def br() -> str:
    return "<w:r><w:br/></w:r>"


def para(
    runs: list[str] | str,
    *,
    align: str | None = None,
    before: int = 0,
    after: int = 120,
    line: int | None = None,
) -> str:
    body = "".join(runs) if isinstance(runs, list) else runs
    ppr = [f'<w:spacing w:before="{before}" w:after="{after}"']
    if line is not None:
        ppr[0] += f' w:line="{line}" w:lineRule="auto"'
    ppr[0] += "/>"
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{body}</w:p>"


def hyperlink(text: str, rid: str = "rIdEmail") -> str:
    return (
        f'<w:hyperlink r:id="{rid}" w:history="1">'
        + run(text, size=20, color="0563C1", underline=True)
        + "</w:hyperlink>"
    )


def build_document_xml() -> str:
    title = (
        "GeoSPIN: Continuous Stress-Path Integral Learning for "
        "Physics-Regularized Rock-Joint Shear Prediction"
    )
    paragraphs: list[str] = []
    paragraphs.append(para([run(title, bold=True, size=34)], align="center", after=280))
    paragraphs.append(
        para(
            [
                run("Yubo Huang", size=24),
                run("a,b", size=18, superscript=True),
                run(", Xiongyu Hu", size=24),
                run("a,b,*", size=18, superscript=True),
                run(", Xin Lai", size=24),
                run("a,b", size=18, superscript=True),
                run(", Xiaobo Zheng", size=24),
                run("a,b", size=18, superscript=True),
                br(),
                run("Yong Fang", size=24),
                run("a,b", size=18, superscript=True),
                run(",  Zixi Wang", size=24),
                run("c", size=18, superscript=True),
                run(", Jingzehua Xu", size=24),
                run("d", size=18, superscript=True),
            ],
            align="center",
            after=280,
        )
    )
    affiliations = [
        (
            "a",
            "School of Civil Engineering, Southwest Jiaotong University, Chengdu, Sichuan, China",
        ),
        (
            "b",
            "Key Laboratory of Transportation Tunnel Engineering, Ministry of Education, "
            "Southwest Jiaotong University, Chengdu, Sichuan, China",
        ),
        (
            "c",
            "School of Information and Software Engineering, University of Electronic Science "
            "and Technology of China, Chengdu, Sichuan, China",
        ),
        (
            "d",
            "Department of Mechanical Engineering, The University of Hong Kong, Pokfulam Road, "
            "Hong Kong, China",
        ),
    ]
    aff_runs: list[str] = []
    for idx, (label, text) in enumerate(affiliations):
        if idx:
            aff_runs.append(br())
        aff_runs.append(run(label, italic=True, size=20, superscript=True))
        aff_runs.append(run(" " + text, italic=True, size=20))
    paragraphs.append(para(aff_runs, align="center", after=300, line=276))

    paragraphs.append(
        para(
            [
                run("*", size=16, superscript=True),
                run("Corresponding author", bold=True, size=20),
                br(),
                run("Xiongyu Hu", size=20),
                br(),
                run(
                    "School of Civil Engineering, Southwest Jiaotong University, Chengdu, "
                    "Sichuan, China",
                    size=20,
                ),
                br(),
                run(
                    "Key Laboratory of Transportation Tunnel Engineering, Ministry of Education, "
                    "Southwest Jiaotong University, Chengdu, Sichuan, China",
                    size=20,
                ),
                br(),
                run("E-mail: ", size=20),
                hyperlink("huxiongyu@swjtu.edu.cn"),
            ],
            after=340,
            line=276,
        )
    )
    paragraphs.append(para([run("Acknowledgments", bold=True, size=24)], after=140))
    paragraphs.append(
        para(
            [
                run(
                    "This research was supported by the National Key R&D Program of China "
                    "(No. 2024YFB2606101), the National Natural Science Foundation of China "
                    "(No.52478418) and the Sichuan Science and Technology Program "
                    "(No. 2025ZNSFSC1294).",
                    size=20,
                )
            ],
            line=276,
        )
    )

    sect_pr = (
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<w:body>{''.join(paragraphs)}{sect_pr}</w:body>"
        "</w:document>"
    )


def write_docx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        "word/_rels/document.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdEmail" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="mailto:huxiongyu@swjtu.edu.cn" TargetMode="External"/>
</Relationships>""",
        "word/document.xml": build_document_xml(),
        "docProps/core.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>GeoSPIN: Continuous Stress-Path Integral Learning for Physics-Regularized Rock-Joint Shear Prediction</dc:title>
  <dc:creator>Yubo Huang; Xiongyu Hu; Xin Lai; Xiaobo Zheng; Yong Fang; Zixi Wang; Jingzehua Xu</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
        "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
</Properties>""",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def main() -> None:
    for path in DOCX_PATHS:
        write_docx(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
