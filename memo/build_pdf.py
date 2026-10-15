"""Render a memo Markdown file to PDF (Markdown -> styled HTML -> headless Chrome).

Usage: python memo/build_pdf.py [memo/memo_v1.md]
Writes the .html and .pdf next to the .md. Images referenced in the Markdown are resolved
relative to the memo folder.
"""
import subprocess
import sys
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.75in 0.85in; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.5pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 17pt; line-height: 1.2; margin: 0 0 2pt; }
h1 + p em { color: #666; font-size: 9.5pt; }
h2 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 12pt; margin: 14pt 0 4pt;
     border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 7pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
table { border-collapse: collapse; margin: 4pt 0 9pt; font-size: 9.5pt; }
th, td { border-bottom: 1px solid #ddd; padding: 3pt 10pt 3pt 0; text-align: left; }
th { border-bottom: 1.5px solid #888; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 600; }
img { max-width: 100%; display: block; margin: 6pt auto 2pt; }
figure { margin: 6pt 0 10pt; break-inside: avoid; }
figcaption { font-size: 8.5pt; color: #555; }
code { font-size: 9pt; }
h2, h3 { break-after: avoid; }
table, li { break-inside: avoid; }
"""


def main(md_path):
    md_path = Path(md_path).resolve()
    body = markdown.markdown(md_path.read_text(), extensions=["tables", "sane_lists"])
    html_path = md_path.with_suffix(".html")
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
                         f"<body>{body}</body></html>")
    pdf_path = md_path.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True)
    print(pdf_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "memo_v1.md")
