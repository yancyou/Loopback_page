"""Assemble the Loopback project page in the OPUS page template.

Inputs (this directory): body.html (markup + inline script), the OPUS stylesheet (Tailwind 3 build + the page's custom
rules, copied from https://yancyou.github.io/OPUS_page/style.css into template/) with its custom `.opus-*` rules renamed to `.lb-*`,
and a few extra rules for the tables, the interactive demo and the buttons.
Outputs: index.html (standalone, GitHub Pages) and artifact.html (the same page without the document skeleton, which the
Claude artifact publisher adds itself).
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OPUS_CSS = os.path.join(HERE, "template", "style.css")   # the OPUS page stylesheet (Tailwind build + its custom rules)

css = open(OPUS_CSS).read().replace(".opus-", ".lb-")
extra = """
/* Loopback additions */
html { scroll-behavior: smooth; color-scheme: light; }
body { background: #ffffff; color: #1f2937; }
.lb-btn { background-color: #0645ad; }
.lb-btn:hover { background-color: #0538a0; }
.lb-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; font-variant-numeric: tabular-nums; }
.lb-table th, .lb-table td { padding: 0.45rem 0.6rem; border-bottom: 1px solid #e5e7eb; text-align: right; white-space: nowrap; }
.lb-table th:first-child, .lb-table td:first-child { text-align: left; }
.lb-table thead th { color: #4b5563; font-weight: 600; font-size: 0.85rem; }
.lb-table td.best { color: #0645ad; font-weight: 700; }
.lb-demo { background: #fff; border: 1px solid #e5e7eb; border-radius: 0.75rem; padding: 1.25rem 1.5rem 1.5rem; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.lb-stagewrap { overflow-x: auto; margin: 0.25rem 0 0.5rem; }
.lb-stagewrap svg { display: block; width: 100%; min-width: 760px; height: auto; }
.lb-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.8rem 1.4rem; margin: 1rem 0 0.4rem; }
.lb-ctl label { display: flex; justify-content: space-between; font-size: 0.9rem; color: #4b5563; }
.lb-ctl label output { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #111827; }
.lb-ctl input[type=range] { width: 100%; accent-color: #0F4D92; }
.lb-ctl select { width: 100%; padding: 0.3rem 0.5rem; border: 1px solid #d1d5db; border-radius: 0.375rem; background: #fff; color: #111827; font: inherit; font-size: 0.9rem; }
.lb-legend { display: flex; flex-wrap: wrap; gap: 0.45rem 1.1rem; font-size: 0.85rem; color: #6b7280; margin: 0.4rem 0 0.2rem; }
.lb-legend span::before { content: ""; display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; background: var(--sw); border: 1.5px solid var(--sb, var(--sw)); box-sizing: border-box; }
.lb-legend .dashed::before { border-style: dashed; }
.lb-legend-lines span::before { width: 18px; height: 3px; border-radius: 2px; border: 0; vertical-align: 3px; }
.lb-tip { position: fixed; pointer-events: none; background: #111827; color: #fff; padding: 6px 9px; border-radius: 6px; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; opacity: 0; transition: opacity .12s; z-index: 50; white-space: nowrap; max-width: 90vw; }
.btn:focus-visible, .btn-sm:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible, a:focus-visible { outline: 2px solid #0645ad; outline-offset: 2px; }
.lb-transport { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem 1rem; margin-bottom: 0.6rem; }
.lb-tbtn { padding: 0.4rem 0.9rem; border: 1px solid #d1d5db; border-radius: 9999px; background: #fff; color: #111827; font: inherit; font-size: 0.95rem; cursor: pointer; }
.lb-tbtn:hover { background: #f3f4f6; }
.lb-tbtn-primary { background: #0F4D92; border-color: #0F4D92; color: #fff; min-width: 6.5rem; }
.lb-tbtn-primary:hover { background: #0C3E76; }
.lb-scrub, .lb-speed { display: flex; flex-direction: column; font-size: 0.85rem; color: #4b5563; min-width: 0; }
.lb-scrub { flex: 1 1 220px; } .lb-speed { flex: 0 1 150px; }
.lb-scrub input, .lb-speed input { width: 100%; accent-color: #0F4D92; }
.lb-scrub output, .lb-speed output { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #111827; }
.lb-h3 { font-family: 'Times New Roman', Times, serif; font-size: 1.25rem; font-weight: 700; color: #111827; margin: 1.4rem 0 0.6rem; }
.lb-ledger { display: grid; grid-template-columns: minmax(150px, 1.4fr) minmax(90px, 1fr) minmax(90px, 1fr) minmax(160px, 2.2fr); gap: 0.35rem 1rem; align-items: center; font-size: 0.95rem; font-variant-numeric: tabular-nums; margin-bottom: 0.8rem; }
.lb-lrow { display: contents; }
.lb-lhead span { font-size: 0.8rem; color: #6b7280; font-weight: 600; padding-bottom: 0.2rem; border-bottom: 1px solid #e5e7eb; }
.lb-lstrong span:first-child { font-weight: 700; color: #0F4D92; }
.lb-meter { display: flex; height: 12px; background: #f3f4f6; border-radius: 6px; overflow: hidden; }
.lb-meter i { display: block; height: 100%; width: 0; background: #4A5560; transition: width .45s ease; }
.lb-meter i + i { background: var(--c); margin-left: 2px; }
.lb-flow path { stroke-dasharray: 5 7; animation: lbflow .5s linear infinite; }
@keyframes lbflow { to { stroke-dashoffset: -12; } }
.lb-formula { color: #374151; margin-top: 0.6rem; overflow-x: auto; }
@media (max-width: 640px) { .lb-demo { padding: 1rem; } .lb-ledger { font-size: 0.85rem; grid-template-columns: minmax(110px, 1.2fr) minmax(70px, 1fr) minmax(70px, 1fr) minmax(120px, 2fr); } }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } * { transition: none !important; } }
"""
body = open(os.path.join(HERE, "body.html")).read().replace("<!--SITE_JS-->", "<script>\n" + open(os.path.join(HERE, "site.js")).read() + "</script>")
title = "<title>Loopback Transformer</title>\n"
style = "<style>\n" + css + extra + "</style>\n"

artifact = title + style + body
open(os.path.join(HERE, "artifact.html"), "w").write(artifact)
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Loopback Transformer: a looped Transformer that stores block updates, keeps the input on its own path, and lets decay set the total update weight while attention allocates it.">\n'
        + title + style + '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
open(os.path.join(HERE, "index.html"), "w").write(page)
print("artifact.html", len(artifact), "bytes; index.html", len(page), "bytes")
