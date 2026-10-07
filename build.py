"""Assemble the Loopback project page.

Inputs (this directory): page.html (markup), style.css (tokens + rules, light and dark), results-data.js (the paper's
tables as data), page.js (theme, nav, lightbox, BibTeX copy, the two data tables), site.js (the interactive loop demo).
Outputs: index.html (standalone, GitHub Pages) and artifact.html (the same page without the document skeleton, which the
Claude artifact publisher adds itself; it also publishes images/ alongside).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
def read(name):
    return open(os.path.join(HERE, name), encoding="utf-8").read()

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=Noto+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@500;600&display=swap">\n')
MATHJAX = '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"></script>\n'
title = "<title>Loopback Transformer</title>\n"
style = "<style>\n" + read("style.css") + "</style>\n"
body = read("page.html")
scripts = ("<script>\n" + read("results-data.js") + "</script>\n"
           "<script>\n" + read("page.js") + "</script>\n"
           "<script>\n" + read("site.js") + "</script>\n")

artifact = title + FONTS + style + body + MATHJAX + scripts
open(os.path.join(HERE, "artifact.html"), "w", encoding="utf-8").write(artifact)
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Loopback Transformer: a looped Transformer that stores block updates, keeps the input on its own path, and lets decay set the total update weight while attention allocates it. Lower loss than dense controls at 191M to 1.4B with 40-68% of the parameters.">\n'
        '<meta name="color-scheme" content="light dark">\n'
        + title + FONTS + style + '</head>\n<body>\n' + body + MATHJAX + scripts + '</body>\n</html>\n')
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(page)
print("artifact.html", len(artifact), "bytes; index.html", len(page), "bytes")
