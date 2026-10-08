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
MATHJAX = '<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"></script>\n'   # deferred: the page's own scripts run first
# the saved light/dark choice is applied before the first paint (page.js keeps it in localStorage 'lb-theme')
THEME = '<script>try{var t=localStorage.getItem("lb-theme");if(t==="dark"||t==="light"){document.documentElement.setAttribute("data-theme",t);}}catch(e){}</script>\n'
SITE = 'https://yancyou.github.io/Loopback_page/'   # GitHub Pages address once the repository is public
ICON = ('<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E'
        '%3Crect width=%2732%27 height=%2732%27 rx=%278%27 fill=%27%230f4d92%27/%3E%3Cpath d=%27M22.5 11.5 A7 7 0 1 0 23 16%27 fill=%27none%27 stroke=%27%23fff%27 '
        'stroke-width=%272.4%27 stroke-linecap=%27round%27/%3E%3Cpath d=%27M19.5 9.2 L23.6 11.2 L21.2 15.1 Z%27 fill=%27%23fff%27/%3E%3C/svg%3E">\n')
title = "<title>Loopback Transformer</title>\n"
style = "<style>\n" + read("style.css") + "</style>\n"
body = read("page.html")
scripts = ("<script>\n" + read("results-data.js") + "</script>\n"
           "<script>\n" + read("page.js") + "</script>\n"
           "<script>\n" + read("site.js") + "</script>\n")

artifact = title + THEME + FONTS + style + body + MATHJAX + scripts
open(os.path.join(HERE, "artifact.html"), "w", encoding="utf-8").write(artifact)
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Loopback Transformer: a looped Transformer that stores block updates, keeps the input on its own path, and lets decay set the total update weight while attention allocates it. Lower loss than dense controls at 191M to 1.4B with 40-68% of the parameters.">\n'
        '<meta name="color-scheme" content="light dark">\n'
        '<meta property="og:type" content="website">\n'
        '<meta property="og:title" content="Loopback Transformer: Towards a Unified View of Residual Looped Transformers">\n'
        '<meta property="og:description" content="Store block updates, keep the input on its own path, let decay set the total update weight and attention allocate it.">\n'
        f'<meta property="og:url" content="{SITE}">\n'
        f'<meta property="og:image" content="{SITE}videos/loopback_promo_poster.jpg">\n'
        '<meta property="og:image:width" content="1920">\n<meta property="og:image:height" content="1080">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:image" content="{SITE}videos/loopback_promo_poster.jpg">\n'
        + ICON + THEME + title + FONTS + style + '</head>\n<body>\n' + body + MATHJAX + scripts + '</body>\n</html>\n')
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(page)
print("artifact.html", len(artifact), "bytes; index.html", len(page), "bytes")
