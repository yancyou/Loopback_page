"""Assemble the Loopback project page.

Inputs (this directory): page.html (markup), style.css (tokens + rules, light and dark), results-data.js (the paper's
tables as data), page.js (theme, nav, lightbox, BibTeX copy, the two data tables), site.js (the interactive loop demo).
Outputs: index.html (standalone, GitHub Pages) and artifact.html (the same page without the document skeleton, which the
Claude artifact publisher adds itself; it also publishes images/ and videos/ alongside, with the captions inline).
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
title = "<title>Loopback Transformer</title>\n"
style = "<style>\n" + read("style.css") + "</style>\n"
body = read("page.html")
scripts = ("<script>\n" + read("results-data.js") + "</script>\n"
           "<script>\n" + read("page.js") + "</script>\n"
           "<script>\n" + read("site.js") + "</script>\n")

# The artifact host serves no .vtt, so the preview carries the captions inline and hands the track a same-origin blob URL;
# index.html keeps the file (videos/loopback_promo.vtt) for GitHub Pages.
# (Only when the page has a caption track: the current video burns its captions into the frames.)
TRACK = '<track kind="captions" src="videos/loopback_promo.vtt" srclang="en" label="English">'
captions, artifact_body = "", body
if TRACK in body:
    vtt = read(os.path.join("videos", "loopback_promo.vtt"))
    assert "</script" not in vtt
    captions = ('<script type="text/vtt" id="lb-vtt">\n' + vtt + '</script>\n'
                '<script>(function(){var t=document.querySelector("track[data-vtt]"),s=document.getElementById("lb-vtt");'
                'if(t&&s&&window.Blob&&window.URL){t.src=URL.createObjectURL(new Blob([s.textContent.trim()+"\\n"],{type:"text/vtt"}));}})();</script>\n')
    artifact_body = body.replace(TRACK, '<track kind="captions" srclang="en" label="English" data-vtt="inline">')
artifact = title + THEME + FONTS + style + artifact_body + captions + MATHJAX + scripts
open(os.path.join(HERE, "artifact.html"), "w", encoding="utf-8").write(artifact)
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Loopback Transformer: a looped Transformer that stores block updates, keeps the input on its own path, and lets decay set the total update weight while attention allocates it. Lower validation loss than standard Transformers from 191M to 3B with 40-75% of their Transformer parameters, at matched training tokens and executed depth.">\n'
        '<meta name="color-scheme" content="light dark">\n'
        '<meta property="og:type" content="website">\n'
        '<meta property="og:title" content="Loopback Transformer: Towards a Unified View of Residual Looped Transformers">\n'
        '<meta property="og:description" content="Store block updates, keep the input on its own path, let decay set the total update weight and attention allocate it.">\n'
        f'<meta property="og:url" content="{SITE}">\n'
        f'<meta property="og:image" content="{SITE}videos/loopback_promo_poster.jpg">\n'
        '<meta property="og:image:width" content="1920">\n<meta property="og:image:height" content="1080">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:image" content="{SITE}videos/loopback_promo_poster.jpg">\n'
        + THEME + title + FONTS + style + '</head>\n<body>\n' + body + MATHJAX + scripts + '</body>\n</html>\n')
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(page)
print("artifact.html", len(artifact), "bytes; index.html", len(page), "bytes")
