# Loopback Transformer project page

A hand-written page in the style of recent model project pages: a title block, key contributions, a performance
overview, the method, an interactive loop demo, and the paper's results as data-driven tables.

- `page.html` — page markup (nav, hero, overview, abstract, contributions, method, demo, results, BibTeX, lightbox)
- `style.css` — design tokens (light and dark) and all rules; figures and the demo sit on a white "paper" surface in both themes
- `results-data.js` — the paper's tables as data (main results at four scales, compute); the page checks the printed means
  against the per-task numbers on load (`window.LOOPBACK_CHECK` is empty when they agree)
- `page.js` — theme toggle, section highlighting, figure lightbox, BibTeX copy, Table 1 (scale selector, Δ vs Dense) and Table 2
- `site.js` — the animated loop demo (unchanged)
- `build.py` — assembles `index.html` (standalone, GitHub Pages) and `artifact.html` (the same page without the document
  skeleton, for the Claude artifact publisher); run `python3 build.py` after editing
- `images/` — figures exported from the paper (pdftoppm, 220 dpi)
- `video/` — the promo video's source: `promo.html` draws every frame as a function of time (`renderAt(t)`, 1920×1080),
  `render.py` screenshots it at 30 fps with Playwright and encodes with ffmpeg (`python3 render.py --preview 3 21 43` for a
  few frames and a contact sheet; `python3 render.py` for the full 64 s video)
- `videos/` — the rendered `loopback_promo.mp4` (H.264, no audio) and its poster frame, shown in the page's video section
- `blog/` — the blog post source (see its own files)

Fonts come from Google Fonts (Manrope, Noto Sans, JetBrains Mono) and MathJax from cdnjs; everything else is inline, so the
page needs no build step to serve. The static tables (configurations, inference memory, ablations, readers) are in `page.html`.

GitHub Pages serves `index.html` from the `main` branch once the repository is public: https://yancyou.github.io/Loopback_page/
A private Claude artifact of the same page is kept for review: https://claude.ai/artifact/5oCZPRShqBX36zwFFDKKBj

Numbers come from the arXiv version of the paper (2026-10-04). Before release, fill in the arXiv link and the code link
in the header buttons and the BibTeX block.
