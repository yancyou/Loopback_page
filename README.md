# Loopback Transformer project page

Built on the OPUS project-page template (its Tailwind stylesheet inlined, custom `.opus-*` rules renamed `.lb-*`).

- `body.html` — page markup (header, abstract, method, demo, results, BibTeX)
- `site.js` — header/mobile nav, the animated loop demo, BibTeX copy (inlined by the build)
- `build.py` — assembles `index.html` (standalone, GitHub Pages) and `artifact.html` (the same page without the
  document skeleton, for the Claude artifact publisher); run `python3 build.py` after editing
- `images/` — figures exported from the paper (pdftoppm, 220 dpi) and the UW logo
- `template/style.css` — the OPUS page stylesheet the build inlines

The demo draws in the paper's figure palette (blue/peach shared blocks, green reader, grey prefix/coda/input,
rose shades for coefficient magnitude, navy Loopback / grey Dense for the chart lines). MathJax is loaded from
cdnjs; everything else is inline, so the page needs no build step to serve.

GitHub Pages serves `index.html` from the `main` branch once the repository is public: https://yancyou.github.io/Loopback_page/
A private Claude artifact of the same page is kept for review: https://claude.ai/artifact/5oCZPRShqBX36zwFFDKKBj

Numbers come from the arXiv version of the paper (2026-10-04). Before release, fill in the arXiv link and the
code link in the header buttons and the BibTeX block.
