# Loopback Transformer project page

A hand-written page in the style of recent model project pages: a title block, key contributions, a performance
overview, the method, an interactive loop demo, and the paper's results as data-driven tables.

- `page.html` — page markup (nav, hero, overview, abstract, contributions, method, demo, results, BibTeX, lightbox)
- `style.css` — design tokens (light and dark) and all rules; figures and the demo sit on a white "paper" surface in both themes
- `results-data.js` — the paper's tables as data (main results at five scales, 191M to 3B; compute); the page checks the printed means
  against the per-task numbers on load (`window.LOOPBACK_CHECK` is empty when they agree)
- `page.js` — theme toggle, section highlighting, figure lightbox, BibTeX copy, Table 1 (scale selector, Δ vs Standard) and Table 2
- `site.js` — the animated loop demo (unchanged)
- `build.py` — assembles `index.html` (standalone, GitHub Pages) and `artifact.html` (the same page without the document
  skeleton, for the Claude artifact publisher); run `python3 build.py` after editing
- `images/` — figures exported from the paper's PDFs (pdftoppm, 2400 px wide)
- `video/` — the promo video's source: `promo.html` draws every frame as a function of time (`renderAt(t)`, 1920×1080),
  `render.py` screenshots it at 30 fps with Playwright and encodes a silent H.264 file with ffmpeg (`python3 render.py
  --preview 3 21 43` for a few frames and a contact sheet; `python3 render.py` for the full 96 s video, with word-level captions burned in from the narration timings, and the two posters)
- `video/voiceover/` — the narration: `script.json` holds one sentence per scene with its time window, the words that are
  spoken and the caption wording. `vc.py source` reads each sentence with Chatterbox's built-in voice and `vc.py convert`
  converts it to the narrator's voice with ChatterboxVC (the reference recording is not part of this repository);
  `synth.py` (Kokoro-82M) and `clone.py` are the earlier routes. `make_av.py` mixes the narration and the music bed onto the
  silent render (loudness-normalized AAC, WebVTT/SRT from the word timings) and `asr_check.py` compares a Whisper transcript
  with the script
- `video/music/` — the music bed: `gen_music.py` generated the track with ACE-Step 1.5 and `cut.py` drops whole bars on the
  beat grid to fit the video (`--drop 24 2` for the current cut); `pad.py` is the earlier synthesized placeholder. Generated
  audio stays out of git (`video/music/build/`)
- `videos/` — `loopback_promo.mp4` (H.264 with narration and music, captions burned in), its captions
  `loopback_promo.vtt`/`.srt`, the title-card poster
  (og:image) and the three-rules poster the hero shows before play
- `blog/` — the blog post source (see its own files)

Fonts come from Google Fonts (Manrope, Noto Sans, JetBrains Mono) and MathJax from cdnjs; everything else is inline, so the
page needs no build step to serve. The static tables (configurations, mathematical and code evaluation, ablations and initialization, readers, inference memory)
are in `page.html`.

GitHub Pages serves `index.html` from the `main` branch once the repository is public: https://yancyou.github.io/Loopback_page/
A private Claude artifact of the same page is kept for review: https://claude.ai/artifact/5oCZPRShqBX36zwFFDKKBj

Numbers come from the arXiv version of the paper (2026-10-29). Before release, fill in the arXiv link and the code link
in the header buttons and the BibTeX block.
