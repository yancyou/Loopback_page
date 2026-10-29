"""Render the Loopback promo video from promo.html.

    python3 render.py --preview 3 10 20 25 33 43 51 57 62     # a few frames as PNGs + a contact sheet
    python3 render.py                                          # every frame at 30 fps -> voiceover/build/loopback_promo_silent.mp4
                                                               # and the two posters in ../videos/

The silent render is the input of voiceover/make_av.py, which writes the published ../videos/loopback_promo.mp4 with sound.
Captions are drawn into the frames: run voiceover/vc.py (or clone.py, synth.py) first, so voiceover/build/timings.json holds the
word times; each phrase of a segment's `display` text becomes a cue whose words appear at the times of the matching
spoken words. The title and outro segments get none (their words are on screen). Posters are drawn without captions.

The page draws every frame as a pure function of t (renderAt), so frames are exact and the output is reproducible.
Needs Playwright (Chromium), network access for Google Fonts and MathJax from cdnjs, and ffmpeg with libx264.
"""
import argparse, asyncio, difflib, json, os, re, shutil, subprocess, sys
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
FPS, DUR = 30, 96.4                 # DUR must match promo.html
POSTER_T, HERO_T = 4.2, 28.5        # title card (og:image); the three-rules frame shown before play
VO = os.path.join(HERE, "voiceover")
NO_CAPTIONS = {"title", "outro"}

def wordmap(dw, ws):
    """For each display word, the index of the spoken word it appears with. Words that agree in their letters and digits
    are paired (difflib); the display words between two pairs share the spoken words between them in order, so "6%"
    comes up with "six" of "six percent" and the words after it stay on their own (a split by position ran up to 1 s off)."""
    key = lambda w: re.sub(r"[^0-9a-z]", "", w.lower())
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [key(d) for d in dw], [key(w["word"]) for w in ws], autojunk=False).get_opcodes():
        out += [min(len(ws) - 1, j1 + (k - i1) * (j2 - j1) // (i2 - i1)) for k in range(i1, i2)]
    return out

def cues():
    """Caption cues from voiceover/script.json (display text) and build/timings.json (spoken word times)."""
    sys.path.insert(0, VO)
    from synth import chunks
    s, tm = (json.load(open(os.path.join(VO, f))) for f in ("script.json", os.path.join("build", "timings.json")))
    if [(g["id"], g["say"]) for g in s["segments"]] != [(g["id"], g["say"]) for g in tm["segments"]]:
        sys.exit("voiceover/build/timings.json does not match script.json; run clone.py or synth.py first")
    out = []
    for g, tg in zip(s["segments"], tm["segments"]):
        if g["id"] in NO_CAPTIONS:
            continue
        for k, text in enumerate(chunks(g["display"])):
            ws, dw = [w for w in tg["words"] if w["chunk"] == k], text.split()
            if not ws:
                sys.exit(f"segment {g['id']!r}: no timed words for {text!r}")
            words = [[d, ws[j]["start"]] for d, j in zip(dw, wordmap(dw, ws))]
            out.append({"start": ws[0]["start"] - 0.08, "last": ws[-1]["end"], "words": words, "seg": g["id"]})
    for a, b in zip(out, out[1:]):   # hold a phrase until the next starts, or 0.6 s past its last word at a scene end
        a["end"] = b["start"] if a["seg"] == b["seg"] else a["last"] + 0.6
    if out:
        out[-1]["end"] = out[-1]["last"] + 0.6
    return out

async def frames(times, out_dir, prefix, captions=()):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        await pg.goto("file://" + os.path.join(HERE, "promo.html"), wait_until="networkidle", timeout=120000)
        await pg.wait_for_function("window.__ready === true", timeout=120000)
        await pg.evaluate("c => { window.CUES = c; }", list(captions))
        stage = await pg.query_selector("#stage")
        paths = []
        for i, t in enumerate(times):
            await pg.evaluate("t => renderAt(t)", t)
            path = os.path.join(out_dir, f"{prefix}{i:05d}.png")
            await stage.screenshot(path=path)
            paths.append(path)
            if i % 150 == 0:
                print(f"frame {i}/{len(times)} t={t:.2f}", flush=True)
        await b.close()
        if errs:
            print("page errors:", errs[:5])
        return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs="*", type=float)
    ap.add_argument("--out", default=os.path.join(HERE, "voiceover", "build", "loopback_promo_silent.mp4"))
    ap.add_argument("--tmp", default=os.path.join(HERE, "_frames"))
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.preview is not None:
        paths = asyncio.run(frames(a.preview, a.tmp, "preview_", cues()))
        from PIL import Image
        ims = [Image.open(q).resize((640, 360)) for q in paths]
        cols = 3; rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (640 * cols, 360 * rows), "white")
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % cols) * 640, (i // cols) * 360))
        sheet.save(os.path.join(a.tmp, "contact.png"))
        print("preview:", paths, "-> contact.png")
        return
    n = int(DUR * FPS)
    asyncio.run(frames([i / FPS for i in range(n)], a.tmp, "f", cues()))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(a.tmp, "f%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-profile:v", "high", "-level:v", "4.1", "-x264-params", "ref=4",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-tune", "animation", a.out], check=True)
    vids = os.path.join(HERE, "..", "videos")
    posters = asyncio.run(frames([POSTER_T, HERO_T], a.tmp, "poster_"))   # no captions on the stills
    for path, name in zip(posters, ("loopback_promo_poster.jpg", "loopback_promo_poster_hero.jpg")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-q:v", "3", os.path.join(vids, name)], check=True)
    print("wrote", a.out, os.path.getsize(a.out), "bytes, and the posters in", os.path.normpath(vids))
    shutil.rmtree(a.tmp)

if __name__ == "__main__":
    main()
