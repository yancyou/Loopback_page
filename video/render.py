"""Render the Loopback promo video from promo.html.

    python3 render.py --preview 3 10 20 25 33 43 51 57 62     # a few frames as PNGs + a contact sheet
    python3 render.py                                          # every frame at 30 fps, then ffmpeg -> ../videos/loopback_promo.mp4

The page draws every frame as a pure function of t (renderAt), so frames are exact and the output is reproducible.
Needs Playwright (Chromium), network access for Google Fonts and MathJax from cdnjs, and ffmpeg with libx264.
"""
import argparse, asyncio, os, shutil, subprocess, sys
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
FPS, DUR = 30, 64.0

async def frames(times, out_dir, prefix):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        await pg.goto("file://" + os.path.join(HERE, "promo.html"), wait_until="networkidle", timeout=120000)
        await pg.wait_for_function("window.__ready === true", timeout=120000)
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
    ap.add_argument("--out", default=os.path.join(HERE, "..", "videos", "loopback_promo.mp4"))
    ap.add_argument("--tmp", default=os.path.join(HERE, "_frames"))
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    if a.preview is not None:
        paths = asyncio.run(frames(a.preview, a.tmp, "preview_"))
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
    asyncio.run(frames([i / FPS for i in range(n)], a.tmp, "f"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(a.tmp, "f%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-profile:v", "high", "-level:v", "4.1", "-x264-params", "ref=4",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-tune", "animation", a.out], check=True)
    poster = os.path.splitext(a.out)[0] + "_poster.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(a.tmp, "f%05d.png" % int(4.2 * FPS)),
                    "-q:v", "3", poster], check=True)
    print("wrote", a.out, os.path.getsize(a.out), "bytes;", poster)
    shutil.rmtree(a.tmp)

if __name__ == "__main__":
    main()
