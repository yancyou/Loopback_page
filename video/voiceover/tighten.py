"""Shorten the pause at each comma of a picked take to at most MAX s (v25, 11-09: the readings paused 0.5-0.8 s at commas such as
"into the stream, the stream carries it", which sounds like a sentence break). The take is transcribed with word times
(faster-whisper small.en, CPU), its words matched to the segment's `say`; at a comma the silent stretch between the two words
(librosa's non-silent intervals, top_db 35, as vc.cap_pauses) loses its middle, with a 10 ms crossfade, down to MAX s.
Sentence breaks are left to vc.cap_pauses (0.55 s). Rewrites the clip in place.

    clone_venv/bin/python tighten.py CLIP.wav script.json ID [MAX]
"""
import difflib, re, sys
import numpy as np, soundfile as sf, librosa
from faster_whisper import WhisperModel

def main():
    path, script, sid = sys.argv[1:4]
    mx = float(sys.argv[4]) if len(sys.argv) > 4 else 0.32
    import json
    say = next(g["say"] for g in json.load(open(script))["segments"] if g["id"] == sid)
    y, sr = sf.read(path, dtype="float32")
    m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
    segs, _ = m.transcribe(librosa.resample(y, orig_sr=sr, target_sr=16000), language="en", beam_size=5, word_timestamps=True,
                           condition_on_previous_text=False)
    W = [w for s in segs for w in s.words]
    key = lambda w: re.sub(r"[^0-9a-z]", "", w.lower())
    toks = say.replace(" ", " ").split()
    pair = {}
    for a, b, n in difflib.SequenceMatcher(None, [key(t) for t in toks], [key(w.word) for w in W], autojunk=False).get_matching_blocks():
        for i in range(n):
            pair[a + i] = b + i
    iv = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
    cuts = []
    for i, t in enumerate(toks[:-1]):
        if not t.endswith(",") or i not in pair or i + 1 not in pair:
            continue
        t0, t1 = W[pair[i]].end, W[pair[i + 1]].start
        mid = (t0 + t1) / 2 * sr
        for (a0, a1), (b0, b1) in zip(iv[:-1], iv[1:]):     # the silent gap nearest the boundary
            if a1 <= mid + 0.15 * sr and b0 >= mid - 0.15 * sr:
                gap = b0 - a1
                if gap > mx * sr:
                    cuts.append((a1, b0, t, toks[i + 1]))
                break
    half, f = int(mx / 2 * sr), int(0.01 * sr)
    x = np.linspace(0, 1, f, dtype=np.float32)
    out, pos = [], 0
    for a1, b0, w0, w1 in cuts:
        left, right = y[pos:a1 + half], y[b0 - half:b0]
        out += [left[:-f], left[-f:] * (1 - x) + right[:f] * x, right[f:]]
        pos = b0
        print(f"  {w0} | {w1}: pause {(b0 - a1) / sr:.2f} s -> {mx:.2f} s")
    out.append(y[pos:])
    o = np.concatenate(out).astype(np.float32)
    sf.write(path, o, sr, subtype="FLOAT")
    print(f"{sid}: {len(cuts)} comma pauses shortened; {len(y) / sr:.2f} -> {len(o) / sr:.2f} s")

if __name__ == "__main__":
    main()
