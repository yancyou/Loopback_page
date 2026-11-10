"""Measure every pause inside a narration clip and bring the long ones into a natural range (v28, 11-10: the user heard
unnatural pauses and tone in v27, whose Kokoro readings were slowed to 0.84-0.88 and paused 0.6-0.7 s at every period).
The clip is transcribed with word times (faster-whisper small.en, CPU) and its words are matched to the segment's `say`
text. Each boundary between two script words is classed by the punctuation after the first word: sentence (. ! ?),
clause (: ; and dashes), comma, or none; its silence is the silent stretch of the audio nearest the recognizer's boundary
(librosa's non-silent intervals, top_db 35, as tighten.py). A silence longer than its class's cap loses its middle, with
a 10 ms crossfade, down to the cap. A pause where the script has no punctuation is listed as a halt.

    clone_venv/bin/python pauses.py CLIP.wav script.json ID [--report] [--sent 0.40] [--clause 0.30] [--comma 0.22] [--none 0.14]
"""
import argparse, difflib, json, re
import numpy as np, soundfile as sf, librosa
from faster_whisper import WhisperModel

def classify(tok):
    t = tok.rstrip("\"')]")
    if t.endswith((".", "!", "?")):
        return "sentence"
    if t.endswith((":", ";", "—", "–")):
        return "clause"
    if t.endswith(","):
        return "comma"
    return "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("script"); ap.add_argument("sid")
    ap.add_argument("--report", action="store_true", help="measure only; leave the clip as it is")
    ap.add_argument("--sent", type=float, default=0.40); ap.add_argument("--clause", type=float, default=0.30)
    ap.add_argument("--comma", type=float, default=0.22); ap.add_argument("--none", type=float, default=0.14)
    a = ap.parse_args()
    caps = {"sentence": a.sent, "clause": a.clause, "comma": a.comma, "none": a.none}
    say = next(g["say"] for g in json.load(open(a.script))["segments"] if g["id"] == a.sid)
    y, sr = sf.read(a.clip, dtype="float32")
    m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
    segs, _ = m.transcribe(librosa.resample(y, orig_sr=sr, target_sr=16000), language="en", beam_size=5, word_timestamps=True,
                           condition_on_previous_text=False)
    W = [w for s in segs for w in s.words]
    key = lambda w: re.sub(r"[^0-9a-z]", "", w.lower())
    toks = say.split()
    heard = {key(w.word) for w in W}
    i = 0                                      # "Hyper Connections" is heard as one word: match the pair as one token
    while i < len(toks) - 1:
        if key(toks[i]) not in heard and key(toks[i] + toks[i + 1]) in heard:
            toks[i:i + 2] = [toks[i] + toks[i + 1]]
        i += 1
    pair = {}
    for i0, j0, n in difflib.SequenceMatcher(None, [key(t) for t in toks], [key(w.word) for w in W], autojunk=False).get_matching_blocks():
        for k in range(n):
            pair[i0 + k] = j0 + k
    iv = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
    gaps = [(int(p1), int(q0)) for (p0, p1), (q0, q1) in zip(iv[:-1], iv[1:])]
    rank = {"sentence": 0, "clause": 1, "comma": 2, "none": 3}
    bnd = [(i, (W[pair[i]].end + W[pair[i + 1]].start) / 2 * sr, classify(toks[i]))
           for i in range(len(toks) - 1) if i in pair and i + 1 in pair]
    own = {}                                   # boundary index -> its gap; each gap goes to one boundary only
    for g in gaps:
        near = [b for b in bnd if g[0] - 0.15 * sr <= b[1] <= g[1] + 0.15 * sr]
        if not near:
            continue
        c = (g[0] + g[1]) / 2
        b = min(near, key=lambda b: (rank[b[2]], abs(b[1] - c)))   # a punctuated boundary first, then the nearest
        if b[0] not in own or (own[b[0]][1] - own[b[0]][0]) < (g[1] - g[0]):
            own[b[0]] = g
    used = set(own.values())
    rows, cuts = [], []
    for i, _, cls in bnd:
        g = own.get(i)
        sil = (g[1] - g[0]) / sr if g else 0.0
        rows.append((cls, sil, toks[i], toks[i + 1]))
        if g and sil > caps[cls]:
            cuts.append((g[0], g[1], caps[cls], cls, toks[i], toks[i + 1]))
    stray = [(g[1] - g[0]) / sr for g in gaps if g not in used and (g[1] - g[0]) / sr >= 0.10]
    dur = len(y) / sr
    print(f"== {a.sid}: {dur:.2f} s, {len(toks)} words, {len(toks) / dur * 60:.0f} wpm, {len(pair)}/{len(toks)} words matched")
    for cls, sil, w0, w1 in rows:
        if sil >= 0.06 or cls != "none":
            flag = "  > cap" if sil > caps[cls] else ""
            print(f"   {cls:<8} {sil:5.2f}  {w0} | {w1}{flag}")
    if stray:
        print(f"   silences away from any word boundary: {', '.join(f'{x:.2f}' for x in stray)}")
    if a.report or not cuts:
        print(f"   {len(cuts)} pauses over their caps" + ("" if a.report else "; clip unchanged"))
        return
    f = int(0.01 * sr)
    x = np.linspace(0, 1, f, dtype=np.float32)
    out, pos = [], 0
    for a1, b0, cap, cls, w0, w1 in sorted(cuts):
        half = int(cap / 2 * sr)
        left, right = y[pos:a1 + half], y[b0 - half:b0]
        out += [left[:-f], left[-f:] * (1 - x) + right[:f] * x, right[f:]]
        pos = b0
    out.append(y[pos:])
    o = np.concatenate(out).astype(np.float32)
    sf.write(a.clip, o, sr, subtype="FLOAT")
    print(f"   {len(cuts)} pauses shortened to their caps; {dur:.2f} -> {len(o) / sr:.2f} s")

if __name__ == "__main__":
    main()
