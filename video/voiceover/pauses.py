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

class _W:                                             # a word with times, as faster-whisper's
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end

def ctc_words(y, sr, toks):
    """Each script token's start and end from a CTC forced alignment (20 ms frames); tokens without letters are skipped."""
    import torch, torchaudio
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
    lab = {c: i for i, c in enumerate(bundle.get_labels())}
    y16 = librosa.resample(y, orig_sr=sr, target_sr=bundle.sample_rate)
    with torch.inference_mode():
        em, _ = model(torch.from_numpy(y16).unsqueeze(0))
        em = torch.log_softmax(em, dim=-1)
    keep = [(t, re.sub(r"[^A-Z']", "", t.upper().replace("-", ""))) for t in toks]
    keep = [(t, w) for t, w in keep if w]
    trans = "|".join(w for _, w in keep)
    ali, sc = torchaudio.functional.forced_align(em, torch.tensor([[lab[c] for c in trans]], dtype=torch.int32), blank=0)
    spans = torchaudio.functional.merge_tokens(ali[0], sc[0].exp())
    sec = len(y16) / bundle.sample_rate / em.size(1)
    out, cur = [], []
    for sp in spans + [None]:
        if sp is None or sp.token == lab["|"]:
            if cur:
                out.append((cur[0].start * sec, cur[-1].end * sec)); cur = []
        else:
            cur.append(sp)
    return [_W(t, a0, a1) for (t, _), (a0, a1) in zip(keep, out)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("script"); ap.add_argument("sid")
    ap.add_argument("--report", action="store_true", help="measure only; leave the clip as it is")
    ap.add_argument("--sent", type=float, default=0.40); ap.add_argument("--clause", type=float, default=0.30)
    ap.add_argument("--comma", type=float, default=0.22); ap.add_argument("--none", type=float, default=0.14)
    ap.add_argument("--ctc", action="store_true", help="word times from a CTC forced alignment of the script text (torchaudio WAV2VEC2_ASR_BASE_960H) instead of the recognizer's")
    a = ap.parse_args()
    caps = {"sentence": a.sent, "clause": a.clause, "comma": a.comma, "none": a.none}
    from synth import plain
    say = plain(next(g["say"] for g in json.load(open(a.script))["segments"] if g["id"] == a.sid))   # [word](/phonemes/) -> word
    y, sr = sf.read(a.clip, dtype="float32")
    if a.ctc:
        W = ctc_words(y, sr, say.split())
    else:
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
    # v29: a silence belongs to the boundary between the last word whose midpoint precedes the silence's centre and the next
    # word (the recognizer stretches short words such as "and" or "at" over a pause, so the nearest punctuated boundary can be
    # the wrong one: af_sarah paused after "and" in "updates, and how", which the v28 rule booked as the comma's pause)
    widx = {pair[i]: i for i in pair}                 # recognizer word -> script token
    bnd = [(i, (W[pair[i]].end + W[pair[i + 1]].start) / 2 * sr, classify(toks[i]))
           for i in range(len(toks) - 1) if i in pair and i + 1 in pair]
    own = {}
    mids = [(w.start + w.end) / 2 * sr for w in W]
    for g in gaps:
        c = (g[0] + g[1]) / 2
        j = max((k for k, m_ in enumerate(mids) if m_ < c), default=None)
        if j is None or j + 1 >= len(W) or j not in widx or (j + 1) not in widx or widx[j + 1] != widx[j] + 1:
            continue
        i = widx[j]
        if i not in own or (own[i][1] - own[i][0]) < (g[1] - g[0]):
            own[i] = g
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
