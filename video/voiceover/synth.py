"""Synthesize the promo voiceover from script.json with Kokoro-82M on CPU.

    python synth.py script.json

Each segment's `say` is read at speed 1.0, then 1.04, 1.08, ... up to 1.20 until it fits in end - start; a segment that
still does not fit is an error. Kokoro's own lead-in and tail silence (about 0.3 s each) is cut to LEAD/TAIL seconds
around the first/last word, and the clip is placed at `start` on a 24 kHz mono track as long as the video.
Writes build/voiceover.wav and build/timings.json (speed, duration, slack and absolute word times per segment).
`say` may use misaki's [word](/phonemes/) markup to force a pronunciation. Paths in script.json are relative to the repo root.
make_av.py and asr_check.py import the helpers below; only main() needs Kokoro and torch (requirements.txt).
"""
import argparse, difflib, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(HERE, "build")
SR = 24000
SPEEDS = [round(1 + 0.04 * k, 2) for k in range(6)]           # 1.00 ... 1.20
LEAD, TAIL = 0.05, 0.10                                        # silence kept before the first / after the last word
PUNCT = re.compile(r"[,;:.?!]+[\"'”’)\]]*(?=\s|$)")         # a chunk ends here; '1.4B' and '1,000' do not split
LINK = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")

def repo(p):
    return os.path.join(ROOT, p)

def plain(text):
    """Text without misaki markup: [word](/phonemes/) -> word."""
    return LINK.sub(r"\1", text)

def chunks(text):
    """Split after , ; : . ? ! (runs like '?!' count once), keeping the punctuation; pieces with no letters join the previous one."""
    out, i = [], 0
    for m in PUNCT.finditer(text):
        out.append(text[i:m.end()]); i = m.end()
    out.append(text[i:])
    res = []
    for c in (c.strip() for c in out):
        if any(ch.isalnum() for ch in c) or not res:
            res.append(c)
        elif c:
            res[-1] += " " + c
    return [c for c in res if c]

def duration(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found")
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)

def load(path):
    """Read script.json and check it: unique ids, 0 <= start < end, segments in order without overlap, and the same
    number of chunks in say and display."""
    s = json.load(open(path))
    for k in ("voice", "video", "out", "segments"):
        if k not in s:
            sys.exit(f"{path}: missing '{k}'")
    seen, prev = set(), None
    for g in s["segments"]:
        for k in ("id", "start", "end", "say", "display"):
            if k not in g:
                sys.exit(f"{path}: segment {g.get('id', '?')!r} has no '{k}'")
        if g["id"] in seen:
            sys.exit(f"{path}: duplicate segment id {g['id']!r}")
        seen.add(g["id"])
        if not 0 <= g["start"] < g["end"]:
            sys.exit(f"segment {g['id']!r}: need 0 <= start < end, got {g['start']} .. {g['end']}")
        if prev and g["start"] < prev["end"]:
            sys.exit(f"segment {g['id']!r} starts at {g['start']} before {prev['id']!r} ends at {prev['end']}")
        a, b = chunks(plain(g["say"])), chunks(g["display"])
        if len(a) != len(b):
            sys.exit(f"segment {g['id']!r}: say has {len(a)} chunks but display has {len(b)} (chunks end at , ; : . ? !)\n"
                     f"  say:     {' | '.join(a)}\n  display: {' | '.join(b)}")
        prev = g
    return s

def align(say, texts):
    """Chunk index of each token text, by matching the letters and digits of the say chunks against the tokens'."""
    a = [(ch.lower(), k) for k, c in enumerate(chunks(plain(say))) for ch in c if ch.isalnum()]
    b = [(ch.lower(), j) for j, t in enumerate(texts) for ch in t if ch.isalnum()]
    owner = {}
    sm = difflib.SequenceMatcher(None, [x for x, _ in a], [x for x, _ in b], autojunk=False)
    for i, j, n in sm.get_matching_blocks():
        for d in range(n):
            owner.setdefault(b[j + d][1], a[i + d][1])
    out, k = [], 0
    for j in range(len(texts)):
        k = owner.get(j, k)
        out.append(k)
    return out

def speak(pipe, text, voice, speed):
    """Audio for text plus its tokens as (text, phonemes, start, end), times in seconds from the start of the audio."""
    import numpy as np, torch
    torch.manual_seed(0)  # the vocoder adds noise; a fixed seed makes reruns identical
    audio, toks, off = [], [], 0.0
    for r in pipe(text, voice=voice, speed=speed):
        for t in r.tokens:
            toks.append((t.text, t.phonemes, None if t.start_ts is None else off + t.start_ts,
                         None if t.end_ts is None else off + t.end_ts))
        audio.append(r.audio.numpy())
        off += len(audio[-1]) / SR
    return np.concatenate(audio), toks

def fit(pipe, g, voice):
    """Lowest speed whose trimmed clip fits the window: (clip, speed, words as (text, start, end, chunk), unspoken)."""
    import numpy as np
    win = g["end"] - g["start"]
    for sp in SPEEDS:
        audio, toks = speak(pipe, g["say"], voice, sp)
        ck = align(g["say"], [t[0] for t in toks])
        words = [(t[0], t[2], t[3], c) for t, c in zip(toks, ck)
                 if any(ch.isalnum() for ch in t[0]) and t[1] and t[2] is not None and t[3] is not None]
        unspoken = [t[0] for t in toks if any(ch.isalnum() for ch in t[0]) and not t[1]]
        if not words:
            sys.exit(f"segment {g['id']!r}: Kokoro produced no words")
        a = max(0, round((words[0][1] - LEAD) * SR))
        b = min(len(audio), round((words[-1][2] + TAIL) * SR))
        clip = np.ascontiguousarray(audio[a:b], dtype=np.float32)
        words = [(w, s - a / SR, e - a / SR, c) for w, s, e, c in words]
        if len(clip) / SR <= win + 1e-6:
            return clip, sp, words, unspoken
    return clip, None, words, unspoken

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    a = ap.parse_args()
    s = load(a.script)
    if s["voice"][:1] not in ("a", "b"):
        sys.exit(f"voice {s['voice']!r}: only the English Kokoro voices (a*, b*) are set up here")
    vdur = duration(repo(s["video"]))
    if s["segments"] and s["segments"][-1]["end"] > vdur:
        sys.exit(f"segment {s['segments'][-1]['id']!r} ends at {s['segments'][-1]['end']} after the video ({vdur:.2f} s)")
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, soundfile as sf, torch
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    from kokoro import KPipeline
    pipe = KPipeline(lang_code=s["voice"][0], repo_id="hexgrad/Kokoro-82M", device="cpu")
    track = np.zeros(round(vdur * SR), np.float32)
    rows, segs, over = [], [], []
    for g in s["segments"]:
        clip, sp, words, unspoken = fit(pipe, g, s["voice"])
        dur, win = len(clip) / SR, g["end"] - g["start"]
        rows.append((g["id"], f"{g['start']:.2f}-{g['end']:.2f}", dur, f"{sp:.2f}" if sp else "-", win - dur))
        if unspoken:
            print(f"warning: segment {g['id']!r}: no phonemes for {unspoken} (silently skipped by Kokoro)")
        if sp is None:
            over.append(f"segment {g['id']!r} is {dur - win:.2f} s over its {win:.2f} s window at speed {SPEEDS[-1]:.2f}")
            continue
        i = round(g["start"] * SR)
        track[i:i + len(clip)] += clip[:len(track) - i]
        segs.append({"id": g["id"], "start": g["start"], "end": g["end"], "speed": sp, "duration": round(dur, 3),
                     "slack": round(win - dur, 3), "say": g["say"],
                     "words": [{"word": w, "start": round(g["start"] + st, 3), "end": round(g["start"] + en, 3), "chunk": c}
                               for w, st, en, c in words]})
    print(f"{'id':<14}{'window':>13}{'dur':>8}{'speed':>7}{'slack':>8}")
    for r in rows:
        print(f"{r[0]:<14}{r[1]:>13}{r[2]:8.2f}{r[3]:>7}{r[4]:8.2f}")
    if over:
        sys.exit("\n".join(over))
    os.makedirs(BUILD, exist_ok=True)
    sf.write(os.path.join(BUILD, "voiceover.wav"), track, SR, subtype="FLOAT")
    json.dump({"voice": s["voice"], "video": s["video"], "video_duration": vdur, "sample_rate": SR, "segments": segs},
              open(os.path.join(BUILD, "timings.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote build/voiceover.wav ({vdur:.2f} s, peak {np.abs(track).max():.2f}) and build/timings.json")

if __name__ == "__main__":
    main()
