"""Pause and drawl check of narration takes (v25, 11-09: the user heard the memory line halt after each word of "write,
transport and read" -- 0.62 s and 0.50 s pauses inside a three-word list). Each take is transcribed with word times
(faster-whisper small.en, CPU); its words are matched to the segment's `say` text, and every boundary is classed by the
script's punctuation: after . ; : ? ! a sentence break (any pause; pick caps it), after a comma (pause at most COMMA s), else
inside a phrase (pause at most PLAIN s; a comma pause up to COMMA s passes because tighten.py shortens it afterwards). A short word (four letters or fewer) inside a phrase held longer than DRAWL s is flagged too, and so is a comma read as a
full stop (the recognizer, which punctuates from the prosody, writes a period or question mark there: falling pitch and a
sentence-length pause, as in the v24 take's "depth:" and v25 takes' "into the stream, the stream"), or puts a full stop where the
script has no punctuation at all (v26: "Hyper-Connections. Weight whole streams"). Because the recognizer's word times can be
wrong (v26: a zero-length "Hyper" hid a 0.54 s pause inside "Hyper-Connections"), the silences are also counted from the audio
(librosa, top_db 35): more pauses of GAP s or longer than the script has punctuation marks inside the line flags the take. Writes
the measures to OUT_DIR/prosody.json and a copy of the voice_sim report in which every flagged take has WER 1.0, so that
`vc.py pick --report` passes over it.

    clone_venv/bin/python prosody_gate.py LIST.json script.json REPORT.json OUT_DIR     # LIST: "id/name" -> wav, as voice_sim.py
"""
import difflib, json, os, re, sys

COMMA, PLAIN, DRAWL, GAP = 0.60, 0.28, 0.55, 0.30   # comma pauses over 0.32 s are then shortened by tighten.py

def main():
    lst, script, report, out = sys.argv[1:5]
    import soundfile as sf, librosa
    from faster_whisper import WhisperModel
    say = {g["id"]: g["say"] for g in json.load(open(script))["segments"]}
    m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 1))
    key = lambda w: re.sub(r"[^0-9a-z]", "", w.lower())
    res = {}
    for k, path in sorted(json.load(open(lst)).items()):
        sid = k.split("/")[0]
        if sid not in say:
            continue
        y, sr = sf.read(path, dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        segs, _ = m.transcribe(librosa.resample(y, orig_sr=sr, target_sr=16000), language="en", beam_size=5,
                               word_timestamps=True, condition_on_previous_text=False)
        W = [w for s in segs for w in s.words]
        toks = [w for t in say[sid].replace(" ", " ").split() for w in re.sub(r"(?<=\w)-(?=\w)", "- ", t).split()]
        toks = [t[:-1] if t.endswith("-") else t for t in toks]   # v26b: "Hyper-Connections" is matched as "Hyper" "Connections"
        kind = ["sentence" if re.search(r"[.;:?!]$", t) else "comma" if t.endswith(",") else "plain" for t in toks]
        sm = difflib.SequenceMatcher(None, [key(t) for t in toks], [key(w.word) for w in W], autojunk=False)
        pair = {}
        for a, b, n in sm.get_matching_blocks():
            for i in range(n):
                pair[a + i] = b + i
        worst = {"comma": (0.0, ""), "plain": (0.0, ""), "drawl": (0.0, "")}
        stops = [toks[i] for i in range(len(toks) - 1) if kind[i] != "sentence" and i in pair and re.search(r"[.?!]$", W[pair[i]].word.strip())]
        iv = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)       # v26b: silences the recognizer's word times miss
        gaps = [round((b0 - a1) / sr, 2) for (a0, a1), (b0, b1) in zip(iv[:-1], iv[1:]) if (b0 - a1) / sr >= GAP]
        npunct = sum(1 for k in kind[:-1] if k != "plain")
        for i in range(len(toks) - 1):
            if i in pair and i + 1 in pair and pair[i + 1] == pair[i] + 1 and kind[i] != "sentence":
                gap = W[pair[i + 1]].start - W[pair[i]].end
                if gap > worst[kind[i]][0]:
                    worst[kind[i]] = (round(gap, 2), f"{toks[i]} | {toks[i + 1]}")
        for i, j in pair.items():       # inside a phrase only: after punctuation the recognizer often folds the pause into the word
            d = W[j].end - W[j].start
            if (i == 0 or kind[i - 1] == "plain") and len(key(toks[i])) <= 4 and d > worst["drawl"][0]:
                worst["drawl"] = (round(d, 2), toks[i])
        flags = [f for f, lim in (("comma", COMMA), ("plain", PLAIN), ("drawl", DRAWL)) if worst[f][0] > lim] + (["stop"] if stops else []) \
            + (["extra"] if len(gaps) > npunct else [])
        res[k] = {"comma": worst["comma"], "plain": worst["plain"], "drawl": worst["drawl"], "stops": stops, "gaps": gaps, "punct": npunct,
                  "matched": len(pair), "words": len(toks),
                  "flags": flags}
        print(f"{k:<22} comma {worst['comma'][0]:4.2f} ({worst['comma'][1]:<24}) plain {worst['plain'][0]:4.2f} ({worst['plain'][1]:<24}) "
              f"drawl {worst['drawl'][0]:4.2f} ({worst['drawl'][1]:<10}) stop {' '.join(stops) or '-':<12} gaps {len(gaps)}/{npunct} "
              f"{'FLAG ' + ','.join(flags) if flags else 'ok'}", flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "prosody.json"), "w"), indent=1)
    R = json.load(open(report))
    for k, v in R.items():
        if k in res and res[k]["flags"] and "wer" in v:
            v["wer"], v["prosody_flags"] = 1.0, res[k]["flags"]
    json.dump(R, open(os.path.join(out, "report_prosody.json"), "w"), indent=1)
    n = sum(1 for v in res.values() if not v["flags"])
    print(f"{n} of {len(res)} takes pass; wrote {out}/prosody.json and {out}/report_prosody.json")

if __name__ == "__main__":
    main()
