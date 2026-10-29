"""The promo voiceover in a reference speaker's voice with standard American pronunciation: Kokoro reads the script,
Chatterbox's voice conversion then gives it the reference timbre. Pronunciation, stress and rhythm are Kokoro's;
voice conversion changes only who it sounds like, so the clone does not pick up the reference recording's accent
(clone.py, which generates from the reference directly, does).

    tts_venv/bin/python   vc.py source  script.json [--voice am_michael]        # Kokoro, CPU
    SIGMA_CLIMB4K=1 CUDA_VISIBLE_DEVICES=0 clone_venv/bin/python vc.py source script.json --engine chatterbox [--takes 6]
    SIGMA_CLIMB4K=1 CUDA_VISIBLE_DEVICES=0 clone_venv/bin/python vc.py convert script.json --ref /path/to/sample.wav
    ... source/convert --only visits eff      # read and convert just these segments; keep the other takes
    clone_venv/bin/python vc.py place script.json          # every previous take at its (moved) start; converts nothing

--only: source reads only the listed segments; convert converts only those and takes every other segment's clip, word
times and scores from the previous build/voiceover.wav and build/timings.json (its `say` must be unchanged), placed at
its start in script.json, which may have moved (its word times move with it). So a script change re-voices only the
lines that changed and the other takes stay exactly as they were.
--seed ID=N (source, chatterbox) reads segment ID with seed N only: the pick above judges a take by how the source voice
is heard, and a take can still blur in the conversion. For the 10-29 "visits" line every take but two came out of
convert as "loop back to grades" (all decoder seeds alike); seed 7 converts cleanly and is the most similar.

source: for each segment, Kokoro speeds SPEEDS are tried and the trimmed reading whose pace is nearest TARGET_WPM while
fitting the window (end - start - 0.15 s) is kept as build/vc/<id>.wav. Pick --voice near the speaker's pitch (am_michael
and am_adam sit at about 110 Hz). --engine chatterbox reads with Chatterbox's built-in voice instead (its own default
speaker, nobody cloned), whose intonation is far livelier than Kokoro's male voices (12-16 semitones against ~8.5, which
the conversion narrows by about a third); --takes seeds are drawn at CB_EXAG / CB_CFG (calmer and slower than 0.5 / 0.5)
and the one nearest TARGET_WPM among those within one word of the best word error rate that fit is kept. That voice
still talks at ~170 wpm, so a faster take is slowed to TARGET_WPM with atempo before conversion; the conversion then
re-synthesizes it from speech tokens, which leaves natural slow speech rather than stretched audio.
convert: each reading goes through ChatterboxVC with the reference (its first 10 s; keep the recording outside the
repository) under --seeds decoder seeds (the flow decoder's noise can blur a vowel: "mean" heard as "main"), then
clone.py's clean-up (POST filter, trim); the conversion with the lowest word error rate, then the highest speaker
similarity, is kept. It is transcribed with faster-whisper small.en for the word
error rate, the caption word times and a speaker-similarity check against the reference. Writes the same
build/voiceover.wav and build/timings.json as synth.py and clone.py, so make_av.py and asr_check.py run unchanged; a
reading the conversion lengthened past its window is sped up by at most clone.MAX_TEMPO.
"""
import argparse, json, os, sys
from synth import BUILD, LEAD, SR, TAIL, align, duration, load, plain

SPEEDS = [0.80, 0.84, 0.88, 0.92, 0.96, 1.00, 1.04, 1.08, 1.12]
TARGET_WPM, MARGIN = 140, 0.15
CB_EXAG, CB_CFG = 0.4, 0.3
VC_DIR = os.path.join(BUILD, "vc")

def picked(s, only):
    """The segments to read or convert: all of them, or those named in --only."""
    if only:
        unknown = set(only) - {g["id"] for g in s["segments"]}
        if unknown:
            sys.exit(f"--only: no segment {sorted(unknown)} in the script")
    return [g for g in s["segments"] if not only or g["id"] in only]

def kept_takes(s, only):
    """Clip, timings and scores of each segment not in `only`, from the previous build/voiceover.wav and timings.json."""
    import numpy as np, soundfile as sf
    tm = json.load(open(os.path.join(BUILD, "timings.json")))
    old, sr = sf.read(os.path.join(BUILD, "voiceover.wav"), dtype="float32")
    if sr != SR:
        sys.exit(f"build/voiceover.wav is {sr} Hz, expected {SR}")
    prev, out = {g["id"]: g for g in tm["segments"]}, {}
    for g in s["segments"]:
        if g["id"] in only:
            continue
        p = prev.get(g["id"])
        if p is None or p["say"] != g["say"]:
            sys.exit(f"segment {g['id']!r}: build/timings.json has no take of this text; add it to --only")
        a, b = round(p["start"] * SR), round(p["end"] * SR)     # the old window holds this clip and nothing else
        nz = np.nonzero(old[a:b])[0]
        out[g["id"]] = (p, np.ascontiguousarray(old[a:a + nz[-1] + 1]))
    return out

def source(s, voice, only=None):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, soundfile as sf, torch
    from kokoro import KPipeline
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M", device="cpu")
    os.makedirs(VC_DIR, exist_ok=True)
    print(f"{'id':<9}{'window':>13}{'speed':>7}{'dur':>7}{'wpm':>6}")
    for g in picked(s, only):
        win, nwords, best = g["end"] - g["start"] - MARGIN, len(plain(g["say"]).split()), None
        for sp in SPEEDS:
            torch.manual_seed(0)
            audio, words = [], []
            for r in pipe(g["say"], voice=voice, speed=sp):
                audio.append(r.audio.numpy())
            audio = np.concatenate(audio)
            n = len(audio) // 240; rms = np.sqrt(np.mean(audio[:n * 240].reshape(n, 240) ** 2, 1) + 1e-12)
            on = np.nonzero(20 * np.log10(rms / rms.max()) > -42)[0]
            clip = audio[max(0, on[0] * 240 - round(LEAD * SR)):min(len(audio), (on[-1] + 1) * 240 + round(TAIL * SR))]
            dur = len(clip) / SR; wpm = nwords / dur * 60
            if dur <= win and (best is None or abs(wpm - TARGET_WPM) < abs(best[2] - TARGET_WPM)):
                best = (sp, clip, wpm)
        if best is None:
            sys.exit(f"segment {g['id']!r} does not fit {win:.2f} s even at speed {SPEEDS[-1]}")
        sp, clip, wpm = best
        sf.write(os.path.join(VC_DIR, f"{g['id']}.wav"), clip.astype(np.float32), SR, subtype="FLOAT")
        json.dump({"say": g["say"], "voice": voice, "speed": sp}, open(os.path.join(VC_DIR, f"{g['id']}.json"), "w"))
        print(f"{g['id']:<9}{g['start']:6.2f}-{g['end']:<6.2f}{sp:7.2f}{len(clip) / SR:7.2f}{wpm:6.0f}")

def source_chatterbox(s, takes, only=None, pins=None):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, soundfile as sf, torch
    from faster_whisper import WhisperModel
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from chatterbox.tts import ChatterboxTTS
    from asr_check import tokens
    from clone import score, tempo, transcribe, trim
    m = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")   # built-in voice: m.conds
    norm, asr = EnglishTextNormalizer({}), WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 1))
    os.makedirs(VC_DIR, exist_ok=True)
    print(f"{'id':<9}{'window':>13}{'seed':>5}{'dur':>7}{'wpm':>6}{'WER':>6}  takes (s)")
    for g in picked(s, only):
        win, nwords, cand = g["end"] - g["start"] - MARGIN, len(plain(g["say"]).split()), []
        for seed in [pins[g["id"]]] if g["id"] in (pins or {}) else range(takes):
            torch.manual_seed(seed)
            clip = trim(m.generate(plain(g["say"]), exaggeration=CB_EXAG, cfg_weight=CB_CFG).squeeze(0).numpy())
            cand.append((seed, clip, score(norm, g["say"], transcribe(asr, clip))))
        best = min(c[2] for c in cand); slack = max(0.05, 1 / max(1, len(tokens(norm, plain(g["say"]))))) + 1e-9
        ok = [c for c in cand if c[2] <= best + slack and len(c[1]) / SR <= win] or [min(cand, key=lambda c: len(c[1]))]
        seed, clip, w = min(ok, key=lambda c: abs(nwords / (len(c[1]) / SR) * 60 - TARGET_WPM))
        raw_wpm = nwords / (len(clip) / SR) * 60
        if raw_wpm > TARGET_WPM:
            clip = tempo(clip, max(TARGET_WPM / raw_wpm, (len(clip) / SR) / win))   # slow down, but stay inside the window
        sf.write(os.path.join(VC_DIR, f"{g['id']}.wav"), clip.astype(np.float32), SR, subtype="FLOAT")
        json.dump({"say": g["say"], "voice": "chatterbox-default", "speed": 1.0, "seed": seed}, open(os.path.join(VC_DIR, f"{g['id']}.json"), "w"))
        print(f"{g['id']:<9}{g['start']:6.2f}-{g['end']:<6.2f}{seed:5d}{len(clip) / SR:7.2f}{nwords / (len(clip) / SR) * 60:6.0f}{w:6.2f}  "
              f"{[round(len(c[1]) / SR, 2) for c in cand]}")

def convert(s, ref, seeds=4, only=None):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, soundfile as sf
    vdur = s.get("duration") or duration(os.path.join(os.path.dirname(os.path.dirname(BUILD)), "..", s["video"]))
    track = np.zeros(round(vdur * SR), np.float32)
    picked(s, only)
    kept = kept_takes(s, only) if only is not None else {}
    if any(g["id"] not in kept for g in s["segments"]):        # something to convert ("place" converts nothing)
        import torch, librosa
        from huggingface_hub import hf_hub_download
        from faster_whisper import WhisperModel
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        from chatterbox.vc import ChatterboxVC
        from chatterbox.models.voice_encoder import VoiceEncoder
        from safetensors.torch import load_file
        from clone import MAX_TEMPO, POST, ffilter, score, tempo, transcribe, trim
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        vc = ChatterboxVC.from_pretrained(dev); vc.set_target_voice(ref)
        ve = VoiceEncoder(); ve.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", "ve.safetensors"))); ve.to(dev).eval()
        emb = lambda w16: ve.embeds_from_wavs([w16], sample_rate=16000).mean(0)
        ref_emb = emb(librosa.load(ref, sr=16000)[0])
        norm = EnglishTextNormalizer({})
        asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 1))
    segs, rows = [], []
    for g in s["segments"]:
        if g["id"] in kept:
            p, clip = kept[g["id"]]
            win, d = g["end"] - g["start"], g["start"] - p["start"]
            if len(clip) / SR > win:
                sys.exit(f"segment {g['id']!r}: its kept take is {len(clip) / SR:.2f} s, longer than its {win:.2f} s window")
            i = round(g["start"] * SR)
            track[i:i + len(clip)] += clip[:len(track) - i]
            segs.append(dict(p, start=g["start"], end=g["end"], slack=round(win - p["duration"], 3),
                             words=[dict(w, start=round(w["start"] + d, 3), end=round(w["end"] + d, 3)) for w in p["words"]]))
            rows.append((g["id"], f"{g['start']:.2f}-{g['end']:.2f}", len(clip) / SR, p["kokoro"][1], p["wer"], p["similarity"],
                         p["speed"], f"kept{d:+.2f} s" if d else "kept"))
            continue
        src = os.path.join(VC_DIR, f"{g['id']}.wav")
        meta = json.load(open(os.path.splitext(src)[0] + ".json"))
        if meta["say"] != g["say"]:
            sys.exit(f"build/vc/{g['id']}.wav was read from other text; rerun vc.py source")
        tries = []
        for seed in range(seeds):
            torch.manual_seed(seed)
            c = trim(ffilter(vc.generate(src).squeeze(0).numpy(), POST))
            ce = emb(librosa.resample(c, orig_sr=SR, target_sr=16000))
            tries.append((score(norm, g["say"], transcribe(asr, c)), -float(np.dot(ce, ref_emb) / (np.linalg.norm(ce) * np.linalg.norm(ref_emb))), seed, c))
        clip = min(tries, key=lambda t: (round(t[0], 3), t[1]))[3]
        win = g["end"] - g["start"]
        factor = max(1.0, len(clip) / SR / (win - 0.02))
        if factor > MAX_TEMPO:
            sys.exit(f"segment {g['id']!r}: converted reading is {len(clip) / SR:.2f} s for a {win:.2f} s window")
        if factor > 1.0:
            clip = tempo(clip, factor)
        words = transcribe(asr, clip)
        e = emb(librosa.resample(clip, orig_sr=SR, target_sr=16000))
        sim = float(np.dot(e, ref_emb) / (np.linalg.norm(e) * np.linalg.norm(ref_emb)))
        w = score(norm, g["say"], words)
        ck = align(g["say"], [x for x, _, _ in words])
        i = round(g["start"] * SR)
        track[i:i + len(clip)] += clip[:len(track) - i]
        dur = len(clip) / SR
        rows.append((g["id"], f"{g['start']:.2f}-{g['end']:.2f}", dur, meta["speed"], w, sim, factor, "new"))
        segs.append({"id": g["id"], "start": g["start"], "end": g["end"], "speed": round(factor, 3), "duration": round(dur, 3),
                     "slack": round(win - dur, 3), "say": g["say"], "kokoro": [meta["voice"], meta["speed"]],
                     "wer": round(w, 3), "similarity": round(sim, 3),
                     "words": [{"word": x, "start": round(g["start"] + st, 3), "end": round(g["start"] + en, 3), "chunk": c}
                               for (x, st, en), c in zip(words, ck)]})
    print(f"{'id':<9}{'window':>13}{'dur':>7}{'kokoro':>8}{'WER':>6}{'sim':>7}{'tempo':>7}  take")
    for r in rows:
        print(f"{r[0]:<9}{r[1]:>13}{r[2]:7.2f}{r[3]:8.2f}{r[4]:6.2f}{r[5]:7.3f}{r[6]:7.3f}  {r[7]}")
    sf.write(os.path.join(BUILD, "voiceover.wav"), track, SR, subtype="FLOAT")
    json.dump({"voice": f"kokoro {segs[0]['kokoro'][0]} -> chatterbox-vc", "video": s["video"], "video_duration": vdur,
               "sample_rate": SR, "segments": segs}, open(os.path.join(BUILD, "timings.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote build/voiceover.wav ({vdur:.2f} s, peak {np.abs(track).max():.2f}) and build/timings.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["source", "convert", "place"]); ap.add_argument("script")
    ap.add_argument("--voice", default="am_michael"); ap.add_argument("--ref")
    ap.add_argument("--engine", choices=["kokoro", "chatterbox"], default="kokoro"); ap.add_argument("--takes", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4, help="convert: decoder seeds tried per segment")
    ap.add_argument("--only", nargs="+", metavar="ID", help="just these segments; convert keeps the others' previous takes")
    ap.add_argument("--seed", action="append", default=[], metavar="ID=N", help="source (chatterbox): read segment ID with seed N only")
    a = ap.parse_args()
    s = load(a.script)
    pins = {k: int(v) for k, v in (p.split("=", 1) for p in a.seed)}
    if a.step == "source":
        source_chatterbox(s, a.takes, a.only, pins) if a.engine == "chatterbox" else source(s, a.voice, a.only)
    elif a.step == "place":
        convert(s, None, 0, [])
    else:
        if not a.ref:
            sys.exit("convert needs --ref")
        convert(s, a.ref, a.seeds, a.only)

if __name__ == "__main__":
    main()
