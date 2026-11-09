"""The promo voiceover in a reference speaker's voice with standard American pronunciation: Kokoro reads the script,
Chatterbox's voice conversion then gives it the reference timbre. Pronunciation, stress and rhythm are Kokoro's;
voice conversion changes only who it sounds like, so the clone does not pick up the reference recording's accent
(clone.py, which generates from the reference directly, does).

    tts_venv/bin/python   vc.py source  script.json [--voice am_michael]        # Kokoro, CPU
    SIGMA_CLIMB4K=1 CUDA_VISIBLE_DEVICES=0 clone_venv/bin/python vc.py source script.json --engine chatterbox [--takes 6]
    SIGMA_CLIMB4K=1 CUDA_VISIBLE_DEVICES=0 clone_venv/bin/python vc.py source script.json --engine hybrid --ref /path/to/sample.wav
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
HY_EXAG, HY_CFG, HY_WPM, HY_MIN_TEMPO = 0.3, 0.3, 148, 0.9   # hybrid: no conversion follows, so slow a take by at most 10%
HY_OUT_WPM, HY_MAX_TEMPO, HY_KEEP_PACE = 165, 1.2, ()
OUT_WPM_ID = {"memory": 150, "memmid": 150, "knobs": 155, "knobshc": 155}   # v23 (11-09): the linear-attention scene was too fast at 165; v25: memmid is its middle   # pick plays the chosen take at about HY_OUT_WPM (v21, 11-08: the
# 140-148 wpm takes sounded slow), sped up with atempo (pitch kept) by at most HY_MAX_TEMPO; ids in HY_KEEP_PACE keep their pace
VC_DIR = os.path.join(BUILD, "vc")
TAKES_HY = os.path.join(BUILD, "takes_hy")

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

def source_chatterbox(s, takes, only=None, pins=None, ref=None, first=0):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, soundfile as sf, torch
    from faster_whisper import WhisperModel
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from chatterbox.tts import ChatterboxTTS
    from asr_check import tokens
    from clone import POST, ffilter, score, tempo, transcribe, trim
    m = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")   # built-in voice: m.conds
    voice = "chatterbox-default"
    if ref:   # --engine hybrid: T3 gets the reference speaker's embedding but the built-in voice's prompt tokens, S3Gen the reference
        from chatterbox.tts import Conditionals, T3Cond
        default = m.conds
        m.prepare_conditionals(ref, exaggeration=CB_EXAG)
        m.conds = Conditionals(T3Cond(speaker_emb=m.conds.t3.speaker_emb, cond_prompt_speech_tokens=default.t3.cond_prompt_speech_tokens,
                                      emotion_adv=HY_EXAG * torch.ones(1, 1, 1)).to(device=m.device), m.conds.gen)
        voice = "hybrid"
    ex, cfg, target = (HY_EXAG, HY_CFG, HY_WPM) if ref else (CB_EXAG, CB_CFG, TARGET_WPM)
    if ref:
        import librosa
        ref16 = librosa.load(ref, sr=16000)[0]
        ref_e = m.ve.embeds_from_wavs([ref16], sample_rate=16000).mean(0)
        def sim(c):
            e = m.ve.embeds_from_wavs([librosa.resample(c, orig_sr=SR, target_sr=16000)], sample_rate=16000).mean(0)
            return float(np.dot(e, ref_e) / (np.linalg.norm(e) * np.linalg.norm(ref_e)))
    norm, asr = EnglishTextNormalizer({}), WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 1))
    os.makedirs(VC_DIR, exist_ok=True)
    listing = {}
    print(f"{'id':<9}{'window':>13}{'seed':>5}{'dur':>7}{'wpm':>6}{'WER':>6}  takes (s)")
    for g in picked(s, only):
        win, nwords, cand = g["end"] - g["start"] - MARGIN, len(plain(g["say"]).split()), []
        for seed in [pins[g["id"]]] if g["id"] in (pins or {}) else range(first, first + takes):   # v22: --first-seed splits takes across processes
            torch.manual_seed(seed)
            clip = m.generate(plain(g["say"]), exaggeration=ex, cfg_weight=cfg).squeeze(0).numpy()
            clip = trim(ffilter(clip, POST) if ref else clip)     # hybrid takes are final: clean them as convert would
            cand.append((seed, clip, score(norm, g["say"], transcribe(asr, clip))))
            if ref:   # every take is kept for `pick` (accent and speaker checks with voice_sim.py)
                os.makedirs(TAKES_HY, exist_ok=True)
                sf.write(os.path.join(TAKES_HY, f"{g['id']}_s{seed}.wav"), clip.astype(np.float32), SR, subtype="FLOAT")
                listing[f"{g['id']}/{g['id']}_s{seed}"] = os.path.join(TAKES_HY, f"{g['id']}_s{seed}.wav")
        best = min(c[2] for c in cand); slack = max(0.05, 1 / max(1, len(tokens(norm, plain(g["say"]))))) + 1e-9
        ok = [c for c in cand if c[2] <= best + slack and len(c[1]) / SR <= win] or [min(cand, key=lambda c: len(c[1]))]
        pace = lambda c: nwords / (len(c[1]) / SR) * 60
        if ref:   # the most similar of the takes within 8 wpm of the slowest that is >= target (or of the nearest)
            near = min(abs(pace(c) - target) for c in ok)
            ok = [c for c in ok if abs(pace(c) - target) <= near + 8]
            sims = {c[0]: sim(c[1]) for c in ok}
            seed, clip, w = max(ok, key=lambda c: sims[c[0]])
        else:
            seed, clip, w = min(ok, key=lambda c: abs(pace(c) - target))
        raw_wpm = nwords / (len(clip) / SR) * 60
        if raw_wpm > target:
            clip = tempo(clip, max(target / raw_wpm, (len(clip) / SR) / win, HY_MIN_TEMPO if ref else 0))   # slow down, inside the window
        sf.write(os.path.join(VC_DIR, f"{g['id']}.wav"), clip.astype(np.float32), SR, subtype="FLOAT")
        json.dump({"say": g["say"], "voice": voice, "speed": 1.0, "seed": seed}, open(os.path.join(VC_DIR, f"{g['id']}.json"), "w"))
        print(f"{g['id']:<9}{g['start']:6.2f}-{g['end']:<6.2f}{seed:5d}{len(clip) / SR:7.2f}{nwords / (len(clip) / SR) * 60:6.0f}{w:6.2f}  "
              f"{[round(len(c[1]) / SR, 2) for c in cand]}")
    if listing:
        old = json.load(open(os.path.join(TAKES_HY, "list.json"))) if os.path.exists(os.path.join(TAKES_HY, "list.json")) else {}
        old.update(listing)
        json.dump(old, open(os.path.join(TAKES_HY, "list.json"), "w"), indent=1)
        print(f"kept {len(listing)} takes in {TAKES_HY} (list.json for voice_sim.py, then vc.py pick)")

ACCENT_MIN = 0.99   # pick: a take counts as a standard American reading when the accent classifier gives P(us) at least this

PAUSE_CAP = 0.55   # pick (v21): a pause inside a take longer than this is shortened to it (the 0.8 s sentence gaps dragged)

def cap_pauses(clip, cap=PAUSE_CAP, top_db=35):
    """The clip with every internal pause longer than cap seconds shortened to cap: the non-silent intervals are librosa's
    (top_db below the peak); the middle of a longer gap is left out, with a 10 ms crossfade where its two halves meet."""
    import numpy as np, librosa
    iv = librosa.effects.split(clip, top_db=top_db, frame_length=1024, hop_length=256)
    if len(iv) < 2:
        return clip
    half, fade = int(cap / 2 * SR), int(0.01 * SR)
    x = np.linspace(0, 1, fade, dtype=np.float32)
    out = [clip[:iv[0][1]]]
    for (a0, a1), (b0, b1) in zip(iv[:-1], iv[1:]):
        if b0 - a1 > 2 * half:
            left, right = clip[a1:a1 + half], clip[b0 - half:b1]
            out += [left[:-fade], left[-fade:] * (1 - x) + right[:fade] * x, right[fade:]]
        else:
            out.append(clip[a1:b1])
    out.append(clip[iv[-1][1]:])
    return np.concatenate(out).astype(np.float32)

def pick(s, report, only=None, exact=False, near_out=False, acc_sim=None):
    """Each segment's take from build/takes_hy (source --engine hybrid) by a voice_sim.py report of those takes: a word error
    rate within one word (or 0.05) of the segment's best take, a standard American accent (P(us) >= ACCENT_MIN), a fit in the
    window once slowed toward HY_WPM (by at most HY_MIN_TEMPO), a pace within 8 wpm of the take nearest HY_WPM; of those,
    the one the two independent speaker-verification models (WavLM, ECAPA) find most like the reference. Writes it, re-paced
    to about HY_OUT_WPM (at most HY_MAX_TEMPO faster; ids in HY_KEEP_PACE keep theirs), as build/vc/<id>.wav for convert --no-vc."""
    import numpy as np, soundfile as sf
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from asr_check import tokens
    from clone import tempo
    R, norm = json.load(open(report)), EnglishTextNormalizer({})
    A = json.load(open(acc_sim)) if acc_sim else {}   # v23: accent_sim.py closeness to the accepted clips ("cen")
    print(f"{'id':<9}{'seed':>5}{'WER':>6}{'P(us)':>7}{'wavlm':>7}{'ecapa':>7}{'wpm':>6}{'tempo':>7}  of (seed: WER, P(us))")
    for g in picked(s, only):
        win, nwords, takes = g["end"] - g["start"] - MARGIN, len(plain(g["say"]).split()), []
        for k, v in R.items():
            if k.split("/")[0] != g["id"] or "wer" not in v:
                continue
            seed = int(k.rsplit("_s", 1)[1])
            clip = sf.read(os.path.join(TAKES_HY, f"{g['id']}_s{seed}.wav"), dtype="float32")[0]
            takes.append({"seed": seed, "clip": clip, "wer": v["wer"], "p_us": v.get("p_us", 0.0), "wavlm": v["wavlm"], "ecapa": v["ecapa"],
                          "wpm": nwords / (len(clip) / SR) * 60, "acc": A.get(k, {}).get("cen", 0.0)})
        if not takes:
            sys.exit(f"segment {g['id']!r}: no scored takes in {report}")
        factor = lambda t: max(HY_MIN_TEMPO, min(1.0, HY_WPM / t["wpm"]))       # selection, as before: slowed toward HY_WPM
        tw = OUT_WPM_ID.get(g["id"], HY_OUT_WPM)
        out = lambda t: 1.0 if g["id"] in HY_KEEP_PACE else max(HY_MIN_TEMPO, min(HY_MAX_TEMPO, tw / t["wpm"]))   # playback pace
        slack = 1e-9 if exact else max(0.05, 1 / max(1, len(tokens(norm, plain(g["say"]))))) + 1e-9   # --exact: best-WER takes only
        target, fit = (tw, out) if near_out else (HY_WPM, factor)   # --near-out: the pace nearest the playback target
        best = min(t["wer"] for t in takes)
        ok = [t for t in takes if t["wer"] <= best + slack and t["p_us"] >= ACCENT_MIN and len(t["clip"]) / SR / fit(t) <= win]
        if not ok:
            sys.exit(f"segment {g['id']!r}: no take passes (WER, accent, window): "
                     + ", ".join(f"s{t['seed']} {t['wer']:.2f} {t['p_us']:.2f}" for t in takes) + "; read more takes")
        near = min(abs(t["wpm"] - target) for t in ok)
        band = 15 if A else 8   # v23: with --accent-sim, accent closeness to the accepted clips decides within a wider pace band
        t = max((t for t in ok if abs(t["wpm"] - target) <= near + band), key=(lambda t: (t["acc"], t["wavlm"] + t["ecapa"])) if A else (lambda t: t["wavlm"] + t["ecapa"]))
        f = out(t)
        clip = tempo(t["clip"], f) if abs(f - 1.0) > 1e-3 else t["clip"]
        clip = cap_pauses(clip)
        sf.write(os.path.join(VC_DIR, f"{g['id']}.wav"), clip.astype(np.float32), SR, subtype="FLOAT")
        json.dump({"say": g["say"], "voice": "hybrid", "speed": round(1 / f, 3), "tempo": round(f, 3), "seed": t["seed"], "p_us": round(t["p_us"], 3),
                   "wavlm": round(t["wavlm"], 3), "ecapa": round(t["ecapa"], 3)}, open(os.path.join(VC_DIR, f"{g['id']}.json"), "w"))
        print(f"{g['id']:<9}{t['seed']:5d}{t['wer']:6.2f}{t['p_us']:7.3f}{t['wavlm']:7.3f}{t['ecapa']:7.3f}{t['wpm']:6.0f}{f:7.3f}  "
              + " ".join(f"{x['seed']}:{x['wer']:.2f},{x['p_us']:.2f}" for x in sorted(takes, key=lambda x: x["seed"])))

def convert(s, ref, seeds=4, only=None, vc_off=False):
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
        vc = None
        if not vc_off:
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
        if vc_off:   # --no-vc: the reading is already in the reference voice (source --engine hybrid); place it as it is
            c = sf.read(src, dtype="float32")[0]
            tries.append((score(norm, g["say"], transcribe(asr, c)), 0.0, 0, c))
        for seed in range(0 if vc_off else seeds):
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
    label = (f"{segs[0]['kokoro'][0]} (no conversion)" if vc_off else
             json.load(open(os.path.join(BUILD, "timings.json")))["voice"] if ref is None else f"kokoro {segs[0]['kokoro'][0]} -> chatterbox-vc")
    json.dump({"voice": label,
               "video": s["video"], "video_duration": vdur,
               "sample_rate": SR, "segments": segs}, open(os.path.join(BUILD, "timings.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote build/voiceover.wav ({vdur:.2f} s, peak {np.abs(track).max():.2f}) and build/timings.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["source", "pick", "convert", "place"]); ap.add_argument("script")
    ap.add_argument("--report", help="pick: voice_sim.py report.json of build/takes_hy")
    ap.add_argument("--exact", action="store_true", help="pick: only takes at the segment's best word error rate")
    ap.add_argument("--near-out", action="store_true", help="pick: prefer the pace nearest HY_OUT_WPM (least re-pacing)")
    ap.add_argument("--accent-sim", help="pick (v23): accent_sim.py output; the take closest in accent to the accepted clips wins")
    ap.add_argument("--voice", default="am_michael"); ap.add_argument("--ref")
    ap.add_argument("--engine", choices=["kokoro", "chatterbox", "hybrid"], default="kokoro"); ap.add_argument("--takes", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4, help="convert: decoder seeds tried per segment")
    ap.add_argument("--first-seed", type=int, default=0, help="source (chatterbox/hybrid): first seed of the --takes range (v22)")
    ap.add_argument("--no-vc", action="store_true", help="convert: place the readings as they are (they are in the reference voice)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="just these segments; convert keeps the others' previous takes")
    ap.add_argument("--seed", action="append", default=[], metavar="ID=N", help="source (chatterbox): read segment ID with seed N only")
    a = ap.parse_args()
    s = load(a.script)
    pins = {k: int(v) for k, v in (p.split("=", 1) for p in a.seed)}
    if a.step == "source":
        if a.engine == "hybrid" and not a.ref:
            sys.exit("--engine hybrid needs --ref")
        if a.engine == "kokoro":
            source(s, a.voice, a.only)
        else:
            source_chatterbox(s, a.takes, a.only, pins, a.ref if a.engine == "hybrid" else None, first=a.first_seed)
    elif a.step == "pick":
        if not a.report:
            sys.exit("pick needs --report")
        pick(s, a.report, a.only, exact=a.exact, near_out=a.near_out, acc_sim=a.accent_sim)
    elif a.step == "place":
        convert(s, None, 0, [])
    else:
        if not a.ref:
            sys.exit("convert needs --ref")
        convert(s, a.ref, a.seeds, a.only, a.no_vc)

if __name__ == "__main__":
    main()
