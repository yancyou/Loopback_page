"""Synthesize the promo voiceover in a reference speaker's own voice with Chatterbox (Resemble AI, MIT) on one GPU.

    SIGMA_CLIMB4K=1 CUDA_VISIBLE_DEVICES=0 python clone.py script.json --ref /path/to/sample.wav [--takes 8] [--plan]

A drop-in alternative to synth.py: it writes the same build/voiceover.wav and build/timings.json, so make_av.py and
asr_check.py run unchanged. Chatterbox takes the voice from the reference recording (its first 10 s for the decoder, all
of it for the speaker embedding); keep the recording outside the repository, nothing here copies it in. A reference
cleaned of room noise first gives cleaner takes: the clone copies the recording's noise floor.
For each segment, --takes seeds are sampled and cached in build/takes/ under a hash of the say text, the reference and
the settings, so a rerun after a timing change reuses them. Each take is cleaned with ffmpeg (POST: a high-pass and
afftdn), trimmed to LEAD/TAIL around its speech and transcribed with faster-whisper small.en. Among the takes within
WER_SLACK of the lowest word error rate against `say` and within SIM_SLACK of the highest speaker similarity to the
reference, the one whose pace is nearest TARGET_WPM wins. A winner longer than its window is sped up with ffmpeg's atempo
(pitch kept) by at most MAX_TEMPO; if a segment still does not fit, nothing is written. --plan prints the chosen takes
and their lengths without fitting or writing, to set the scene timing from. Caption word times come from the same transcription,
aligned to `say` as synth.py aligns Kokoro's tokens. Chatterbox marks its output with Resemble's inaudible Perth watermark.
Needs chatterbox-tts 0.1.7, faster-whisper and setuptools<81 (perth imports pkg_resources).
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile
from synth import BUILD, LEAD, SR, TAIL, align, duration, load, plain, repo
from asr_check import rejoin, tokens, wer

MAX_TEMPO = 1.12
WER_SLACK = 0.05                         # takes this close to the best WER (or one word off) all count as correct
SIM_SLACK = 0.02                         # ... and this close to the most similar take all sound like the speaker
TARGET_WPM = 140
EXAG, CFG, TEMP = 0.3, 0.3, 0.8          # calmer and slower than Chatterbox's 0.5 / 0.5 (175 -> 147 wpm on this voice)
POST = "highpass=f=70,afftdn=nr=12:nf=-55:tn=1"
GAP_DB = 42                              # speech: 10 ms frames within this many dB of the loudest frame

def trim(audio):
    import numpy as np
    f = SR // 100
    n = len(audio) // f
    rms = np.sqrt(np.mean(audio[:n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    on = np.nonzero(20 * np.log10(rms / rms.max()) > -GAP_DB)[0]
    a, b = max(0, on[0] * f - round(LEAD * SR)), min(len(audio), (on[-1] + 1) * f + round(TAIL * SR))
    return np.ascontiguousarray(audio[a:b], dtype=np.float32)

def ffilter(audio, chain):
    """The clip through an ffmpeg audio filter chain."""
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        sf.write(i, audio, SR, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", i, "-af", chain, "-ar", str(SR), "-c:a", "pcm_f32le", o],
                       check=True)
        return sf.read(o, dtype="float32")[0]

def tempo(audio, factor):
    """The clip sped up by factor with ffmpeg's atempo, which keeps the pitch."""
    return ffilter(audio, f"atempo={factor:.5f}")

def transcribe(asr, audio):
    """Words of a 24 kHz clip as (text, start, end) in seconds from its start."""
    import librosa
    segs, _ = asr.transcribe(librosa.resample(audio, orig_sr=SR, target_sr=16000), language="en", beam_size=5,
                             word_timestamps=True, condition_on_previous_text=False)
    return [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]

def score(norm, say, words):
    ref = tokens(norm, plain(say))
    return wer(ref, rejoin(tokens(norm, " ".join(w for w, _, _ in words)), ref))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--ref", required=True, help="the speaker's recording, outside the repository")
    ap.add_argument("--takes", type=int, default=8)
    ap.add_argument("--plan", action="store_true", help="print the chosen takes' lengths; fit and write nothing")
    a = ap.parse_args()
    s = load(a.script)
    vdur = s.get("duration") or duration(repo(s["video"]))   # "duration" lets the voice come before the render that uses its timing
    if s["segments"] and s["segments"][-1]["end"] > vdur:
        sys.exit(f"segment {s['segments'][-1]['id']!r} ends at {s['segments'][-1]['end']} after the video ({vdur:.2f} s)")
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, soundfile as sf, torch, librosa
    from faster_whisper import WhisperModel
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from chatterbox.tts import ChatterboxTTS
    norm = EnglishTextNormalizer({})
    asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 1))
    ref_id = hashlib.sha256(open(a.ref, "rb").read()).hexdigest()[:12]
    cache = os.path.join(BUILD, "takes"); os.makedirs(cache, exist_ok=True)
    model, ref_emb = None, None
    def tts():
        nonlocal model, ref_emb
        if model is None:
            model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
            model.prepare_conditionals(a.ref, exaggeration=EXAG)
            wav16 = librosa.load(a.ref, sr=16000)[0]
            ref_emb = model.ve.embeds_from_wavs([wav16], sample_rate=16000).mean(0)
        return model
    def similarity(clip):
        tts()
        e = model.ve.embeds_from_wavs([librosa.resample(clip, orig_sr=SR, target_sr=16000)], sample_rate=16000).mean(0)
        return float(np.dot(e, ref_emb) / (np.linalg.norm(e) * np.linalg.norm(ref_emb)))
    track = np.zeros(round(vdur * SR), np.float32)
    rows, segs, over = [], [], []
    for g in s["segments"]:
        key = hashlib.sha256(json.dumps([plain(g["say"]), ref_id, EXAG, CFG, TEMP, POST]).encode()).hexdigest()[:12]
        takes = []
        for seed in range(a.takes):
            wav_p, meta_p = (os.path.join(cache, f"{g['id']}_{key}_s{seed}.{x}") for x in ("wav", "json"))
            if os.path.exists(meta_p):
                clip, meta = sf.read(wav_p, dtype="float32")[0], json.load(open(meta_p))
            else:
                torch.manual_seed(seed)
                raw = tts().generate(plain(g["say"]), exaggeration=EXAG, cfg_weight=CFG, temperature=TEMP).squeeze(0).numpy()
                clip = trim(ffilter(raw, POST))
                words = transcribe(asr, clip)
                meta = {"seed": seed, "duration": len(clip) / SR, "wer": score(norm, g["say"], words),
                        "similarity": similarity(clip), "asr": " ".join(w for w, _, _ in words)}
                sf.write(wav_p, clip, SR, subtype="FLOAT"); json.dump(meta, open(meta_p, "w"), indent=1)
            takes.append((meta, clip))
        win = g["end"] - g["start"]
        need = lambda m: max(1.0, m["duration"] / (win - 0.02))
        nwords = len(plain(g["say"]).split())
        wpm = lambda m: nwords / m["duration"] * 60
        best = min(m["wer"] for m, _ in takes)
        slack = max(WER_SLACK, 1 / max(1, len(tokens(norm, plain(g["say"]))))) + 1e-9   # ASR quirks: "$374 million"
        ok = [t for t in takes if t[0]["wer"] <= best + slack]
        top = max(m["similarity"] for m, _ in ok)
        meta, clip = min((t for t in ok if t[0]["similarity"] >= top - SIM_SLACK), key=lambda t: abs(wpm(t[0]) - TARGET_WPM))
        dur, factor = meta["duration"], need(meta)
        rows.append((g["id"], f"{g['start']:.2f}-{g['end']:.2f}", dur, meta["seed"], meta["wer"], meta["similarity"],
                     factor, wpm(meta), [round(t[0]["duration"], 2) for t in takes]))
        if a.plan:
            continue
        if factor > MAX_TEMPO:
            over.append(f"segment {g['id']!r} needs a {dur / MAX_TEMPO + 0.02:.2f} s window (has {win:.2f} s)")
            continue
        if factor > 1.0:
            clip = tempo(clip, factor)
        words = transcribe(asr, clip)
        ck = align(g["say"], [w for w, _, _ in words])
        i = round(g["start"] * SR)
        track[i:i + len(clip)] += clip[:len(track) - i]
        segs.append({"id": g["id"], "start": g["start"], "end": g["end"], "speed": round(factor, 3),
                     "duration": round(len(clip) / SR, 3), "slack": round(win - len(clip) / SR, 3), "say": g["say"],
                     "seed": meta["seed"], "wer": round(meta["wer"], 3), "similarity": round(meta["similarity"], 3),
                     "words": [{"word": w, "start": round(g["start"] + st, 3), "end": round(g["start"] + en, 3), "chunk": c}
                               for (w, st, en), c in zip(words, ck)]})
    print(f"{'id':<9}{'window':>13}{'dur':>7}{'seed':>5}{'WER':>6}{'sim':>6}{'tempo':>7}{'wpm':>5}  takes (s)")
    for r in rows:
        print(f"{r[0]:<9}{r[1]:>13}{r[2]:7.2f}{r[3]:5d}{r[4]:6.2f}{r[5]:6.3f}{r[6]:7.3f}{r[7]:5.0f}  {r[8]}")
    if a.plan:
        return
    if over:
        sys.exit("\n".join(over))
    sf.write(os.path.join(BUILD, "voiceover.wav"), track, SR, subtype="FLOAT")
    json.dump({"voice": f"chatterbox:{ref_id}", "video": s["video"], "video_duration": vdur, "sample_rate": SR, "segments": segs},
              open(os.path.join(BUILD, "timings.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote build/voiceover.wav ({vdur:.2f} s, peak {np.abs(track).max():.2f}) and build/timings.json")

if __name__ == "__main__":
    main()
