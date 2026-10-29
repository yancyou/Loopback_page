"""Generate the background-music candidates for the Loopback promo with ACE-Step 1.5, locally, and measure them.

    CUDA_VISIBLE_DEVICES=1 SIGMA_CLIMB4K=1 /mnt/localssd/loopresidual/tools/music_venv/bin/python gen_music.py
        # build/cand{1,2,3}.wav (110 s, 48 kHz stereo, 24-bit), build/contact.wav, build/metrics.json
    /mnt/localssd/loopresidual/tools/music_venv/bin/python gen_music.py --measure [WAV ...]   # metrics only, CPU

ACE-Step 1.5 (github.com/ace-step/ACE-Step-1.5 @ ca1e85f, MIT code and weights) lives in VENV (checkout in src/, weights
in checkpoints/). Its 4B LM plans the 110 s from the caption, the section tags in LYRICS and 116 bpm 4/4; the XL turbo
DiT renders that in 8 steps (guidance-distilled, so guidance 1.0); the output is peak-normalized to PEAK_DB. The seeds
were picked from 72 by the metrics below. The LM samples from the global CUDA RNG (under nano-vllm it ignores its seed),
so each candidate first seeds torch and clears nano-vllm's prefix cache; it is then bit-identical across runs and orders.
Existing candidates are kept unless --force.

Metrics: loudness, LRA and true peak (ffmpeg ebur128); tempo (librosa beat_track, whose grid has 117.5 but not 116, and
60 / mean interval of its beats over 6-85 s); words heard by faster-whisper small.en with vad_filter=True in the TTS venv,
fed 16 kHz mono float32 decoded here by ffmpeg (its decode_audio breaks on new PyAV): any word flags the candidate;
structure from the ebur128 loudness against the groove's median (intro 0-6 s, groove-in time, deepest dip and highest
peak up to 85 s, fade-out time); spectral centroid and 1-4 kHz power share over 6-85 s on 16 kHz mono (the reference
video, voice included, gives 1.69 kHz and 13%). contact.wav is EXCERPT of each candidate, back to back, 1 s apart.
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
VENV = "/mnt/localssd/loopresidual/tools/music_venv"
SRC = os.path.join(VENV, "src", "ACE-Step-1.5")
TTS_PY = "/mnt/localssd/loopresidual/tools/tts_venv/bin/python"   # faster-whisper 1.2.1
DIT, LM = "acestep-v15-xl-turbo", "acestep-5Hz-lm-4B"
STEPS, CFG, SHIFT, PEAK_DB = 8, 1.0, 3.0, -1.5
BPM, DUR = 116, 110.0
BASE = ("minimal modern electronic, instrumental, 116 bpm, steady soft kick and pulse, warm sub bass, airy synth pads, "
        "subtle arpeggio, uplifting, confident, cinematic tech launch, clean mix, no vocals")
LYRICS = "[Intro - soft ambient pads]\n\n[Instrumental]\n\n[Outro - resolving]"
CANDS = [("cand1", 101, BASE),
         ("cand2", 126, BASE + ", steady groove throughout with gentle lifts"),
         ("cand3", 143, BASE)]
INTRO, GROOVE, EXCERPT = (0.0, 6.0), (6.0, 85.0), (40.0, 60.0)

def generate(force):
    todo = [c for c in CANDS if force or not os.path.exists(os.path.join(BUILD, c[0] + ".wav"))]
    if not todo:
        return
    os.environ.setdefault("ACESTEP_PROJECT_ROOT", SRC)
    os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", os.path.join(VENV, "checkpoints"))
    import torch, soundfile as sf
    from loguru import logger
    logger.remove(); logger.add(sys.stderr, level="WARNING")
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    dit, lm = AceStepHandler(), LLMHandler()
    msg, ok = dit.initialize_service(project_root=SRC, config_path=DIT, device="cuda", use_flash_attention=True)
    if not ok:
        sys.exit(f"DiT init failed: {msg}")
    msg, ok = lm.initialize(checkpoint_dir=os.environ["ACESTEP_CHECKPOINTS_DIR"], lm_model_path=LM, backend="vllm", device="cuda")
    if not ok or lm.llm_backend != "vllm":
        sys.exit(f"LM init failed (or not on nano-vllm): {msg}")
    for name, seed, caption in todo:
        torch.manual_seed(seed)
        lm.llm.reset()
        p = GenerationParams(caption=caption, lyrics=LYRICS, instrumental=True, bpm=BPM, timesignature="4", duration=DUR,
                             inference_steps=STEPS, guidance_scale=CFG, shift=SHIFT, seed=seed, normalization_db=PEAK_DB,
                             thinking=True, use_cot_caption=False, use_cot_language=False)
        r = generate_music(dit, lm, p, GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed]), save_dir=None)
        if not r.success:
            sys.exit(f"{name}: {r.error}")
        a = r.audios[0]
        sf.write(os.path.join(BUILD, name + ".wav"), a["tensor"].numpy().T, a["sample_rate"], subtype="PCM_24")
        print(f"{name}: seed {seed}, {(r.extra_outputs.get('lm_metadata') or {}).get('keyscale', '?')}", flush=True)

def ebur128(path):
    err = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    frames = [(float(t), float(m), float(s)) for t, m, s in
              re.findall(r" t: ([\d.]+)\s+TARGET:\S+ LUFS\s+M:\s*(-?[\d.]+) S:\s*(-?[\d.]+)", err)]
    tail = err[err.rfind("Summary:"):]
    num = lambda pat: float(re.search(pat, tail).group(1))
    return num(r"I:\s+(-?[\d.]+) LUFS"), num(r"LRA:\s+([\d.]+) LU"), num(r"Peak:\s+(-?[\d.]+|-inf) dBFS"), frames

def decode(path, sr, ch):
    import numpy as np
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, ch)

def asr(paths):
    """Runs under TTS_PY: Whisper segments per file, as JSON on stdout."""
    import warnings
    warnings.filterwarnings("ignore")
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 1))
    out = {}
    for p in paths:
        segs, _ = model.transcribe(decode(p, 16000, 1)[:, 0], language="en", beam_size=5, vad_filter=True,
                                   condition_on_previous_text=False)
        out[p] = [(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segs]
    print(json.dumps(out))

def measure(paths):
    import numpy as np, librosa
    r = subprocess.run([TTS_PY, "-I", os.path.abspath(__file__), "--asr", *paths], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ASR failed:\n{r.stderr[-2000:]}")
    heard = json.loads(r.stdout.strip().splitlines()[-1])
    rows = {}
    for p in paths:
        lufs, lra, tp, frames = ebur128(p)
        t, M, S = (np.array(v) for v in zip(*frames))
        on = lambda lo, hi: (t >= lo) & (t <= hi)
        med = float(np.median(S[on(15, GROOVE[1])]))                                 # groove level
        Ms = np.array([np.median(M[max(0, i - 5):i + 5]) for i in range(len(M))])    # 400 ms loudness, 1 s running median
        start, end = float(t[np.argmax(Ms >= med - 3)]), float(t[np.nonzero(Ms >= med - 20)[0][-1]])
        sg = S[on(start + 3, GROOVE[1])]
        intro = 10 * np.log10(np.mean(10 ** (M[on(0.4, INTRO[1])] / 10))) - med
        tempo, beats = librosa.beat.beat_track(y=decode(p, 22050, 1)[:, 0], sr=22050, units="time")
        tempo, bt = float(np.atleast_1d(tempo)[0]), beats[(beats >= GROOVE[0]) & (beats <= GROOVE[1])]
        ibi = 60 / float(np.mean(np.diff(bt))) if len(bt) > 2 else float("nan")
        g = decode(p, 16000, 1)[int(GROOVE[0] * 16000):int(GROOVE[1] * 16000), 0]
        cent = float(np.mean(librosa.feature.spectral_centroid(y=g, sr=16000)))
        P = np.abs(librosa.stft(g, n_fft=4096)) ** 2
        f = librosa.fft_frequencies(sr=16000, n_fft=4096)
        voice = float(P[(f >= 1000) & (f <= 4000)].sum() / P.sum())
        segs = heard[p]
        words = [w for s in segs for w in re.findall(r"[A-Za-z']+", re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s[2]))]
        rows[os.path.basename(p)] = dict(
            lufs=lufs, true_peak_dbtp=tp, lra_lu=lra, tempo_bpm=round(tempo, 1), groove_beat_bpm=round(ibi, 1),
            intro_vs_groove_lu=round(float(intro), 1), groove_in_s=round(start, 1), groove_dip_lu=round(float(sg.min()) - med, 1),
            groove_peak_lu=round(float(sg.max()) - med, 1), music_end_s=round(end, 1), centroid_hz=round(cent),
            voice_band_pct=round(100 * voice, 1), words=len(words), whisper=segs)
    print(f"{'file':<14}{'LUFS':>6}{'TP':>6}{'LRA':>5}{'BPM':>7}{'beats':>7}{'intro':>6}{'in@':>6}{'dip':>6}{'peak':>5}{'end@':>7}"
          f"{'cent':>6}{'1-4k%':>6}  words")
    for k, v in rows.items():
        print(f"{k:<14}{v['lufs']:6.1f}{v['true_peak_dbtp']:6.1f}{v['lra_lu']:5.1f}{v['tempo_bpm']:7.1f}{v['groove_beat_bpm']:7.1f}"
              f"{v['intro_vs_groove_lu']:6.1f}{v['groove_in_s']:6.1f}{v['groove_dip_lu']:6.1f}{v['groove_peak_lu']:5.1f}{v['music_end_s']:7.1f}"
              f"{v['centroid_hz']:6d}{v['voice_band_pct']:6.1f}  "
              + (f"FLAG {v['words']}: " + " | ".join(f"{s[0]:.1f}s {s[2]}" for s in v["whisper"]) if v["words"] else "none"))
    return rows

def contact(paths, out):
    import numpy as np, soundfile as sf
    parts = []
    for p in paths:
        x, sr = sf.read(p, dtype="float32", start=int(EXCERPT[0] * 48000), stop=int(EXCERPT[1] * 48000))
        ramp = np.linspace(0, 1, int(0.05 * sr))[:, None]
        x[:len(ramp)] *= ramp; x[-len(ramp):] *= ramp[::-1]
        parts += [x, np.zeros((sr, x.shape[1]), np.float32)]
    sf.write(out, np.concatenate(parts[:-1]), sr, subtype="PCM_24")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", nargs="*", help="only measure these WAVs (none given: the candidates)")
    ap.add_argument("--asr", nargs="+", help=argparse.SUPPRESS)
    ap.add_argument("--force", action="store_true", help="regenerate existing candidates")
    a = ap.parse_args()
    if a.asr:
        return asr(a.asr)
    os.makedirs(BUILD, exist_ok=True)
    if a.measure:
        return measure(a.measure)
    if a.measure is None:
        generate(a.force)
    paths = [os.path.join(BUILD, c[0] + ".wav") for c in CANDS]
    rows = measure(paths)
    contact(paths, os.path.join(BUILD, "contact.wav"))
    settings = dict(model=f"ACE-Step 1.5 @ ca1e85f: LM {LM}, DiT {DIT}", steps=STEPS, guidance=CFG, shift=SHIFT,
                    peak_db=PEAK_DB, bpm=BPM, timesignature="4/4", duration_s=DUR, lyrics=LYRICS,
                    candidates={n: dict(seed=s, caption=c) for n, s, c in CANDS}, contact_excerpt_s=EXCERPT)
    with open(os.path.join(BUILD, "metrics.json"), "w") as f:
        json.dump(dict(settings=settings, metrics=rows), f, indent=1)
    print("wrote", BUILD)

if __name__ == "__main__":
    main()
