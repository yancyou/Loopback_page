"""How much does a narration sound like the reference speaker? Two speaker-verification models that the voice conversion
never sees (vc.py picks takes with Chatterbox's own VoiceEncoder, which its conditioning is built around, so its score
is not an independent check): WavLM-base-plus-sv (x-vector head, transformers) and SpeechBrain's ECAPA-TDNN on VoxCeleb.
Each clip's embedding is compared by cosine with the whole reference recording's. Anchors from the same two models:
  ceiling: each ~10 s window of the reference against the rest of the reference (a genuine clip of the same speaker);
  floor:   the source readings before conversion (another speaker saying the same words).
F0 median (Hz) and range (5-95th percentile, semitones) come from librosa's pYIN.

    clone_venv/bin/python  voice_sim.py embed wavlm  LIST.json OUT_DIR      # LIST.json: {"label": "/path/clip.wav", ...}
    accent_venv/bin/python voice_sim.py embed ecapa  LIST.json OUT_DIR
    accent_venv/bin/python voice_sim.py accent LIST.json OUT_DIR            # CommonAccent ECAPA: P(us) = softmax(30 cos)
    clone_venv/bin/python  voice_sim.py asr    LIST.json OUT_DIR script.json  # faster-whisper small.en: WER, words/min
    clone_venv/bin/python  voice_sim.py report LIST.json OUT_DIR --ref LABEL

Labels of the form "<group>/<name>" are averaged per group in the report; for asr, <name> is a segment id of script.json,
optionally followed by _s<seed>. The ref label must be in LIST.json; the "ref_w*" labels (reference windows) are
scored leave-one-out against the other windows' concatenation.
"""
import json, os, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

SR = 16000

def load16(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        w = resample_poly(w, SR // g, sr // g).astype(np.float32)
    return w

def embedder(kind):
    import torch
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    if kind == "wavlm":
        from transformers import AutoFeatureExtractor, WavLMForXVector
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        def f(w):
            x = fe(w, sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                return m(**x).embeddings[0].numpy()
        return f
    from speechbrain.inference.speaker import EncoderClassifier
    m = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=os.path.expanduser("~/.cache/speechbrain/spkrec-ecapa-voxceleb"))
    def f(w):
        with torch.no_grad():
            return m.encode_batch(torch.from_numpy(w)[None]).squeeze().numpy()
    return f

def embed(kind, lst, out):
    clips = json.load(open(lst))
    f = embedder(kind)
    E = {k: f(load16(p)) for k, p in clips.items()}
    w = [load16(p) for k, p in clips.items() if k.startswith("ref_w")]
    for i, k in enumerate(k for k in clips if k.startswith("ref_w")):   # leave-one-out reference for each window
        E["loo:" + k] = f(np.concatenate([x for j, x in enumerate(w) if j != i]))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"{kind}.npz"), **{k.replace("/", "|"): v for k, v in E.items()})
    print(f"{kind}: {len(E)} embeddings -> {out}/{kind}.npz")

def accent(lst, out):
    """P(us) and the most likely other accent: the classifier's outputs are cosine scores of an AAM-softmax head, so the
    probabilities are softmax(30 cos)."""
    import torch
    torch.set_num_threads(min(16, os.cpu_count() or 1))      # the default (every core) crawls when the machine is busy
    from speechbrain.inference.classifiers import EncoderClassifier
    m = EncoderClassifier.from_hparams("Jzuluaga/accent-id-commonaccent_ecapa",
                                       savedir=os.path.expanduser("~/.cache/speechbrain/accent-id-commonaccent_ecapa"))
    labs = [m.hparams.label_encoder.decode_ndim(i) for i in range(len(m.hparams.label_encoder))]
    R = {}
    for k, p in json.load(open(lst)).items():
        with torch.no_grad():
            cos = m.classify_batch(torch.from_numpy(load16(p))[None])[0][0]
        pr = torch.softmax(30 * cos, -1).numpy()
        o = sorted((x for x in range(len(labs)) if labs[x] != "us"), key=lambda x: -pr[x])[0]
        R[k] = {"p_us": float(pr[labs.index("us")]), "other": labs[o], "p_other": float(pr[o])}
    os.makedirs(out, exist_ok=True)
    json.dump(R, open(os.path.join(out, "accent.json"), "w"), indent=1)
    print(f"accent: {len(R)} clips -> {out}/accent.json")

def asr(lst, out, script):
    import re
    from faster_whisper import WhisperModel
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from clone import score, transcribe
    say = {g["id"]: g["say"] for g in json.load(open(script))["segments"]}
    norm = EnglishTextNormalizer({})
    m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 1))
    R = {}
    for k, p in json.load(open(lst)).items():
        seg = re.sub(r"_s\d+$", "", k.split("/")[-1])
        if "/" not in k or seg not in say:
            continue
        w, sr = sf.read(p, dtype="float32", always_2d=True)
        w = w.mean(1)
        if sr != 24000:
            g = np.gcd(sr, 24000); w = resample_poly(w, 24000 // g, sr // g).astype(np.float32)
        words = transcribe(m, w)
        R[k] = {"wer": float(score(norm, say[seg], words)), "wpm": len(say[seg].split()) / (len(w) / 24000) * 60,
                "heard": " ".join(x for x, _, _ in words)}
    json.dump(R, open(os.path.join(out, "asr.json"), "w"), indent=1)
    print(f"asr: {len(R)} clips -> {out}/asr.json")

def f0(w):
    import librosa
    f, v, _ = librosa.pyin(w, fmin=60, fmax=320, sr=SR, frame_length=1024)
    f = f[v & np.isfinite(f)]
    if len(f) < 20:
        return float("nan"), float("nan")
    lo, hi = np.percentile(f, [5, 95])
    return float(np.median(f)), float(12 * np.log2(hi / lo))

def report(lst, out, ref):
    clips = json.load(open(lst))
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    S = {}
    for kind in ("wavlm", "ecapa"):
        p = os.path.join(out, f"{kind}.npz")
        if not os.path.exists(p):
            continue
        E = {k.replace("|", "/"): v for k, v in np.load(p).items()}
        for k in clips:
            r = E["loo:" + k] if k.startswith("ref_w") else E[ref]
            S.setdefault(k, {})[kind] = cos(E[k], r)
    for k, p in clips.items():
        S.setdefault(k, {})["f0"], S[k]["range"] = f0(load16(p))
    for name in ("accent", "asr"):
        p = os.path.join(out, f"{name}.json")
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                if k in S:
                    S[k].update(v)
    groups = {}
    for k in clips:
        groups.setdefault(k.split("/")[0] if "/" in k else k, []).append(k)
    kinds = [x for x in ("wavlm", "ecapa", "p_us", "wer", "wpm") if any(x in S[k] for k in S)]
    print(f"{'group':<26}{'n':>3}" + "".join(f"{x:>8}" for x in kinds) + f"{'F0 Hz':>8}{'range st':>10}  min p_us")
    for g, ks in groups.items():
        row = [np.mean([S[k][x] for k in ks if x in S[k]]) if any(x in S[k] for k in ks) else float("nan") for x in kinds]
        lo = min((S[k]["p_us"] for k in ks if "p_us" in S[k]), default=float("nan"))
        print(f"{g:<26}{len(ks):>3}" + "".join(f"{v:8.3f}" for v in row)
              + f"{np.nanmedian([S[k]['f0'] for k in ks]):8.1f}{np.nanmean([S[k]['range'] for k in ks]):10.1f}  {lo:.3f}")
    json.dump(S, open(os.path.join(out, "report.json"), "w"), indent=1)

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "embed":
        embed(a[1], a[2], a[3])
    elif a[0] == "accent":
        accent(a[1], a[2])
    elif a[0] == "asr":
        asr(a[1], a[2], a[3])
    elif a[0] == "report":
        report(a[1], a[2], a[a.index("--ref") + 1])
    else:
        sys.exit(__doc__)
