"""Transcribe the muxed promo with faster-whisper small.en and score each script segment against its `say` text.

    python asr_check.py script.json     # after make_av.py

The audio of `out` is decoded with ffmpeg to 16 kHz mono (faster_whisper.decode_audio breaks on new PyAV) and transcribed
on CPU, int8, with the Silero VAD on (without it Whisper hallucinates words like "You" in the long silences). Prints the
transcript with times, then per segment the word error rate of the ASR words whose midpoint lies in its window (or
within TOL s of it, the nearest window winning: Whisper's word times drift) against `say`. Both sides go through
Whisper's English normalizer (numbers to digits, punctuation dropped), runs of single letters are joined ("M H C" =
"MHC") and split compounds are rejoined ("loop back" = "loopback"). Segments with WER > 0.15 are flagged; any flag
makes the exit status 1.
"""
import argparse, os, subprocess, sys
from synth import load, plain, repo

FLAG, TOL = 0.15, 0.5

def decode(path):
    import numpy as np
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-f", "f32le", "-ac", "1",
                        "-ar", "16000", "-"], capture_output=True)
    if r.returncode:
        sys.exit(f"ffmpeg could not decode the audio of {path}:\n{r.stderr.decode()[-1000:]}")
    return np.frombuffer(r.stdout, np.float32)

def tokens(norm, text):
    out = []
    for w in norm(text).split():
        if len(w) == 1 and w.isalpha() and out and len(out[-1][0]) == out[-1][1]:
            out[-1] = (out[-1][0] + w, out[-1][1] + 1)   # still a run of single letters
        else:
            out.append((w, 1 if len(w) == 1 and w.isalpha() else 0))
    return [w for w, _ in out]

def rejoin(a, b):
    """Merge adjacent words of a whose concatenation is a word of b."""
    vocab, out, i = set(b), [], 0
    while i < len(a):
        if i + 1 < len(a) and a[i] not in vocab and a[i] + a[i + 1] in vocab:
            out.append(a[i] + a[i + 1]); i += 2
        else:
            out.append(a[i]); i += 1
    return out

def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    a = ap.parse_args()
    s = load(a.script)
    out = repo(s["out"])
    if not os.path.exists(out):
        sys.exit(f"{s['out']} not found; run make_av.py first")
    audio = decode(out)
    import warnings
    warnings.filterwarnings("ignore")
    from faster_whisper import WhisperModel
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    norm = EnglishTextNormalizer({})
    model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 1))
    segs, _ = model.transcribe(audio, language="en", beam_size=5, word_timestamps=True, vad_filter=True,
                               condition_on_previous_text=False)
    words = []
    print(f"transcript of {s['out']}:")
    for sg in segs:
        print(f"  [{sg.start:6.2f} - {sg.end:6.2f}] {sg.text.strip()}")
        words += sg.words
    gap = lambda g, t: max(0.0, g["start"] - t, t - g["end"])
    mine = {g["id"]: [] for g in s["segments"]}
    stray = []
    for w in words:
        mid = (w.start + w.end) / 2
        g = min(s["segments"], key=lambda g: gap(g, mid))
        (mine[g["id"]] if gap(g, mid) <= TOL else stray).append(w)
    flagged = 0
    print(f"\n{'id':<14}{'window':>13}{'WER':>7}{'words':>7}")
    for g in s["segments"]:
        ref = tokens(norm, plain(g["say"]))
        hyp = tokens(norm, "".join(w.word for w in mine[g["id"]]))
        hyp = rejoin(hyp, ref); ref = rejoin(ref, hyp)
        e = wer(ref, hyp)
        flagged += e > FLAG
        print(f"{g['id']:<14}{g['start']:>7.2f}-{g['end']:<5.2f}{e:7.2f}{len(ref):7d}" + ("   <-- FLAG" if e > FLAG else ""))
        print(f"    say: {' '.join(ref)}\n    asr: {' '.join(hyp)}")
    if stray:
        print("outside every window:", " ".join(f"{w.word.strip()}@{w.start:.2f}" for w in stray))
    if flagged:
        sys.exit(f"{flagged} segment(s) with WER > {FLAG}")

if __name__ == "__main__":
    main()
