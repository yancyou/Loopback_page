"""v25 memory take (11-09, user: "这里停顿的语气很有问题"): keep the v24 take's first sentence ("Linear attention is a memory over
tokens ... later tokens read from it.") and last sentence ("Unrolled, every read ... at one."), and put a new reading of the
middle (segment memmid: "The residual stream does the same over depth. Each block writes ... read from it.") between them.
The joints are the silent gaps after the first sentence and before "Unrolled", found from the recognizer's word times and
librosa's non-silent intervals (top_db 35, as vc.cap_pauses); each joint keeps the v24 take's own pause there. The middle is
trimmed to its speech and scaled to the v24 take's speech loudness (RMS over non-silent frames, clamped to 0.7-1.4x).

    clone_venv/bin/python splice.py V24_MEMORY.wav MEMMID.wav OUT.wav
"""
import re, sys
import numpy as np, soundfile as sf, librosa
from faster_whisper import WhisperModel

def speech(y, sr):
    return librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)

def rms(y, iv):
    return float(np.sqrt(np.mean(np.concatenate([y[a:b] for a, b in iv]) ** 2)))

def main():
    a_path, m_path, out = sys.argv[1:4]
    y, sr = sf.read(a_path, dtype="float32"); mid, sr2 = sf.read(m_path, dtype="float32")
    assert sr == sr2, (sr, sr2)
    asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
    segs, _ = asr.transcribe(librosa.resample(y, orig_sr=sr, target_sr=16000), language="en", beam_size=5, word_timestamps=True,
                             condition_on_previous_text=False)
    W = [(re.sub(r"[^a-z]", "", w.word.lower()), w.start, w.end) for s in segs for w in s.words]
    i_it = next(i for i in range(2, len(W)) if W[i][0] == "it" and W[i - 1][0] == "from")       # "... read from it."
    i_un = next(i for i in range(len(W)) if W[i][0] == "unrolled")
    iv = speech(y, sr)
    def gap_after(t):    # the silent gap that follows the non-silent interval holding time t (s): (end of speech, start of next)
        k = max(j for j in range(len(iv)) if iv[j][0] <= t * sr + 0.05 * sr)
        return iv[k][1], iv[k + 1][0]
    e1, s2 = gap_after(W[i_it][2] - 0.05)
    e3, s4 = gap_after(W[i_un][1] - 0.15)
    p1, p2 = s2 - e1, s4 - e3
    A, C = y[:e1 + p1 // 2], y[s4 - p2 // 2:]
    mi = speech(mid, sr)
    M = mid[max(0, mi[0][0] - int(0.02 * sr)):min(len(mid), mi[-1][1] + int(0.04 * sr))]
    g = float(np.clip(rms(y, iv) / rms(mid, mi), 0.7, 1.4)); M = M * g
    f = int(0.01 * sr); ramp = np.linspace(0, 1, f, dtype=np.float32)
    M[:f] *= ramp; M[-f:] *= ramp[::-1]; A[-f:] *= ramp[::-1]; C[:f] *= ramp
    z1, z2 = np.zeros(max(0, p1 - p1 // 2 - int(0.02 * sr)), np.float32), np.zeros(max(0, p2 - p2 // 2 - int(0.04 * sr)), np.float32)
    o = np.concatenate([A, z1, M, z2, C]).astype(np.float32)
    sf.write(out, o, sr, subtype="FLOAT")
    print(f"first sentence ends {e1 / sr:.2f} s (pause {p1 / sr:.2f}), 'Unrolled' starts {s4 / sr:.2f} s (pause {p2 / sr:.2f}); "
          f"middle {len(M) / sr:.2f} s at gain {g:.2f}; out {len(o) / sr:.2f} s (v24 {len(y) / sr:.2f} s)")

if __name__ == "__main__":
    main()
