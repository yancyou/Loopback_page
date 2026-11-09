"""Replace the middle of a take with another reading (v26b, 11-09: the knobs take is kept except its Hyper-Connections sentence).
A is cut in two silent gaps: the one holding time T1 (s) ends the head, the one holding T2 starts the tail (librosa's non-silent
intervals, top_db 35, as vc.cap_pauses). MID is trimmed to its speech and scaled to A's speech loudness (RMS over non-silent
frames, clamped to 0.7-1.4x). Each joint is a PAUSE s silence; 10 ms fades at every cut.

    clone_venv/bin/python splice_gap.py A.wav MID.wav OUT.wav T1 T2 [PAUSE]
"""
import sys
import numpy as np, soundfile as sf, librosa

def speech(y):
    return librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)

def rms(y, iv):
    return float(np.sqrt(np.mean(np.concatenate([y[a:b] for a, b in iv]) ** 2)))

def gap_at(iv, t):     # (end of the speech before, start of the speech after) of the silent gap holding sample t
    for (a0, a1), (b0, b1) in zip(iv[:-1], iv[1:]):
        if a1 <= t <= b0:
            return a1, b0
    sys.exit(f"no silent gap at {t} samples")

def main():
    a_path, m_path, out = sys.argv[1:4]
    t1, t2 = float(sys.argv[4]), float(sys.argv[5]); pause = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    y, sr = sf.read(a_path, dtype="float32"); mid, sr2 = sf.read(m_path, dtype="float32")
    assert sr == sr2, (sr, sr2)
    iv, mi = speech(y), speech(mid)
    e1, _ = gap_at(iv, int(t1 * sr)); _, s2 = gap_at(iv, int(t2 * sr))
    keep = int(min(0.25, pause / 2) * sr)
    A, C = y[:e1 + keep].copy(), y[s2 - keep:].copy()
    M = mid[max(0, mi[0][0] - int(0.02 * sr)):min(len(mid), mi[-1][1] + int(0.04 * sr))].copy()
    g = float(np.clip(rms(y, iv) / rms(mid, mi), 0.7, 1.4)); M *= g
    f = int(0.01 * sr); ramp = np.linspace(0, 1, f, dtype=np.float32)
    A[-f:] *= ramp[::-1]; C[:f] *= ramp; M[:f] *= ramp; M[-f:] *= ramp[::-1]
    z1 = np.zeros(max(0, int(pause * sr) - keep - int(0.02 * sr)), np.float32)
    z2 = np.zeros(max(0, int(pause * sr) - keep - int(0.04 * sr)), np.float32)
    o = np.concatenate([A, z1, M, z2, C]).astype(np.float32)
    sf.write(out, o, sr, subtype="FLOAT")
    print(f"head ends {e1 / sr:.2f} s, tail starts {s2 / sr:.2f} s; middle {len(M) / sr:.2f} s at gain {g:.2f}; joints {pause:.2f} s; "
          f"out {len(o) / sr:.2f} s (was {len(y) / sr:.2f} s)")

if __name__ == "__main__":
    main()
