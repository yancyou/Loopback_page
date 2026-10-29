"""Shorten a generated music bed by whole bars, so its own ending lands where the video ends.

    python cut.py build/cand1.wav build/bed.wav --bpm 116 --phase 0.034 --drop 24 2

Removes bars [first, first + count) of a 4/4 track whose beat grid is --bpm with its first beat at --phase seconds (read
off librosa's beat_track: the grid fits cand1's beats to 18 ms over 9-98 s), joining the two sides with a 40 ms
equal-power crossfade on the downbeat. For cand1 (110 s, ending from bar 48 at 99.4 s; from bar 8 the chords change
every 2 bars in a 4-bar cycle): dropping bars 24-25 moves the ending to 95.2 s of the 96.4 s video. The seam (49.69 s,
where the scale line pauses after "parameters") joins two bars of the same chord, so the B-flat chord is held for four
bars once; kick onsets stay within 15 ms of the grid around it. The 84 s cut dropped the whole 8 bars 24-31 (ending
82.8 s). numpy and soundfile.
"""
import argparse
import numpy as np, soundfile as sf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--bpm", type=float, required=True); ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--drop", type=int, nargs=2, metavar=("FIRST", "COUNT"), required=True)
    a = ap.parse_args()
    x, sr = sf.read(a.src, dtype="float32", always_2d=True)
    bar = 4 * 60 / a.bpm
    i, j = (round((a.phase + b * bar) * sr) for b in (a.drop[0], a.drop[0] + a.drop[1]))
    xf = round(0.04 * sr)
    w = np.sin(np.linspace(0, np.pi / 2, xf))[:, None]                        # equal power
    seam = x[i - xf // 2:i + xf - xf // 2] * w[::-1] + x[j - xf // 2:j + xf - xf // 2] * w
    y = np.concatenate([x[:i - xf // 2], seam, x[j + xf - xf // 2:]])
    sf.write(a.dst, y, sr, subtype="FLOAT")
    print(f"dropped bars {a.drop[0]}-{sum(a.drop) - 1} ({(j - i) / sr:.2f} s, {i / sr:.2f}-{j / sr:.2f} s): {len(x) / sr:.1f} s -> {len(y) / sr:.1f} s")

if __name__ == "__main__":
    main()
