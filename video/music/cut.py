"""Shorten or lengthen a generated music bed by whole bars, so its own ending lands where the video ends.

    python cut.py build/cand1.wav build/bed.wav --bpm 116 --phase 0.034 --bars 0:40 24:     # bars 24-39 play twice
    python cut.py build/cand1.wav build/bed.wav --bpm 116 --phase 0.034 --drop 24 2         # bars 24-25 left out

--bars plays the given half-open bar ranges in order ("a:b" is bars a..b-1, "a:" runs to the end of the file, a range
from bar 0 starts at the file's first sample); --drop FIRST COUNT is short for --bars 0:FIRST FIRST+COUNT:. The beat grid
is --bpm with its first beat at --phase seconds (read off librosa's beat_track: the grid fits cand1's beats to 18 ms over
9-98 s), and every join is a 40 ms equal-power crossfade on the downbeat.
cand1 (110 s, ending from bar 48 at 99.4 s): from bar 8 the chords change every 2 bars in a 4-bar cycle, and the 8-bar
sections from bar 8 to bar 39 match each other closely, so a jump back by a multiple of 4 bars to a phrase start keeps
the harmony and the phrasing. The 133.7 s video uses --bars 0:36 20: (bars 20-35 twice, 143.1 s, ending from 132.4 s; the
join at 74.5 s keeps the kick within 14 ms of the grid). The 96.4 s cut dropped bars 24-25 (ending 95.2 s); the 84 s cut
dropped bars 24-31.
numpy and soundfile.
"""
import argparse
import numpy as np, soundfile as sf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--bpm", type=float, required=True); ap.add_argument("--phase", type=float, default=0.0)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--drop", type=int, nargs=2, metavar=("FIRST", "COUNT"))
    g.add_argument("--bars", nargs="+", metavar="A:B")
    a = ap.parse_args()
    x, sr = sf.read(a.src, dtype="float32", always_2d=True)
    bar = 4 * 60 / a.bpm
    spec = [f"0:{a.drop[0]}", f"{sum(a.drop)}:"] if a.drop else a.bars
    at = lambda b: 0 if b == 0 else round((a.phase + b * bar) * sr)
    ranges = []
    for r in spec:
        b0, b1 = r.split(":")
        ranges.append((int(b0), int(b1) if b1 else None))
    xf = round(0.04 * sr)
    w = np.sin(np.linspace(0, np.pi / 2, xf))[:, None]                        # equal power
    out, pos = [], at(ranges[0][0])
    for (b0, b1), nxt in zip(ranges, ranges[1:] + [None]):
        end = len(x) if b1 is None else at(b1)
        if nxt is None:
            out.append(x[pos:end]); break
        j = at(nxt[0])
        out.append(x[pos:end - xf // 2])
        out.append(x[end - xf // 2:end + xf - xf // 2] * w[::-1] + x[j - xf // 2:j + xf - xf // 2] * w)
        pos = j + xf - xf // 2
    y = np.concatenate(out)
    sf.write(a.dst, y, sr, subtype="FLOAT")
    t, joins = 0.0, []
    for (b0, b1), nxt in zip(ranges, ranges[1:]):
        t += ((at(b1) if b1 is not None else len(x)) - at(b0)) / sr
        joins.append(f"bar {b1 - 1}->{nxt[0]} at {t:.2f} s")
    print(f"bars {' '.join(spec)}: {len(x) / sr:.1f} s -> {len(y) / sr:.1f} s; joins: {'; '.join(joins) or 'none'}")

if __name__ == "__main__":
    main()
