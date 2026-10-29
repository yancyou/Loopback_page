"""A plain procedural music bed for the promo, until a generated track replaces it.

    python pad.py [--out build/pad.wav] [--seconds 84]

116 BPM, Am7 - Fmaj7 - Cadd9 - G6, two bars each: an additive pad (harmonics rolled off above ~1.5 kHz, three detuned
voices per note, left and right apart), a sub bass on the root, a soft kick on the beat, an off-beat noise tick, and a
plucked arpeggio an octave up with a dotted-eighth ping-pong delay; pad and arpeggio share a synthetic reverb and the
whole bed pumps lightly on the beat. Sections follow the video's scenes (promo.html SC): pad alone under the title,
bass and tick from the loop scene, kick and arpeggio from the three rules, kick and arpeggio out for the outro. The
1-4 kHz band stays thin so the voice sits on top. numpy, scipy and soundfile; 48 kHz stereo float WAV.
"""
import argparse, os
import numpy as np, soundfile as sf
from scipy.signal import butter, oaconvolve, sosfilt

SR, BPM = 48000, 116
BEAT = 60 / BPM
CHORDS = [(45, [57, 60, 64, 67]), (41, [53, 57, 60, 64]), (48, [60, 64, 67, 74]), (43, [55, 59, 62, 64])]  # (bass, pad)
BARS_PER_CHORD = 2
SECTIONS = {"bass": (6.5, 80.0), "tick": (6.5, 75.5), "kick": (17.0, 75.5), "arp": (17.0, 75.5)}

def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def ramp(t, a, b, fade=1.5):
    """1 inside [a, b], with linear fades of `fade` s inward at both ends."""
    return np.clip((t - a) / fade, 0, 1) * np.clip((b - t) / fade, 0, 1)

def pad_note(f, n, rng):
    """Additive pad: harmonics with 1/k amplitude rolled off above ~1.5 kHz, three detuned voices, L/R apart."""
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for ch, cents in enumerate(((-7, 0, 5), (-4, 0, 8))):
        for c in cents:
            fc, ph = f * 2 ** (c / 1200), rng.uniform(0, 2 * np.pi)
            k = np.arange(1, int(3000 // fc) + 1)
            amp = (1 / k) * np.exp(-k * fc / 1500)
            out[:, ch] += (amp[:, None] * np.sin(2 * np.pi * fc * k[:, None] * t + ph * k[:, None])).sum(0)
    return out / 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "pad.wav"))
    ap.add_argument("--seconds", type=float, default=84.0)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    n = int(a.seconds * SR); t = np.arange(n) / SR
    pad, bass, arp, drums = (np.zeros((n, 2)) for _ in range(4))
    seg = BARS_PER_CHORD * 4 * BEAT
    for i in range(int(np.ceil(a.seconds / seg))):
        root, notes = CHORDS[i % len(CHORDS)]
        s0 = int(i * seg * SR); s1 = min(n, int(((i + 1) * seg + 1.0) * SR))   # 1 s overlap into the next chord
        m = s1 - s0
        if m <= 0:
            continue
        env = np.minimum(1, np.arange(m) / (0.8 * SR)) * np.minimum(1, (m - np.arange(m)) / (1.0 * SR))
        for f in notes:
            pad[s0:s1] += pad_note(hz(f), m, rng) * env[:, None] * 0.25
        tb = np.arange(m) / SR
        bass[s0:s1] += (np.sin(2 * np.pi * hz(root) * tb) + 0.18 * np.sin(4 * np.pi * hz(root) * tb))[:, None] * env[:, None] * 0.5
        arp_notes = [notes[j] + 12 for j in (0, 1, 2, 3, 2, 1, 2, 3)]
        for j in range(int(seg / (BEAT / 2))):                                  # eighth notes
            start = int((i * seg + j * BEAT / 2) * SR)
            L = int(0.45 * SR)
            if start + L > n:
                break
            tt = np.arange(L) / SR; f = hz(arp_notes[j % 8])
            arp[start:start + L] += ((np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 0.12))[:, None] * 0.22
    for b in range(int(a.seconds / BEAT)):
        s0 = int(b * BEAT * SR)
        L = min(int(0.35 * SR), n - s0); tt = np.arange(L) / SR
        kick = np.sin(2 * np.pi * (48 * tt + (110 - 48) * 0.035 * (1 - np.exp(-tt / 0.035)))) * np.exp(-tt / 0.16)
        drums[s0:s0 + L] += (kick * ramp(t[s0:s0 + L], *SECTIONS["kick"]))[:, None] * 0.55
        h0 = int((b + 0.5) * BEAT * SR); Lh = min(int(0.05 * SR), n - h0)
        if Lh > 0:
            hat = rng.standard_normal(Lh) * np.exp(-np.arange(Lh) / (0.012 * SR))
            drums[h0:h0 + Lh] += (hat * ramp(t[h0:h0 + Lh], *SECTIONS["tick"]))[:, None] * 0.11
    drums = sosfilt(butter(2, 9000, "lowpass", fs=SR, output="sos"), drums, axis=0)
    hp = butter(2, 6000, "highpass", fs=SR, output="sos")
    tick_band = sosfilt(hp, drums, axis=0); drums = drums - sosfilt(butter(2, 300, "highpass", fs=SR, output="sos"), drums, axis=0) + tick_band
    bass *= ramp(t, *SECTIONS["bass"], fade=2.0)[:, None]
    arp *= ramp(t, *SECTIONS["arp"], fade=3.0)[:, None]
    d = int(0.75 * BEAT * SR)                                                  # ping-pong delay, dotted eighth
    for k in range(1, 5):
        g = 0.38 ** k; sh = arp[:n - k * d].copy()
        if k % 2:
            sh = sh[:, ::-1]
        arp[k * d:] += g * sh
    arp = sosfilt(butter(2, 2600, "lowpass", fs=SR, output="sos"), arp, axis=0)
    pad = sosfilt(butter(2, [90, 2200], "bandpass", fs=SR, output="sos"), pad, axis=0)
    pad *= (0.55 + 0.45 * ramp(t, 3.0, a.seconds - 2.0, fade=4.0))[:, None]  # darker under the title
    L = int(2.8 * SR); ir_t = np.arange(L) / SR
    ir = rng.standard_normal((L, 2)) * np.exp(-ir_t / 0.55)[:, None]
    ir = sosfilt(butter(2, 3500, "lowpass", fs=SR, output="sos"), ir, axis=0); ir /= np.sqrt((ir ** 2).sum(0))
    bus = pad + 0.8 * arp
    wet = np.stack([oaconvolve(bus[:, c], ir[:, c])[:n] for c in range(2)], 1)
    pump = 1 - 0.28 * np.exp(-((t % BEAT) / 0.11)) * ramp(t, *SECTIONS["kick"])  # light sidechain feel on the beat
    mix = (0.75 * bus + 0.45 * wet + bass) * pump[:, None] + drums
    mix *= ramp(t, 0.0, a.seconds, fade=2.5)[:, None]
    mix = np.tanh(1.1 * mix / np.abs(mix).max()) * 0.8                          # gentle saturation, -2 dBFS peak area
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sf.write(a.out, mix.astype(np.float32), SR, subtype="FLOAT")
    print(f"wrote {a.out}: {a.seconds:.1f} s, {SR} Hz stereo, peak {np.abs(mix).max():.2f}")

if __name__ == "__main__":
    main()
