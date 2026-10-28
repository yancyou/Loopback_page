"""Loudness-normalize the voiceover, mux it into the promo video, and write subtitles.

    python make_av.py script.json     # after synth.py

Two-pass ffmpeg loudnorm (-16 LUFS integrated, -1.5 dBTP, LRA 11) on build/voiceover.wav taken as dual-mono stereo,
so the targets hold for the stereo file people hear; then 48 kHz AAC 160k, muxed with the video stream copied
(+faststart) into `out`. Also writes <out stem>.vtt and .srt: one cue per display chunk, from the first word to the last
word (+0.15 s, never into the next cue) of the matching say chunk in build/timings.json; a chunk over 84 characters,
or one that does not wrap into 2 lines of 42, is split at the space nearest its middle, times shared by character count.
Standard library only; needs ffmpeg and ffprobe.
"""
import argparse, json, os, re, subprocess, sys
from synth import BUILD, chunks, duration, load, repo

LN = "loudnorm=I=-16:TP=-1.5:LRA=11"
UPMIX = "pan=stereo|c0=c0|c1=c0"
PAD, MAXLEN, LINE = 0.15, 84, 42

def ffjson(args):
    """Run ffmpeg and return the last JSON object it prints (loudnorm's print_format=json report)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-nostats", *args], capture_output=True, text=True)
    if r.returncode:
        sys.exit("ffmpeg failed:\n" + r.stderr[-2000:])
    return json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])

def wrap(text):
    """One line if it fits, else the most balanced split at a space into 2 lines of <= LINE chars; None if impossible."""
    if len(text) <= LINE:
        return [text]
    cuts = [(max(i, len(text) - i - 1), i) for i, ch in enumerate(text) if ch == " " and i <= LINE and len(text) - i - 1 <= LINE]
    if not cuts:
        return None
    i = min(cuts)[1]
    return [text[:i], text[i + 1:]]

def pieces(text, t0, t1):
    """[(t0, t1, lines)] for one display chunk, split at the space nearest the middle until every piece fits."""
    if (len(text) <= MAXLEN and wrap(text)) or " " not in text:
        return [(t0, t1, wrap(text) or [text])]
    i = min((i for i, ch in enumerate(text) if ch == " "), key=lambda i: abs(i - len(text) / 2))
    a, b = text[:i].rstrip(), text[i + 1:].lstrip()
    tm = t0 + (t1 - t0) * len(a) / (len(a) + len(b))
    return pieces(a, t0, tm) + pieces(b, tm, t1)

def cues(script, timings):
    out = []
    for g, tm in zip(script["segments"], timings["segments"]):
        for k, text in enumerate(chunks(g["display"])):
            ws = [w for w in tm["words"] if w["chunk"] == k]
            if not ws:
                sys.exit(f"segment {g['id']!r}: no timed words for chunk {k} ({text!r}); check synth.py warnings")
            out += [[t0, t1, lines, g["id"]] for t0, t1, lines in pieces(text, ws[0]["start"], ws[-1]["end"])]
    for c, nxt in zip(out, out[1:] + [None]):
        c[1] = min(c[1] + PAD, nxt[0] if nxt else timings["video_duration"])
    return out

def stamp(t, sep):
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    a = ap.parse_args()
    s = load(a.script)
    wav, tpath = os.path.join(BUILD, "voiceover.wav"), os.path.join(BUILD, "timings.json")
    if not (os.path.exists(wav) and os.path.exists(tpath)):
        sys.exit("build/voiceover.wav or build/timings.json missing; run synth.py first")
    tm = json.load(open(tpath))
    video, out = repo(s["video"]), repo(s["out"])
    if [(g["id"], g["say"]) for g in s["segments"]] != [(g["id"], g["say"]) for g in tm["segments"]]:
        sys.exit("build/timings.json does not match the script (ids or say text changed); rerun synth.py")
    if abs(duration(video) - tm["video_duration"]) > 0.001:
        sys.exit(f"{s['video']} is not {tm['video_duration']} s long any more; rerun synth.py")
    if os.path.abspath(out) == os.path.abspath(video):
        sys.exit("out must not be the input video")
    cs = cues(s, tm)
    m = ffjson(["-i", wav, "-af", f"{UPMIX},{LN}:print_format=json", "-f", "null", "-"])
    print(f"measured (stereo): I {m['input_i']} LUFS, TP {m['input_tp']} dBTP, LRA {m['input_lra']} LU, thresh {m['input_thresh']}")
    if m["input_i"] == "-inf":
        sys.exit("voiceover.wav is silent")
    apply = (f"{LN}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
             f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true:print_format=json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = ffjson(["-y", "-i", video, "-i", wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-af", f"{UPMIX},{apply},aresample=48000", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart", out])
    print(f"applied: {r['normalization_type']} normalization")
    stem = os.path.splitext(out)[0]
    esc = lambda x: x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    with open(stem + ".vtt", "w") as f:
        f.write("WEBVTT\n")
        for i, (t0, t1, lines, gid) in enumerate(cs, 1):
            f.write(f"\n{gid}-{i}\n{stamp(t0, '.')} --> {stamp(t1, '.')}\n" + "\n".join(map(esc, lines)) + "\n")
    with open(stem + ".srt", "w") as f:
        for i, (t0, t1, lines, gid) in enumerate(cs, 1):
            f.write(f"{i}\n{stamp(t0, ',')} --> {stamp(t1, ',')}\n" + "\n".join(lines) + "\n\n")
    q = ffjson(["-i", out, "-map", "0:a:0", "-af", f"{LN}:print_format=json", "-f", "null", "-"])
    print(f"wrote {s['out']} ({os.path.getsize(out)} bytes): integrated {q['input_i']} LUFS, true peak {q['input_tp']} dBTP, LRA {q['input_lra']} LU")
    print(f"wrote {os.path.relpath(stem, repo(''))}.vtt and .srt: {len(cs)} cues")

if __name__ == "__main__":
    main()
