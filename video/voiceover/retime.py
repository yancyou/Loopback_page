"""Lay the scenes out around the narration: each scene is as long as its line needs (a 0.5 s lead-in, the take, a pause)
and at least as long as its animation needs (MINLEN), and the spare time up to --dur is shared out as extra pause, so the
music bed's own ending can land at the end of the video. Writes the segment windows (start = scene start + LEAD,
end = scene end - TAIL_GAP) and duration into script.json, and SC / DUR into ../promo.html; then run vc.py place to move
the takes to their new starts.

    python3 retime.py script.json [--dur 142.0]          # takes from build/timings.json (any earlier window layout)
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LEAD, TAIL_GAP, PAUSE, END_HOLD = 0.5, 0.3, 0.8, 1.3
MINLEN = {"title": 5.0, "memory": 14.5, "knobs": 14.5, "method": 12.0, "live": 10.6, "scale": 10.4, "nine": 9.4, "math": 9.6, "outro": 7.4}   # v22: insight scenes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script"); ap.add_argument("--dur", type=float)
    a = ap.parse_args()
    s = json.load(open(a.script))
    tm = {g["id"]: g for g in json.load(open(os.path.join(HERE, "build", "timings.json")))["segments"]}
    ids = [g["id"] for g in s["segments"]]
    if set(ids) - set(tm):
        sys.exit(f"build/timings.json has no take for {sorted(set(ids) - set(tm))}")
    need = []
    for k, i in enumerate(ids):
        tail = END_HOLD if k == len(ids) - 1 else PAUSE
        need.append(max(MINLEN.get(i, 0.0), LEAD + tm[i]["duration"] + tail))
    total = sum(need)
    extra = 0.0
    if a.dur:
        if a.dur < total:
            sys.exit(f"the scenes need {total:.2f} s, more than --dur {a.dur}")
        extra = (a.dur - total) / (len(ids) - 1)       # shared by every scene but the outro
    sc, t = {}, 0.0
    for k, (i, n) in enumerate(zip(ids, need)):
        n += extra if k < len(ids) - 1 else 0.0
        sc[i] = [round(t, 2), round(t + n, 2)]
        t += n
    dur = round(round(t * 30) / 30, 3)          # a whole number of 30 fps frames, as render.py writes them
    for g in s["segments"]:
        g["start"], g["end"] = round(sc[g["id"]][0] + LEAD, 2), round(sc[g["id"]][1] - TAIL_GAP, 2)
        if g["start"] + tm[g["id"]]["duration"] > g["end"]:
            sys.exit(f"segment {g['id']!r}: take {tm[g['id']]['duration']:.2f} s does not fit {g['start']}..{g['end']}")
    s["duration"] = dur
    json.dump(s, open(a.script, "w"), indent=1, ensure_ascii=False)
    p = os.path.join(HERE, "..", "promo.html")
    page = open(p, encoding="utf-8").read()
    body = ", ".join(f"{i}: [{sc[i][0]:g}, {sc[i][1]:g}]" for i in ids)
    new = "var SC = { " + body.replace(", math:", ",\n           math:", 1) + " };"
    page, n1 = re.subn(r"var SC = \{[^}]*\};", lambda m: new, page)
    page, n2 = re.subn(r"var DUR = [0-9.]+;", f"var DUR = {dur:g};", page)
    if n1 != 1 or n2 != 1:
        sys.exit("promo.html: SC or DUR not found once")
    open(p, "w", encoding="utf-8").write(page)
    print(f"{'id':<9}{'scene':>16}{'take':>7}{'pause':>7}")
    for i in ids:
        st = sc[i][0] + LEAD
        print(f"{i:<9}{sc[i][0]:8.2f}-{sc[i][1]:<7.2f}{tm[i]['duration']:7.2f}{sc[i][1] - st - tm[i]['duration']:7.2f}")
    print(f"duration {dur:.2f} s (scenes need {total:.2f} s); wrote {a.script} and promo.html; next: vc.py place {a.script}")

if __name__ == "__main__":
    main()
