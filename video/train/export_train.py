"""Training curves for the video's `train` scene: the paper's Figure 4 (training loss of the standard Transformer and Loopback
at 191M, 374M, 778M, 1.4B and 3B), from the same inputs and with the same smoothing as the paper's generator
(figure/train_loss/fig_train_loss_5.py: 100-step moving averages of the loss logged every 10 steps; the 1.4B and 3B curves are
stitched from their 2-, 5- and 20-TPP rungs at the forks). Delta is the paper's: the mean of the last 100 steps' logged losses,
Loopback minus Standard. Writes train/train_curves.json (the values) and ../train_data.js (var TRAIN, loaded by promo.html).

    python3 train/export_train.py [FIGURE_DIR]        # FIGURE_DIR: the directory of fig_train_loss_5.py (default below)
"""
import json, os, sys
import numpy as np

FIG = sys.argv[1] if len(sys.argv) > 1 else "/home/sigma/xuano/figure/train_loss"
sys.path.insert(0, FIG)
from fig_train_loss import SMOOTH, smooth          # noqa: E402  (the paper figure's own smoothing)
from fig_train_loss_5 import stitched_3b           # noqa: E402  (and its 3B stitching)

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(FIG, "train_loss_data.json")))
data["3B"] = stitched_3b()
out = []
for tag, scale in zip("abcde", ("191M", "374M", "778M", "1.4B", "3B")):
    d = data[scale]; tps = d["tokens_per_step"]; cur = {}
    for m in ("Dense", "Loopback"):
        p = d["runs"][m]; st = np.array(sorted(int(s) for s in p))
        cur[m] = (st, smooth([p[str(s)] for s in st], SMOOTH))
    assert np.array_equal(cur["Dense"][0], cur["Loopback"][0]), scale
    st = cur["Dense"][0]
    last = max(int(k) for k in d["runs"]["Dense"])
    tail = lambda m: float(np.mean([v for k, v in d["runs"][m].items() if int(k) > last - 100]))
    delta = round(tail("Loopback") - tail("Dense"), 4)
    x_end = float(st[-1] * tps / 1e9)
    hi = max(float(np.interp(0.08 * x_end, st * tps / 1e9, y)) for _, y in cur.values())     # the paper's top of the panel
    lo = min(float(y.min()) for _, y in cur.values()) - 0.03
    out.append({"tag": tag, "scale": scale, "tokens_b": [round(float(s * tps / 1e9), 4) for s in st],
                "std": [round(float(v), 4) for v in cur["Dense"][1]], "lb": [round(float(v), 4) for v in cur["Loopback"][1]],
                "delta": delta, "lo": round(lo, 4), "hi": round(hi, 4), "x_end": round(x_end, 4),
                "std_tail": round(tail("Dense"), 4), "lb_tail": round(tail("Loopback"), 4)})
    print(f"{scale:5s} {len(st)} points, {x_end:.2f}B tokens, panel {lo:.3f}-{hi:.3f}, last-100-step Standard {tail('Dense'):.4f} "
          f"Loopback {tail('Loopback'):.4f} delta {delta:+.4f}")
json.dump({"note": "Paper Figure 4 (fig:train-loss): 100-step moving averages (SMOOTH=%d logged points of 10 steps); "
           "delta = mean of the last 100 steps, Loopback minus Standard. Source: %s" % (SMOOTH, FIG), "panels": out},
          open(os.path.join(HERE, "train_curves.json"), "w"))
with open(os.path.join(HERE, "..", "train_data.js"), "w") as f:
    f.write("// written by train/export_train.py from train/train_curves.json (paper Figure 4); do not edit\n")
    f.write("var TRAIN = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("wrote train/train_curves.json and train_data.js")
