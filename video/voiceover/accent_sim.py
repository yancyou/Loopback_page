"""Accent closeness to the takes the user has already accepted (v23, 2026-11-09).

The CommonAccent classifier gives P(us) > 0.999 to takes the user hears as accented, so its decision is not enough.
This scores every take by the cosine between its CommonAccent ECAPA embedding and the centroid of the accepted clips
(build/vc/<id>.wav for --accepted ids), and also reports P(indian) = softmax(30 cos)[indian].
    accent_venv/bin/python accent_sim.py LIST.json OUT.json --accepted title hook live scale nine math outro
"""
import argparse, json, os
import numpy as np
import torch
from voice_sim import load16

ap = argparse.ArgumentParser()
ap.add_argument("lst"); ap.add_argument("out"); ap.add_argument("--accepted", nargs="+", required=True)
a = ap.parse_args()
torch.set_num_threads(min(16, os.cpu_count() or 1))
from speechbrain.inference.classifiers import EncoderClassifier
m = EncoderClassifier.from_hparams("Jzuluaga/accent-id-commonaccent_ecapa",
                                   savedir=os.path.expanduser("~/.cache/speechbrain/accent-id-commonaccent_ecapa"))
labs = [m.hparams.label_encoder.decode_ndim(i) for i in range(len(m.hparams.label_encoder))]
ind = [i for i, l in enumerate(labs) if "ind" in l.lower()]

def emb_prob(path):
    w = torch.from_numpy(load16(path))[None]
    with torch.no_grad():
        e = m.encode_batch(w)[0, 0].numpy()
        cos = m.classify_batch(w)[0][0]
    pr = torch.softmax(30 * cos, -1).numpy()
    return e / np.linalg.norm(e), pr

acc = [emb_prob(os.path.join("build", "vc", i + ".wav"))[0] for i in a.accepted]
cen = np.mean(acc, 0); cen /= np.linalg.norm(cen)
R = {"labels": labs, "indian_labels": [labs[i] for i in ind], "accepted": {}}
for i, e in zip(a.accepted, acc):
    R["accepted"][i] = round(float(e @ cen), 4)
for k, p in json.load(open(a.lst)).items():
    e, pr = emb_prob(p)
    R[k] = {"cen": round(float(e @ cen), 4), "max": round(float(max(e @ x for x in acc)), 4),
            "p_ind": round(float(sum(pr[i] for i in ind)), 6)}
json.dump(R, open(a.out, "w"), indent=1)
print("accepted vs centroid:", R["accepted"], "| indian labels:", R["indian_labels"])
