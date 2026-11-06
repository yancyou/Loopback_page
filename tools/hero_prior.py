"""hero_prior.py -- the hero card of the project page, in two states that the card cycles between (page.js):

  Decay prior  c0[s][i] = lam ** (floor((s+1)/K) - floor(i/K))           (paper eq. boundary-schedule)
  Attention    c[s][i]  = B_s * softmax_i(z[s][i] + log c0[s][i])          (paper eq. instance; B_s = sum_i c0[s][i])

for K = 2 shared blocks over three loop iterations with lam = 1/4 (the paper's Figure 4b, extended by one iteration). Row s is
the read after block application s, column i the stored block update delta_i. A cell's rose tint is its weight (lighter =
smaller), every iteration end multiplies the stored weights by lam, and the right column is the row total B_s, which attention
leaves unchanged. The scores z are illustrative (the paper reports no per-update weights for this toy): the queries favour the
first update delta_0 and, inside every iteration, the f0 update over the f1 one -- the two things the paper says attention can
do (favour an older update; split equal priors). The first read has a single update and no learned query, so row 0 keeps c = 1.

Prints the <figure class="prior-card"> markup to stdout (pasted into page.html by tools/patch_hero.py) and, with --table, the
two coefficient tables for checking.
"""
import math
import sys
from fractions import Fraction

K, R, LAM = 2, 3, Fraction(1, 4)
N = K * R
Z = [2.3, 0.0, 0.6, 0.0, 0.6, 0.0]           # illustrative content scores z_i = q^T k_i / sqrt(d), the same query in every row
cell, gap, x0, y0 = 46, 6, 74, 70
BX = x0 + N * (cell + gap) + 80              # centre of the B_s column
W = BX + 44                                  # room for the totals band and its label
H = y0 + N * (cell + gap) + 4


def prior(s, i):
    return LAM ** ((s + 1) // K - i // K)


def label(v):
    if v == 1:
        return "1"
    for p in range(1, 4):
        if v == LAM ** p:
            return "λ" if p == 1 else f"λ{'²³'[p - 2]}"
    return str(v)


def opacity(v):
    return min(1.0, max(0.12, float(v) ** 0.55))


def num(v):
    return f"{v:.2f}"


def total(v):
    return f"{float(v):.3f}".rstrip("0").rstrip(".")


rows = []
for s in range(N):
    c0 = [prior(s, i) for i in range(s + 1)]
    B = sum(c0)
    if s == 0:                                # one update: no learned query, softmax over one entry
        ca = [float(B)]
    else:
        w = [float(c) * math.exp(Z[i]) for i, c in enumerate(c0)]
        ca = [float(B) * x / sum(w) for x in w]
    assert abs(sum(ca) - float(B)) < 1e-12
    rows.append((c0, ca, B))

if "--table" in sys.argv:
    for s, (c0, ca, B) in enumerate(rows):
        print(f"s={s} B={float(B):.5f} | prior " + " ".join(f"{float(c):.4f}" for c in c0) + " | attn " + " ".join(f"{c:.4f}" for c in ca)
              + f" | sum {sum(ca):.5f}")
    sys.exit(0)

svg = [f'<svg class="prior" viewBox="0 0 {W} {H}" role="img" aria-labelledby="prior-title prior-desc" xmlns="http://www.w3.org/2000/svg">',
       '<title id="prior-title">Loopback’s read weights: decay prior and attention</title>',
       '<desc id="prior-desc">Read weights for two shared blocks over three loop iterations with decay 1/4. Decay prior: after every '
       'application the new block update enters with weight 1, and at the end of every iteration all stored weights are multiplied '
       'by 1/4, so older updates fade while the total stays bounded. Attention: each row is reweighted by content with illustrative '
       'scores, moving weight toward the first update and toward each iteration’s first block, while every row total stays the same.</desc>',
       # the band behind the B_s column, shown in the attention state: the row totals do not change
       f'<rect class="prior__bband" x="{BX - 34}" y="{y0 - 44}" width="68" height="{N * (cell + gap) + 46}" rx="12"/>',
       f'<text class="prior__bnote" x="{BX}" y="{y0 - 50}" text-anchor="middle">unchanged</text>']
for i in range(N):
    cx = x0 + i * (cell + gap) + cell / 2
    svg.append(f'<text class="prior__col" x="{cx:.0f}" y="{y0 - 16}" text-anchor="middle">δ<tspan dy="4" font-size="11">{i}</tspan></text>')
svg.append(f'<text class="prior__col prior__col--b" x="{BX}" y="{y0 - 16}" text-anchor="middle">B<tspan dy="4" font-size="11">s</tspan></text>')
for s, (c0, ca, B) in enumerate(rows):
    y = y0 + s * (cell + gap)
    ly = y + cell / 2 + 5
    svg.append(f'<g class="prior__row" style="--d:{s}">')
    svg.append(f'<text class="prior__row-label prior__lab--p" x="{x0 - 14}" y="{ly:.0f}" text-anchor="end">c<tspan dy="-7" font-size="10">0</tspan><tspan dy="11" font-size="10">{s}</tspan></text>')
    svg.append(f'<text class="prior__row-label prior__lab--a" x="{x0 - 14}" y="{ly:.0f}" text-anchor="end">c<tspan dy="4" font-size="10">{s}</tspan></text>')
    for i, (p, a) in enumerate(zip(c0, ca)):
        x = x0 + i * (cell + gap)
        op, oa = opacity(p), opacity(a)
        up = s > 0 and a > 1.25 * float(p)
        svg.append(f'<rect class="prior__cell{" prior__cell--up" if up else ""}" x="{x}" y="{y}" width="{cell}" height="{cell}" rx="7" style="--o:{op:.3f};--oa:{oa:.3f}"/>')
        tx, ty = x + cell / 2, y + cell / 2 + 6
        svg.append(f'<text class="prior__val prior__val--p{" prior__val--light" if op < 0.5 else ""}" x="{tx:.0f}" y="{ty:.0f}" text-anchor="middle">{label(p)}</text>')
        alab = label(p) if s == 0 else num(a)   # the first read keeps its single weight 1 (no query)
        svg.append(f'<text class="prior__val prior__val--a{"" if s == 0 else " prior__val--num"}{" prior__val--light" if oa < 0.5 else ""}" x="{tx:.0f}" y="{ty:.0f}" text-anchor="middle">{alab}</text>')
    svg.append(f'<text class="prior__total" x="{BX}" y="{ly:.0f}" text-anchor="middle">{total(B)}</text>')
    if (s + 1) % K == 0:
        svg.append(f'<text class="prior__decay" x="{x0 + (s + 1) * (cell + gap) + 4:.0f}" y="{ly:.0f}">× λ</text>')
    svg.append('</g>')
svg.append('</svg>')

out = ['<figure class="prior-card" aria-labelledby="prior-cap">',
       '<div class="prior-card__head">',
       '<div class="prior-tabs" role="group" aria-label="Read weights shown">'
       '<button type="button" class="prior-tab is-on" data-state="decay" aria-pressed="true">Decay prior</button>'
       '<button type="button" class="prior-tab" data-state="attn" aria-pressed="false">Attention</button></div>',
       '<span class="prior-card__params"><i>K</i> = 2, <i>&lambda;</i> = 1/4</span>',
       '</div>',
       "".join(svg),
       '<figcaption id="prior-cap" aria-live="polite">',
       '<span class="prior-cap prior-cap--decay">Each read after a block application gives the new update weight 1; at the end of every '
       'loop iteration all stored weights shrink by <i>&lambda;</i>, so the total <i>B</i> stays below <i>K&lambda;</i>/(1&minus;<i>&lambda;</i>) = 2/3.</span>',
       '<span class="prior-cap prior-cap--attn"><span class="nowrap"><i>c<sub>s</sub></i> = <i>B<sub>s</sub></i>&thinsp;softmax(<i>z<sub>s</sub></i> + log&thinsp;<i>c</i><sup>0</sup><sub><i>s</i></sub>)</span>: '
       'attention reweights each row by content, here moving weight toward the older <i>&delta;</i><sub>0</sub> (illustrative scores), while every row total <i>B<sub>s</sub></i> stays the same.</span>',
       '</figcaption>',
       '</figure>']
print("\n".join(out))
