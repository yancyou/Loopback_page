"""hero_prior.py -- the hero card of the project page, drawn like panels (b) and (c) of the paper's Loopback figure
(figures/fig_loopback.pdf), in two states that the card cycles between (page.js):

  Decay prior  c0[s][i] = lam ** (floor((s+1)/K) - floor(i/K))           (paper eq. boundary-schedule)
  Attention    c[s][i]  = B_s * softmax_i(z[s][i] + log c0[s][i])          (paper eq. instance; B_s = sum_i c0[s][i])

for K = 2 shared blocks over three loop iterations with lam = 1/4 (panel (b) shows two iterations; the card adds a third).
As in the paper, the updates of one iteration share one prior weight ("[1] -> [lam, lam] -> [lam, lam, 1] -> [lam^2, lam^2,
lam, lam]"; section 4.2: "updates from the same iteration ... receiving equal prior weights"); only attention separates them.

Drawing follows panel (b): flat cells, ink symbols, bold c with stacked indices, "x lambda" in the figure's rose (#975b59) in a
column right of the grid, and the figure's shading, which deepens with every iteration-end decay: 1 -> #efe6e6, lam -> #dbc6c5,
lam^2 -> #c19d9b (sampled from the figure), lam^3 one step further. In the attention state a weight c takes the shade of its
decay depth d = ln c / ln lam (so a weight that grows gets lighter), and the right column B_s is marked unchanged. The scores z
are illustrative (the paper reports no per-update weights for this toy): the queries favour the first update delta_0 and, inside
every iteration, the f0 update over the f1 one -- the two things the paper says attention can do. The first read has a single
update and no learned query, so row 0 keeps weight 1.

Prints the <figure class="prior-card"> markup (pasted into page.html by tools/patch_hero.py); --table prints both tables.
"""
import math
import sys
from fractions import Fraction

K, R, LAM = 2, 3, Fraction(1, 4)
N = K * R
Z = [2.3, 0.0, 0.6, 0.0, 0.6, 0.0]           # illustrative content scores z_i = q^T k_i / sqrt(d), the same query in every row
CW, CH, GX, GY = 54, 40, 6, 12               # cell width / height and gaps (panel (b): 212 x 130 px cells, wide row gaps); 40 high so a
                                             # cell can hold the prior symbol over attention's weight
PX, PY = CW + GX, CH + GY
x0, y0 = 72, 62
XL = x0 + N * PX + 6                          # the "x lambda" column
BX = XL + 76                                  # centre of the B_s column
W = BX + 40
H = y0 + N * PY - GY + 10
STOPS = ["#efe6e6", "#dbc6c5", "#c19d9b", "#ac8583"]   # decay depth 0, 1, 2 (paper) and 3 (one step further)


def prior(s, i):
    return LAM ** ((s + 1) // K - i // K)


def label(v):
    if v == 1:
        return "1"
    for p in range(1, 4):
        if v == LAM ** p:
            return "λ" if p == 1 else f"λ{'²³'[p - 2]}"
    return str(v)


def label_svg(v):
    """the cell symbol in SVG: an italic lambda (as in the paper's math) with its power"""
    t = label(v)
    return t if not t.startswith("λ") else '<tspan class="prior__lam">λ</tspan>' + t[1:]


def shade(c):
    """the panel's colour for weight c: interpolated along the decay depth d = ln c / ln lam, clamped to 0..3"""
    d = min(3.0, max(0.0, math.log(c) / math.log(float(LAM))))
    i = min(2, int(d))
    t = d - i
    a = [int(STOPS[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(STOPS[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


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
    for i in range(s + 1):                    # equal prior weights inside an iteration (paper sections 4.1 / 4.2)
        assert c0[i] == c0[i - i % K]
    rows.append((c0, ca, B))
assert [[label(c) for c in r[0]] for r in rows[:4]] == [["1"], ["λ", "λ"], ["λ", "λ", "1"], ["λ²", "λ²", "λ", "λ"]]   # panel (b)

if "--table" in sys.argv:
    for s, (c0, ca, B) in enumerate(rows):
        print(f"s={s} B={float(B):.5f} | prior " + " ".join(f"{label(c):>3}" for c in c0) + " | attn " + " ".join(f"{c:.4f}" for c in ca)
              + " | shades " + " ".join(shade(float(c)) for c in c0) + " -> " + " ".join(shade(c) for c in ca))
    sys.exit(0)


def clabel(x, y, s, prior_state):
    """bold c with the paper's stacked indices: superscript 0 (prior only) over subscript s"""
    cls = "prior__row-label " + ("prior__lab--p" if prior_state else "prior__lab--a")
    out = f'<text class="{cls}" x="{x}" y="{y}"><tspan class="prior__c">c</tspan>'
    if prior_state:
        out += f'<tspan x="{x + 10.5}" y="{y - 8}" font-size="11">0</tspan>'
    out += f'<tspan x="{x + 10.5}" y="{y + 5}" font-size="11">{s}</tspan></text>'
    return out


svg = [f'<svg class="prior" viewBox="0 0 {W} {H}" role="img" aria-labelledby="prior-title prior-desc" xmlns="http://www.w3.org/2000/svg">',
       '<title id="prior-title">Loopback’s read weights: decay prior and attention</title>',
       '<desc id="prior-desc">Read weights for two shared blocks over three loop iterations with decay 1/4, drawn like the paper’s figure. '
       'Decay prior: each new update is appended with weight 1 and every iteration end scales all entries by 1/4, so the two updates of '
       'one iteration share one weight and older iterations fade (darker shades). Attention: each row is reweighted by content with '
       'illustrative scores, moving weight toward the first update and toward each iteration’s first block, while every row total stays the same.</desc>',
       f'<rect class="prior__bband" x="{BX - 32}" y="{y0 - 40}" width="64" height="{N * PY - GY + 46}" rx="10"/>',
       f'<text class="prior__bnote" x="{BX}" y="{y0 - 47}" text-anchor="middle">unchanged</text>',
       f'<text class="prior__params" x="6" y="{y0 - 34}"><tspan font-style="italic">K</tspan> = 2, <tspan font-style="italic">λ</tspan> = 1/4</text>']
for i in range(N):
    cx = x0 + i * PX + CW / 2
    svg.append(f'<text class="prior__col" x="{cx:.0f}" y="{y0 - 12}" text-anchor="middle"><tspan class="prior__d">δ</tspan><tspan dy="4" font-size="11">{i}</tspan></text>')
svg.append(f'<text class="prior__col prior__col--b" x="{BX}" y="{y0 - 12}" text-anchor="middle">B<tspan dy="4" font-size="11">s</tspan></text>')
for s, (c0, ca, B) in enumerate(rows):
    y = y0 + s * PY
    cy = y + CH / 2
    svg.append(f'<g class="prior__row" style="--d:{s}">')
    svg.append(clabel(x0 - 40, round(cy + 6), s, True))
    svg.append(clabel(x0 - 40, round(cy + 6), s, False))
    for i, (p, a) in enumerate(zip(c0, ca)):
        x = x0 + i * PX
        up = s > 0 and a > 1.25 * float(p)
        svg.append(f'<rect class="prior__cell{" prior__cell--up" if up else ""}" x="{x}" y="{y}" width="{CW}" height="{CH}" rx="1.5" style="--f:{shade(float(p))};--fa:{shade(a)}"/>')
        tx, ty = x + CW / 2, round(cy + 6.5)
        # the prior symbol (centred; in the attention phase it shrinks to the top of the cell and stays) and attention's
        # weight under it; the first read has one update and no query, so row 0 keeps its single 1
        svg.append(f'<text class="prior__val prior__val--p{" prior__val--keep" if s == 0 else ""}" x="{tx:.0f}" y="{ty}" text-anchor="middle">{label_svg(p)}</text>')
        if s > 0:
            svg.append(f'<text class="prior__val prior__val--a prior__val--num" x="{tx:.0f}" y="{round(y + CH - 7)}" text-anchor="middle">{num(a)}</text>')
    if (s + 1) % K == 0:
        svg.append(f'<text class="prior__decay" x="{XL}" y="{round(cy + 6.5)}">×<tspan class="prior__lam">λ</tspan></text>')
    svg.append(f'<text class="prior__total" x="{BX}" y="{round(cy + 5)}" text-anchor="middle">{total(B)}</text>')
    svg.append('</g>')
svg.append('</svg>')

out = ['<figure class="prior-card" aria-labelledby="prior-cap">',
       '<div class="prior-card__head">',
       '<ol class="prior-steps" aria-label="Two steps of every read">'
       '<li class="prior-step is-on" data-phase="decay"><span class="prior-step__n">1</span>Decay sets the prior</li>'
       '<li class="prior-step" data-phase="attn"><span class="prior-step__n">2</span>Attention allocates it</li></ol>',
       '</div>',
       "".join(svg),
       '<figcaption id="prior-cap">',
       '<b>Decay sets the prior</b> (symbols): append 1, scale all entries by <i>&lambda;</i> at iteration ends; one iteration&rsquo;s updates '
       'share a weight and <i>B</i> stays below <i>K&lambda;</i>/(1&minus;<i>&lambda;</i>) = 2/3. <b>Attention allocates that total</b> '
       '(numbers; illustrative scores): <span class="nowrap"><i>c<sub>s</sub></i> = <i>B<sub>s</sub></i>&thinsp;softmax(<i>z<sub>s</sub></i> + log&thinsp;<i>c</i><sup>0</sup><sub><i>s</i></sub>)</span>, '
       'so updates can differ while every <i>B<sub>s</sub></i> stays the same.',
       '</figcaption>',
       '</figure>']
print("\n".join(out))
