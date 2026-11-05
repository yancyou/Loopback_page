"""hero_prior.py -- the hero illustration of the project page: Loopback's decay prior c0 for K = 2 shared blocks and three
loop iterations (the paper's Figure 4b, extended by one iteration), with lambda = 1/4 as in the paper. Row s is the read
after block application s; column i is the stored block update delta_i; the cell's rose tint is the prior coefficient
(lighter = smaller), every iteration end multiplies all stored weights by lambda, and the right column is the row total
B_s. Writes the SVG markup to stdout; build.py does not run it (the markup is pasted into page.html)."""
from fractions import Fraction

K, R, LAM = 2, 3, Fraction(1, 4)
N = K * R
cell, gap, x0, y0 = 46, 6, 74, 52
W = x0 + N * (cell + gap) + 124
H = y0 + N * (cell + gap) + 4


def label(v):
    if v == 1:
        return "1"
    for p in range(1, 4):
        if v == LAM ** p:
            return "λ" if p == 1 else f"λ{'²³'[p - 2]}"
    return str(v)


rows = []
for s in range(N):
    coeffs = []
    for i in range(s + 1):
        # iterations completed after application i up to and including application s
        it_i = i // K
        ends = sum(1 for e in range(it_i, s // K + 1) if (e + 1) * K - 1 <= s)
        coeffs.append(LAM ** ends)
    rows.append(coeffs)

out = [f'<svg class="prior" viewBox="0 0 {W} {H}" role="img" aria-labelledby="prior-title prior-desc" xmlns="http://www.w3.org/2000/svg">',
       '<title id="prior-title">Loopback’s decay prior</title>',
       '<desc id="prior-desc">Prior read coefficients for two shared blocks over three loop iterations with decay 1/4: after every '
       'application the new block update enters with coefficient 1, and at the end of every iteration all stored coefficients '
       'are multiplied by 1/4, so older updates fade while the total stays bounded.</desc>']
for i in range(N):
    cx = x0 + i * (cell + gap) + cell / 2
    out.append(f'<text class="prior__col" x="{cx:.0f}" y="{y0 - 16}" text-anchor="middle">δ<tspan dy="4" font-size="11">{i}</tspan></text>')
out.append(f'<text class="prior__col prior__col--b" x="{x0 + N * (cell + gap) + 92:.0f}" y="{y0 - 16}" text-anchor="middle">B<tspan dy="4" font-size="11">s</tspan></text>')
for s, cs in enumerate(rows):
    y = y0 + s * (cell + gap)
    end = (s + 1) % K == 0
    out.append(f'<g class="prior__row" style="--d:{s}">')
    out.append(f'<text class="prior__row-label" x="{x0 - 14}" y="{y + cell / 2 + 5:.0f}" text-anchor="end">c<tspan dy="-7" font-size="10">0</tspan><tspan dy="11" font-size="10">{s}</tspan></text>')
    for i, v in enumerate(cs):
        x = x0 + i * (cell + gap)
        op = max(0.12, float(v) ** 0.55)
        out.append(f'<rect class="prior__cell" x="{x}" y="{y}" width="{cell}" height="{cell}" rx="7" style="--o:{op:.3f}"/>')
        out.append(f'<text class="prior__val{" prior__val--light" if op < 0.5 else ""}" x="{x + cell / 2:.0f}" y="{y + cell / 2 + 6:.0f}" text-anchor="middle">{label(v)}</text>')
    total = sum(cs)
    out.append(f'<text class="prior__total" x="{x0 + N * (cell + gap) + 92:.0f}" y="{y + cell / 2 + 5:.0f}" text-anchor="middle">{f"{float(total):.3f}".rstrip("0").rstrip(".")}</text>')
    if end:
        out.append(f'<text class="prior__decay" x="{x0 + (s + 1) * (cell + gap) + 4:.0f}" y="{y + cell / 2 + 5:.0f}">× λ</text>')
    out.append('</g>')
out.append('</svg>')
print("".join(out))
