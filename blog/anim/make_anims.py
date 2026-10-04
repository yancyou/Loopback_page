"""Animated figures for the blog (GIF + MP4), drawn in the paper's palette with matplotlib and encoded with ffmpeg.
Usage: python3 make_anims.py [names...]   (default: all)"""
import json, os, shutil, subprocess, sys, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'images')
NAVY, DENSE, ROSE, INK, MUTED, GRID = '#0F4D92', '#767676', '#B64342', '#202020', '#6B6B6B', '#E2E2E2'
BLUE_F, BLUE_S, PEACH_F, PEACH_S, GREEN_F, GREEN_S, GRAY_F = '#D3E4F1', '#6F93B4', '#F1D9CD', '#C98F74', '#DFEBD3', '#6F9A5E', '#EAEAEA'
RAMP = ['#F6CFCB', '#E9A6A1', '#CF7F7A', '#B64342']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.edgecolor': INK, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.titleweight': 'bold', 'axes.titlesize': 13, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
FPS = 10

def ramp(v):
    t = min(1.0, max(0.0, math.sqrt(max(0.0, v)))) * 3; i = min(2, int(t)); f = t - i
    a = np.array(matplotlib.colors.to_rgb(RAMP[i])); b = np.array(matplotlib.colors.to_rgb(RAMP[i + 1])); return tuple(a + (b - a) * f)

def encode(name, frames_dir, w=1000):
    gif = os.path.join(OUT, name + '.gif'); mp4 = os.path.join(OUT, name + '.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(frames_dir, 'f%04d.png'),
                    '-vf', f'scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3', gif], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(frames_dir, 'f%04d.png'),
                    '-vf', 'scale=1200:-2:flags=lanczos,format=yuv420p', '-c:v', 'libx264', '-crf', '22', '-movflags', '+faststart', mp4], check=True)
    shutil.rmtree(frames_dir); print(name, 'gif', os.path.getsize(gif) // 1024, 'KB')

class Frames:
    def __init__(self, name): self.dir = os.path.join(HERE, '_' + name); shutil.rmtree(self.dir, ignore_errors=True); os.makedirs(self.dir); self.i = 0; self.name = name
    def save(self, fig, hold=1):
        for _ in range(hold): fig.savefig(os.path.join(self.dir, f'f{self.i:04d}.png'), dpi=100); self.i += 1
    def done(self): encode(self.name, self.dir)

# ---------- shared model ----------
def coeffs(K, lam, s, focus=0, gain=0.0, pre=False):
    ends = (s // K) if pre else ((s + 1) // K)
    prior = np.array([lam ** (ends - i // K) for i in range(s + 1)])
    z = np.array([gain if (focus > 0 and i // K == focus - 1) else 0.0 for i in range(s + 1)])
    logits = z + np.log(prior); w = np.exp(logits - logits.max()); w /= w.sum()
    return prior, prior.sum() * w, prior.sum()

# ---------- A. three rules as depth grows ----------
def anim_rules():
    K, R, lam = 4, 4, 0.25; N = K * R; bound = K * lam / (1 - lam)
    add = np.arange(1, N + 1, dtype=float); sm_in = 1 / (np.arange(N) + 2); sm_tot = 1 - sm_in; lb = np.array([coeffs(K, lam, s)[2] for s in range(N)])
    F = Frames('anim_rules')
    for s in range(N):
        for sub in range(6 if (s + 1) % K == 0 else 4):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={'wspace': 0.32})
            for ax, title in zip(axes, ['Input coefficient', 'Total weight of the stored updates']):
                ax.set_title(title, loc='left'); ax.set_xlim(-0.5, N - 0.5); ax.set_xticks(range(0, N, 4)); ax.set_xlabel('block application s'); ax.grid(axis='y', color=GRID, lw=0.8)
                for v in range(1, R): ax.axvline(v * K - 0.5, color=GRID, lw=1, ls=':')
            x = np.arange(s + 1); ax0, ax1 = axes
            ax0.set_ylim(0, 1.15); ax0.plot(x, np.ones(s + 1), color=DENSE, lw=5, solid_capstyle='round', label='residual addition'); ax0.plot(x, np.ones(s + 1), color=NAVY, lw=2.2, label='Loopback')
            ax0.plot(x, sm_in[:s + 1], color=ROSE, lw=2.2, label='joint softmax (Attention Residuals)'); ax0.scatter([s], [sm_in[s]], color=ROSE, zorder=5); ax0.scatter([s], [1], color=NAVY, zorder=6, s=28)
            ax0.legend(loc='center right', frameon=False, fontsize=10)
            ax1.set_ylim(0, 6.2); ax1.axhline(bound, color=NAVY, ls='--', lw=1); ax1.text(N - 0.6, bound + 0.12, f'visit-end bound Kλ/(1−λ) = {bound:.2f}', color=NAVY, ha='right', fontsize=10)
            ax1.plot(x, add[:s + 1], color=DENSE, lw=2.2); ax1.plot(x, sm_tot[:s + 1], color=ROSE, lw=2.2); ax1.plot(x, lb[:s + 1], color=NAVY, lw=1.2)
            ve = [i for i in range(s + 1) if (i + 1) % K == 0]; ax1.plot(ve, lb[ve], color=NAVY, lw=2.6, marker='o', ms=5)
            ax1.scatter([s], [min(add[s], 6.1)], color=DENSE, zorder=5, clip_on=False); ax1.scatter([s], [sm_tot[s]], color=ROSE, zorder=5); ax1.scatter([s], [lb[s]], color=NAVY, zorder=6, s=28)
            if add[s] > 6: ax1.annotate(f'addition: {int(add[s])}, off the scale', (s, 6.0), xytext=(-8, -14), textcoords='offset points', ha='right', color=DENSE, fontsize=10)
            fig.text(0.5, 0.955, f'K = {K} shared blocks, R = {R} visits, λ = {lam}   —   after application s = {s}: ' + (f'visit {(s+1)//K} ends, every stored weight × λ' if (s + 1) % K == 0 else f'visit {s//K+1}'), ha='center', fontsize=11.5, color=ROSE if (s + 1) % K == 0 and sub >= 2 else INK)
            fig.subplots_adjust(top=0.82, bottom=0.17, left=0.07, right=0.98); F.save(fig); plt.close(fig)
    F.save(fig if False else plt.figure(figsize=(0.1, 0.1)), 0); F.done()

# ---------- B. the decay prior, row by row ----------
def anim_prior():
    K, R, lam = 2, 3, 0.25; N = K * R
    F = Frames('anim_prior')
    def draw(upto, flash_row=None, pre=False):
        fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_axis_off(); ax.set_xlim(-1.4, N + 4.6); ax.set_ylim(-0.6, N + 0.6)
        ax.text(-0.9, N + 0.25, 'read s', ha='center', va='center', fontsize=11, color=MUTED); ax.text(N + 0.4, N + 0.25, 'total B', va='center', fontsize=11, color=MUTED); 
        for i in range(N): ax.text(i + 0.5, N + 0.25, f'δ{i}', ha='center', va='center', fontsize=12, color=INK)
        for s in range(upto + 1):
            prior, _, B = coeffs(K, lam, s, pre=(pre and s == upto)); y = N - 1 - s
            ax.text(-0.9, y + 0.5, f's = {s}', ha='center', va='center', fontsize=11.5, color=INK)
            for i in range(s + 1):
                c = prior[i]; ax.add_patch(FancyBboxPatch((i + 0.08, y + 0.1), 0.84, 0.8, boxstyle='round,pad=0.01,rounding_size=0.08', fc=ramp(c), ec='white', lw=1))
                e = round(math.log(c, lam)) if c < 1 else 0; ax.text(i + 0.5, y + 0.5, '1' if e == 0 else ('λ' if e == 1 else f'λ{"²³⁴"[e-2]}'), ha='center', va='center', fontsize=13, color='white' if c > 0.4 else INK)
            ax.text(N + 0.4, y + 0.5, f'B = {B:.3f}' + (f'  (= {s+1} × λ{"" if (s+1)//K==1 else ""})' if False else ''), va='center', fontsize=11.5, color=NAVY)
            if flash_row == s: ax.text(N + 2.4, y + 0.5, '× λ', va='center', fontsize=15, color=ROSE, fontweight='bold')
        fig.text(0.5, 0.96, f'K = {K}, R = {R}, λ = {lam}: a new update enters with weight 1; each visit end multiplies every stored weight by λ', ha='center', fontsize=11.5)
        fig.text(0.5, 0.03, 'lighter = smaller prior · B = total update weight of that read · the input is not in the table: its coefficient is always 1', ha='center', fontsize=10, color=MUTED)
        fig.subplots_adjust(top=0.9, bottom=0.08, left=0.03, right=0.99); return fig
    for s in range(N):
        if (s + 1) % K == 0:
            fig = draw(s, pre=True); F.save(fig, 5); plt.close(fig); fig = draw(s, flash_row=s); F.save(fig, 7); plt.close(fig)
        else:
            fig = draw(s); F.save(fig, 5); plt.close(fig)
    fig = draw(N - 1); F.save(fig, 12); plt.close(fig); F.done()

# ---------- C. attention reallocates a conserved total ----------
def anim_alloc():
    K, R, lam = 4, 4, 0.25; N = K * R; s = N - 1
    F = Frames('anim_alloc')
    sched = [(0, 0.0)] * 8 + [(2, g) for g in np.linspace(0, 3, 22)] + [(2, 3.0)] * 8 + [(2, g) for g in np.linspace(3, 0, 12)] + [(4, g) for g in np.linspace(0, 3, 22)] + [(4, 3.0)] * 8 + [(4, g) for g in np.linspace(3, 0, 12)] + [(0, 0.0)] * 8
    for focus, gain in sched:
        prior, post, B = coeffs(K, lam, s, focus=focus, gain=gain)
        fig, ax = plt.subplots(figsize=(10, 4.4)); ax.set_xlim(-1.6, N + 0.5); ax.set_ylim(0, 1.1); ax.set_ylabel('coefficient in the final read'); ax.grid(axis='y', color=GRID, lw=0.8)
        ax.bar([-1.1], [1.0], width=0.7, color=INK); ax.text(-1.1, 1.03, 'input', ha='center', fontsize=10.5)
        for v in range(R): ax.axvspan(v * K - 0.5, v * K + K - 0.5, color=NAVY, alpha=0.05 if v % 2 == 0 else 0.0, lw=0); ax.text(v * K + K / 2 - 0.5, 1.03, f'visit {v+1}', ha='center', fontsize=10.5, color=NAVY if focus == v + 1 and gain > 0.05 else MUTED)
        ax.bar(range(N), prior, width=0.7, fc='none', ec=ROSE, ls='--', lw=1); ax.bar(range(N), post, width=0.7, color=[ramp(p) for p in post], ec=INK, lw=0.5)
        ax.set_xticks(range(N)); ax.set_xticklabels([f'δ{i}' for i in range(N)], fontsize=9.5)
        msg = 'zero queries: the read is the decay prior itself' if gain < 0.05 else f'query favors visit {focus} (score +{gain:.1f}): weight moves there; the total B does not change'
        ax.set_title(f'K = {K}, R = {R}, λ = {lam}, read after the last application   ·   input coefficient 1   ·   total B = {B:.3f} (bound {K*lam/(1-lam):.3f})', loc='left', fontsize=11.5, fontweight='normal')
        fig.text(0.5, 0.05, msg, ha='center', fontsize=11, color=ROSE if gain >= 0.05 else MUTED); fig.text(0.5, 0.012, 'dashed outline = decay prior · filled = after attention', ha='center', fontsize=9.5, color=MUTED)
        fig.subplots_adjust(top=0.86, bottom=0.24, left=0.08, right=0.98); F.save(fig); plt.close(fig)
    F.done()

# ---------- D. training curves drawn progressively ----------
def anim_train():
    d = json.load(open('/home/sigma/xuano/figure/train_loss/train_loss_data.json')); scales = ['191M', '374M', '778M', '1.4B']
    series = {}
    for sc in scales:
        tps = d[sc]['tokens_per_step']; runs = d[sc]['runs']; ser = {}
        for name, r in runs.items():
            st = np.array(sorted(int(k) for k in r)); y = np.array([r[str(k)] for k in st]); x = st * tps / 1e9
            w = 25; ys = np.convolve(y, np.ones(w) / w, mode='valid'); xs = x[w - 1:]   # light smoothing, same for both runs
            ser[name] = (xs, ys)
        series[sc] = ser
    gaps = {'191M': -0.0046, '374M': -0.0235, '778M': -0.0192, '1.4B': -0.0212}
    F = Frames('anim_train'); steps = 50
    for t in range(steps + 1):
        frac = (t / steps) ** 1.4
        fig, axes = plt.subplots(1, 4, figsize=(10, 3.6), gridspec_kw={'wspace': 0.42})
        for ax, sc in zip(axes, scales):
            ser = series[sc]; names = list(ser.keys()); dense = [n for n in names if 'ense' in n][0]; loop = [n for n in names if n != dense][0]
            xmax = max(ser[dense][0][-1], ser[loop][0][-1]); cut = xmax * frac
            for name, color, lw in ((dense, DENSE, 1.6), (loop, NAVY, 2.0)):
                xs, ys = ser[name]; m = xs <= max(cut, xs[0] + 1e-9); ax.plot(xs[m], ys[m], color=color, lw=lw, label='Dense' if name == dense else 'Loopback')
            ax.set_title(sc, loc='left'); ax.set_xlim(0, xmax * 1.02); lo = min(ser[dense][1].min(), ser[loop][1].min()); ax.set_ylim(lo - 0.05, lo + 0.9); ax.set_xlabel('tokens (B)'); ax.grid(color=GRID, lw=0.6)
            if t == steps: ax.text(0.97, 0.9, f'Δ = {gaps[sc]:+.4f}', transform=ax.transAxes, ha='right', color=NAVY, fontweight='bold', fontsize=11)
        axes[0].set_ylabel('training loss'); axes[0].legend(frameon=False, fontsize=10, loc='lower left')
        fig.text(0.5, 0.96, 'Training loss, Dense vs Loopback at four scales (same tokens, same depth); Δ = final gap, Loopback − Dense', ha='center', fontsize=11.5)
        fig.subplots_adjust(top=0.82, bottom=0.2, left=0.07, right=0.98); F.save(fig, 12 if t == steps else 1); plt.close(fig)
    F.done()

# ---------- E. parameters and quality at four scales ----------
def anim_scaling():
    rows = [('191M', 141.59, 56.65, 2.6234, 2.6187, 39.81, 40.16), ('374M', 308.33, 179.88, 2.4198, 2.3971, 43.28, 44.27), ('778M', 679.55, 453.06, 2.2521, 2.2334, 47.83, 48.73), ('1.4B', 1284.61, 873.57, 2.1283, 2.1073, 52.11, 52.98)]
    F = Frames('anim_scaling'); steps = 30
    for t in range(steps + 1):
        e = 1 - (1 - t / steps) ** 3
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={'wspace': 0.3, 'width_ratios': [1.25, 1]})
        ax = axes[0]; x = np.arange(4); ax.set_title('Transformer parameters (M)', loc='left'); ax.grid(axis='y', color=GRID, lw=0.8)
        ax.bar(x - 0.2, [r[1] * e for r in rows], width=0.38, color=DENSE, label='Dense'); ax.bar(x + 0.2, [r[2] * e for r in rows], width=0.38, color=NAVY, label='Loopback')
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 1450); ax.legend(frameon=False, loc='upper left')
        if t == steps:
            for i, r in enumerate(rows): ax.text(i + 0.2, r[2] + 25, f'{100*r[2]/r[1]:.0f}%', ha='center', color=NAVY, fontweight='bold', fontsize=11)
        ax = axes[1]; ax.set_title('Validation loss (lower is better)', loc='left'); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_ylim(2.05, 2.7)
        for i, r in enumerate(rows):
            ax.plot([i - 0.2, i + 0.2], [r[3], r[3] + (r[4] - r[3]) * e], color=GRID, lw=2, zorder=1); ax.scatter([i - 0.2], [r[3]], color=DENSE, s=60, zorder=3); ax.scatter([i + 0.2], [r[3] + (r[4] - r[3]) * e], color=NAVY, s=60, zorder=4)
            if t == steps: ax.text(i + 0.2, r[4] - 0.03, f'{r[4]:.4f}', ha='center', color=NAVY, fontsize=10, fontweight='bold'); ax.text(i - 0.2, r[3] + 0.018, f'{r[3]:.4f}', ha='center', color=DENSE, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
        fig.text(0.5, 0.96, 'Same tokens, same depth: 40–68% of the Transformer parameters, lower loss at every scale', ha='center', fontsize=11.5)
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.98); F.save(fig, 14 if t == steps else 1); plt.close(fig)
    F.done()

ALL = {'anim_rules': anim_rules, 'anim_prior': anim_prior, 'anim_alloc': anim_alloc, 'anim_train': anim_train, 'anim_scaling': anim_scaling}
if __name__ == '__main__':
    for n in (sys.argv[1:] or ALL): ALL[n]()
