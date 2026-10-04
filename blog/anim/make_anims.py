"""Animated figures for the blog: 2000x880 frames at 30 fps -> MP4 (crisp, for the HTML page) and a 1400-px GIF at 15 fps
(for Notion image blocks, which autoplay GIFs). Motion is tweened per frame rather than held, so playback is smooth.
Usage: python3 make_anims.py [names...]   (default: all)"""
import json, os, shutil, subprocess, sys, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'images')
NAVY, DENSE, ROSE, INK, MUTED, GRID = '#0F4D92', '#767676', '#B64342', '#202020', '#6B6B6B', '#E2E2E2'
RAMP = ['#F6CFCB', '#E9A6A1', '#CF7F7A', '#B64342']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.edgecolor': INK, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.titleweight': 'bold', 'axes.titlesize': 13, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'path.simplify': True})
FPS, DPI = 30, 200

def ease(u): return 1 - (1 - min(1.0, max(0.0, u))) ** 3
def ramp(v):
    t = min(1.0, max(0.0, math.sqrt(max(0.0, v)))) * 3; i = min(2, int(t)); f = t - i
    a = np.array(matplotlib.colors.to_rgb(RAMP[i])); b = np.array(matplotlib.colors.to_rgb(RAMP[i + 1])); return tuple(a + (b - a) * f)

def encode(name, frames_dir, gif_width=1400, gif_fps=15):
    mp4 = os.path.join(OUT, name + '.mp4'); gif = os.path.join(OUT, name + '.gif'); poster = os.path.join(OUT, name + '_poster.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(frames_dir, 'f%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-movflags', '+faststart', mp4], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', os.path.join(frames_dir, 'f%04d.png'),
                    '-vf', f'fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle', gif], check=True)
    shutil.copy(os.path.join(frames_dir, 'f0000.png'), poster)
    shutil.rmtree(frames_dir); print(name, 'mp4', os.path.getsize(mp4) // 1024, 'KB, gif', os.path.getsize(gif) // 1024, 'KB')

class Frames:
    def __init__(self, name): self.dir = os.path.join(HERE, '_' + name); shutil.rmtree(self.dir, ignore_errors=True); os.makedirs(self.dir); self.i = 0; self.name = name
    def save(self, fig, hold=1):
        for _ in range(hold): fig.savefig(os.path.join(self.dir, f'f{self.i:04d}.png'), dpi=DPI); self.i += 1
    def done(self): encode(self.name, self.dir)

def coeffs(K, lam, s, focus=0, gain=0.0, pre=False):
    ends = (s // K) if pre else ((s + 1) // K)
    prior = np.array([lam ** (ends - i // K) for i in range(s + 1)])
    z = np.array([gain if (focus > 0 and i // K == focus - 1) else 0.0 for i in range(s + 1)])
    logits = z + np.log(prior); w = np.exp(logits - logits.max()); w /= w.sum()
    return prior, prior.sum() * w, prior.sum()

# ---------- A. three rules as depth grows (continuous reveal) ----------
def anim_rules():
    K, R, lam = 4, 4, 0.25; N = K * R; bound = K * lam / (1 - lam)
    add = np.arange(1, N + 1, dtype=float); sm_in = 1 / (np.arange(N) + 2); sm_tot = 1 - sm_in; lb = np.array([coeffs(K, lam, s)[2] for s in range(N)])
    def interp(vals, t):   # piecewise-linear value at fractional position t
        i = int(math.floor(t)); f = t - i
        return vals[i] if i >= N - 1 else vals[i] + (vals[i + 1] - vals[i]) * f
    F = Frames('anim_rules'); total_s = 12.0; n_frames = int(total_s * FPS)
    for fr in range(n_frames + FPS * 2):
        t = min(N - 1, (fr / n_frames) * (N - 1)); s = int(math.floor(t + 1e-9)); flash = ((s + 1) % K == 0) and (t - s < 0.35 or s == N - 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={'wspace': 0.32})
        for ax, title in zip(axes, ['Input coefficient', 'Total weight of the stored updates']):
            ax.set_title(title, loc='left'); ax.set_xlim(-0.5, N - 0.5); ax.set_xticks(range(0, N, 4)); ax.set_xlabel('block application s'); ax.grid(axis='y', color=GRID, lw=0.8)
            for v in range(1, R): ax.axvline(v * K - 0.5, color=GRID, lw=1, ls=':')
        ax0, ax1 = axes; xs = list(np.arange(0, s + 1, dtype=float)) + ([t] if t > s else [])
        ax0.set_ylim(0, 1.15); ax0.plot(xs, [1] * len(xs), color=DENSE, lw=5, solid_capstyle='round', label='residual addition'); ax0.plot(xs, [1] * len(xs), color=NAVY, lw=2.2, label='Loopback')
        ax0.plot(xs, [interp(sm_in, x) for x in xs], color=ROSE, lw=2.2, label='joint softmax (Attention Residuals)')
        ax0.scatter([t], [interp(sm_in, t)], color=ROSE, zorder=5, s=36); ax0.scatter([t], [1], color=NAVY, zorder=6, s=30); ax0.legend(loc='center right', frameon=False, fontsize=10)
        ax1.set_ylim(0, 6.2); ax1.axhline(bound, color=NAVY, ls='--', lw=1); ax1.text(N - 0.6, bound + 0.12, f'visit-end bound Kλ/(1−λ) = {bound:.2f}', color=NAVY, ha='right', fontsize=10)
        ax1.plot(xs, [interp(add, x) for x in xs], color=DENSE, lw=2.2); ax1.plot(xs, [interp(sm_tot, x) for x in xs], color=ROSE, lw=2.2); ax1.plot(xs, [interp(lb, x) for x in xs], color=NAVY, lw=1.2)
        ve = [i for i in range(s + 1) if (i + 1) % K == 0]; ax1.plot(ve, lb[ve], color=NAVY, lw=2.6, marker='o', ms=5)
        ax1.scatter([t], [min(interp(add, t), 6.1)], color=DENSE, zorder=5, s=36, clip_on=False); ax1.scatter([t], [interp(sm_tot, t)], color=ROSE, zorder=5, s=36); ax1.scatter([t], [interp(lb, t)], color=NAVY, zorder=6, s=30)
        if interp(add, t) > 6: ax1.annotate(f'addition: {int(round(interp(add, t)))}, off the scale', (t, 6.0), xytext=(-8, -14), textcoords='offset points', ha='right', color=DENSE, fontsize=10)
        fig.text(0.5, 0.955, f'K = {K} shared blocks, R = {R} visits, λ = {lam}   —   application s = {s}, visit {min(R, s // K + 1)}' + ('   ·   visit end: every stored weight × λ' if flash else ''), ha='center', fontsize=11.5, color=ROSE if flash else INK)
        fig.subplots_adjust(top=0.82, bottom=0.17, left=0.07, right=0.98); F.save(fig); plt.close(fig)
    F.done()

# ---------- B. the decay prior, row by row (fade-in cells, decay flash) ----------
def anim_prior():
    K, R, lam = 2, 3, 0.25; N = K * R
    F = Frames('anim_prior')
    def draw(s_now, u_new, decay_u, flash):
        """rows 0..s_now; the newest row's new cell fades in with u_new; decay_u in [0,1] blends the pre-decay and decayed values of the newest row"""
        fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_axis_off(); ax.set_xlim(-1.4, N + 4.6); ax.set_ylim(-0.6, N + 0.6)
        ax.text(-0.9, N + 0.25, 'read s', ha='center', va='center', fontsize=11, color=MUTED); ax.text(N + 0.4, N + 0.25, 'total B', va='center', fontsize=11, color=MUTED)
        for i in range(N): ax.text(i + 0.5, N + 0.25, f'δ{i}', ha='center', va='center', fontsize=12, color=INK)
        for s in range(s_now + 1):
            pre, _, _ = coeffs(K, lam, s, pre=True); post, _, _ = coeffs(K, lam, s); y = N - 1 - s
            vals = post if s < s_now else pre + (post - pre) * decay_u
            ax.text(-0.9, y + 0.5, f's = {s}', ha='center', va='center', fontsize=11.5, color=INK)
            for i in range(s + 1):
                c = vals[i]; alpha = 1.0 if not (s == s_now and i == s) else u_new
                ax.add_patch(FancyBboxPatch((i + 0.08, y + 0.1), 0.84, 0.8, boxstyle='round,pad=0.01,rounding_size=0.08', fc=ramp(c), ec='white', lw=1, alpha=alpha))
                e = int(round(math.log(c, lam))) if c < 0.999 else 0
                ax.text(i + 0.5, y + 0.5, '1' if e == 0 else ('λ' if e == 1 else 'λ' + '²³⁴'[e - 2]), ha='center', va='center', fontsize=13, color='white' if c > 0.4 else INK, alpha=alpha)
            ax.text(N + 0.4, y + 0.5, f'B = {vals.sum():.3f}', va='center', fontsize=11.5, color=NAVY)
            if flash and s == s_now: ax.text(N + 2.4, y + 0.5, '× λ', va='center', fontsize=15, color=ROSE, fontweight='bold')
        fig.text(0.5, 0.96, f'K = {K}, R = {R}, λ = {lam}: a new update enters with weight 1; each visit end multiplies every stored weight by λ', ha='center', fontsize=11.5)
        fig.text(0.5, 0.03, 'lighter = smaller prior · B = total update weight of that read · the input is not in the table: its coefficient is always 1', ha='center', fontsize=10, color=MUTED)
        fig.subplots_adjust(top=0.9, bottom=0.08, left=0.03, right=0.99); return fig
    for s in range(N):
        for fr in range(int(0.4 * FPS)):   # new cell fades in
            fig = draw(s, ease(fr / (0.4 * FPS)), 0.0, False); F.save(fig); plt.close(fig)
        if (s + 1) % K == 0:
            fig = draw(s, 1.0, 0.0, True); F.save(fig, int(0.5 * FPS)); plt.close(fig)
            for fr in range(int(0.6 * FPS)):   # decay blends in
                fig = draw(s, 1.0, ease(fr / (0.6 * FPS)), True); F.save(fig); plt.close(fig)
            fig = draw(s, 1.0, 1.0, True); F.save(fig, int(0.5 * FPS)); plt.close(fig)
        else:
            fig = draw(s, 1.0, 1.0, False); F.save(fig, int(0.7 * FPS)); plt.close(fig)
    fig = draw(N - 1, 1.0, 1.0, False); F.save(fig, 2 * FPS); plt.close(fig); F.done()

# ---------- C. attention reallocates a conserved total ----------
def anim_alloc():
    K, R, lam = 4, 4, 0.25; N = K * R; s = N - 1
    F = Frames('anim_alloc')
    def seg(f, g, n): return [(f, g)] * n
    sched = seg(0, 0.0, FPS) + [(2, 3 * ease(i / (2.5 * FPS))) for i in range(int(2.5 * FPS))] + seg(2, 3.0, FPS) + [(2, 3 * (1 - ease(i / (1.2 * FPS)))) for i in range(int(1.2 * FPS))] \
          + [(4, 3 * ease(i / (2.5 * FPS))) for i in range(int(2.5 * FPS))] + seg(4, 3.0, FPS) + [(4, 3 * (1 - ease(i / (1.2 * FPS)))) for i in range(int(1.2 * FPS))] + seg(0, 0.0, FPS)
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
        tps = d[sc]['tokens_per_step']; ser = {}
        for name, r in d[sc]['runs'].items():
            st = np.array(sorted(int(k) for k in r)); y = np.array([r[str(k)] for k in st]); x = st * tps / 1e9
            w = 25; ys = np.convolve(y, np.ones(w) / w, mode='valid'); xs = x[w - 1:]
            ser[name] = (xs, ys)
        series[sc] = ser
    gaps = {'191M': -0.0046, '374M': -0.0235, '778M': -0.0192, '1.4B': -0.0212}
    F = Frames('anim_train'); n = int(6 * FPS)
    for t in range(n + 1):
        frac = (t / n) ** 1.4
        fig, axes = plt.subplots(1, 4, figsize=(10, 3.6), gridspec_kw={'wspace': 0.42})
        for ax, sc in zip(axes, scales):
            ser = series[sc]; dense = 'Dense'; loop = [k for k in ser if k != 'Dense'][0]
            xmax = max(ser[dense][0][-1], ser[loop][0][-1]); cut = xmax * frac
            for name, color, lw in ((dense, DENSE, 1.6), (loop, NAVY, 2.0)):
                xs, ys = ser[name]; m = xs <= max(cut, xs[0] + 1e-9); ax.plot(xs[m], ys[m], color=color, lw=lw, label='Dense' if name == dense else 'Loopback')
            ax.set_title(sc, loc='left'); ax.set_xlim(0, xmax * 1.02); lo = min(ser[dense][1].min(), ser[loop][1].min()); ax.set_ylim(lo - 0.05, lo + 0.9); ax.set_xlabel('tokens (B)'); ax.grid(color=GRID, lw=0.6)
            if t == n: ax.text(0.97, 0.9, f'Δ = {gaps[sc]:+.4f}', transform=ax.transAxes, ha='right', color=NAVY, fontweight='bold', fontsize=11)
        axes[0].set_ylabel('training loss'); axes[0].legend(frameon=False, fontsize=10, loc='lower left')
        fig.text(0.5, 0.96, 'Training loss, Dense vs Loopback at four scales (same tokens, same depth); Δ = final gap, Loopback − Dense', ha='center', fontsize=11.5)
        fig.subplots_adjust(top=0.82, bottom=0.2, left=0.07, right=0.98); F.save(fig, 3 * FPS if t == n else 1); plt.close(fig)
    F.done()

# ---------- E. parameters and quality at four scales ----------
def anim_scaling():
    rows = [('191M', 141.59, 56.65, 2.6234, 2.6187), ('374M', 308.33, 179.88, 2.4198, 2.3971), ('778M', 679.55, 453.06, 2.2521, 2.2334), ('1.4B', 1284.61, 873.57, 2.1283, 2.1073)]
    F = Frames('anim_scaling'); n = int(2.5 * FPS)
    for t in range(n + 1):
        e = ease(t / n)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={'wspace': 0.3, 'width_ratios': [1.25, 1]})
        ax = axes[0]; x = np.arange(4); ax.set_title('Transformer parameters (M)', loc='left'); ax.grid(axis='y', color=GRID, lw=0.8)
        ax.bar(x - 0.2, [r[1] * e for r in rows], width=0.38, color=DENSE, label='Dense'); ax.bar(x + 0.2, [r[2] * e for r in rows], width=0.38, color=NAVY, label='Loopback')
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 1450); ax.legend(frameon=False, loc='upper left')
        if t == n:
            for i, r in enumerate(rows): ax.text(i + 0.2, r[2] + 25, f'{100*r[2]/r[1]:.0f}%', ha='center', color=NAVY, fontweight='bold', fontsize=11)
        ax = axes[1]; ax.set_title('Validation loss (lower is better)', loc='left'); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_ylim(2.05, 2.7)
        for i, r in enumerate(rows):
            yl = r[3] + (r[4] - r[3]) * e
            ax.plot([i - 0.2, i + 0.2], [r[3], yl], color=GRID, lw=2, zorder=1); ax.scatter([i - 0.2], [r[3]], color=DENSE, s=60, zorder=3); ax.scatter([i + 0.2], [yl], color=NAVY, s=60, zorder=4)
            if t == n: ax.text(i + 0.2, r[4] - 0.03, f'{r[4]:.4f}', ha='center', color=NAVY, fontsize=10, fontweight='bold'); ax.text(i - 0.2, r[3] + 0.018, f'{r[3]:.4f}', ha='center', color=DENSE, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
        fig.text(0.5, 0.96, 'Same tokens, same depth: 40–68% of the Transformer parameters, lower loss at every scale', ha='center', fontsize=11.5)
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.98); F.save(fig, 3 * FPS if t == n else 1); plt.close(fig)
    F.done()

ALL = {'anim_rules': anim_rules, 'anim_prior': anim_prior, 'anim_alloc': anim_alloc, 'anim_train': anim_train, 'anim_scaling': anim_scaling}
if __name__ == '__main__':
    for n in (sys.argv[1:] or ALL): ALL[n]()
