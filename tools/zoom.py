import sys, numpy as np, librosa, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
path, out, t0, t1, g0, g = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:7])
lo, hi = int(sys.argv[7]), int(sys.argv[8])
y, sr = librosa.load(path, sr=22050, mono=True)
hop = 128
C = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.midi_to_hz(lo), n_bins=(hi - lo) * 3, bins_per_octave=36)), ref=np.max)
tt = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop)
m = (tt >= t0) & (tt <= t1)
fig, ax = plt.subplots(figsize=(22, 10))
ax.imshow(C[:, m], origin="lower", aspect="auto", extent=[t0, t1, lo - 0.5, hi - 0.5], vmin=-45, cmap="magma")
for k in range(int((t1 - g0) / g) + 2):
    x = g0 + k * g
    if t0 <= x <= t1:
        ax.axvline(x, color="cyan" if k % 16 == 0 else ("white" if k % 4 == 0 else "gray"), lw=1.5 if k % 4 == 0 else 0.4, alpha=0.7)
        if k % 4 == 0: ax.text(x, hi - 1, str((k // 4) % 4 + 1) if k % 16 else f"bar{k // 16}", color="cyan", fontsize=9)
ax.set_yticks(range(lo, hi)); ax.set_yticklabels([librosa.midi_to_note(p) for p in range(lo, hi)], fontsize=7)
ax.grid(True, axis="y", alpha=0.15)
plt.tight_layout(); plt.savefig(out, dpi=70)
