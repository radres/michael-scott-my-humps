import sys, numpy as np, librosa, librosa.display, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
path, out, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
y, sr = librosa.load(path, sr=22050, mono=True, offset=t0, duration=t1 - t0)
C = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr, hop_length=256, fmin=librosa.note_to_hz("C2"), n_bins=60 * 2, bins_per_octave=24)), ref=np.max)
fig, ax = plt.subplots(figsize=(max(12, (t1 - t0) * 1.6), 9))
librosa.display.specshow(C, sr=sr, hop_length=256, x_axis="time", y_axis="cqt_note", fmin=librosa.note_to_hz("C2"), bins_per_octave=24, ax=ax, vmin=-50)
ax.set_xticks(np.arange(0, t1 - t0, 0.5))
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig(out, dpi=80)
