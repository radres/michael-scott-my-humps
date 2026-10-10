import sys, numpy as np, librosa, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
path, out = sys.argv[1], sys.argv[2]
marks = [float(x) for x in sys.argv[3:]]
y, sr = librosa.load(path, sr=11025, mono=True)
S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=2048)[0]
t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=2048)
fig, (a, b) = plt.subplots(2, 1, figsize=(24, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
a.imshow(S[:400], origin="lower", aspect="auto", extent=[0, len(y) / sr, 0, 400 * sr / 2048], vmin=-70, cmap="magma")
b.plot(t, 20 * np.log10(rms + 1e-9)); b.set_ylim(-50, 0)
for mk in marks:
    a.axvline(mk, color="cyan", lw=1); b.axvline(mk, color="red", lw=1)
plt.tight_layout(); plt.savefig(out, dpi=60)
