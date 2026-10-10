import sys, numpy as np, soundfile as sf, librosa
x, sr = sf.read(sys.argv[1]); m = x.mean(1).astype(np.float32)
oenv = librosa.onset.onset_strength(y=m, sr=sr, hop_length=64)
on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=64, units="time")
st = 60 / 124.03 / 4
t0 = on[0]
for it in range(3):
    k = np.round((on - t0) / st)
    A = np.stack([np.ones_like(k), k], 1)
    (t0, st), *_ = np.linalg.lstsq(A, on, rcond=None)
    r = on - (t0 + k * st)
    keep = np.abs(r) < 0.02
    on, k = on[keep], k[keep]
bar = st * 16
print("16th %.6f  bar %.6f  bpm %.4f  t0 %.5f  resid sd %.2f ms  n=%d" % (st, bar, 60 / (st * 4), t0, r[keep].std() * 1000, keep.sum()))
S = librosa.power_to_db(librosa.feature.melspectrogram(y=m, sr=sr, n_fft=2048, hop_length=256, n_mels=96))
fpb = bar * sr / 256
feats = []
for b in range(12):
    i = int(round((t0 + b * bar) * sr / 256))
    feats.append(S[:, i:i + int(fpb)][:, : int(fpb) - 1].ravel())
n = min(len(f) for f in feats)
F = np.array([f[:n] for f in feats]); F = (F - F.mean(1, keepdims=True)) / F.std(1, keepdims=True)
C = F @ F.T / n
print("bar similarity (x100):")
print("    " + " ".join("%4d" % j for j in range(12)))
for i in range(12):
    print("%3d " % i + " ".join("%4d" % round(100 * C[i, j]) for j in range(12)))
