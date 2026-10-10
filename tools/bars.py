import sys, numpy as np, soundfile as sf, librosa
x, sr = sf.read(sys.argv[1]); m = x.mean(1)
t0g, barg = float(sys.argv[2]), float(sys.argv[3])
def seg(t, d): return m[int(t * sr): int((t + d) * sr)]
ref = seg(t0g + 4 * barg, 3 * barg)
i0 = int((t0g + 4 * barg) * sr)
lo, hi = int((4 * barg - 0.08) * sr), int((4 * barg + 0.08) * sr)
L = len(ref)
win = m[i0 + lo: i0 + hi + L]
c = np.correlate(win, ref, mode="valid")
norm = np.sqrt(np.convolve(win ** 2, np.ones(L), mode="valid") * (ref ** 2).sum())
k = np.argmax(c / norm)
period4 = (lo + k) / sr
print("4-bar period %.5f s -> bar %.5f s, bpm %.4f, corr %.4f" % (period4, period4 / 4, 960 / period4, (c / norm)[k]))
bar = period4 / 4
oenv = librosa.onset.onset_strength(y=m.astype(np.float32), sr=sr, hop_length=64)
ot = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=64)
on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=64, units="time", backtrack=False)
print("first onsets", np.round(on[:6], 4))
t0 = on[0]
res = []
for o in on:
    k = round((o - t0) / (bar / 8))
    res.append(o - (t0 + k * bar / 8))
res = np.array(res)
print("8th-grid residual ms: mean %.2f median %.2f sd %.2f" % (res.mean() * 1000, np.median(res) * 1000, res.std() * 1000))
t0c = t0 + np.median(res)
print("t0 %.4f  downbeats:" % t0c, " ".join("%.3f" % (t0c + i * bar) for i in range(13)))
