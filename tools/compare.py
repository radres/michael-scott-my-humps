import sys, numpy as np, soundfile as sf
from scipy.signal import correlate, welch
a, sr = sf.read(sys.argv[1]); b, _ = sf.read(sys.argv[2])
for name, x in (("A", a), ("B", b)):
    m = x.mean(1)
    f, p = welch(m, sr, nperseg=4096)
    tot = p.sum()
    print(name, "len %.3f" % (len(x) / sr), "peak %.3f" % np.abs(x).max(),
          "LR corr %.4f" % np.corrcoef(x[:, 0], x[:, 1])[0, 1],
          "energy>11k %.2e" % (p[f > 11000].sum() / tot), ">15k %.2e" % (p[f > 15000].sum() / tot), ">16.5k %.2e" % (p[f > 16500].sum() / tot),
          "first sound %.3f" % (np.argmax(np.abs(m) > 0.01) / sr))
ma, mb = a.mean(1), b.mean(1)
c = correlate(ma, mb, mode="full", method="fft")
lag = np.argmax(c) - (len(mb) - 1)
print("lag A vs B (s): %.4f" % (lag / sr))
