import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, soundfile as sf, librosa
import extend as E
y = E.load(); db = E.downbeats(y)
out, sr = sf.read(E.OUT / f"{E.NAME}.wav")
pre = int(E.PRE * sr)
n_same = int(round(db[11] * sr)) - int(round(db[0] * sr))
src0 = y[int(round(db[0] * sr)) - pre:]
print("first %.2fs identical to source: max diff %.2e" % (n_same / sr, np.abs(out[:n_same] - src0[:n_same]).max()))
rs = E.runs(E.SONG)
bar = np.diff(db).mean()
t, joins = 0.0, []
for r in rs[:-1]:
    t += len(r) * bar
    joins.append((t, r[-1], None))
for i, r in enumerate(rs[1:]):
    joins[i] = (joins[i][0], joins[i][1], r[0])
d = np.abs(np.diff(out))
hf = np.abs(np.diff(out, 2))
glob = np.percentile(hf, 99.9)
print("global 99.9%% 2nd-diff %.4f" % glob)
for tj, a, b in joins[:8]:
    i = int((tj - E.PRE) * sr)
    w = hf[i - 2000: i + 2000]
    print(f"  join {a:2d}->{b:2d} at {tj:7.2f}s  max 2nd-diff near cut {w.max():.4f} ({w.max() / glob:.2f}x of global 99.9%)")
oenv = librosa.onset.onset_strength(y=out.astype(np.float32), sr=sr, hop_length=64)
on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=64, units="time")
step = bar / 8
g0 = E.PRE
r = on - (g0 + np.round((on - g0) / step) * step)
print("onset residual vs 8th grid across whole output: median %.1f ms, sd %.1f ms" % (np.median(r) * 1000, r.std() * 1000))
r0 = librosa.onset.onset_detect(y=y.astype(np.float32), sr=sr, hop_length=64, units="time")
rr = r0 - (db[0] + np.round((r0 - db[0]) / step) * step)
print("same metric on original ringtone:            median %.1f ms, sd %.1f ms" % (np.median(rr) * 1000, rr.std() * 1000))
