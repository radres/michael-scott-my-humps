import sys, numpy as np, librosa
path = sys.argv[1]
y, sr = librosa.load(path, sr=22050, mono=True)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
bt = librosa.frames_to_time(beats, sr=sr, hop_length=256)
print("tempo", tempo, "beats", np.round(bt, 3).tolist())
hop = 256
C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C2"), n_bins=72, bins_per_octave=12))
D = librosa.amplitude_to_db(C, ref=np.max)
times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop)
step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.125
names = [librosa.midi_to_note(36 + i) for i in range(72)]
t = 0.0
while t < times[-1]:
    m = (times >= t) & (times < t + step)
    if m.any():
        v = D[:, m].mean(axis=1)
        def top(lo, hi, n):
            idx = np.arange(lo, hi)
            sel = idx[np.argsort(v[lo:hi])[::-1][:n]]
            return " ".join(f"{names[i]}({v[i]:.0f})" for i in sorted(sel) if v[i] > -38)
        print(f"{t:6.3f} | bass {top(0, 18, 1):12s} | mid {top(18, 30, 3):30s} | mel {top(30, 54, 2)}")
    t += step
