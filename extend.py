import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SRC_URL = "https://soundcloud.com/kieranbellamy/michael-scott-the-office-my"
SRC = ROOT / "source" / "sc" / "sc.mp3"
OUT = ROOT / "out"
NAME = "my-humps-extended"
SR = 44100
PRE = 0.045
XF = 0.025
FADE_BARS = 2

HOOK = [0, 1, 2, 3]
CH1 = [4, 5, 6, 7]
CH2 = [8, 9, 10, 7]
VAMP = [0, 1, 2, 1, 2, 1, 2, 3]
SONG = (HOOK + CH1 + CH2) + (VAMP + CH1 + CH2 + HOOK + CH1 + CH2) * 3


def load():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["yt-dlp", "-q", "-f", "http_mp3_0_0", "-o", str(SRC), SRC_URL], check=True)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(SRC), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def downbeats(y):
    t0 = np.argmax(np.abs(y) > 0.02) / SR
    guess = 60 / 124 * 4
    a = y[int((t0 + 4 * guess) * SR) : int((t0 + 5 * guess) * SR)]
    lo, hi = int((t0 + 8 * guess - 0.06) * SR), int((t0 + 8 * guess + 0.06) * SR)
    win = y[lo : hi + len(a)]
    c = np.correlate(win, a, "valid") / np.sqrt(np.convolve(win**2, np.ones(len(a)), "valid") * (a**2).sum())
    bar = ((lo + np.argmax(c)) / SR - (t0 + 4 * guess)) / 4
    n = int((len(y) / SR - t0) / bar) + 1
    print(f"t0 {t0:.4f}s  bar {bar:.5f}s  bpm {240 / bar:.3f}  usable bars {n}")
    return t0 + bar * np.arange(n + 1)


def runs(seq):
    out = [[seq[0]]]
    for b in seq[1:]:
        if b == out[-1][-1] + 1:
            out[-1].append(b)
        else:
            out.append([b])
    return out


def splice(y, db, seq):
    pre, xf = int(PRE * SR), int(XF * SR)
    pos = [int(round(t * SR)) for t in db]
    chunks = []
    for r in runs(seq):
        a, b = pos[r[0]] - pre, pos[r[-1] + 1] - pre
        chunks.append(y[max(a, 0) : b + xf].copy())
    fade_in, fade_out = np.linspace(0, 1, xf), np.linspace(1, 0, xf)
    total = sum(len(c) - xf for c in chunks) + xf
    out = np.zeros(total)
    cur = 0
    for i, c in enumerate(chunks):
        if i:
            c[:xf] *= fade_in
        c[-xf:] *= fade_out
        out[cur : cur + len(c)] += c
        cur += len(c) - xf
    return out[: cur + xf]


def main():
    OUT.mkdir(exist_ok=True)
    y = load()
    db = downbeats(y)
    assert (max(SONG) + 2) * SR * float(np.diff(db).mean()) < len(y) + SR
    out = splice(y, db, SONG)
    bar = float(np.median(np.diff(db)))
    n = int(FADE_BARS * bar * SR)
    out[-n:] *= np.cos(np.linspace(0, np.pi / 2, n)) ** 2
    print(f"{len(SONG)} bars, {len(out) / SR:.1f}s")
    wav = OUT / f"{NAME}.wav"
    sf.write(wav, out, SR, subtype="FLOAT")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "2", "-c:a", "libmp3lame", "-q:a", "0",
            "-metadata", "title=My Humps (Michael Scott ringtone, extended)", str(ROOT / f"{NAME}.mp3"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
