import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
BPM = 124
SPB = 60 / BPM
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
NAME = "my-humps-synth-remix"
rng = np.random.default_rng(124)

LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def m(name):
    acc = {"#": 1, "b": -1}.get(name[1], 0)
    return 12 * (int(name[2:] if acc else name[1:]) + 1) + LETTERS[name[0]] + acc


def hz(p):
    return 440.0 * 2 ** ((np.asarray(p, dtype=float) - 69) / 12)


def ph(text):
    out = []
    for tok in text.split():
        f = tok.split(":")
        out.append((float(f[0]), float(f[1]), m(f[2]), float(f[3]) if len(f) > 3 else 1.0, float(f[4]) if len(f) > 4 else 0.0))
    return out


def shift(seq, beats=0.0, tr=0, vel=1.0, stretch=1.0):
    return [(b * stretch + beats, d * stretch, p + tr, v * vel, g) for b, d, p, v, g in seq]


HOOK = ph("1:1:A4 2:1:B4 3:1:C5 4:1.5:B4 5.5:2:A4 8.5:.5:A4 9:1.75:B4 12.5:.5:A4 13:1:B4 14:.5:F5 14.5:.5:E5 15:.5:F5 15.5:.5:E5")
HOOK_WHOOP = ph(
    "1.5:.4:G5:.8:-3 2.5:.4:G5:.8:-3 3.5:.4:G5:.8:-3 "
    "4:.5:G5:.9:2 5.5:.4:G5:.8:-3 6:.25:E5:.6 6.5:.4:G5:.8:-3 7.5:.25:E5:.6 "
    "8:.5:D5:.9:2 9:.4:G5:.8:-3 10:.4:G5:.8:-3 11:.4:G5:.8:-3 "
    "12:.5:G5:.9:2"
)
CHORUS_HEAD = "1:1:A4 2:1:B4 3:1:C5 "
CHORUS_FILL = "6.5:.5:G5 7:.5:A5 7.5:.5:E5 8:.5:A5 8.5:.5:A5 9:.5:A5 9.5:.5:A5 10:.5:F5 10.5:.5:E5 11:.5:F5 11.5:.5:E5 "
CHORUS_A = ph(CHORUS_HEAD + "4:1:B4 5.5:2.5:A4 " + CHORUS_FILL + "12.5:.5:D5 13:.5:D5 13.5:.5:E5 14:.5:F5 14.5:.5:E5 15:.5:F5 15.5:.75:E5")
CHORUS_B = ph(CHORUS_HEAD + "4:.5:B4 4.5:1:B4 5.5:2.5:A4 " + CHORUS_FILL + "12.5:.5:F5 13:.5:F5 13.5:1:E5 14.5:.5:D5 15:.5:C5 15.5:.5:B4")
VERSE = ph(
    ".5:.5:E5 1:.5:E5 1.5:.5:D5 2:1:C5 3:.5:B4 3.5:.5:C5 4:1.5:A4 6:.5:G4 6.5:.5:A4 7:1:C5 "
    "8.5:.5:A4 9:.5:C5 9.5:.5:F5 10:.5:E5 10.5:.5:F5 11:1:E5 12.5:.5:D5 13:.5:D5 13.5:.5:E5 14:.5:D5 14.5:.5:B4 15:1:G4 "
    "16.5:.5:E5 17:.5:E5 17.5:.5:D5 18:1:C5 19:.5:B4 19.5:.5:C5 20:1.5:A4 22:.5:G4 22.5:.5:A4 23:1:E5 "
    "24.5:.5:A4 25:.5:C5 25.5:.5:F5 26:.5:E5 26.5:.5:F5 27:1:A5 28:.5:E5 28.5:.5:D5 29:.5:C5 29.5:.5:B4 30:1:G#4 31:1:B4"
)
SOLO = ph(
    "1:1:A4 2:1:B4 3:1:C5 4:1:B4 5.5:2.5:A4 9:1:C5 10:1:D5 11:1:E5 12:1:D5 13.5:2.5:C5 "
    "17:1:E5 18:1:F5 19:1:G5 20:1:F5 21.5:2.5:E5 "
    "24:.5:A5 24.5:.5:A5 25:.5:A5 25.5:.5:A5 26:.5:F5 26.5:.5:E5 27:.5:F5 27.5:.5:E5 "
    "28.5:.5:D5 29:.5:D5 29.5:.5:E5 30:.5:D5 30.5:.5:C5 31:.5:B4 31.5:.5:G#4"
)
OUTRO = ph(
    CHORUS_HEAD + "4:1:B4 5.5:2.5:A4 8:.5:A5 8.5:.5:A5 9:.5:A5 9.5:.5:A5 10:.5:F5 10.5:.5:E5 11:.5:F5 11.5:.5:E5 "
    "12.5:.5:D5 13:.5:C5 13.5:.5:B4 14:2:G#4 17:1:A4 18:1:B4 19:1:C5 20:1:B4 21.5:2.5:A4 24:4:A4"
)

CHORDS = {
    "Am": (["A3", "C4", "E4"], "A1"),
    "F": (["A3", "C4", "F4"], "F1"),
    "Dm": (["A3", "D4", "F4"], "D2"),
    "G": (["G3", "B3", "D4"], "G1"),
    "E": (["G#3", "B3", "E4"], "E1"),
    "C": (["G3", "C4", "E4"], "C2"),
    "F#": (["F#3", "A#3", "C#4"], "F#1"),
}
GROOVE = [(0, 0.6, 0), (0.75, 0.25, 0), (1.5, 0.4, 12), (2, 0.5, 0), (2.5, 0.4, 7), (3, 0.3, 0), (3.5, 0.4, 12)]
STABS = [0, 1.5, 2.5]
ARP = [0, 1, 2, 3, 2, 1, 2, 3, 0, 1, 2, 3, 2, 1, 3, 2]
SCALE = [9, 11, 0, 2, 4, 5, 7]


def third_above(seq):
    out = []
    for b, d, p, v, g in seq:
        pc = p % 12
        nxt = SCALE[(SCALE.index(pc) + 2) % 7] if pc in SCALE else 11
        out.append((b, d, p + (nxt - pc) % 12, v, g))
    return out


class Song:
    def __init__(self):
        self.ev = defaultdict(list)
        self.drums = []
        self.risers = []
        self.sections = []
        self.at = 0.0

    def notes(self, track, seq, at, tr=0, vel=1.0):
        self.ev[track] += shift(seq, at, tr, vel)

    def hit(self, kind, beat, vel=1.0):
        self.drums.append((beat, kind, vel))

    def progression(self, at, names, tr=0, stabs=True, bass="groove", vel=1.0, arp=False):
        t = at
        for item in names:
            name, length = (item, 4) if isinstance(item, str) else item
            voicing, root = CHORDS[name]
            pitches = tuple(m(p) + tr for p in voicing)
            self.ev["pad"].append((t, length, pitches, vel, 0))
            if stabs:
                for x in STABS:
                    if x < length:
                        self.ev["stab"].append((t + x, 0.3, pitches, vel, 0))
            r = m(root) + tr
            if bass == "groove":
                for b, d, o in GROOVE:
                    if b < length:
                        self.ev["bass"].append((t + b, d, r + o, vel, 0))
            elif bass == "long":
                self.ev["bass"].append((t, length, r, 0.8 * vel, 0))
            if arp:
                tones = [p + 12 for p in pitches] + [pitches[0] + 24]
                for k in range(int(length * 4)):
                    self.ev["arp"].append((t + k / 4, 0.2, tones[ARP[k % 16]], 0.7 + 0.3 * (k % 4 == 0), 0))
            t += length

    def beat_full(self, at, nbars, fill=True, busy=False, crash=True):
        if crash:
            self.hit("crash", at)
        for k in range(nbars):
            b0 = at + 4 * k
            last = fill and k == nbars - 1
            for x in (0, 0.75, 2, 2.5):
                if not (last and x >= 3):
                    self.hit("kick", b0 + x)
            for x in (1, 3):
                self.hit("clap", b0 + x)
            for i, x in enumerate(np.arange(0, 4, 0.5)):
                if last and x >= 3:
                    continue
                if x == 3.5:
                    self.hit("ohat", b0 + x, 0.8)
                else:
                    self.hit("chat", b0 + x, 0.95 if i % 2 == 0 else 0.6)
            if busy:
                for x in (0.25, 1.75, 2.25, 2.75):
                    self.hit("chat", b0 + x, 0.35)
            if last:
                for j, x in enumerate((3, 3.25, 3.5, 3.75)):
                    self.hit("snare", b0 + x, 0.55 + 0.15 * j)

    def beat_verse(self, at, nbars):
        self.hit("crash", at, 0.6)
        for k in range(nbars):
            b0 = at + 4 * k
            last = k == nbars - 1
            for x in (0, 1.75, 2.5):
                self.hit("kick", b0 + x, 0.85)
            for x in (1, 3):
                self.hit("clap", b0 + x, 0.7)
            for x in np.arange(0, 4, 0.25):
                if not (last and x >= 3):
                    self.hit("shaker", b0 + x, 0.9 if x % 0.5 == 0 else 0.5)
            if last:
                for j, x in enumerate((3, 3.25, 3.5, 3.75)):
                    self.hit("snare", b0 + x, 0.5 + 0.15 * j)

    def mark(self, name, bars):
        self.sections.append((name, self.at, bars))
        start = self.at
        self.at += 4 * bars
        return start

    def intro(self):
        at = self.mark("intro", 8)
        self.progression(at, ["Am", "Am", "F", "Dm"], stabs=False, bass=None, vel=0.8)
        self.notes("bell", CHORUS_A, at, vel=0.8)
        for x in np.arange(8, 16, 0.5):
            self.hit("chat", at + x, 0.5 if x % 1 == 0 else 0.3)
        at2 = at + 16
        self.progression(at2, ["Am", "Am", "F", ("Dm", 2), ("E", 2)], stabs=False, bass="groove", vel=0.9)
        self.notes("bell", CHORUS_B, at2)
        self.notes("whoop", HOOK_WHOOP, at2, vel=0.6)
        for k in range(4):
            b0 = at2 + 4 * k
            for x in (0, 2, 2.5):
                if not (k == 3 and x >= 3):
                    self.hit("kick", b0 + x, 0.9)
            for x in np.arange(0, 4, 0.5):
                if not (k == 3 and x >= 3):
                    self.hit("chat", b0 + x, 0.8 if x % 1 == 0 else 0.5)
            self.hit("clap", b0 + 3, 0.6)
        for j, x in enumerate(np.arange(3, 4, 0.25)):
            self.hit("snare", at2 + 12 + x, 0.5 + 0.15 * j)

    def hook(self, tr=0, layers=()):
        at = self.mark("hook", 4)
        self.notes("lead", HOOK, at, tr)
        self.notes("whoop", HOOK_WHOOP, at, tr, 0.8)
        if "octave" in layers:
            self.notes("bell", HOOK, at, tr + 12, 0.5)
        if "harm" in layers:
            self.notes("harm", third_above(HOOK), at, tr, 0.7)
        self.progression(at, ["Am", "Am", "Am", "F"], tr, arp="arp" in layers)
        self.beat_full(at, 4, busy="busy" in layers)

    def chorus(self, variant, tr=0, layers=()):
        at = self.mark("chorus", 4)
        mel = CHORUS_A if variant == "A" else CHORUS_B
        self.notes("lead", mel, at, tr)
        if "octave" in layers:
            self.notes("bell", mel, at, tr + 12, 0.5)
        if "harm" in layers:
            self.notes("harm", third_above(mel), at, tr, 0.7)
        bass = ph("0:.5:A1 1:1:A1 2:1:B1 3:1:C2 4:1:B1 5.5:1:A1 6.5:.4:A2 7:.4:E2 7.5:.4:G2")
        self.notes("bass", bass, at, tr)
        self.progression(at, ["Am", "Am"], tr, bass=None, arp="arp" in layers)
        tail = ["F", "Dm"] if variant == "A" else ["F", ("Dm", 2), ("E", 2)]
        self.progression(at + 8, tail, tr, arp="arp" in layers)
        self.beat_full(at, 4, busy="busy" in layers, crash=variant == "A")

    def verse(self, tr=0):
        at = self.mark("verse", 8)
        self.notes("bell", VERSE, at, tr)
        self.notes("harm", VERSE, at, tr - 12, 0.35)
        self.progression(at, ["Am", "Am", "F", "G", "Am", "Am", "F", "E"], tr, stabs=False, vel=0.85)
        self.beat_verse(at, 8)

    def solo(self, tr=0):
        at = self.mark("solo", 8)
        self.notes("lead", SOLO, at, tr)
        self.notes("bell", SOLO, at, tr + 12, 0.45)
        self.notes("whoop", HOOK_WHOOP, at + 16, tr, 0.6)
        self.progression(at, ["Am", "Am", "F", "Dm", "Am", "Am", "F", "E"], tr, arp=True)
        self.beat_full(at, 4, fill=False)
        self.beat_full(at + 16, 4, busy=True, crash=False)

    def breakdown(self, tr=0):
        at = self.mark("breakdown", 8)
        self.notes("bell", [e for e in shift(HOOK, stretch=2) if e[0] < 24], at, tr)
        self.notes("harm", [e for e in shift(HOOK, tr=-12, stretch=2) if e[0] < 24], at, tr, 0.45)
        self.progression(at, ["Am", "G", "F", "Am", "Dm", "E", "F", "F#"], tr, stabs=False, bass="long")
        self.risers.append((at + 24, 8))
        for x in np.arange(24, 28, 0.5):
            self.hit("snare", at + x, 0.25 + 0.04 * (x - 24))
        for x in np.arange(28, 32, 0.25):
            self.hit("snare", at + x, 0.45 + 0.13 * (x - 28))
        self.hit("kick", at + 28, 0.8)
        self.hit("kick", at + 30, 0.8)

    def outro(self, tr=0):
        at = self.mark("outro", 8)
        self.notes("bell", OUTRO, at, tr)
        self.notes("lead", [e for e in OUTRO if e[0] < 16], at, tr, 0.6)
        self.notes("lead", [e for e in OUTRO if e[0] >= 24], at, tr, 0.8)
        self.notes("bell", [e for e in OUTRO if e[0] >= 24], at, tr + 12, 0.6)
        self.progression(at, ["Am", "Am", "F", "E", "Am", "Am"], tr, stabs=False)
        self.ev["pad"].append((at + 24, 6, tuple(m(p) + tr for p in ("A3", "C4", "E4", "A4")), 1.0, 0))
        self.ev["bass"].append((at + 24, 3, m("A1") + tr, 1.0, 0))
        self.beat_full(at, 4, crash=True)
        for k in range(2):
            b0 = at + 16 + 4 * k
            for x in (0, 2):
                self.hit("kick", b0 + x, 0.8)
            for x in np.arange(0, 4, 0.5):
                self.hit("chat", b0 + x, 0.5)
        self.hit("kick", at + 24, 1.0)
        self.hit("crash", at + 24, 1.0)
        self.hit("clap", at + 24, 0.8)


def arrange():
    s = Song()
    s.intro()
    s.hook()
    s.chorus("A")
    s.chorus("B")
    s.verse()
    s.hook(layers=("octave",))
    s.chorus("A", layers=("octave",))
    s.chorus("B", layers=("octave",))
    s.solo()
    s.chorus("A", layers=("harm", "arp"))
    s.chorus("B", layers=("harm", "arp"))
    s.breakdown()
    full = ("octave", "harm", "arp", "busy")
    s.chorus("A", 2, full)
    s.chorus("B", 2, full)
    s.hook(2, full)
    s.chorus("A", 2, full)
    s.chorus("B", 2, full)
    s.outro(2)
    return s


_sos = {}


def filt(x, kind, f, order=2):
    key = (kind, tuple(np.atleast_1d(np.round(f))), order)
    if key not in _sos:
        _sos[key] = butter(order, f, btype=kind, fs=SR, output="sos")
    return sosfilt(_sos[key], x, axis=-1)


def lowpass(x, f, order=2):
    return filt(x, "lowpass", min(f, SR * 0.45), order)


def highpass(x, f, order=2):
    return filt(x, "highpass", f, order)


def bandpass(x, lo, hi, order=2):
    return filt(x, "bandpass", [lo, hi], order)


def tvec(n):
    return np.arange(n) / SR


def blep(t, dt):
    y = np.zeros_like(t)
    a = t < dt
    x = t[a] / dt[a]
    y[a] = x + x - x * x - 1
    b = t > 1 - dt
    x = (t[b] - 1) / dt[b]
    y[b] = x * x + x + x + 1
    return y


def saw(phase, dt):
    t = phase % 1.0
    return 2 * t - 1 - blep(t, dt)


def sqr(phase, dt, pw=0.5):
    t = phase % 1.0
    return np.where(t < pw, 1.0, -1.0) + blep(t, dt) - blep((t + 1 - pw) % 1.0, dt)


def adsr(n, gate, a, d, s, r):
    t = tvec(n)
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / d))
    g = int(gate * SR)
    if g < n:
        e[g:] = e[max(g - 1, 0)] * np.clip(1 - (t[g:] - gate) / r, 0, 1)
    return e


def noise(n):
    return rng.standard_normal(n)


def v_lead(p, dur, vel, glide, bright=1.0):
    gate = dur * SPB
    n = int((gate + 0.12) * SR)
    t = tvec(n)
    tau = 0.05 if glide < 0 else 0.12
    pc = p + glide * np.exp(-t / tau)
    if dur >= 1:
        pc = pc + 0.13 * np.sin(2 * np.pi * 5.3 * t) * np.clip((t - 0.28) / 0.25, 0, 1)
    f = hz(pc)
    phase = np.cumsum(f / SR)
    dt = f / SR
    sub = np.sin(np.pi * phase)
    x = np.stack([0.55 * saw(phase * 1.004, dt * 1.004) + 0.3 * sub, 0.55 * saw(phase / 1.004, dt / 1.004) + 0.3 * sub])
    w = np.clip((0.3 + 0.7 * np.exp(-t / 0.2)) * bright, 0, 1)
    x = lowpass(x, 1400) * (1 - w) + lowpass(x, 7000) * w
    return x * adsr(n, gate, 0.004, 0.3, 0.65, 0.1) * vel


def v_bell(p, dur, vel, glide):
    gate = dur * SPB
    n = int((gate + 1.0) * SR)
    t = tvec(n)
    f = float(hz(p))
    idx = 2.0 * np.exp(-t / 0.22) + 0.35
    y = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    y += 0.25 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t / 0.06)
    y *= np.exp(-t / 0.9) * np.clip(1 - (t - gate - 0.15) / 0.6, 0, 1) * np.clip(t / 0.002, 0, 1)
    return np.stack([0.92 * y, y]) * vel


def v_bass(p, dur, vel, glide):
    gate = dur * SPB
    n = int((gate + 0.04) * SR)
    t = tvec(n)
    f = float(hz(p))
    phase = f * t
    dt = np.full(n, f / SR)
    fe = np.exp(-t / 0.07)
    sw = saw(phase, dt)
    sw = lowpass(sw, 500) * (1 - fe) + lowpass(sw, 2800) * fe
    y = np.tanh(1.8 * (0.85 * np.sin(2 * np.pi * phase) + 0.45 * sw))
    return y * adsr(n, gate, 0.003, 0.25, 0.8, 0.035) * vel


def v_pad(pitches, dur, vel, glide):
    gate = dur * SPB
    n = int((gate + 0.7) * SR)
    t = tvec(n)
    x = np.zeros((2, n))
    for p in pitches:
        for k, det in enumerate((-0.12, -0.04, 0.05, 0.13)):
            f = float(hz(p + det))
            x[k % 2] += saw(f * t + rng.random(), np.full(n, f / SR))
    return x / (2 * len(pitches)) * adsr(n, gate, 0.35, 1, 1, 0.7) * vel


def v_stab(pitches, dur, vel, glide):
    n = int(0.35 * SR)
    t = tvec(n)
    x = np.zeros((2, n))
    for p in pitches:
        for k, det in enumerate((-0.08, 0.08)):
            f = float(hz(p + det))
            x[k] += saw(f * t, np.full(n, f / SR))
    fe = np.exp(-t / 0.05)
    x = lowpass(x, 900) * (1 - fe) + lowpass(x, 5000) * fe
    return x / len(pitches) * np.exp(-t / 0.11) * np.clip(t / 0.002, 0, 1) * vel


def v_arp(p, dur, vel, glide):
    n = int(0.25 * SR)
    t = tvec(n)
    f = float(hz(p))
    y = sqr(f * t, np.full(n, f / SR), 0.3)
    fe = np.exp(-t / 0.04)
    y = lowpass(y, 1200) * (1 - fe) + lowpass(y, 6000) * fe
    return y * np.exp(-t / 0.09) * np.clip(t / 0.002, 0, 1) * vel


VOICES = {
    "lead": lambda p, d, v, g: v_lead(p, d, v, g),
    "harm": lambda p, d, v, g: v_lead(p, d, v, g, bright=0.6),
    "whoop": lambda p, d, v, g: v_lead(p, d, v, g, bright=1.3),
    "bell": v_bell,
    "bass": v_bass,
    "pad": v_pad,
    "stab": v_stab,
    "arp": v_arp,
}
PANS = {"harm": 0.35, "whoop": -0.35, "bell": -0.1, "chat": 0.25, "ohat": 0.3, "shaker": -0.35, "crash": -0.2, "snare": 0.05}


def drum(kind):
    if kind == "kick":
        n = int(0.5 * SR)
        t = tvec(n)
        f = 48 + 120 * np.exp(-t / 0.035) + 30 * np.exp(-t / 0.004)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.32)
        click = bandpass(noise(n), 1500, 6000) * np.exp(-t / 0.004) * 0.5
        return np.tanh(2.2 * (body + click))
    if kind == "clap":
        n = int(0.45 * SR)
        t = tvec(n)
        env = sum((t >= o) * np.exp(-np.clip(t - o, 0, None) / 0.0045) for o in (0, 0.009, 0.018, 0.027))
        env = env + (t >= 0.03) * np.exp(-np.clip(t - 0.03, 0, None) / 0.13) * 0.7
        return bandpass(noise(n), 900, 3000) * env + 0.3 * np.sin(2 * np.pi * 210 * t) * np.exp(-t / 0.05)
    if kind == "snare":
        n = int(0.3 * SR)
        t = tvec(n)
        return 0.8 * bandpass(noise(n), 1200, 9000) * np.exp(-t / 0.11) + 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.07)
    if kind == "chat":
        n = int(0.08 * SR)
        return highpass(noise(n), 7500, 4) * np.exp(-tvec(n) / 0.022)
    if kind == "ohat":
        n = int(0.45 * SR)
        return highpass(noise(n), 6500, 4) * np.exp(-tvec(n) / 0.17)
    if kind == "crash":
        n = int(2.5 * SR)
        t = tvec(n)
        return highpass(noise(n), 4500, 4) * np.exp(-t / 0.9) + 0.4 * bandpass(noise(n), 3000, 6000) * np.exp(-t / 0.3)
    if kind == "shaker":
        n = int(0.09 * SR)
        t = tvec(n)
        env = np.where(t < 0.012, t / 0.012, np.exp(-(t - 0.012) / 0.035))
        return bandpass(noise(n), 5000, 11000) * env
    raise ValueError(kind)


def riser(beats):
    n = int(beats * SPB * SR)
    t = tvec(n)
    T = t[-1]
    centers = np.geomspace(400, 9000, 8)
    pos = t / T * (len(centers) - 1)
    out = np.zeros((2, n))
    for c in range(2):
        src = noise(n)
        bank = [bandpass(src, fc * 0.7, fc * 1.4) for fc in centers]
        lo = np.clip(np.floor(pos).astype(int), 0, len(centers) - 2)
        fr = pos - lo
        y = np.zeros(n)
        for k in range(len(centers) - 1):
            mk = lo == k
            y[mk] = bank[k][mk] * (1 - fr[mk]) + bank[k + 1][mk] * fr[mk]
        out[c] = y
    sweep = 0.3 * np.sin(2 * np.pi * np.cumsum(200 * 10 ** (t / T)) / SR)
    return (out + sweep) * (t / T) ** 2


def pan(x, p):
    th = (p + 1) * np.pi / 4
    return np.stack([x[0] * np.cos(th), x[1] * np.sin(th)]) * np.sqrt(2)


def place(buf, x, beat, jitter=0.0):
    if x.ndim == 1:
        x = np.stack([x, x])
    i = max(0, int(round((beat * SPB + jitter) * SR)))
    j = min(buf.shape[1], i + x.shape[1])
    buf[:, i:j] += x[:, : j - i]


def active_rms(x):
    mono = x.mean(0)
    frame = 2048
    k = len(mono) // frame
    r = np.sqrt((mono[: k * frame].reshape(k, frame) ** 2).mean(1) + 1e-20)
    act = r > r.max() * 10 ** (-40 / 20)
    return np.sqrt((r[act] ** 2).mean())


def duck(kicks, n, depth, rel=0.13):
    e = np.ones(n)
    L = int(0.4 * SR)
    shape = 1 - depth * np.exp(-tvec(L) / rel)
    for b in kicks:
        i = int(b * SPB * SR)
        seg = e[i : i + L]
        np.minimum(seg, shape[: len(seg)], out=seg)
    return e


def make_ir(sec=2.2, decay=0.45):
    n = int(sec * SR)
    ir = np.stack([noise(n), noise(n)]) * np.exp(-tvec(n) / decay)
    ir = lowpass(ir, 5000)
    ir[:, : int(0.012 * SR)] = 0
    return ir / np.sqrt((ir**2).sum(1, keepdims=True))


def pingpong(x, beats=0.75, fb=0.38, taps=5):
    d = int(beats * SPB * SR)
    y = np.zeros_like(x)
    for k in range(1, taps + 1):
        src = x[::-1] if k % 2 else x
        y[:, k * d :] += fb**k * src[:, : x.shape[1] - k * d]
    return lowpass(highpass(y, 300), 4500)


def sweep(x, b0, b1, f0, f1):
    i0, i1 = int(b0 * SPB * SR), int(b1 * SPB * SR)
    seg = x[:, i0:i1]
    n = seg.shape[1]
    bank_f = np.geomspace(f0, f1, 10)
    bank = [lowpass(seg, f) for f in bank_f]
    pos = np.linspace(0, len(bank_f) - 1, n) ** 1.0
    lo = np.clip(np.floor(pos).astype(int), 0, len(bank_f) - 2)
    fr = pos - lo
    out = np.zeros_like(seg)
    for k in range(len(bank_f) - 1):
        mk = lo == k
        out[:, mk] = bank[k][:, mk] * (1 - fr[mk]) + bank[k + 1][:, mk] * fr[mk]
    x[:, i0:i1] = out


TARGETS = {
    "kick": -14, "bass": -15, "lead": -16, "bell": -19, "clap": -20, "snare": -22, "harm": -23, "whoop": -23,
    "pad": -25, "stab": -26, "arp": -26, "chat": -29, "ohat": -29, "crash": -29, "shaker": -30, "riser": -24,
}
REVERB = {"lead": 0.18, "bell": 0.32, "harm": 0.22, "whoop": 0.3, "pad": 0.35, "stab": 0.2, "arp": 0.2, "clap": 0.18, "snare": 0.15, "ohat": 0.08}
DELAY = {"lead": 0.14, "bell": 0.22, "whoop": 0.25, "arp": 0.22}
DRUMS = {"kick", "clap", "snare", "chat", "ohat", "crash", "shaker"}


def render(s):
    total_beats = s.at
    N = int((total_beats * SPB + 4.0) * SR)
    tracks = {}
    for name, events in s.ev.items():
        buf = np.zeros((2, N))
        voice = VOICES[name]
        for k, (b, d, p, v, g) in enumerate(events):
            x = voice(p, d, v, g)
            if name == "arp":
                x = pan(np.stack([x, x]), 0.45 if k % 2 else -0.45)
            place(buf, x, b)
        tracks[name] = buf
    for kind in DRUMS:
        buf = np.zeros((2, N))
        for b, k, v in s.drums:
            if k == kind:
                place(buf, drum(kind) * v * (1 + rng.normal(0, 0.04)), b, rng.normal(0, 0.002))
        tracks[kind] = buf
    if s.risers:
        buf = np.zeros((2, N))
        for b, d in s.risers:
            place(buf, riser(d), b)
        tracks["riser"] = buf

    tracks["pad"] = highpass(lowpass(tracks["pad"], 1800), 150)
    tracks["stab"] = highpass(tracks["stab"], 200)
    for k in ("lead", "harm", "whoop", "bell", "arp"):
        if k in tracks:
            tracks[k] = highpass(tracks[k], 150)
    tracks["bass"] = highpass(lowpass(tracks["bass"], 4000), 30)

    for k, x in tracks.items():
        if k in PANS:
            tracks[k] = pan(x, PANS[k])
        r = active_rms(tracks[k])
        tracks[k] = tracks[k] * (10 ** (TARGETS[k] / 20) / r)

    kicks = [b for b, k, v in s.drums if k == "kick"]
    for k, depth in (("pad", 0.55), ("stab", 0.4), ("arp", 0.35), ("bass", 0.35)):
        if k in tracks:
            tracks[k] = tracks[k] * duck(kicks, N, depth)

    ir = make_ir()
    rsend = sum(tracks[k] * a for k, a in REVERB.items() if k in tracks)
    wet = np.stack([fftconvolve(rsend[c], ir[c])[:N] for c in range(2)]) * 0.4
    dsend = sum(tracks[k] * a for k, a in DELAY.items() if k in tracks)
    echo = pingpong(dsend)

    music = sum(x for k, x in tracks.items() if k not in DRUMS) + wet + echo
    sweep(music, 0, 16, 350, 18000)
    master = music + sum(x for k, x in tracks.items() if k in DRUMS)
    master = highpass(master, 25)
    master /= np.abs(master).max()
    master = np.tanh(1.6 * master) / np.tanh(1.6)
    master *= 0.95 / np.abs(master).max()

    for k, x in sorted(tracks.items()):
        print(f"  {k:7s} active rms {20 * np.log10(active_rms(x) + 1e-12):6.1f} dB")
    return master


PROGRAMS = {"lead": 81, "harm": 81, "whoop": 80, "bell": 10, "bass": 38, "pad": 89, "stab": 62, "arp": 80}
GM_DRUMS = {"kick": 36, "clap": 39, "snare": 38, "chat": 42, "ohat": 46, "crash": 49, "shaker": 70}


def export_midi(s, path):
    tpb = 480
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="My Humps (extended)", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
    prev = 0
    for name, start, bars in s.sections:
        tick = int(round(start * tpb))
        meta.append(mido.MetaMessage("marker", text=name, time=tick - prev))
        prev = tick
    mid.tracks.append(meta)

    def track(name, ch, items, program=None):
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if program is not None:
            tr.append(mido.Message("program_change", program=program, channel=ch, time=0))
        msgs = []
        for b, d, p, v in items:
            vel = int(np.clip(v * 100, 1, 127))
            msgs.append((int(round(b * tpb)), 1, p, vel))
            msgs.append((int(round((b + d) * tpb)), 0, p, 0))
        msgs.sort()
        now = 0
        for tick, on, p, vel in msgs:
            tr.append(mido.Message("note_on" if on else "note_off", note=int(p), velocity=vel, channel=ch, time=tick - now))
            now = tick
        mid.tracks.append(tr)

    for ch, name in enumerate(n for n in PROGRAMS if n in s.ev):
        items = []
        for b, d, p, v, g in s.ev[name]:
            for q in p if isinstance(p, tuple) else (p,):
                items.append((b, d, q, v))
        track(name, ch, items, PROGRAMS[name])
    track("drums", 9, [(b, 0.25, GM_DRUMS[k], v) for b, k, v in s.drums])
    mid.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    s = arrange()
    print(f"{s.at / 4:.0f} bars, {s.at * SPB:.1f}s at {BPM} bpm")
    for name, start, bars in s.sections:
        print(f"  {start * SPB:6.1f}s  {name} ({bars} bars)")
    export_midi(s, ROOT / f"{NAME}.mid")
    audio = render(s)
    wav = OUT / f"{NAME}.wav"
    sf.write(wav, audio.T, SR, subtype="PCM_24")
    if "--no-mp3" not in sys.argv:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(wav), "-af", "loudnorm=I=-14:TP=-1.2:LRA=11:print_format=json", "-f", "null", "-"],
            capture_output=True, text=True, check=True,
        ).stderr
        meas = json.loads(probe[probe.rindex("{") : probe.rindex("}") + 1])
        norm = (
            "loudnorm=I=-14:TP=-1.2:LRA=11:linear=true:"
            f"measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:measured_LRA={meas['input_lra']}:"
            f"measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}"
        )
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", norm, "-ar", str(SR),
                "-c:a", "libmp3lame", "-b:a", "320k", "-metadata", "title=My Humps (Michael Scott ringtone, extended)",
                "-metadata", "artist=radres", str(ROOT / f"{NAME}.mp3"),
            ],
            check=True,
        )
    print("done")


if __name__ == "__main__":
    main()
