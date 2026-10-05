#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FuryCube V68 - Audio Visualizer (GUI)

Protokol (hasil sadapan USB):
  feature report ID 0x06, 520 byte, interface 1, usage page 0xFF00
  header 06 08 00 00 01 00 7a 01, lalu 90 slot x 3 byte (R, G, B)
  slot = kolom * 6 + baris  (15 kolom x 6 baris, baris 0 = atas)

Instalasi:
  pip install hidapi numpy soundcard
  (Linux/Arch: butuh paket 'tk', dan udev rule untuk akses device)

Tutup driver bawaan (OemDrv.exe) sebelum menjalankan.
"""
import json
import os
import sys
import threading
import time

import numpy as np

try:
    import hid
except Exception:
    hid = None
try:
    import soundcard as sc
except Exception:
    sc = None
try:
    import tkinter as tk
    from tkinter import ttk, colorchooser
except Exception:
    tk = None

VID = 0x258A
COLS, ROWS = 15, 6
N_LED = COLS * ROWS
HEADER = bytes.fromhex("0608000001007a01")
RATE = 48000
CHUNK = 1024
FFT_N = 4096
FPS = 30
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "furycube_config.json")
PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

# ---------------------------------------------------------------- palet
PALETTES = {
    "Klasik (hijau > merah)": [(0, 255, 60), (180, 255, 0), (255, 200, 0), (255, 90, 0), (255, 0, 0)],
    "Neon": [(0, 255, 255), (120, 0, 255), (255, 0, 200)],
    "Api": [(120, 0, 0), (255, 0, 0), (255, 140, 0), (255, 230, 100), (255, 255, 255)],
    "Laut": [(0, 40, 120), (0, 160, 255), (0, 255, 200), (200, 255, 255)],
    "Sunset": [(80, 0, 160), (255, 0, 120), (255, 140, 0), (255, 230, 80)],
    "Matrix": [(0, 60, 0), (0, 255, 70), (200, 255, 200)],
}
RAINBOW = "Pelangi (bergeser)"
CUSTOM = "Kustom (2 warna)"
RAINBOW_STOPS = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
                 (0, 0, 255), (255, 0, 255), (255, 0, 0)]
PALETTE_NAMES = list(PALETTES) + [RAINBOW, CUSTOM]


class Palette:
    """pal(x) dengan x 0..1 (skalar atau array) -> RGB 0..1 (shape x.shape + (3,))."""

    def __init__(self, S):
        self.S = S
        self.t = 0.0

    def __call__(self, x):
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        name = self.S.palette
        if name == RAINBOW:
            stops = RAINBOW_STOPS
            x = (x * 0.85 + self.t * 0.08) % 1.0
        elif name == CUSTOM:
            stops = [self.S.c1, self.S.c2]
        else:
            stops = PALETTES.get(name, PALETTES[PALETTE_NAMES[0]])
        st = np.asarray(stops, dtype=float) / 255.0
        pos = np.linspace(0.0, 1.0, len(st))
        return np.stack([np.interp(x, pos, st[:, i]) for i in range(3)], axis=-1)


# ---------------------------------------------------------------- analisis audio
class Analyzer:
    def __init__(self):
        freqs = np.fft.rfftfreq(FFT_N, 1.0 / RATE)
        edges = np.geomspace(40, 16000, COLS + 1)
        self.slices = []
        for i in range(COLS):
            lo = int(np.searchsorted(freqs, edges[i]))
            hi = max(int(np.searchsorted(freqs, edges[i + 1])), lo + 1)
            self.slices.append((lo, hi))
        centers = np.sqrt(edges[:-1] * edges[1:])
        self.tilt = (centers / 300.0) ** 0.5          # kompensasi nada tinggi
        self.win = np.hanning(FFT_N)
        self.bass_lo = int(np.searchsorted(freqs, 40))
        self.bass_hi = int(np.searchsorted(freqs, 150))
        self.ref = 1.0
        self.bass_ref = 1.0
        self.bass_avg = 0.0
        self.last_beat = 0.0
        self.t = 0.0
        self.level = np.zeros(COLS)
        self.vol = 0.0

    def process(self, x, dt, sens, decay):
        """Return (level[COLS] 0..1, vol 0..1, bass 0..1, beat bool). Index 0 = bass."""
        self.t += dt
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        beat = False
        if rms < 1e-4:
            norm = np.zeros(COLS)
            vol_t = 0.0
            self.bass_avg *= 0.9
        else:
            spec = np.abs(np.fft.rfft(x * self.win))
            mag = np.array([spec[lo:hi].mean() for lo, hi in self.slices]) * self.tilt
            fall = 0.5 ** (dt / 6.0)
            self.ref = max(1.0, float(mag.max()), self.ref * fall)
            ratio = mag / (self.ref / max(sens, 1e-3))
            db = 20.0 * np.log10(ratio + 1e-6)
            norm = np.clip((db + 32.0) / 32.0, 0.0, 1.0) ** 1.5
            vol_t = float(np.clip((20.0 * np.log10(rms * sens + 1e-9) + 50.0) / 45.0, 0.0, 1.0))
            be = float(spec[self.bass_lo:self.bass_hi].mean())
            self.bass_ref = max(1.0, be, self.bass_ref * fall)
            self.bass_avg += (be - self.bass_avg) * min(1.0, dt / 1.0)
            if (be > 1.4 * self.bass_avg and be > 0.25 * self.bass_ref and be > 3.0
                    and self.t - self.last_beat > 0.18):
                beat = True
                self.last_beat = self.t
        self.level = np.maximum(norm, self.level - decay * dt)
        self.vol = max(vol_t, self.vol - decay * dt)
        return self.level.copy(), self.vol, float(self.level[:3].mean()), beat


class Audio(threading.Thread):
    """Menangkap audio sistem (loopback) ke ring buffer."""

    def __init__(self, S):
        super().__init__(daemon=True)
        self.S = S
        self.buf = np.zeros(FFT_N, dtype=np.float32)
        self.lock = threading.Lock()
        self.last = 0.0
        self.running = True
        self.reopen = False
        self.status = "audio: mulai..."

    def latest(self):
        with self.lock:
            fresh = (time.time() - self.last) < 0.25
            return self.buf.copy() if fresh else np.zeros(FFT_N, dtype=np.float32)

    def _pick(self):
        name = self.S.audio
        if name != "(otomatis)":
            try:
                return sc.get_microphone(name, include_loopback=True)
            except Exception:
                pass
        spk = sc.default_speaker()
        try:
            return sc.get_microphone(spk.name, include_loopback=True)
        except Exception:
            for m in sc.all_microphones(include_loopback=True):
                if getattr(m, "isloopback", False) and spk.name in m.name:
                    return m
            raise

    def run(self):
        if sc is None:
            self.status = "audio: library soundcard belum terpasang (pip install soundcard)"
            return
        while self.running:
            try:
                mic = self._pick()
                self.reopen = False
                self.status = f"audio: {mic.name}"
                with mic.recorder(samplerate=RATE, blocksize=CHUNK) as rec:
                    while self.running and not self.reopen:
                        data = rec.record(numframes=CHUNK)
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        with self.lock:
                            self.buf = np.concatenate((self.buf, mono.astype(np.float32)))[-FFT_N:]
                            self.last = time.time()
            except Exception as e:
                self.status = f"audio error: {e}"
                time.sleep(2.0)


# ---------------------------------------------------------------- efek
CC, RR = np.meshgrid(np.arange(COLS), np.arange(ROWS), indexing="ij")


class Ctx:
    pass


def _bars(ctx, levels, dim=1.0):
    out = np.zeros((COLS, ROWS, 3))
    for c in range(COLS):
        rows = ctx.rows_bu[c]
        n = len(rows)
        if n == 0:
            continue
        h = float(levels[c]) * n
        full = int(h)
        frac = h - full
        for k in range(min(full + 1, n)):
            b = 1.0 if k < full else frac
            if b < 0.03:
                continue
            x = (ROWS - 1 - rows[k]) / (ROWS - 1)
            out[c, rows[k]] = ctx.pal(x) * b * dim
    return out


def fx_bars(ctx, st):
    return _bars(ctx, ctx.bands)


def fx_peak(ctx, st):
    peak = st.setdefault("peak", np.zeros(COLS))
    hold = st.setdefault("hold", np.zeros(COLS))
    b = ctx.bands
    up = b >= peak
    peak[up] = b[up]
    hold[up] = 0.4
    hold[~up] -= ctx.dt
    drop = (~up) & (hold <= 0)
    peak[drop] = np.maximum(peak[drop] - 1.1 * ctx.dt, b[drop])
    out = _bars(ctx, b, 0.55)
    for c in range(COLS):
        rows = ctx.rows_bu[c]
        n = len(rows)
        if n == 0:
            continue
        k = int(peak[c] * n + 0.5) - 1
        if k < 0:
            continue
        k = min(k, n - 1)
        x = (ROWS - 1 - rows[k]) / (ROWS - 1)
        out[c, rows[k]] = 0.5 * ctx.pal(x) + 0.5
    return out


def fx_butterfly(ctx, st):
    out = np.zeros((COLS, ROWS, 3))
    order = [2, 3, 1, 4, 0, 5]
    for c in range(COLS):
        d = abs(c - 7)
        h = float(ctx.raw[2 * d]) * ROWS
        for k, r in enumerate(order):
            b = min(1.0, h - k)
            if b <= 0.03:
                break
            if not ctx.valid[c, r]:
                continue
            x = 0.1 + 0.9 * abs(r - 2.5) / 2.5
            out[c, r] = ctx.pal(x) * b
    return out


def fx_waterfall(ctx, st):
    hist = st.setdefault("hist", np.zeros((COLS, ROWS)))
    st["acc"] = st.get("acc", 0.0) + ctx.dt
    if st["acc"] >= 0.07:
        st["acc"] = 0.0
        hist[:-1] = hist[1:].copy()
        groups = np.array_split(np.arange(COLS), ROWS)
        col = np.array([ctx.raw[g].mean() for g in groups])   # 0 = bass
        hist[-1, :] = col[::-1]                                # baris bawah = bass
    v = np.clip(hist * 1.2, 0, 1)
    return ctx.pal(v) * v[..., None]


def fx_ripple(ctx, st):
    rings = st.setdefault("rings", [])
    if ctx.beat:
        rings.append([np.random.uniform(3, 11), np.random.uniform(1.0, 4.0), 0.0, np.random.rand()])
    out = np.zeros((COLS, ROWS, 3))
    for ring in rings:
        ring[2] += ctx.dt
        cx, cy, age, hue = ring
        dist = np.hypot(CC - cx, (RR - cy) * 1.3)
        inten = np.exp(-age * 2.0) * np.exp(-(((dist - age * 12.0) / 1.0) ** 2))
        out += inten[..., None] * ctx.pal((hue + ctx.pal.t * 0.05) % 1.0)
    st["rings"] = [r for r in rings if np.exp(-r[2] * 2.0) > 0.04]
    out[:, ROWS - 1, :] += ctx.pal(0.0)[None, :] * ctx.bands[:, None] * 0.5
    return out


def fx_fire(ctx, st):
    h = st.get("heat")
    if h is None:
        h = np.zeros((COLS, ROWS))
    new = np.zeros_like(h)
    active = 1.0 if ctx.vol > 0.02 else 0.0
    src = np.clip(ctx.bands * 1.3 + ctx.vol * 0.2 + np.random.rand(COLS) * 0.15 * active, 0, 1)
    new[:, ROWS - 1] = np.maximum(src, h[:, ROWS - 1] * 0.5)
    for r in range(ROWS - 2, -1, -1):
        below = h[:, r + 1]
        side = (np.roll(below, 1) + np.roll(below, -1)) * 0.5
        new[:, r] = np.clip(below * 0.68 + side * 0.20 - np.random.rand(COLS) * 0.09, 0, 1)
    st["heat"] = new
    return ctx.pal(new) * np.clip(new * 2.2, 0, 1)[..., None]


def fx_rain(ctx, st):
    drops = st.setdefault("drops", [])
    p = (ctx.bands ** 1.5) * 7.0 * ctx.dt
    for c in np.nonzero(np.random.rand(COLS) < p)[0]:
        drops.append([int(c), -0.5, np.random.uniform(7, 13), float(ctx.bands[c])])
    if ctx.beat:
        for c in np.random.choice(COLS, 4, replace=False):
            drops.append([int(c), -0.5, np.random.uniform(11, 16), 1.0])
    out = np.zeros((COLS, ROWS, 3))
    keep = []
    for d in drops:
        d[1] += d[2] * ctx.dt
        head = int(round(d[1]))
        for i in range(4):
            r = head - i
            if 0 <= r < ROWS:
                col = ctx.pal(d[0] / (COLS - 1)) * (0.5 + 0.5 * d[3]) * (1.0 - 0.28 * i)
                out[d[0], r] = np.maximum(out[d[0], r], col)
        if head - 4 < ROWS:
            keep.append(d)
    st["drops"] = keep
    return out


def fx_aurora(ctx, st):
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (0.6 + 4.0 * ctx.vol)
    ph = st["ph"]
    v = (np.sin(CC * 0.55 + ph) + np.sin(RR * 0.8 - ph * 0.7)
         + np.sin((CC + RR) * 0.35 + ph * 0.5)
         + np.sin(np.hypot(CC - 7, RR - 2.5) * 0.6 - ph)) / 4.0 * 0.5 + 0.5
    lvl = 0.25 + 0.75 * np.clip(ctx.bands[:, None] * 1.2 + ctx.bass * 0.3, 0, 1)
    return ctx.pal(v) * lvl[..., None]


def fx_pulse(ctx, st):
    flash = st.get("flash", 0.0) * np.exp(-ctx.dt * 5.0)
    if ctx.beat:
        flash = 1.0
    st["flash"] = flash
    dist = np.hypot(CC - 7, (RR - 2.5) * 1.6) / 8.0
    inten = np.clip((0.12 + 0.55 * ctx.bass + 0.6 * flash) - dist * 0.55, 0, 1)
    return ctx.pal(np.clip(0.1 + dist * 0.9, 0, 1)) * inten[..., None]


# ---------------------------------------------------------------- visualizer tambahan
def fx_wave(ctx, st):
    """Gelombang horizontal yang mengikuti level tiap band."""
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (1.5 + ctx.vol * 5.0)
    out = np.zeros((COLS, ROWS, 3))
    for c in range(COLS):
        amp = 0.35 + ctx.bands[c] * 2.8
        center = 2.5 + np.sin(c * 0.65 + st["ph"]) * amp
        for r in range(ROWS):
            d = abs(r - center)
            glow = np.exp(-d * 1.55) * (0.25 + 0.75 * ctx.bands[c])
            out[c, r] = ctx.pal((c / max(1, COLS - 1) + st["ph"] * 0.035) % 1.0) * glow
    return out


def fx_burst(ctx, st):
    """Ledakan dari pusat saat bass/beat naik."""
    pulse = st.get("pulse", 0.0) * np.exp(-ctx.dt * 4.5)
    if ctx.beat:
        pulse = 1.0
    st["pulse"] = pulse

    dist = np.hypot(CC - 7, (RR - 2.5) * 1.25)
    radius = 0.7 + pulse * 7.0 + ctx.bass * 2.5
    ring = np.exp(-((dist - radius) ** 2) / 1.5)
    core = np.exp(-(dist ** 2) / 8.0) * (0.15 + pulse * 0.8)
    return ctx.pal(np.clip(dist / 8.0 + ctx.pal.t * 0.04, 0, 1)) * (
        ring + core
    )[..., None]


def fx_matrix(ctx, st):
    """Hujan pixel vertikal dengan intensitas audio."""
    drops = st.setdefault("drops", np.random.uniform(-ROWS, 0, COLS))
    speeds = st.setdefault("speeds", np.random.uniform(2.0, 5.5, COLS))
    out = np.zeros((COLS, ROWS, 3))

    drops += speeds * (0.35 + ctx.bands * 1.4) * ctx.dt
    drops[drops > ROWS + 2] = np.random.uniform(-5, -1, np.count_nonzero(drops > ROWS + 2))

    for c in range(COLS):
        head = drops[c]
        strength = 0.25 + ctx.bands[c] * 0.9
        for tail in range(4):
            r = int(head) - tail
            if 0 <= r < ROWS:
                fade = (1.0 - tail / 4.5) * strength
                out[c, r] = ctx.pal(c / max(1, COLS - 1)) * fade

    return out


def fx_vortex(ctx, st):
    """Spiral/vortex kecil yang berputar dan diperkuat bass."""
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (1.2 + ctx.bass * 4.0)
    ph = st["ph"]

    dx = CC - 7
    dy = (RR - 2.5) * 1.35
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    spiral = np.sin(angle * 3.0 + radius * 1.8 - ph * 3.0) * 0.5 + 0.5
    mask = np.exp(-radius * 0.22)
    audio = 0.2 + 0.8 * np.clip(ctx.bands[:, None] + ctx.bass * 0.35, 0, 1)
    return ctx.pal((spiral + ph * 0.025) % 1.0) * (spiral * mask * audio)[..., None]


def fx_heartbeat(ctx, st):
    """Pulse seperti detak jantung, sinkron dengan beat/bass."""
    phase = st.get("phase", 0.0)
    if ctx.beat:
        phase = 1.0
    phase = max(0.0, phase - ctx.dt * (2.8 + ctx.vol * 4.0))
    st["phase"] = phase

    # Dua puncak kecil membuat pulse terasa seperti lub-dub.
    t = max(0.0, min(1.0, phase))
    pulse = max(np.exp(-((t - 0.72) / 0.11) ** 2),
                0.72 * np.exp(-((t - 0.48) / 0.08) ** 2))
    pulse *= 0.35 + ctx.bass * 0.65

    dist = np.hypot(CC - 7, (RR - 2.5) * 1.35)
    glow = np.exp(-dist * 0.30)
    return ctx.pal(np.clip(dist / 8.0, 0, 1)) * (pulse * glow)[..., None]




# ---------------------------------------------------------------- visualizer batch 2
def fx_plasma(ctx, st):
    """Plasma cair multi-gelombang, intensitas mengikuti audio."""
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (0.8 + ctx.vol * 3.0)
    ph = st["ph"]
    v = (
        np.sin(CC * 0.72 + ph)
        + np.sin(RR * 1.05 - ph * 0.8)
        + np.sin((CC + RR) * 0.48 + ph * 0.55)
        + np.sin(np.hypot(CC - 7, RR - 2.5) * 0.85 - ph * 1.2)
    ) / 4.0
    v = v * 0.5 + 0.5
    audio = 0.18 + 0.82 * np.clip(
        ctx.bands[:, None] * 1.15 + ctx.bass * 0.35, 0, 1
    )
    return ctx.pal((v + ph * 0.025) % 1.0) * (v * audio)[..., None]


def fx_galaxy(ctx, st):
    """Bintang/pixel yang berkedip dan bergerak perlahan."""
    stars = st.get("stars")
    if stars is None:
        rng = np.random.default_rng(42)
        stars = np.column_stack([
            rng.uniform(0, COLS - 1, 24),
            rng.uniform(0, ROWS - 1, 24),
            rng.uniform(0.5, 1.8, 24),
            rng.uniform(0, 1, 24),
        ])
        st["stars"] = stars

    out = np.zeros((COLS, ROWS, 3))
    stars[:, 0] += np.sin(stars[:, 3] * 7 + st.get("ph", 0)) * ctx.dt * 0.25
    stars[:, 1] += np.cos(stars[:, 3] * 5 + st.get("ph", 0)) * ctx.dt * 0.18
    stars[:, 0] %= COLS
    stars[:, 1] %= ROWS
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (0.5 + ctx.bass * 2.0)

    for x, y, speed, hue in stars:
        c = int(round(x))
        r = int(round(y))
        if 0 <= c < COLS and 0 <= r < ROWS:
            twinkle = 0.35 + 0.65 * (
                0.5 + 0.5 * np.sin(st["ph"] * speed * 2.5 + hue * 12)
            )
            intensity = twinkle * (0.25 + 0.9 * ctx.vol)
            out[c, r] = np.maximum(
                out[c, r],
                ctx.pal((hue + st["ph"] * 0.02) % 1.0) * intensity
            )

    # Bass creates a subtle galactic core.
    dist = np.hypot(CC - 7, (RR - 2.5) * 1.35)
    core = np.exp(-dist * 0.65) * ctx.bass * 0.5
    out += ctx.pal(0.55 + st["ph"] * 0.015) * core[..., None]
    return np.clip(out, 0, 1)


def fx_lightning(ctx, st):
    """Kilat bercabang ketika beat terdeteksi."""
    flash = st.get("flash", 0.0) * np.exp(-ctx.dt * 8.0)
    if ctx.beat:
        flash = 1.0
        st["bolt"] = np.random.default_rng().integers(0, ROWS, COLS)
    st["flash"] = flash

    bolt = st.get("bolt")
    out = np.zeros((COLS, ROWS, 3))
    if bolt is not None:
        for c in range(COLS):
            center = int(bolt[c])
            for r in range(ROWS):
                d = abs(r - center)
                if d <= 1:
                    out[c, r] = ctx.pal(
                        (c / max(1, COLS - 1) + st.get("flash", 0) * 0.08) % 1
                    ) * flash * (1.0 if d == 0 else 0.35)

    # Ambient electric field.
    out += ctx.pal(0.6) * (ctx.vol * 0.08)
    return np.clip(out, 0, 1)


def fx_comet(ctx, st):
    """Komet bergerak dengan trail, kecepatannya mengikuti musik."""
    pos = st.get("pos", -4.0)
    pos += ctx.dt * (2.0 + ctx.vol * 8.0)
    if pos > COLS + 4:
        pos = -4.0
    st["pos"] = pos

    out = np.zeros((COLS, ROWS, 3))
    center = 2.5 + np.sin(st.get("ph", 0.0)) * (0.8 + ctx.bass * 1.4)
    st["ph"] = st.get("ph", 0.0) + ctx.dt * 1.2

    for c in range(COLS):
        d = abs(c - pos)
        trail = np.exp(-d * 0.75)
        if trail < 0.02:
            continue
        for r in range(ROWS):
            dy = abs(r - center)
            glow = np.exp(-dy * 1.4) * trail
            out[c, r] = ctx.pal((c / max(1, COLS - 1) + st["ph"] * 0.04) % 1) * glow

    return out * (0.35 + 0.65 * ctx.vol)


def fx_dna(ctx, st):
    """Dua heliks pixel yang bergerak dan berinteraksi dengan audio."""
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (1.4 + ctx.bass * 3.5)
    ph = st["ph"]
    out = np.zeros((COLS, ROWS, 3))

    for c in range(COLS):
        t = c / max(1, COLS - 1)
        y1 = 2.5 + np.sin(t * np.pi * 3.0 + ph) * (1.5 + ctx.bands[c] * 1.0)
        y2 = 5.0 - y1
        for y, hue in ((y1, 0.05), (y2, 0.55)):
            r = int(round(y))
            if 0 <= r < ROWS:
                out[c, r] = ctx.pal((hue + t + ph * 0.025) % 1) * (
                    0.35 + ctx.bands[c] * 0.65
                )

        # Connecting rung.
        a, b = sorted((int(round(y1)), int(round(y2))))
        for r in range(max(0, a), min(ROWS, b + 1)):
            out[c, r] = np.maximum(
                out[c, r],
                ctx.pal((t + ph * 0.02) % 1) * 0.18
            )
    return out


def fx_orbital(ctx, st):
    """Partikel mengorbit pusat dengan radius reaktif."""
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (1.0 + ctx.vol * 4.0)
    ph = st["ph"]
    out = np.zeros((COLS, ROWS, 3))

    center_x, center_y = 7.0, 2.5
    for i in range(4):
        angle = ph * (1.0 + i * 0.22) + i * np.pi / 2
        rx = 2.0 + i * 0.75 + ctx.bass * 1.5
        ry = 0.7 + i * 0.35 + ctx.bass * 0.5
        x = center_x + np.cos(angle) * rx
        y = center_y + np.sin(angle) * ry

        dist = np.hypot(CC - x, (RR - y) * 1.2)
        glow = np.exp(-dist * 1.7) * (0.35 + ctx.vol * 0.8)
        out += ctx.pal((i / 4 + ph * 0.035) % 1) * glow[..., None]

    core = np.exp(-np.hypot(CC - center_x, (RR - center_y) * 1.2) * 1.4)
    out += ctx.pal(0.15 + ph * 0.03) * core[..., None] * ctx.bass * 0.55
    return np.clip(out, 0, 1)


def fx_mirror(ctx, st):
    """Spectrum mirror: energi tumbuh dari sumbu tengah."""
    out = np.zeros((COLS, ROWS, 3))
    half = ROWS / 2.0

    for c in range(COLS):
        level = float(ctx.bands[c])
        height = level * half
        for r in range(ROWS):
            d = abs((r + 0.5) - half)
            if d <= height:
                edge = 1.0 - max(0.0, d / max(height, 0.01))
                out[c, r] = ctx.pal(d / max(1.0, half)) * edge

    return out


def fx_explosion(ctx, st):
    """Pixel explosion dari pusat ketika beat."""
    particles = st.setdefault("particles", [])

    if ctx.beat:
        rng = np.random.default_rng()
        for _ in range(14):
            angle = rng.uniform(0, np.pi * 2)
            speed = rng.uniform(2.5, 7.0)
            particles.append([
                7.0, 2.5,
                np.cos(angle) * speed,
                np.sin(angle) * speed,
                rng.uniform(0, 1),
                0.0,
            ])

    out = np.zeros((COLS, ROWS, 3))
    keep = []
    for p in particles:
        p[0] += p[2] * ctx.dt
        p[1] += p[3] * ctx.dt
        p[5] += ctx.dt
        if p[5] < 1.5 and -1 <= p[0] < COLS + 1 and -1 <= p[1] < ROWS + 1:
            keep.append(p)
            c = int(round(p[0]))
            r = int(round(p[1]))
            if 0 <= c < COLS and 0 <= r < ROWS:
                fade = max(0.0, 1.0 - p[5] / 1.5)
                out[c, r] = ctx.pal(
                    (p[4] + p[5] * 0.15) % 1.0
                ) * fade

    st["particles"] = keep
    # Keep a faint center pulse between beats.
    core = np.exp(-np.hypot(CC - 7, (RR - 2.5) * 1.2) * 1.7)
    out += ctx.pal(0.0) * core[..., None] * ctx.bass * 0.22
    return np.clip(out, 0, 1)


def fx_soundwave(ctx, st):
    """Gelombang sinus horizontal yang dipengaruhi band audio."""
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (1.5 + ctx.vol * 5.0)
    ph = st["ph"]
    out = np.zeros((COLS, ROWS, 3))

    for c in range(COLS):
        amp = 0.45 + ctx.bands[c] * 1.6
        center = 2.5 + np.sin(c * 0.8 + ph) * amp
        for r in range(ROWS):
            d = abs(r - center)
            glow = np.exp(-d * 2.2)
            out[c, r] = ctx.pal(
                (c / max(1, COLS - 1) + ph * 0.03) % 1
            ) * glow * (0.2 + ctx.bands[c] * 0.9)

    return out


def fx_tunnel(ctx, st):
    """Energy tunnel: cincin konsentris menuju titik pusat."""
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (1.0 + ctx.vol * 3.5)
    ph = st["ph"]

    dist = np.hypot(CC - 7, (RR - 2.5) * 1.35)
    rings = np.sin(dist * 3.0 - ph * 5.0) * 0.5 + 0.5
    depth = np.exp(-dist * 0.22)
    audio = 0.18 + 0.82 * np.clip(
        ctx.bass * 0.65 + ctx.bands[:, None] * 0.7, 0, 1
    )

    return ctx.pal((rings + ph * 0.025) % 1.0) * (
        rings * depth * audio
    )[..., None]


NEW_MODES_BATCH_2 = [
    ("Plasma", fx_plasma),
    ("Galaxy", fx_galaxy),
    ("Lightning", fx_lightning),
    ("Comet", fx_comet),
    ("DNA", fx_dna),
    ("Orbital", fx_orbital),
    ("Spectrum Mirror", fx_mirror),
    ("Pixel Explosion", fx_explosion),
    ("Soundwave", fx_soundwave),
    ("Energy Tunnel", fx_tunnel),
]

NEW_MODES = [
    ("Spectrum Wave", fx_wave),
    ("Center Burst", fx_burst),
    ("Matrix Rain", fx_matrix),
    ("Vortex", fx_vortex),
    ("Heartbeat", fx_heartbeat),
]

MODES = [
    ("Spektrum Klasik", fx_bars),
    ("Peak Hold", fx_peak),
    ("Kupu-kupu (simetris)", fx_butterfly),
    ("Air Terjun (spektrogram)", fx_waterfall),
    ("Riak Beat", fx_ripple),
    ("Api", fx_fire),
    ("Hujan Neon", fx_rain),
    ("Aurora", fx_aurora),
    ("Denyut Tengah", fx_pulse),
] + NEW_MODES + NEW_MODES_BATCH_2
MODE_FUNCS = dict(MODES)
TEST_ALL = "[TES] Semua LED putih"
TEST_SWEEP = "[TES] Sapu satu per satu"
MODE_NAMES = [n for n, _ in MODES] + [TEST_ALL, TEST_SWEEP]


# ---------------------------------------------------------------- pengaturan
class Settings:
    KEYS = ("mode", "palette", "c1", "c2", "mirror", "sens", "bright", "decay", "audio")

    def __init__(self):
        self.mode = MODE_NAMES[0]
        self.palette = PALETTE_NAMES[0]
        self.c1 = (0, 255, 255)
        self.c2 = (255, 0, 200)
        self.mirror = True          # True = bass di kanan
        self.sens = 1.0
        self.bright = 1.0
        self.decay = 3.0
        self.audio = "(otomatis)"
        self.empty = set()          # {(kolom, baris)} slot tanpa tombol

    def load(self):
        try:
            with open(CONFIG, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return
        for k in self.KEYS:
            if k in d:
                setattr(self, k, d[k])
        self.c1, self.c2 = tuple(self.c1), tuple(self.c2)
        self.empty = {tuple(x) for x in d.get("empty", [])}
        if self.mode not in MODE_NAMES:
            self.mode = MODE_NAMES[0]
        if self.palette not in PALETTE_NAMES:
            self.palette = PALETTE_NAMES[0]

    def save(self):
        try:
            d = {k: getattr(self, k) for k in self.KEYS}
            d["empty"] = sorted(self.empty)
            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
        except Exception:
            pass

    def snapshot(self):
        d = {k: getattr(self, k) for k in self.KEYS}
        d["empty"] = sorted(self.empty)
        return d

    def apply(self, d):
        for k in self.KEYS:
            if k in d:
                setattr(self, k, d[k])
        self.c1, self.c2 = tuple(self.c1), tuple(self.c2)
        self.empty = {tuple(x) for x in d.get("empty", [])}
        if self.mode not in MODE_NAMES:
            self.mode = MODE_NAMES[0]
        if self.palette not in PALETTE_NAMES:
            self.palette = PALETTE_NAMES[0]

    def save_preset(self, name):
        os.makedirs(PRESETS_DIR, exist_ok=True)
        safe = re.sub(r"[^a-zA-Z0-9 _-]+", "", name).strip()
        if not safe:
            return False
        path = os.path.join(PRESETS_DIR, safe + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)
        return True

    def load_preset(self, name):
        path = os.path.join(PRESETS_DIR, name + ".json")
        with open(path, "r", encoding="utf-8") as f:
            self.apply(json.load(f))

    @staticmethod
    def preset_names():
        try:
            os.makedirs(PRESETS_DIR, exist_ok=True)
            return sorted(
                os.path.splitext(x)[0]
                for x in os.listdir(PRESETS_DIR)
                if x.lower().endswith(".json")
            )
        except Exception:
            return []


# ---------------------------------------------------------------- mesin (thread render + HID)
class Engine(threading.Thread):
    def __init__(self, S, audio):
        super().__init__(daemon=True)
        self.S, self.audio = S, audio
        self.an = Analyzer()
        self.pal = Palette(S)
        self.dev = None
        self.status = "keyboard: belum terhubung"
        self.preview = np.zeros((COLS, ROWS, 3), np.uint8)
        self.sweep = 0
        self.states = {}
        self.running = True

    @staticmethod
    def make_buf(rgb):
        buf = bytearray(520)
        buf[0:8] = HEADER
        buf[8:8 + N_LED * 3] = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        return bytes(buf)

    def connect(self):
        if hid is None:
            self.status = "keyboard: library hidapi belum terpasang (pip install hidapi)"
            return False
        blank = self.make_buf(np.zeros((COLS, ROWS, 3), np.uint8))
        try:
            devs = hid.enumerate(VID, 0)
        except Exception as e:
            self.status = f"keyboard: error enumerate ({e})"
            return False
        for d in devs:
            if d.get("usage_page") != 0xFF00:
                continue
            dev = hid.device()
            try:
                dev.open_path(d["path"])
                if dev.send_feature_report(blank) == 520:
                    self.dev = dev
                    self.status = f"keyboard terhubung (PID {d['product_id']:04X})"
                    return True
            except Exception:
                pass
            try:
                dev.close()
            except Exception:
                pass
        self.status = "keyboard tidak ketemu - tutup OemDrv.exe lalu klik 'Sambung ulang'"
        return False

    def disconnect(self):
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:
                pass
            self.dev = None

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
        if self.dev is not None:
            try:
                self.dev.send_feature_report(self.make_buf(np.zeros((COLS, ROWS, 3), np.uint8)))
            except Exception:
                pass
            self.disconnect()

    def run(self):
        S = self.S
        ctx = Ctx()
        ctx.pal = self.pal
        last = time.time()
        next_try = 0.0
        while self.running:
            now = time.time()
            dt = min(now - last, 0.2)
            last = now
            self.pal.t += dt

            raw, vol, bass, beat = self.an.process(self.audio.latest(), dt, S.sens, S.decay)
            valid = np.ones((COLS, ROWS), bool)
            for c, r in list(S.empty):
                if 0 <= c < COLS and 0 <= r < ROWS:
                    valid[c, r] = False
            ctx.dt, ctx.t = dt, now
            ctx.raw = raw
            ctx.bands = raw[::-1].copy() if S.mirror else raw
            ctx.vol, ctx.bass, ctx.beat = vol, bass, beat
            ctx.valid = valid
            ctx.rows_bu = [[r for r in range(ROWS - 1, -1, -1) if valid[c, r]] for c in range(COLS)]

            mode = S.mode
            try:
                if mode == TEST_ALL:
                    frame = np.ones((COLS, ROWS, 3))
                elif mode == TEST_SWEEP:
                    idx = int(now * 2.5) % N_LED
                    self.sweep = idx
                    frame = np.zeros((COLS, ROWS, 3))
                    frame[idx // ROWS, idx % ROWS] = 1.0
                else:
                    fn = MODE_FUNCS.get(mode, fx_bars)
                    st = self.states.setdefault(mode, {})
                    frame = fn(ctx, st) * valid[..., None]
            except Exception as e:           # jangan sampai thread mati karena satu efek
                frame = np.zeros((COLS, ROWS, 3))
                self.status = f"efek error: {e}"
            frame = np.clip(frame * S.bright, 0.0, 1.0)
            rgb = (frame * 255.0 + 0.5).astype(np.uint8)
            self.preview = rgb

            if self.dev is not None:
                try:
                    if self.dev.send_feature_report(self.make_buf(rgb)) != 520:
                        raise IOError("write gagal")
                except Exception:
                    self.disconnect()
                    self.status = "keyboard: koneksi putus, mencoba menyambung ulang..."
            elif now >= next_try:
                self.connect()
                next_try = now + 2.0

            time.sleep(max(0.0, 1.0 / FPS - (time.time() - now)))



# ---------------------------------------------------------------- GUI
# UI v2: dashboard modern, visualizer library, audio monitor, dan LED preview.
BG = "#0b0b10"
SURFACE = "#12121a"
SURFACE_2 = "#181821"
SURFACE_3 = "#20202b"
BORDER = "#292936"
FG = "#f1f1f7"
MUTED = "#8d8da0"
ACCENT = "#8b5cf6"
ACCENT_2 = "#22d3ee"
GOOD = "#34d399"
DANGER = "#fb7185"


class App:
    LED_CELL = 42
    LED_GAP = 7
    LED_PAD = 18

    def __init__(self):
        self.S = Settings()
        self.S.load()
        self.audio = Audio(self.S)
        self.eng = Engine(self.S, self.audio)

        self.root = tk.Tk()
        self.root.title("FuryCube V68 • Visualizer")
        self.root.configure(bg=BG)
        self.root.minsize(1120, 760)
        self.root.geometry("1180x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._style()
        self._build()

        self.preview_glow = True
        self.preview_zoom = 1.0
        self.on_palette()
        self.audio.start()
        self.eng.start()
        self.refresh()

    # ---- styling ---------------------------------------------------------
    def _style(self):
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except Exception:
            pass

        s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=SURFACE)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Card.TLabel", background=SURFACE, foreground=FG)
        s.configure("Muted.Card.TLabel", background=SURFACE, foreground=MUTED)
        s.configure("Title.TLabel", background=BG, foreground=FG,
                    font=("Segoe UI Semibold", 18))
        s.configure("Section.TLabel", background=SURFACE, foreground=FG,
                    font=("Segoe UI Semibold", 11))
        s.configure("Metric.TLabel", background=SURFACE, foreground=FG,
                    font=("Segoe UI Semibold", 16))
        s.configure("Small.TLabel", background=SURFACE, foreground=MUTED,
                    font=("Segoe UI", 9))

        s.configure("TButton", background=SURFACE_2, foreground=FG,
                    borderwidth=0, padding=(12, 8))
        s.map("TButton", background=[("active", SURFACE_3)])

        s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    borderwidth=0, padding=(12, 8))
        s.map("Accent.TButton", background=[("active", "#7c3aed")])

        s.configure("TCombobox", fieldbackground=SURFACE_2,
                    background=SURFACE_2, foreground=FG,
                    arrowcolor=FG, borderwidth=0)
        s.map("TCombobox", fieldbackground=[("readonly", SURFACE_2)],
              foreground=[("readonly", FG)])
        s.configure("Horizontal.TScale", background=SURFACE)

        self.root.option_add("*TCombobox*Listbox.background", SURFACE_2)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

    def _card(self, parent, **kwargs):
        f = ttk.Frame(parent, style="Card.TFrame", **kwargs)
        return f

    def _build(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=22, pady=18)

        # Header
        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x", pady=(0, 16))

        title_box = tk.Frame(header, bg=BG)
        title_box.pack(side="left")
        tk.Label(title_box, text="FURYCUBE", bg=BG, fg=FG,
                 font=("Segoe UI Semibold", 21)).pack(anchor="w")
        tk.Label(title_box, text="V68  /  AUDIO VISUALIZER", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")

        status_box = tk.Frame(header, bg=SURFACE, padx=12, pady=7)
        status_box.pack(side="right")
        self.device_dot = tk.Label(status_box, text="●", bg=SURFACE,
                                   fg=MUTED, font=("Segoe UI", 11))
        self.device_dot.pack(side="left", padx=(0, 7))
        self.device_status = tk.Label(status_box, text="Connecting...",
                                      bg=SURFACE, fg=MUTED,
                                      font=("Segoe UI Semibold", 9))
        self.device_status.pack(side="left")

        # Main 2-column area
        body = tk.Frame(outer, bg=BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_preview(body)
        self._build_controls(body)

        # Bottom audio monitor
        self._build_audio_monitor(outer)

    def _build_preview(self, parent):
        card = self._card(parent)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        top = tk.Frame(card, bg=SURFACE)
        top.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 4))

        tk.Label(top, text="LIVE MATRIX", bg=SURFACE, fg=FG,
                 font=("Segoe UI Semibold", 12)).pack(side="left")
        self.preview_info = tk.Label(top, text="15 × 6", bg=SURFACE,
                                     fg=MUTED, font=("Segoe UI", 9))
        self.preview_info.pack(side="right")

        holder = tk.Frame(card, bg="#09090d")
        holder.grid(row=1, column=0, sticky="nsew", padx=18, pady=(8, 18))

        w = self.LED_PAD * 2 + COLS * (self.LED_CELL + self.LED_GAP)
        h = self.LED_PAD * 2 + ROWS * (self.LED_CELL + self.LED_GAP)

        self.canvas = tk.Canvas(holder, width=w, height=h, bg="#09090d",
                                highlightthickness=0)
        self.canvas.pack(expand=True, pady=20)

        self.rects, self.texts = {}, {}
        for c in range(COLS):
            for rr in range(ROWS):
                x0 = self.LED_PAD + c * (self.LED_CELL + self.LED_GAP)
                y0 = self.LED_PAD + rr * (self.LED_CELL + self.LED_GAP)
                self.rects[c, rr] = self.canvas.create_rectangle(
                    x0, y0, x0 + self.LED_CELL, y0 + self.LED_CELL,
                    fill="#171720", outline="#272733", width=1)
                self.texts[c, rr] = self.canvas.create_text(
                    x0 + self.LED_CELL / 2, y0 + self.LED_CELL / 2,
                    text="", fill=MUTED, font=("Segoe UI", 8, "bold"))
        self.canvas.bind("<Button-1>", self.on_click)

        tk.Label(card,
                 text="Klik LED untuk menandai slot kosong • gunakan TEST untuk validasi mapping",
                 bg=SURFACE, fg=MUTED, font=("Segoe UI", 8)).grid(
                     row=2, column=0, pady=(0, 12))

    def _build_controls(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(9, 0))

        self._build_visualizer_card(right)
        self._build_palette_card(right)
        self._build_settings_card(right)
        self._build_actions(right)

    def _build_visualizer_card(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 9))

        header = tk.Frame(card, bg=SURFACE)
        header.pack(fill="x", padx=16, pady=(14, 8))

        tk.Label(header, text="VISUALIZER LIBRARY", bg=SURFACE, fg=FG,
                 font=("Segoe UI Semibold", 11)).pack(side="left")

        self.mode_count = tk.Label(
            header, text=f"{len(MODES)} MODES", bg=SURFACE,
            fg=MUTED, font=("Segoe UI", 8))
        self.mode_count.pack(side="right")

        # Scrollable card grid. Small enough for the current window, but
        # scrollable so adding more visualizers won't destroy the layout.
        wrap = tk.Frame(card, bg=SURFACE)
        wrap.pack(fill="x", padx=12, pady=(0, 8))

        self.viz_canvas = tk.Canvas(
            wrap, height=170, bg=SURFACE, highlightthickness=0)
        self.viz_scroll = ttk.Scrollbar(
            wrap, orient="vertical", command=self.viz_canvas.yview)
        self.viz_inner = tk.Frame(self.viz_canvas, bg=SURFACE)

        self.viz_inner.bind(
            "<Configure>",
            lambda e: self.viz_canvas.configure(
                scrollregion=self.viz_canvas.bbox("all")))
        self.viz_canvas.create_window(
            (0, 0), window=self.viz_inner, anchor="nw", width=390)
        self.viz_canvas.configure(yscrollcommand=self.viz_scroll.set)

        self.viz_canvas.pack(side="left", fill="both", expand=True)
        self.viz_scroll.pack(side="right", fill="y")

        self.mode_cards = {}
        self._build_visualizer_cards()

        # Keep the old selector as a compact accessibility/fallback control.
        bottom = tk.Frame(card, bg=SURFACE)
        bottom.pack(fill="x", padx=16, pady=(2, 13))
        self.mode_var = tk.StringVar(value=self.S.mode)
        self.mode_combo = ttk.Combobox(
            bottom, textvariable=self.mode_var, values=MODE_NAMES,
            state="readonly")
        self.mode_combo.pack(fill="x")
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode)

    def _build_visualizer_cards(self):
        for child in self.viz_inner.winfo_children():
            child.destroy()

        names = [n for n, _ in MODES]
        for i, name in enumerate(names):
            rr, cc = divmod(i, 2)
            card = tk.Frame(
                self.viz_inner, bg=SURFACE_2, highlightthickness=1,
                highlightbackground=BORDER, cursor="hand2")
            card.grid(row=rr, column=cc, padx=4, pady=4, sticky="ew")
            self.viz_inner.columnconfigure(cc, weight=1)

            thumb = tk.Canvas(card, width=72, height=42, bg="#0a0a0f",
                              highlightthickness=0)
            thumb.pack(side="left", padx=7, pady=7)
            self._draw_viz_thumbnail(thumb, i)

            info = tk.Frame(card, bg=SURFACE_2)
            info.pack(side="left", fill="both", expand=True, padx=(0, 5))
            tk.Label(info, text=name, bg=SURFACE_2, fg=FG,
                     font=("Segoe UI Semibold", 8),
                     anchor="w").pack(fill="x", pady=(8, 0))
            reactive = "BEAT" if any(x in name.lower()
                                     for x in ("beat", "burst", "heartbeat", "riak")) else "AUDIO"
            tk.Label(info, text=reactive, bg=SURFACE_2, fg=MUTED,
                     font=("Segoe UI", 7), anchor="w").pack(fill="x")

            for widget in (card, thumb, info):
                widget.bind("<Button-1>",
                            lambda e, n=name: self._select_visualizer(n))

            self.mode_cards[name] = card

        self._highlight_visualizer(self.S.mode)

    def _draw_viz_thumbnail(self, canvas, idx):
        # Static mini-preview; actual LED preview remains live.
        canvas.delete("all")
        w, h = 72, 42
        seed = [
            [0.15, 0.35, 0.8, 0.55, 0.25],
            [0.2, 0.55, 0.35, 0.75, 0.3],
            [0.8, 0.45, 0.25, 0.45, 0.8],
            [0.15, 0.75, 0.3, 0.9, 0.2],
        ][idx % 4]
        for i, v in enumerate(seed):
            x = 6 + i * 15
            hh = v * 28
            canvas.create_rectangle(x, h - hh - 5, x + 8, h - 5,
                                    fill=ACCENT if i % 2 else ACCENT_2,
                                    outline="")
        canvas.create_oval(30, 14, 42, 26, outline="#ffffff",
                           width=1)

    def _select_visualizer(self, name):
        self.S.mode = name
        self.mode_var.set(name)
        self.S.save()
        self._highlight_visualizer(name)

    def _highlight_visualizer(self, name):
        for n, card in self.mode_cards.items():
            selected = n == name
            card.config(
                bg=ACCENT if selected else SURFACE_2,
                highlightbackground=ACCENT_2 if selected else BORDER)
            for child in card.winfo_children():
                if isinstance(child, tk.Frame):
                    child.config(bg=ACCENT if selected else SURFACE_2)
                    for sub in child.winfo_children():
                        if isinstance(sub, tk.Label):
                            sub.config(bg=ACCENT if selected else SURFACE_2)

    def _build_palette_card(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 9))

        tk.Label(card, text="COLOR & PALETTE", bg=SURFACE, fg=FG,
                 font=("Segoe UI Semibold", 11)).pack(
                     anchor="w", padx=16, pady=(14, 8))

        row = tk.Frame(card, bg=SURFACE)
        row.pack(fill="x", padx=16, pady=(0, 8))
        self.pal_var = tk.StringVar(value=self.S.palette)
        self.pal_combo = ttk.Combobox(
            row, textvariable=self.pal_var, values=PALETTE_NAMES,
            state="readonly")
        self.pal_combo.pack(fill="x")
        self.pal_combo.bind("<<ComboboxSelected>>", self.on_palette)

        colors = tk.Frame(card, bg=SURFACE)
        colors.pack(fill="x", padx=16, pady=(2, 14))
        self.b1 = tk.Button(colors, text="A", width=5, relief="flat",
                            bd=0, command=lambda: self.pick("c1"))
        self.b2 = tk.Button(colors, text="B", width=5, relief="flat",
                            bd=0, command=lambda: self.pick("c2"))
        self.b1.pack(side="left")
        self.b2.pack(side="left", padx=(7, 0))
        tk.Label(colors, text="Custom gradient", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=10)
        self._paint_color_buttons()

    def _build_settings_card(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 9))

        tk.Label(card, text="RESPONSE", bg=SURFACE, fg=FG,
                 font=("Segoe UI Semibold", 11)).pack(
                     anchor="w", padx=16, pady=(14, 5))

        self.slider(card, "Sensitivity", "sens", 0.3, 3.0)
        self.slider(card, "Brightness", "bright", 0.05, 1.0)
        self.slider(card, "Decay", "decay", 0.5, 8.0)

        mirror = tk.Frame(card, bg=SURFACE)
        mirror.pack(fill="x", padx=16, pady=(4, 14))
        self.mirror_var = tk.BooleanVar(value=self.S.mirror)
        ttk.Checkbutton(
            mirror, text="Bass on right", variable=self.mirror_var,
            command=self.on_mirror).pack(side="left")

    def slider(self, parent, label, attr, lo, hi):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", padx=16, pady=4)

        top = tk.Frame(row, bg=SURFACE)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=SURFACE, fg=FG,
                 font=("Segoe UI", 9)).pack(side="left")
        value = tk.Label(top, width=5, bg=SURFACE, fg=MUTED,
                         font=("Segoe UI", 9))
        value.pack(side="right")
        value.config(text=f"{getattr(self.S, attr):.2f}")

        def cb(v):
            val = float(v)
            setattr(self.S, attr, val)
            value.config(text=f"{val:.2f}")

        scale = ttk.Scale(row, from_=lo, to=hi, orient="horizontal",
                          command=cb)
        scale.set(getattr(self.S, attr))
        scale.pack(fill="x", pady=(2, 0))
        scale.bind("<ButtonRelease-1>", lambda e: self.S.save())

    def _build_actions(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, 5))

        ttk.Button(row, text="Reconnect", command=self.reconnect,
                   style="Accent.TButton").pack(side="left", fill="x",
                                                expand=True, padx=(0, 4))
        ttk.Button(row, text="Clear ✕", command=self.clear_empty).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

        # Presets
        preset = self._card(parent)
        preset.pack(fill="x", pady=(4, 0))

        head = tk.Frame(preset, bg=SURFACE)
        head.pack(fill="x", padx=14, pady=(10, 6))
        tk.Label(head, text="PRESETS", bg=SURFACE, fg=FG,
                 font=("Segoe UI Semibold", 10)).pack(side="left")
        self.preset_status = tk.Label(head, text="", bg=SURFACE, fg=MUTED,
                                      font=("Segoe UI", 8))
        self.preset_status.pack(side="right")

        line = tk.Frame(preset, bg=SURFACE)
        line.pack(fill="x", padx=14, pady=(0, 10))

        self.preset_var = tk.StringVar()
        self.preset_combo = ttk.Combobox(
            line, textvariable=self.preset_var,
            values=Settings.preset_names(), state="readonly")
        self.preset_combo.pack(side="left", fill="x", expand=True)

        ttk.Button(line, text="Load",
                   command=self.load_preset).pack(side="left", padx=(6, 0))
        ttk.Button(line, text="Save",
                   command=self.save_preset).pack(side="left", padx=(6, 0))

        preview = tk.Frame(preset, bg=SURFACE)
        preview.pack(fill="x", padx=14, pady=(0, 11))
        self.glow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preview, text="LED glow",
                        variable=self.glow_var).pack(side="left")
        ttk.Button(preview, text="Reset preview",
                   command=self.reset_preview).pack(side="right")

    def _build_audio_monitor(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(12, 0))

        left = tk.Frame(card, bg=SURFACE)
        left.pack(side="left", fill="x", expand=True, padx=16, pady=12)

        tk.Label(left, text="AUDIO", bg=SURFACE, fg=FG,
                 font=("Segoe UI Semibold", 10)).pack(anchor="w")
        self.audio_name = tk.Label(left, text="Starting...",
                                   bg=SURFACE, fg=MUTED,
                                   font=("Segoe UI", 8))
        self.audio_name.pack(anchor="w", pady=(2, 5))

        self.audio_bar = ttk.Progressbar(left, maximum=100, length=300)
        self.audio_bar.pack(fill="x")

        metrics = tk.Frame(card, bg=SURFACE)
        metrics.pack(side="right", padx=16, pady=12)

        self.vol_metric = self._metric(metrics, "VOL")
        self.bass_metric = self._metric(metrics, "BASS")
        self.beat_metric = self._metric(metrics, "BEAT")

    def _metric(self, parent, name):
        box = tk.Frame(parent, bg=SURFACE_2, padx=10, pady=5)
        box.pack(side="left", padx=3)
        tk.Label(box, text=name, bg=SURFACE_2, fg=MUTED,
                 font=("Segoe UI", 7, "bold")).pack()
        value = tk.Label(box, text="0%", bg=SURFACE_2, fg=FG,
                         font=("Segoe UI Semibold", 10))
        value.pack()
        return value

    # ---- events ----------------------------------------------------------
    def on_mode(self, _=None):
        self.S.mode = self.mode_var.get()
        self.S.save()
        if hasattr(self, "mode_cards"):
            self._highlight_visualizer(self.S.mode)

    def on_palette(self, _=None):
        self.S.palette = self.pal_var.get()
        state = "normal" if self.S.palette == CUSTOM else "disabled"
        self.b1.config(state=state)
        self.b2.config(state=state)
        self.S.save()

    def on_mirror(self):
        self.S.mirror = bool(self.mirror_var.get())
        self.S.save()

    def pick(self, key):
        cur = "#%02x%02x%02x" % getattr(self.S, key)
        res = colorchooser.askcolor(color=cur, parent=self.root)
        if res and res[0]:
            setattr(self.S, key, tuple(int(v) for v in res[0]))
            self._paint_color_buttons()
            self.S.save()

    def _paint_color_buttons(self):
        for b, k in ((self.b1, "c1"), (self.b2, "c2")):
            col = getattr(self.S, k)
            b.config(bg="#%02x%02x%02x" % col,
                     fg="#000000" if sum(col) > 380 else "#ffffff")

    def on_click(self, ev):
        c = int((ev.x - self.LED_PAD) // (self.LED_CELL + self.LED_GAP))
        r = int((ev.y - self.LED_PAD) // (self.LED_CELL + self.LED_GAP))
        if 0 <= c < COLS and 0 <= r < ROWS:
            self.S.empty.symmetric_difference_update({(c, r)})
            self.S.save()

    def clear_empty(self):
        self.S.empty.clear()
        self.S.save()

    def reconnect(self):
        self.eng.disconnect()
        self.eng.status = "keyboard: reconnecting..."

    def refresh(self):
        frame = self.eng.preview
        test = self.S.mode in (TEST_ALL, TEST_SWEEP)

        for c in range(COLS):
            for r in range(ROWS):
                R, G, B = (int(v) for v in frame[c, r])
                active = R + G + B >= 20

                if active:
                    col = "#%02x%02x%02x" % (R, G, B)
                    outline = col
                    if self.glow_var.get():
                        # Bright LED gets a slightly brighter border to
                        # simulate bloom without adding heavy canvas objects.
                        edge = tuple(min(255, int(v * 1.18 + 8)) for v in (R, G, B))
                        outline = "#%02x%02x%02x" % edge
                else:
                    col = "#171720"
                    outline = "#272733"

                self.canvas.itemconfig(
                    self.rects[c, r], fill=col, outline=outline)

                if (c, r) in self.S.empty:
                    txt = "×"
                    txt_color = DANGER
                elif test:
                    txt = str(c * ROWS + r)
                    txt_color = "#111118"
                else:
                    txt = ""
                    txt_color = MUTED

                self.canvas.itemconfig(
                    self.texts[c, r], text=txt, fill=txt_color)

        if self.eng.status.startswith("keyboard terhubung"):
            self.device_dot.config(fg=GOOD)
            self.device_status.config(text="DEVICE CONNECTED", fg=GOOD)
        else:
            self.device_dot.config(fg=DANGER if "tidak" in self.eng.status else MUTED)
            self.device_status.config(text="DEVICE OFFLINE", fg=MUTED)

        self.audio_name.config(text=self.audio.status)
        self.audio_bar["value"] = max(0.0, min(100.0, self.eng.an.vol * 100))
        self.vol_metric.config(text=f"{int(self.eng.an.vol * 100)}%")
        self.bass_metric.config(text=f"{int(self.eng.an.level[:3].mean() * 100)}%")

        if self.eng.an.last_beat > self.eng.an.t - 0.18:
            self.beat_metric.config(text="BEAT", fg=ACCENT_2)
        else:
            self.beat_metric.config(text="—", fg=FG)

        if self.S.mode == TEST_SWEEP:
            self.preview_info.config(text=f"TEST • slot {self.eng.sweep}")
        elif self.S.mode == TEST_ALL:
            self.preview_info.config(text="TEST • ALL LEDs")
        else:
            self.preview_info.config(text=f"15 × 6  •  {self.S.mode}")

        self.root.after(50, self.refresh)

    def save_preset(self):
        import tkinter.simpledialog as simpledialog
        name = simpledialog.askstring(
            "Save preset", "Preset name:", parent=self.root)
        if not name:
            return
        try:
            if self.S.save_preset(name):
                self.preset_var.set(name.strip())
                self.preset_combo["values"] = Settings.preset_names()
                self.preset_status.config(text="Saved")
        except Exception as e:
            self.preset_status.config(text=f"Error: {e}")

    def load_preset(self):
        name = self.preset_var.get().strip()
        if not name:
            return
        try:
            self.S.load_preset(name)
            self.mode_var.set(self.S.mode)
            self.pal_var.set(self.S.palette)
            self.mirror_var.set(self.S.mirror)
            self._highlight_visualizer(self.S.mode)
            self.on_palette()
            self._paint_color_buttons()
            self.S.save()
            self.preset_status.config(text=f"Loaded • {name}")
        except Exception as e:
            self.preset_status.config(text=f"Error: {e}")

    def reset_preview(self):
        self.glow_var.set(True)
        self.preset_status.config(text="Preview reset")

    def on_close(self):
        self.S.save()
        self.audio.running = False
        self.eng.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if tk is None:
        sys.exit("tkinter tidak tersedia (Arch: sudo pacman -S tk)")
    App().run()
