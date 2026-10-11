#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FuryCube V68 - Audio Visualizer (GUI v2, berbasis layout fisik V68)

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
    import tkinter.font as tkfont
    from tkinter import ttk, colorchooser, messagebox
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

# ================================================================ palet
PALETTES = {
    "Klasik (hijau > merah)": [(0, 255, 60), (180, 255, 0), (255, 200, 0), (255, 90, 0), (255, 0, 0)],
    "Neon": [(0, 255, 255), (120, 0, 255), (255, 0, 200)],
    "Api": [(120, 0, 0), (255, 0, 0), (255, 140, 0), (255, 230, 100), (255, 255, 255)],
    "Laut": [(0, 40, 120), (0, 160, 255), (0, 255, 200), (200, 255, 255)],
    "Sunset": [(80, 0, 160), (255, 0, 120), (255, 140, 0), (255, 230, 80)],
    "Matrix": [(0, 60, 0), (0, 255, 70), (200, 255, 200)],
    "Cinta (pink-merah)": [(110, 0, 40), (255, 20, 90), (255, 90, 150), (255, 200, 220)],
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


# ================================================================ analisis audio
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


# ================================================================ efek
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
    order = [3, 2, 4, 1, 5, 0]          # dari tengah area tombol (baris 1-5) ke luar
    for c in range(COLS):
        d = abs(c - 7)
        h = float(ctx.raw[2 * d]) * 5.0
        k = 0
        for r in order:
            if r == 0:
                continue                 # baris 0 tidak punya tombol
            b = min(1.0, h - k)
            k += 1
            if b <= 0.03:
                break
            if not ctx.valid[c, r]:
                continue
            x = 0.1 + 0.9 * abs(r - 3) / 2.0
            out[c, r] = ctx.pal(x) * b
    return out


def fx_waterfall(ctx, st):
    hist = st.setdefault("hist", np.zeros((COLS, ROWS)))
    st["acc"] = st.get("acc", 0.0) + ctx.dt
    if st["acc"] >= 0.07:
        st["acc"] = 0.0
        hist[:-1] = hist[1:].copy()
        groups = np.array_split(np.arange(COLS), ROWS - 1)     # 5 baris tombol
        col = np.array([ctx.raw[g].mean() for g in groups])    # 0 = bass
        hist[-1, 1:] = col[::-1]                                # baris 5 (bawah) = bass
        hist[-1, 0] = 0.0
    v = np.clip(hist * 1.2, 0, 1)
    return ctx.pal(v) * v[..., None]


def fx_ripple(ctx, st):
    rings = st.setdefault("rings", [])
    if ctx.beat:
        rings.append([np.random.uniform(3, 11), np.random.uniform(1.5, 4.5), 0.0, np.random.rand()])
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
        drops.append([int(c), 0.5, np.random.uniform(7, 13), float(ctx.bands[c])])
    if ctx.beat:
        for c in np.random.choice(COLS, 4, replace=False):
            drops.append([int(c), 0.5, np.random.uniform(11, 16), 1.0])
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
         + np.sin(np.hypot(CC - 7, RR - 3) * 0.6 - ph)) / 4.0 * 0.5 + 0.5
    lvl = 0.25 + 0.75 * np.clip(ctx.bands[:, None] * 1.2 + ctx.bass * 0.3, 0, 1)
    return ctx.pal(v) * lvl[..., None]


def fx_pulse(ctx, st):
    flash = st.get("flash", 0.0) * np.exp(-ctx.dt * 5.0)
    if ctx.beat:
        flash = 1.0
    st["flash"] = flash
    dist = np.hypot(CC - 7, (RR - 3) * 1.6) / 8.0
    inten = np.clip((0.12 + 0.55 * ctx.bass + 0.6 * flash) - dist * 0.55, 0, 1)
    return ctx.pal(np.clip(0.1 + dist * 0.9, 0, 1)) * inten[..., None]


# ---------------------------------------------------------------- efek Love (area keycap khusus berbentuk hati)
HEART_NAMES = ["8", "9", "-", "=",
               "U", "I", "O", "P", "[", "]",
               "J", "K", "L", ";", "'",
               "M", ",", ".", "/",
               "Alt (kanan)"]
HEART_CX, HEART_CY = 10.5, 1.9           # pusat hati (satuan u, sama dengan layout GUI)


def _heart_cells(km):
    """[(kolom, baris, x_u, y_u)] untuk tiap tombol area hati, dari peta tombol saat ini."""
    pts = []
    for name in HEART_NAMES:
        key = km.by_name.get(name)
        if key is None:
            continue
        sl = km.slots.get(key.id) or []
        if sl:
            pts.append((sl[0][0], sl[0][1], key.x + key.w / 2.0, key.y + 0.5))
    return pts


def _heart_t(ctx):
    """0 = ujung bawah hati, 1 = lekuk atas."""
    if len(ctx.hy) == 0:
        return ctx.hy
    y0, y1 = ctx.hy.min(), ctx.hy.max()
    return (y1 - ctx.hy) / max(y1 - y0, 1e-6)


def _love_out(ctx, col):
    """col: (n,3) warna tiap tombol hati. Tambahkan latar spektrum redup di luar hati (opsional)."""
    out = np.zeros((COLS, ROWS, 3))
    if len(ctx.hc):
        out[ctx.hc, ctx.hr] = col
    if ctx.love_bg:
        bg = _bars(ctx, ctx.bands, 0.30)
        if len(ctx.hc):
            bg[ctx.hc, ctx.hr] = 0.0
        out = out + bg
    return out


def fx_love_beat(ctx, st):
    since = st.get("since", 9.0) + ctx.dt
    if ctx.beat or since > 1.25:           # tanpa beat tetap berdetak pelan
        since = 0.0
    st["since"] = since
    e = min(1.0, np.exp(-((since - 0.06) / 0.07) ** 2) + 0.7 * np.exp(-((since - 0.30) / 0.09) ** 2))
    amp = (0.22 + 0.78 * e) * (0.75 + 0.25 * ctx.vol)
    d = np.hypot(ctx.hx - HEART_CX, ctx.hy - HEART_CY) / 3.2
    val = amp * np.clip(1.0 - 0.4 * d, 0, 1)
    col = ctx.pal(np.clip(0.2 + 0.7 * e - 0.15 * d, 0, 1)) * val[:, None]
    return _love_out(ctx, col)


def fx_love_fill(ctx, st):
    lvl = float(np.clip(0.55 * ctx.vol + 0.65 * ctx.bass, 0, 1))
    cur = max(lvl, st.get("lvl", 0.0) - 1.3 * ctx.dt)
    st["lvl"] = cur
    t = _heart_t(ctx)
    b = np.clip((cur * 1.15 - t) * 5.0, 0, 1)
    col = ctx.pal(np.clip(0.2 + 0.8 * t, 0, 1)) * (0.07 + 0.93 * b)[:, None]
    return _love_out(ctx, col)


def fx_love_ripple(ctx, st):
    rings = st.setdefault("rings", [])
    since = st.get("since", 9.0) + ctx.dt
    if ctx.beat or since > 1.4:
        rings.append([0.0, np.random.rand()])
        since = 0.0
    st["since"] = since
    d = np.hypot(ctx.hx - HEART_CX, ctx.hy - HEART_CY)
    col = np.zeros((len(d), 3)) + (0.10 + 0.15 * ctx.bass) * ctx.pal(0.25)[None, :]
    for ring in rings:
        ring[0] += ctx.dt
        age, hue = ring
        inten = np.exp(-age * 2.2) * np.exp(-(((d - age * 6.0) / 0.8) ** 2))
        col += inten[:, None] * ctx.pal((hue + ctx.pal.t * 0.05) % 1.0)
    st["rings"] = [r for r in rings if np.exp(-r[0] * 2.2) > 0.05]
    return _love_out(ctx, np.clip(col, 0, 1))


def fx_love_spectrum(ctx, st):
    t = _heart_t(ctx)
    if len(t) == 0:
        return _love_out(ctx, np.zeros((0, 3)))
    x0, x1 = ctx.hx.min(), ctx.hx.max()
    u = (ctx.hx - x0) / max(x1 - x0, 1e-6)
    lvl = np.interp(u * (COLS - 1), np.arange(COLS), ctx.bands)
    b = np.clip((lvl * 1.4 - t) * 4.0, 0, 1)
    col = ctx.pal(np.clip(0.2 + 0.8 * t, 0, 1)) * (0.07 + 0.93 * b)[:, None]
    return _love_out(ctx, col)


def fx_love_shimmer(ctx, st):
    st["ph"] = st.get("ph", 0.0) + ctx.dt * (0.8 + 3.5 * ctx.vol)
    ph = st["ph"]
    v = 0.5 + 0.5 * np.sin(ctx.hx * 1.1 + ctx.hy * 0.8 - ph * 2.0)
    w = 0.5 + 0.5 * np.sin(ctx.hx * 2.3 - ctx.hy * 1.7 + ph * 1.3)
    flash = st.get("flash", 0.0) * np.exp(-ctx.dt * 6.0)
    if ctx.beat:
        flash = 1.0
    st["flash"] = flash
    val = np.clip(0.30 + 0.45 * w + 0.35 * flash, 0, 1)
    col = np.clip(ctx.pal(v) * val[:, None] + 0.30 * flash, 0, 1)
    return _love_out(ctx, col)


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
    ("Love - Detak jantung", fx_love_beat),
    ("Love - Isi (VU hati)", fx_love_fill),
    ("Love - Riak beat", fx_love_ripple),
    ("Love - Spektrum hati", fx_love_spectrum),
    ("Love - Berkilau", fx_love_shimmer),
]
LOVE_MODES = {n for n, _ in MODES if n.startswith("Love")}
MODE_FUNCS = dict(MODES)
TEST_ALL = "[TES] Semua LED putih"
TEST_SWEEP = "[TES] Kalibrasi LED (sapu)"
MODE_NAMES = [n for n, _ in MODES] + [TEST_ALL, TEST_SWEEP]


# ================================================================ layout fisik V68 & peta slot
class Key:
    def __init__(self, kid, label, name, x, y, w, default, kind="key"):
        self.id, self.label, self.name = kid, label, name
        self.x, self.y, self.w = x, y, w
        self.default = default
        self.cap = max(1, len(default))      # jumlah LED maksimum (spasi = 3)
        self.kind = kind


# (label, lebar dalam u)  |  None = jarak kosong.  Daftar kolom = kolom matriks LED tiap tombol
# (list di dalam list = tombol dengan beberapa LED, mis. spasi).  Baris GUI = baris fisik + 1.
_ROWS = [
    ([("Esc", 1)] + [(ch, 1) for ch in "1234567890-="] + [("Bksp", 2)],
     list(range(14))),
    ([("Tab", 1.5)] + [(ch, 1) for ch in "QWERTYUIOP[]"] + [("\\", 1.5), (None, 0.6), ("Del", 1)],
     list(range(15))),
    ([("Caps", 1.75)] + [(ch, 1) for ch in "ASDFGHJKL;'"] + [("Enter", 2.25), (None, 0.6), ("PgUp", 1)],
     list(range(12)) + [13, 14]),
    ([("Shift", 2.25)] + [(ch, 1) for ch in "ZXCVBNM,./"] + [("Shift", 2.1), (None, 0.25), ("\u2191", 1), ("PgDn", 1)],
     list(range(12)) + [13, 14]),
    ([("Ctrl", 1.25), ("Win", 1.25), ("Alt", 1.25), ("Space", 6.25), ("Alt", 1.1), ("Fn", 1.1),
      ("Ctrl", 1.1), (None, 0.3), ("\u2190", 1), ("\u2193", 1), ("\u2192", 1)],
     [0, 1, 2, [3, 5, 7], 8, 9, 10, 12, 13, 14]),
]
LAYOUT_W = 16.6          # lebar total dalam satuan u
MAP_VERSION = 3          # naikkan jika peta bawaan berubah (peta tersimpan versi lama dibuang)
KNOB_ID = "knob"


def build_layout():
    keys = []
    for ri, (items, cols) in enumerate(_ROWS):
        x, ki = 0.0, 0
        for label, w in items:
            if label is None:
                x += w
                continue
            col = cols[ki]
            default = [(c, ri + 1) for c in (col if isinstance(col, list) else [col])]
            keys.append(Key(f"r{ri}k{ki}", label, label, x, ri, w, default))
            x += w
            ki += 1
        assert ki == len(cols), (ri, ki, len(cols))
    # nama unik untuk tombol kembar
    seen = {}
    for k in keys:
        seen.setdefault(k.label, []).append(k)
    for label, lst in seen.items():
        if len(lst) == 2:
            lst[0].name, lst[1].name = f"{label} (kiri)", f"{label} (kanan)"
    keys.append(Key(KNOB_ID, "Knob", "Knob (rotary)", 15.6, -0.15, 1.0, [], kind="knob"))
    return keys


class Keymap:
    """Peta tombol <-> slot LED. Slot yang tidak dimiliki tombol manapun = kosong."""

    def __init__(self):
        self.keys = build_layout()
        self.by_id = {k.id: k for k in self.keys}
        self.by_name = {k.name: k for k in self.keys}
        self.reset()

    def reset(self):
        self.slots = {k.id: [tuple(s) for s in k.default] for k in self.keys}
        self._reown()

    def _reown(self):
        self.owner = {}
        for kid, lst in self.slots.items():
            for s in lst:
                self.owner[tuple(s)] = kid

    def unassign(self, slot):
        slot = tuple(slot)
        kid = self.owner.get(slot)
        if kid is not None:
            self.slots[kid].remove(slot)
        self._reown()

    def assign(self, slot, kid):
        slot = tuple(slot)
        self.unassign(slot)
        lst = self.slots[kid]
        lst.append(slot)
        while len(lst) > self.by_id[kid].cap:
            lst.pop(0)
        lst.sort()
        self._reown()

    def empty_slots(self):
        return {(c, r) for c in range(COLS) for r in range(ROWS) if (c, r) not in self.owner}

    def to_json(self):
        return {kid: [list(s) for s in lst] for kid, lst in self.slots.items()}

    def from_json(self, d):
        used, slots = set(), {}
        for k in self.keys:
            raw = d.get(k.id)
            if raw is None:
                raw = k.default
            lst = []
            for s in raw:
                try:
                    t = (int(s[0]), int(s[1]))
                except Exception:
                    continue
                if 0 <= t[0] < COLS and 0 <= t[1] < ROWS and t not in used:
                    lst.append(t)
                    used.add(t)
            slots[k.id] = sorted(lst)[:k.cap]
        self.slots = slots
        self._reown()


# ================================================================ pengaturan
class Settings:
    KEYS = ("mode", "palette", "c1", "c2", "mirror", "sens", "bright", "decay", "audio", "love_bg_on")

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
        self.love_bg_on = False     # mode Love: True = sisa keyboard menampilkan spektrum redup
        self.keymap = Keymap()
        self.empty = self.keymap.empty_slots()

    def refresh_empty(self):
        self.empty = self.keymap.empty_slots()

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
        if d.get("map_version") == MAP_VERSION and isinstance(d.get("keymap"), dict):
            self.keymap.from_json(d["keymap"])
        self.refresh_empty()
        if self.mode not in MODE_NAMES:
            self.mode = MODE_NAMES[0]
        if self.palette not in PALETTE_NAMES:
            self.palette = PALETTE_NAMES[0]

    def save(self):
        try:
            d = {k: getattr(self, k) for k in self.KEYS}
            d["keymap"] = self.keymap.to_json()
            d["map_version"] = MAP_VERSION
            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
        except Exception:
            pass


# ================================================================ mesin (thread render + HID)
class Engine(threading.Thread):
    def __init__(self, S, audio):
        super().__init__(daemon=True)
        self.S, self.audio = S, audio
        self.an = Analyzer()
        self.pal = Palette(S)
        self.dev = None
        self.status = "keyboard: belum terhubung"
        self.preview = np.zeros((COLS, ROWS, 3), np.uint8)
        self.states = {}
        self.running = True
        # alat kalibrasi
        self.test_slot = 0
        self.test_auto = False
        self.test_row = None
        self._sweep_t = 0.0

    @staticmethod
    def make_buf(rgb):
        buf = bytearray(520)
        buf[0:8] = HEADER
        buf[8:8 + N_LED * 3] = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        return bytes(buf)

    def step_slot(self, d):
        i = self.test_slot
        for _ in range(N_LED):
            i = (i + d) % N_LED
            if self.test_row is None or i % ROWS == self.test_row:
                break
        self.test_slot = i
        self._sweep_t = time.time()

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
            if mode in LOVE_MODES:
                pts = _heart_cells(S.keymap)
                a = np.array(pts, dtype=float).reshape(-1, 4)
                ctx.hc, ctx.hr = a[:, 0].astype(int), a[:, 1].astype(int)
                ctx.hx, ctx.hy = a[:, 2], a[:, 3]
                ctx.love_bg = S.love_bg_on
            try:
                if mode == TEST_ALL:
                    frame = np.ones((COLS, ROWS, 3))
                elif mode == TEST_SWEEP:
                    if self.test_auto and now - self._sweep_t >= 0.6:
                        self.step_slot(1)
                    idx = self.test_slot
                    frame = np.zeros((COLS, ROWS, 3))
                    frame[idx // ROWS, idx % ROWS] = 1.0
                else:
                    fn = MODE_FUNCS.get(mode, fx_bars)
                    st = self.states.setdefault(mode, {})
                    frame = fn(ctx, st) * valid[..., None]
            except Exception as e:           # satu efek error tidak boleh mematikan thread
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


# ================================================================ GUI
BG, CARD, FG, MUTED, ACC = "#0d0d13", "#15151e", "#e8e8f2", "#8a8a9c", "#8b5cf6"
OFF, EDGE, WARN, HL, FIELD = "#1a1a24", "#2e2e3e", "#9a6a22", "#ffd23f", "#20202c"
ROW_CHOICES = ["Semua baris", "Baris 0 (atas)", "Baris 1 (Esc)", "Baris 2 (Tab)",
               "Baris 3 (Caps)", "Baris 4 (Shift)", "Baris 5 (Ctrl/Space)"]


def enable_dpi_awareness():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def style_titlebar(root, bg=BG, fg=FG, border=EDGE):
    """Windows: warnai title bar native (dark mode + warna caption/teks/border).
    Win11 (build 22000+) mendapat warna persis; Win10 hanya dark mode. Platform lain: tidak berbuat apa-apa.
    Tetap native, jadi geser/snap/minimize tetap mulus."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32, dwm = ctypes.windll.user32, ctypes.windll.dwmapi
        user32.GetParent.restype = wintypes.HWND
        user32.GetParent.argtypes = [wintypes.HWND]
        root.update_idletasks()
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()

        def colorref(h):                       # COLORREF = 0x00BBGGRR
            return int(h[1:3], 16) | (int(h[3:5], 16) << 8) | (int(h[5:7], 16) << 16)

        def put(attr, value):
            v = ctypes.c_int(value)
            return dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), ctypes.c_uint(attr),
                                             ctypes.byref(v), ctypes.sizeof(v))

        if put(20, 1) != 0:                    # DWMWA_USE_IMMERSIVE_DARK_MODE (19 di Win10 lama)
            put(19, 1)
        put(35, colorref(bg))                  # DWMWA_CAPTION_COLOR (Win11)
        put(36, colorref(fg))                  # DWMWA_TEXT_COLOR    (Win11)
        put(34, colorref(border))              # DWMWA_BORDER_COLOR  (Win11)
        # paksa non-client area digambar ulang (NOSIZE|NOMOVE|NOZORDER|FRAMECHANGED)
        user32.SetWindowPos(wintypes.HWND(hwnd), None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0020)
        return True
    except Exception:
        return False


def make_icon(size=32):
    """Ikon keycap ungu kecil (digambar lewat tk.PhotoImage, tanpa file/PIL)."""
    img = tk.PhotoImage(width=size, height=size)

    def inside(x, y, m, rad):                  # persegi bulat dengan margin m
        lo, hi = m, size - 1 - m
        if not (lo <= x <= hi and lo <= y <= hi):
            return False
        cx = min(max(x, lo + rad), hi - rad)
        cy = min(max(y, lo + rad), hi - rad)
        return (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad
    for y in range(size):
        for x in range(size):
            if inside(x, y, 1, 7):
                img.put("#8b5cf6" if not inside(x, y, 6, 3) else "#15151e", to=(x, y))
    return img


class App:
    BASE_W, BASE_H = 900, 700

    def __init__(self):
        enable_dpi_awareness()
        self.S = Settings()
        self.S.load()
        self.audio = Audio(self.S)
        self.eng = Engine(self.S, self.audio)

        r = self.root = tk.Tk()
        r.title("FuryCube V68 Visualizer")
        r.configure(bg=BG)
        r.resizable(False, False)          # tanpa maximize/resize = ringan

        k = max(1.0, min(2.0, r.winfo_fpixels("1i") / 96.0))
        k = min(k, (r.winfo_screenwidth() - 60) / self.BASE_W, (r.winfo_screenheight() - 120) / self.BASE_H)
        self.k = k = max(0.7, k)

        win = sys.platform.startswith("win")
        fam = "Segoe UI" if win else tkfont.nametofont("TkDefaultFont").actual("family")

        def F(px, bold=False):
            return tkfont.Font(family=fam, size=-max(8, int(round(px * k))), weight="bold" if bold else "normal")

        self.f_norm, self.f_small, self.f_key = F(12), F(10), F(10, True)
        self.f_title, self.f_sub = F(22, True), F(11)
        for nm in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(nm).configure(family=fam, size=-int(12 * k))
            except Exception:
                pass
        self._style()

        pad = int(14 * k)

        # ---- header
        hd = tk.Frame(r, bg=BG)
        hd.pack(fill="x", padx=pad, pady=(pad, 4))
        tk.Label(hd, text="FURYCUBE", bg=BG, fg=FG, font=self.f_title).pack(side="left")
        tk.Label(hd, text="  V68 · AUDIO VISUALIZER", bg=BG, fg=MUTED, font=self.f_sub).pack(side="left", pady=(int(8 * k), 0))
        self.conn = tk.Label(hd, text="● TIDAK TERHUBUNG", bg=BG, fg="#ff6b6b", font=self.f_small)
        self.conn.pack(side="right")

        # ---- keyboard canvas
        self.U = 50.0 * k
        self.m = 18.0 * k
        cw = int(self.m * 2 + LAYOUT_W * self.U)
        ch = int(self.m * 2 + 5.3 * self.U)
        self.canvas = tk.Canvas(r, width=cw, height=ch, bg=CARD, highlightthickness=0)
        self.canvas.pack(padx=pad, pady=4)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Left>", lambda e: self.eng.step_slot(-1))
        self.canvas.bind("<Right>", lambda e: self.eng.step_slot(1))
        self.segs, self.key_boxes, self.last = [], [], {}
        self.rebuild_keys()

        body = tk.Frame(r, bg=BG)
        body.pack(fill="x", padx=pad, pady=4)
        body.columnconfigure(0, weight=1, uniform="c")
        body.columnconfigure(1, weight=1, uniform="c")

        # ---- kartu kiri: mode & warna
        left = self.card(body, 0, 0)
        self.label(left, "MODE").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value=self.S.mode)
        cb = ttk.Combobox(left, textvariable=self.mode_var, values=MODE_NAMES, state="readonly", width=28)
        cb.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)
        cb.bind("<<ComboboxSelected>>", self.on_mode)

        self.label(left, "PALET").grid(row=1, column=0, sticky="w")
        self.pal_var = tk.StringVar(value=self.S.palette)
        cb2 = ttk.Combobox(left, textvariable=self.pal_var, values=PALETTE_NAMES, state="readonly", width=28)
        cb2.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3)
        cb2.bind("<<ComboboxSelected>>", self.on_palette)

        self.label(left, "KUSTOM").grid(row=2, column=0, sticky="w")
        cf = tk.Frame(left, bg=CARD)
        cf.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=3)
        self.b1 = tk.Button(cf, text="A", width=6, relief="flat", bd=0, font=self.f_small,
                            command=lambda: self.pick("c1"))
        self.b2 = tk.Button(cf, text="B", width=6, relief="flat", bd=0, font=self.f_small,
                            command=lambda: self.pick("c2"))
        self.b1.pack(side="left", padx=(0, 6))
        self.b2.pack(side="left")
        self.mirror_var = tk.BooleanVar(value=self.S.mirror)
        tk.Checkbutton(cf, text="Bass di kanan", variable=self.mirror_var, command=self.on_mirror,
                       bg=CARD, fg=FG, selectcolor=FIELD, activebackground=CARD, activeforeground=FG,
                       font=self.f_small, bd=0, highlightthickness=0).pack(side="left", padx=(14, 0))
        self.bg_var = tk.BooleanVar(value=self.S.love_bg_on)
        tk.Checkbutton(left, text="Mode Love: tampilkan spektrum redup di luar hati", variable=self.bg_var,
                       command=self.on_love_bg, bg=CARD, fg=FG, selectcolor=FIELD, activebackground=CARD,
                       activeforeground=FG, font=self.f_small, bd=0, highlightthickness=0
                       ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        left.columnconfigure(1, weight=1)
        self._paint_color_buttons()

        # ---- kartu kanan: slider
        right = self.card(body, 0, 1)
        right.columnconfigure(1, weight=1)
        self.slider(right, 0, "SENSITIVITAS", "sens", 0.3, 3.0)
        self.slider(right, 1, "KECERAHAN", "bright", 0.05, 1.0)
        self.slider(right, 2, "KEJATUHAN BAR", "decay", 0.5, 8.0)

        # ---- baris audio
        au = tk.Frame(r, bg=CARD, padx=int(12 * k), pady=int(8 * k))
        au.pack(fill="x", padx=pad, pady=4)
        au.columnconfigure(1, weight=1)
        self.label(au, "AUDIO").grid(row=0, column=0, sticky="w")
        self.dev_map = {"(otomatis)": "(otomatis)"}
        try:
            for m in sc.all_microphones(include_loopback=True):
                lab = ("[loopback] " if getattr(m, "isloopback", False) else "[mic] ") + m.name
                self.dev_map[lab] = m.name
        except Exception:
            pass
        cur = next((kk for kk, v in self.dev_map.items() if v == self.S.audio), "(otomatis)")
        self.dev_var = tk.StringVar(value=cur)
        cb3 = ttk.Combobox(au, textvariable=self.dev_var, values=list(self.dev_map), state="readonly")
        cb3.grid(row=0, column=1, sticky="ew", padx=8)
        cb3.bind("<<ComboboxSelected>>", self.on_audio)
        self.button(au, "Sambung ulang", self.reconnect).grid(row=0, column=2, padx=(0, 6))
        self.button(au, "Reset peta LED", self.reset_map).grid(row=0, column=3)

        # ---- panel kalibrasi (hanya tampil di mode kalibrasi)
        self.test = tk.Frame(r, bg=CARD, padx=int(12 * k), pady=int(8 * k))
        self.button(self.test, "◀", lambda: self.eng.step_slot(-1), w=3).grid(row=0, column=0)
        self.button(self.test, "▶", lambda: self.eng.step_slot(1), w=3).grid(row=0, column=1, padx=(4, 8))
        self.auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.test, text="Auto", variable=self.auto_var, command=self.on_auto,
                       bg=CARD, fg=FG, selectcolor=FIELD, activebackground=CARD, activeforeground=FG,
                       font=self.f_small, bd=0, highlightthickness=0).grid(row=0, column=2)
        self.row_var = tk.StringVar(value=ROW_CHOICES[0])
        cb4 = ttk.Combobox(self.test, textvariable=self.row_var, values=ROW_CHOICES, state="readonly", width=20)
        cb4.grid(row=0, column=3, padx=8)
        cb4.bind("<<ComboboxSelected>>", self.on_row)
        self.button(self.test, "Slot ini kosong", self.mark_empty).grid(row=0, column=4)
        self.info = tk.Label(self.test, text="", bg=CARD, fg=HL, font=self.f_small, anchor="w")
        self.info.grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))
        tk.Label(self.test, bg=CARD, fg=MUTED, font=self.f_small, anchor="w", justify="left",
                 text="Satu LED putih berjalan di keyboard. Klik tombol di gambar yang MENYALA di keyboard asli "
                      "(spasi: klik 3x untuk 3 LED-nya). Kalau tidak ada yang menyala, klik 'Slot ini kosong'."
                 ).grid(row=2, column=0, columnspan=6, sticky="w")

        self.status = tk.StringVar()
        tk.Label(r, textvariable=self.status, bg=BG, fg=MUTED, font=self.f_small, anchor="w", justify="left",
                 wraplength=int(self.BASE_W * k) - 2 * pad).pack(fill="x", padx=pad, pady=(2, pad))

        self.on_palette()
        self._apply_mode_panel()
        self._quiet_until = 0.0
        self._tick = 0
        r.bind("<Configure>", self._on_configure)
        r.protocol("WM_DELETE_WINDOW", self.on_close)
        try:
            self._icon = make_icon()
            r.iconphoto(True, self._icon)
        except Exception:
            pass
        style_titlebar(r)
        r.after(120, lambda: style_titlebar(r))     # ulangi sekali setelah window tampil
        self.audio.start()
        self.eng.start()
        self.refresh()

    # ---- helper widget
    def _style(self):
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TCombobox", fieldbackground=FIELD, background=FIELD, foreground=FG,
                    arrowcolor=FG, bordercolor=EDGE, lightcolor=FIELD, darkcolor=FIELD,
                    selectbackground=FIELD, selectforeground=FG)
        s.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", FG)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", FG)])
        s.configure("Card.Horizontal.TScale", background=CARD, troughcolor="#262633",
                    bordercolor=CARD, lightcolor=ACC, darkcolor=ACC)
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACC)
        self.root.option_add("*TCombobox*Listbox.font", self.f_norm)

    def card(self, parent, row, col):
        f = tk.Frame(parent, bg=CARD, padx=int(12 * self.k), pady=int(10 * self.k))
        f.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else int(4 * self.k), int(4 * self.k) if col == 0 else 0))
        return f

    def label(self, parent, text):
        return tk.Label(parent, text=text, bg=CARD, fg=MUTED, font=self.f_small)

    def button(self, parent, text, cmd, w=None):
        b = tk.Button(parent, text=text, command=cmd, bg="#262633", fg=FG, activebackground=ACC,
                      activeforeground="#ffffff", relief="flat", bd=0, font=self.f_small,
                      padx=int(10 * self.k), pady=int(4 * self.k))
        if w:
            b.config(width=w)
        return b

    def slider(self, parent, row, text, attr, lo, hi):
        self.label(parent, text).grid(row=row, column=0, sticky="w", pady=3)
        val = tk.Label(parent, width=5, bg=CARD, fg=FG, font=self.f_small)
        val.grid(row=row, column=2, padx=(8, 0))
        val.config(text=f"{getattr(self.S, attr):.2f}")

        def cb(v):
            x = float(v)
            setattr(self.S, attr, x)
            val.config(text=f"{x:.2f}")

        sc_ = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", command=cb, style="Card.Horizontal.TScale")
        sc_.set(getattr(self.S, attr))
        sc_.grid(row=row, column=1, sticky="ew", padx=8)
        sc_.bind("<ButtonRelease-1>", lambda e: self.S.save())

    # ---- gambar keyboard
    def rebuild_keys(self):
        c = self.canvas
        c.delete("all")
        self.segs, self.key_boxes, self.last = [], [], {}
        U, m, k = self.U, self.m, self.k
        g = max(1, int(round(2 * k)))
        km = self.S.keymap
        for key in km.keys:
            x0 = m + key.x * U + g
            y0 = m + (key.y + 0.15) * U + g
            x1 = m + (key.x + key.w) * U - g
            y1 = m + (key.y + 1.15) * U - g
            slots = km.slots[key.id]
            self.key_boxes.append((key, x0, y0, x1, y1))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if key.kind == "knob":
                it = c.create_oval(x0, y0, x1, y1, fill=OFF, outline=EDGE, width=max(1, int(k)))
                lab = c.create_text(cx, cy, text="KNOB", fill="#9a9ab0", font=self.f_small)
                if slots:
                    self.segs.append((it, slots[0][0], slots[0][1], lab))
                continue
            n = max(1, len(slots))
            lab = None
            for i in range(n):
                sx0 = x0 + (x1 - x0) * i / n + (g / 2 if i else 0)
                sx1 = x0 + (x1 - x0) * (i + 1) / n - (g / 2 if i < n - 1 else 0)
                it = c.create_rectangle(sx0, y0, sx1, y1, fill=OFF,
                                        outline=EDGE if slots else WARN,
                                        dash=() if slots else (3, 3), width=max(1, int(k)))
                if slots:
                    self.segs.append((it, slots[i][0], slots[i][1], None))
            lab = c.create_text(cx, cy, text=key.label, fill="#9a9ab0",
                                font=self.f_key if len(key.label) <= 2 else self.f_small)
            if n == 1 and slots:                        # label ikut kontras pada tombol 1 LED
                it, cc, rr, _ = self.segs[-1]
                self.segs[-1] = (it, cc, rr, lab)
        self.hl = c.create_rectangle(0, 0, 0, 0, outline=HL, width=max(2, int(2.5 * k)), state="hidden")

    def refresh(self):
        delay = 40
        try:
            now = time.time()
            if self.root.state() == "iconic" or now < self._quiet_until:
                delay = 250
            else:
                fl = self.eng.preview.tolist()
                cv = self.canvas
                last = self.last
                for it, c, r, lab in self.segs:
                    R, G, B = fl[c][r]
                    t = (R, G, B)
                    if last.get(it) == t:
                        continue
                    last[it] = t
                    on = (R + G + B) >= 24
                    cv.itemconfig(it, fill=("#%02x%02x%02x" % t) if on else OFF)
                    if lab is not None:
                        lum = 0.3 * R + 0.59 * G + 0.11 * B
                        cv.itemconfig(lab, fill="#101018" if (on and lum > 150) else "#9a9ab0")
                self._update_highlight()
                self._tick += 1
                if self._tick % 8 == 0:
                    self._update_status()
        except Exception:
            pass
        finally:
            self.root.after(delay, self.refresh)

    def _update_highlight(self):
        if self.S.mode != TEST_SWEEP:
            self.canvas.itemconfig(self.hl, state="hidden")
            return
        i = self.eng.test_slot
        kid = self.S.keymap.owner.get((i // ROWS, i % ROWS))
        if kid is None:
            self.canvas.itemconfig(self.hl, state="hidden")
            return
        for key, x0, y0, x1, y1 in self.key_boxes:
            if key.id == kid:
                self.canvas.coords(self.hl, x0 - 1, y0 - 1, x1 + 1, y1 + 1)
                self.canvas.itemconfig(self.hl, state="normal")
                self.canvas.tag_raise(self.hl)
                break

    def _update_status(self):
        ok = self.eng.dev is not None
        self.conn.config(text="● TERHUBUNG" if ok else "● TIDAK TERHUBUNG", fg="#4ade80" if ok else "#ff6b6b")
        used = N_LED - len(self.S.empty)
        self.status.set(f"{self.eng.status}   |   {self.audio.status}   |   LED terpakai {used}/{N_LED}, kosong {len(self.S.empty)}")
        if self.S.mode == TEST_SWEEP:
            i = self.eng.test_slot
            c, r = i // ROWS, i % ROWS
            kid = self.S.keymap.owner.get((c, r))
            who = f"tombol \u00ab{self.S.keymap.by_id[kid].name}\u00bb" if kid else "KOSONG (tidak ada tombol)"
            self.info.config(text=f"Slot {i}  \u00b7  kolom {c}, baris {r}  \u2192  {who}")

    def _on_configure(self, e):
        if e.widget is self.root:
            self._quiet_until = time.time() + 0.25

    # ---- event
    def _apply_mode_panel(self):
        if self.S.mode == TEST_SWEEP:
            self.test.pack(fill="x", padx=int(14 * self.k), pady=4, before=self.status_widget())
        else:
            self.test.pack_forget()

    def status_widget(self):
        for w in self.root.pack_slaves():
            if isinstance(w, tk.Label) and str(w.cget("textvariable")) == str(self.status):
                return w
        return None

    def on_mode(self, _=None):
        self.S.mode = self.mode_var.get()
        self._apply_mode_panel()
        self.S.save()

    def on_palette(self, _=None):
        self.S.palette = self.pal_var.get()
        st = "normal" if self.S.palette == CUSTOM else "disabled"
        self.b1.config(state=st)
        self.b2.config(state=st)
        self.S.save()

    def on_mirror(self):
        self.S.mirror = bool(self.mirror_var.get())
        self.S.save()

    def on_love_bg(self):
        self.S.love_bg_on = bool(self.bg_var.get())
        self.S.save()

    def on_audio(self, _=None):
        self.S.audio = self.dev_map.get(self.dev_var.get(), "(otomatis)")
        self.audio.reopen = True
        self.S.save()

    def on_auto(self):
        self.eng.test_auto = bool(self.auto_var.get())
        self.eng._sweep_t = time.time()

    def on_row(self, _=None):
        idx = ROW_CHOICES.index(self.row_var.get())
        self.eng.test_row = None if idx == 0 else idx - 1
        if self.eng.test_row is not None and self.eng.test_slot % ROWS != self.eng.test_row:
            self.eng.step_slot(1)

    def pick(self, key):
        cur = "#%02x%02x%02x" % getattr(self.S, key)
        res = colorchooser.askcolor(color=cur, parent=self.root)
        if res and res[0]:
            setattr(self.S, key, tuple(int(v) for v in res[0]))
            self._paint_color_buttons()
            self.S.save()

    def _paint_color_buttons(self):
        for b, kk in ((self.b1, "c1"), (self.b2, "c2")):
            col = getattr(self.S, kk)
            b.config(bg="#%02x%02x%02x" % col, fg="#000000" if sum(col) > 380 else "#ffffff",
                     activebackground="#%02x%02x%02x" % col)

    def reconnect(self):
        self.eng.disconnect()
        self.eng.status = "keyboard: menyambung ulang..."

    # ---- kalibrasi
    def _map_changed(self):
        self.S.refresh_empty()
        self.S.save()
        self.rebuild_keys()

    def _stop_auto(self):
        self.eng.test_auto = False
        self.auto_var.set(False)

    def on_click(self, ev):
        self.canvas.focus_set()
        if self.S.mode != TEST_SWEEP:
            return
        for key, x0, y0, x1, y1 in self.key_boxes:
            if x0 <= ev.x <= x1 and y0 <= ev.y <= y1:
                self.assign_current(key.id)
                return

    def assign_current(self, kid):
        self._stop_auto()
        i = self.eng.test_slot
        self.S.keymap.assign((i // ROWS, i % ROWS), kid)
        self._map_changed()
        self.eng.step_slot(1)

    def mark_empty(self):
        self._stop_auto()
        i = self.eng.test_slot
        self.S.keymap.unassign((i // ROWS, i % ROWS))
        self._map_changed()
        self.eng.step_slot(1)

    def reset_map(self):
        if messagebox.askyesno("Reset peta LED", "Kembalikan peta tombol ke bawaan layout V68?", parent=self.root):
            self.S.keymap.reset()
            self._map_changed()

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