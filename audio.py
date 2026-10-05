import hid
import numpy as np
import soundcard as sc

VID, PID = 0x258A, 0x010C
COLS, ROWS = 15, 6
RATE, BLOCK = 48000, 2048
HEADER = bytes.fromhex("0608000001007a01")

MIRROR = True          # True = bass di kanan
EMPTY = set()          # slot tanpa tombol, format (kolom, baris), mis. {(4, 5), (5, 5)}

# warna per tinggi bar, dari bawah ke atas (R, G, B)
PALETTE = [(0, 255, 60), (0, 255, 60), (180, 255, 0),
           (255, 200, 0), (255, 90, 0), (255, 0, 0)]

FLOOR_DB, CEIL_DB = -10, 35
DECAY = 0.12

# baris yang valid tiap kolom, urut dari bawah ke atas
ROWS_BOTTOM_UP = [[r for r in range(ROWS - 1, -1, -1) if (c, r) not in EMPTY]
                  for c in range(COLS)]


def frame(levels):
    buf = bytearray(520)
    buf[0:8] = HEADER
    for band in range(COLS):
        col = COLS - 1 - band if MIRROR else band
        rows = ROWS_BOTTOM_UP[col]
        n = min(int(levels[band]), len(rows))
        for k in range(n):
            i = 8 + (col * ROWS + rows[k]) * 3
            buf[i:i + 3] = bytes(PALETTE[k])
    return bytes(buf)


def open_dev():
    blank = frame([0] * COLS)
    for d in hid.enumerate(VID, PID):
        dev = hid.device()
        try:
            dev.open_path(d["path"])
            if dev.send_feature_report(blank) == 520:
                return dev
        except Exception:
            pass
        dev.close()
    raise SystemExit("device tidak ketemu (driver masih terbuka?)")


dev = open_dev()
speaker = sc.default_speaker()
mic = sc.get_microphone(speaker.name, include_loopback=True)

freqs = np.fft.rfftfreq(BLOCK, 1 / RATE)
edges = np.geomspace(40, 16000, COLS + 1)
bands = []
for i in range(COLS):
    lo = np.searchsorted(freqs, edges[i])
    hi = max(np.searchsorted(freqs, edges[i + 1]), lo + 1)
    bands.append((lo, hi))
window = np.hanning(BLOCK)
tilt = np.linspace(0, 12, COLS)
level = np.zeros(COLS)

try:
    with mic.recorder(samplerate=RATE, blocksize=BLOCK) as rec:
        while True:
            data = rec.record(numframes=BLOCK).mean(axis=1)
            spec = np.abs(np.fft.rfft(data * window))
            lv = np.array([spec[lo:hi].mean() for lo, hi in bands])
            db = 20 * np.log10(lv + 1e-9) + tilt
            norm = np.clip((db - FLOOR_DB) / (CEIL_DB - FLOOR_DB), 0, 1)
            level = np.maximum(norm, level - DECAY)
            dev.send_feature_report(frame(np.round(level * ROWS)))
except KeyboardInterrupt:
    dev.send_feature_report(frame([0] * COLS))