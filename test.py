import hid, time

VID = 0x258A
HEADER = bytes.fromhex("0608000001007a01")

def frame(leds):  # leds: {slot: (r, g, b)}
    buf = bytearray(520)
    buf[0:8] = HEADER
    for slot, (r, g, b) in leds.items():
        buf[8 + slot*3 : 11 + slot*3] = bytes((r, g, b))
    return bytes(buf)

def find_dev():
    probe = frame({0: (255, 0, 0)})
    for d in hid.enumerate(VID):
        dev = hid.device()
        try:
            dev.open_path(d["path"])
            if dev.send_feature_report(probe) == 520:
                print("PID=%04x" % d["product_id"],
                      "iface=", d["interface_number"],
                      "usage_page=%04x" % d["usage_page"],
                      "usage=%04x" % d["usage"])
                print("path =", d["path"])
                return dev
        except Exception:
            pass
        dev.close()
    raise SystemExit("tidak ada entri yang bisa ditulis")

dev = find_dev()
input("Cek: apakah ada LED merah menyala? Enter untuk mulai sweep...")

for slot in range(90):
    dev.send_feature_report(frame({slot: (255, 0, 0)}))
    print(slot, end=" ", flush=True)
    time.sleep(0.4)
dev.send_feature_report(frame({}))