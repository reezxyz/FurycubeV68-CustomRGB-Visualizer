import usb.core
import usb.util

# Ganti dengan VID dan PID dari keyboard Anda
VID = 0x258A  # Contoh VID
PID = 0x013B  # Contoh PID

# Temukan perangkat USB
device = usb.core.find(idVendor=VID, idProduct=PID)

if device is None:
    raise ValueError("Keyboard tidak ditemukan!")

# Atur konfigurasi perangkat
device.set_configuration()

# Kirim perintah ke perangkat (contoh perintah)
try:
    # Data di bawah ini adalah contoh. Anda perlu mengganti dengan data yang cocok dengan protokol perangkat Anda.
    data = [0x00, 0x01, 0xFF, 0x00, 0x00]
    device.write(1, data)  # Endpoint 1 untuk pengiriman data
    print("Perintah berhasil dikirim!")
except Exception as e:
    print(f"Error: {e}")
