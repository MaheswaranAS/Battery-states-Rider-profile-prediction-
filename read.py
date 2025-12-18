import serial
import time

# ===== CONFIGURE SERIAL PORT =====
PORT = "COM9"       # Change to your Arduino COM port
BAUD = 9600         # Must match Serial.begin(9600)

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)   # allow Arduino to reset
    print(f"[OK] Connected to {PORT} at {BAUD} baud.")
except Exception as e:
    print("[ERROR] Could not open serial port:", e)
    exit()

# ===== READ LOOP =====
try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(line)
except KeyboardInterrupt:
    print("\n[STOP] Exiting...")
finally:
    ser.close()
