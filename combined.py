import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
import pandas as pd
import threading
import serial
import time
import re

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------------------------------
# LOAD SCALER + MODEL FOR SOH (AND OTHER OUTPUTS)
# -----------------------------------------------------
scaler = joblib.load("input_scaler.pkl")

try:
    model = joblib.load("best_model.pkl")
    is_keras = False
except:
    model = load_model("model_neuralnetwork.h5")
    is_keras = True

# -----------------------------------------------------
# LOAD CSV TO GET OUTPUT LABELS (FOR SOH)
# -----------------------------------------------------
df = pd.read_csv("dataset_500.csv")
output_labels = df.select_dtypes(include=['float64', 'int64']) \
                  .drop(['Temperature_C', 'Voltage_V', 'Current_A'], axis=1).columns.tolist()

# Try to find an SoH output column (case-insensitive)
soh_index = None
for i, name in enumerate(output_labels):
    if "soh" in name.lower():
        soh_index = i
        break

# -----------------------------------------------------
# SOME CONSTANTS
# -----------------------------------------------------
SERIAL_PORT = "COM3"   # <--- change if needed
BAUD = 9600

BASE_RANGE_KM = 40.0   # assumed max range at 100% SoC under normal driving

# thresholds for rash driving conditions
THROTTLE_DELTA_RISE = 50.0      # change per sample
ACCEL_DELTA_RISE = 10.0         # change in acceleration magnitude
CURRENT_RISE_THRESH = 0.5       # A
VOLT_DROP_THRESH = 0.3          # V

# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def compute_soc_from_voltage(voltage):
    """
    Map voltage to SoC%:
    13 V -> 100%
    6 V  -> 40%
    Linear in between, clamped to [0, 100].
    """
    v_min = 6.0
    v_max = 13.0
    soc_min = 40.0
    soc_max = 100.0

    if voltage <= v_min:
        return soc_min
    elif voltage >= v_max:
        return soc_max
    else:
        soc = soc_min + (voltage - v_min) * (soc_max - soc_min) / (v_max - v_min)
        return max(0.0, min(100.0, soc))

def classify_speed_type(raw_speed):
    """
    Manual mapping:
    0          -> Low speed
    ~0.03,0.07 -> Normal speed
    >0.07      -> High speed
    """
    # small tolerance because floats may not be exact
    if abs(raw_speed) < 1e-3:
        return "Low Speed"
    elif raw_speed <= 0.07:
        return "Normal Speed"
    else:
        return "High Speed"

def detect_rash_reason(curr_data, prev_data):
    """
    Determines if rash driving occurs based on 4 categories:
    1. Throttle rise alone
    2. Accelerometer change alone
    3. Throttle + accelerometer together
    4. Voltage + current rise
    Returns: (is_rash, reason_string or None)
    """
    if prev_data is None:
        return False, None

    # unpack
    throttle = curr_data["Throttle"]
    prev_throttle = prev_data["Throttle"]

    ax_mag = curr_data["AccelMag"]
    prev_ax_mag = prev_data["AccelMag"]

    voltage = curr_data["Voltage"]
    prev_voltage = prev_data["Voltage"]

    current = curr_data["Current"]
    prev_current = prev_data["Current"]

    throttle_rise = (throttle - prev_throttle) > THROTTLE_DELTA_RISE
    accel_change = abs(ax_mag - prev_ax_mag) > ACCEL_DELTA_RISE
    current_rise = (current - prev_current) > CURRENT_RISE_THRESH
    voltage_drop = (prev_voltage - voltage) > VOLT_DROP_THRESH
    power_abnormal = current_rise and voltage_drop

    # 3. Throttle + accelerometer together
    if throttle_rise and accel_change:
        return True, "Throttle + Accel Change"

    # 1. Throttle rise alone
    if throttle_rise and not accel_change and not power_abnormal:
        return True, "Throttle Rise"

    # 2. Accelerometer change alone
    if accel_change and not throttle_rise and not power_abnormal:
        return True, "Accel Change"

    # 4. Voltage and current rise
    if power_abnormal:
        return True, "Voltage + Current Change"

    return False, None

def compute_distance_remaining(soc_percent, speed_type, is_rash):
    """
    Use SoC and driving style to estimate remaining distance.
    Simple heuristic:
    - base_range at 100% SoC under normal = BASE_RANGE_KM
    - factor depends on style:
        Low speed  -> 1.1
        Normal     -> 1.0
        Rash       -> 0.7
    """
    soc_fraction = soc_percent / 100.0

    if is_rash:
        factor = 0.7
    else:
        if speed_type == "Low Speed":
            factor = 1.1
        elif speed_type == "Normal Speed":
            factor = 1.0
        else:  # High speed but not rash by our rules
            factor = 0.8

    return soc_fraction * BASE_RANGE_KM * factor

# -----------------------------------------------------
# GUI WINDOW
# -----------------------------------------------------
root = tk.Tk()
root.title("Real-Time EV Battery & Rider Profile System")
root.geometry("1100x800")
root.configure(bg="black")

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="black", foreground="white", font=("Arial", 16))
style.configure("TEntry", font=("Arial", 16))
style.configure("TButton", font=("Arial", 16), foreground="black")

# TITLE
tk.Label(root,
         text="Real-Time EV Battery & Rider Profile System",
         bg="black", fg="cyan",
         font=("Arial", 24, "bold")).pack(pady=10)

main_frame = tk.Frame(root, bg="black")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# -----------------------------------------------------
# CENTER FRAME: Temp, Voltage, Current, SoC, SoH
# -----------------------------------------------------
center_frame = tk.LabelFrame(main_frame, text="Battery Data & Health",
                             bg="black", fg="cyan", font=("Arial", 16))
center_frame.place(x=50, y=50, width=450, height=350)

center_labels = ["Temperature (°C)", "Voltage (V)", "Current (A)",
                 "SoC (%)", "SoH (%)"]
center_entries = {}

for i, lbl in enumerate(center_labels):
    tk.Label(center_frame, text=lbl, bg="black", fg="white").place(x=20, y=30 + i * 50)
    ent = ttk.Entry(center_frame, width=18)
    ent.place(x=230, y=30 + i * 50)
    center_entries[lbl] = ent

# -----------------------------------------------------
# RIGHT FRAME: Rider Profile, Speed, Throttle, Distance Remaining
# -----------------------------------------------------
right_frame = tk.LabelFrame(main_frame, text="Rider Profile & Range",
                            bg="black", fg="cyan", font=("Arial", 16))
right_frame.place(x=550, y=50, width=500, height=350)

rider_labels = ["Throttle", "Speed (raw)", "Speed Profile",
                "Rider Profile", "Distance Remaining (km)"]
rider_entries = {}

for i, lbl in enumerate(rider_labels):
    tk.Label(right_frame, text=lbl, bg="black", fg="white").place(x=20, y=30 + i * 50)
    ent = ttk.Entry(right_frame, width=22)
    ent.place(x=230, y=30 + i * 50)
    rider_entries[lbl] = ent

# -----------------------------------------------------
# PLOT FRAME FOR MODEL OUTPUTS
# -----------------------------------------------------
plot_frame = tk.LabelFrame(main_frame, text="Model Outputs (e.g., SoH & others)",
                           bg="black", fg="cyan", font=("Arial", 16))
plot_frame.place(x=50, y=430, width=1000, height=320)

fig, ax = plt.subplots(figsize=(8, 3), facecolor="black")
ax.set_facecolor("black")
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

# -----------------------------------------------------
# GLOBALS FOR PREVIOUS VALUES (for rash detection)
# -----------------------------------------------------
prev_data = None

# regex pattern for serial line:
pattern = re.compile(
    r"X:([\d.]+).*?Y:([\d.]+).*?Z:([\d.]+).*?"
    r"Voltage:\s*([\d.]+).*?"
    r"Current:\s*([\d.]+).*?"
    r"Temp:\s*([\d.]+).*?"
    r"Throttle:\s*([\d.]+).*?"
    r"Speed:\s*([\d.]+)"
)

# -----------------------------------------------------
# PREDICTION & GUI UPDATE
# -----------------------------------------------------
def do_prediction_and_update_gui(temp, volt, curr, throttle, speed_raw, ax_mag, is_rash, rash_reason, speed_type):
    # ----- SoC from voltage -----
    soc = compute_soc_from_voltage(volt)

    # ----- Model prediction for SoH (if available) -----
    X = np.array([[temp, volt, curr]])
    X_scaled = scaler.transform(X)

    if is_keras:
        pred = model.predict(X_scaled)[0]
    else:
        pred = model.predict(X_scaled)[0]

    soh_value = None
    if soh_index is not None:
        soh_value = float(pred[soh_index])

    # ----- Distance remaining from SoC + style -----
    distance_remaining = compute_distance_remaining(soc, speed_type, is_rash)

    # ----- Update center entries (Temp, Volt, Curr, SoC, SoH) -----
    center_entries["Temperature (°C)"].delete(0, tk.END)
    center_entries["Temperature (°C)"].insert(0, f"{temp:.2f}")

    center_entries["Voltage (V)"].delete(0, tk.END)
    center_entries["Voltage (V)"].insert(0, f"{volt:.2f}")

    center_entries["Current (A)"].delete(0, tk.END)
    center_entries["Current (A)"].insert(0, f"{curr:.2f}")

    center_entries["SoC (%)"].delete(0, tk.END)
    center_entries["SoC (%)"].insert(0, f"{soc:.1f}")

    center_entries["SoH (%)"].delete(0, tk.END)
    if soh_value is not None:
        center_entries["SoH (%)"].insert(0, f"{soh_value:.1f}")
    else:
        center_entries["SoH (%)"].insert(0, "N/A")

    # ----- Update rider side (Throttle, Speed, Profile, Range) -----
    rider_entries["Throttle"].delete(0, tk.END)
    rider_entries["Throttle"].insert(0, f"{throttle:.1f}")

    rider_entries["Speed (raw)"].delete(0, tk.END)
    rider_entries["Speed (raw)"].insert(0, f"{speed_raw:.3f}")

    rider_entries["Speed Profile"].delete(0, tk.END)
    rider_entries["Speed Profile"].insert(0, speed_type)

    rider_entries["Distance Remaining (km)"].delete(0, tk.END)
    rider_entries["Distance Remaining (km)"].insert(0, f"{distance_remaining:.2f}")

    # Rider profile text
    rider_entries["Rider Profile"].delete(0, tk.END)
    if is_rash:
        if rash_reason is not None:
            rider_entries["Rider Profile"].insert(0, f"Rash Driving ({rash_reason})")
        else:
            rider_entries["Rider Profile"].insert(0, "Rash Driving")
    else:
        rider_entries["Rider Profile"].insert(0, speed_type)

    # ----- Plot all model outputs -----
    ax.clear()
    ax.set_facecolor("black")
    bars = ax.bar(range(len(pred)), pred, color="cyan")
    ax.set_title("Predicted Model Outputs", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.set_xticks(range(len(pred)))
    ax.set_xticklabels(output_labels, color="white", fontsize=10, rotation=45, ha="right")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f"{height:.1f}",
                ha="center", va="bottom",
                color="white", fontsize=9)

    canvas.draw()

# -----------------------------------------------------
# SERIAL READER THREAD
# -----------------------------------------------------
def serial_reader():
    global prev_data

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        time.sleep(2)
        print(f"[OK] Serial connected on {SERIAL_PORT}.")
    except Exception as e:
        print(f"[ERROR] Cannot open {SERIAL_PORT}:", e)
        return

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Debug print
            print(line)

            match = pattern.search(line)
            if match:
                X_val = float(match.group(1))
                Y_val = float(match.group(2))
                Z_val = float(match.group(3))
                volt = float(match.group(4))
                curr = float(match.group(5))
                temp = float(match.group(6))
                throttle = float(match.group(7))
                speed_raw = float(match.group(8))

                # acceleration magnitude (for rash detection)
                ax_mag = np.sqrt(X_val**2 + Y_val**2 + Z_val**2)

                curr_data = {
                    "Throttle": throttle,
                    "AccelMag": ax_mag,
                    "Voltage": volt,
                    "Current": curr
                }

                # detect rash based on 4 categories
                is_rash, rash_reason = detect_rash_reason(curr_data, prev_data)

                # classify speed type from raw speed
                speed_type = classify_speed_type(speed_raw)

                prev_data = curr_data

                # schedule GUI update & prediction
                root.after(
                    0,
                    lambda t=temp, v=volt, c=curr,
                           thr=throttle, s=speed_raw, a=ax_mag,
                           r=is_rash, rr=rash_reason, sp=speed_type:
                        do_prediction_and_update_gui(t, v, c, thr, s, a, r, rr, sp)
                )

        except Exception as e:
            print("Serial read error:", e)

# Start reader thread
threading.Thread(target=serial_reader, daemon=True).start()

# -----------------------------------------------------
# START GUI
# -----------------------------------------------------
root.mainloop()
