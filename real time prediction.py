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
# LOAD MODEL + SCALER
# -----------------------------------------------------
scaler = joblib.load("input_scaler.pkl")

try:
    model = joblib.load("best_model.pkl")
    is_keras = False
except:
    model = load_model("model_neuralnetwork.h5")
    is_keras = True

# -----------------------------------------------------
# LOAD CSV to get feature names
# -----------------------------------------------------
df = pd.read_csv("dataset_500.csv")
output_labels = df.select_dtypes(include=['float64', 'int64']) \
                  .drop(['Temperature_C','Voltage_V','Current_A'], axis=1).columns.tolist()

# -----------------------------------------------------
# GUI WINDOW
# -----------------------------------------------------
root = tk.Tk()
root.title("Battery SOC & SOH Real-Time Prediction")
root.geometry("650x900")
root.configure(bg="black")

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="black", foreground="white", font=("Arial", 18))
style.configure("TEntry", font=("Arial", 18))
style.configure("TButton", font=("Arial", 18), foreground="black")

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------
tk.Label(root, text="Real-Time Battery Prediction GUI",
         bg="black", fg="cyan",
         font=("Arial", 26, "bold")).pack(pady=20)

# -----------------------------------------------------
# LIVE SERIAL INPUT DISPLAY
# -----------------------------------------------------
info_frame = tk.Frame(root, bg="black")
info_frame.pack(pady=10)

labels = ["Temperature (°C)", "Voltage (V)", "Current (A)", "STATUS"]
entries = []
status_label = None

for i, text in enumerate(labels):
    lbl = ttk.Label(info_frame, text=text)
    lbl.pack(pady=10)

    if text == "STATUS":
        status_label = tk.Label(info_frame, text="WAITING...",
                                bg="black", fg="yellow",
                                font=("Arial", 20, "bold"))
        status_label.pack(pady=10)
    else:
        entry = ttk.Entry(info_frame, width=22, font=("Arial", 20))
        entry.pack(pady=10)
        entries.append(entry)

# -----------------------------------------------------
# MATPLOTLIB PLOT AREA
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
ax.set_facecolor("black")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=20)

# -----------------------------------------------------
# SAFETY CONDITIONS
# -----------------------------------------------------
def check_conditions(temp, volt, curr):
    """
    Update GUI status color according to safety rules.
    """

    if temp > 40:
        status_label.config(text="DANGER: HIGH TEMP", fg="red")
        return

    if volt < 5:
        status_label.config(text="LOW VOLTAGE", fg="orange")
        return

    if volt > 13:
        status_label.config(text="OVER VOLTAGE", fg="red")
        return

    if curr > 5:
        status_label.config(text="HIGH CURRENT", fg="red")
        return

    status_label.config(text="NORMAL", fg="lightgreen")

# -----------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------
def do_prediction(temp, volt, curr):
    X = np.array([[temp, volt, curr]])
    X_scaled = scaler.transform(X)

    if is_keras:
        pred = model.predict(X_scaled)[0]
    else:
        pred = model.predict(X_scaled)[0]

    # Plot
    ax.clear()
    ax.set_facecolor("black")
    bars = ax.bar(range(len(pred)), pred, color="cyan")

    ax.set_title("Predicted Outputs", color="white", fontsize=18)
    ax.tick_params(colors="white")
    ax.set_xticks(range(len(pred)))
    ax.set_xticklabels(output_labels, color="white", fontsize=14)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f"{height:.1f}",
                ha="center", color="white", fontsize=14)

    canvas.draw()

# -----------------------------------------------------
# SERIAL READING THREAD
# -----------------------------------------------------
SERIAL_PORT = "COM9"    # <---- Your port
BAUD = 9600

def serial_reader():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        time.sleep(2)
        print("[OK] Serial connected on COM9.")
    except:
        print("[ERROR] Cannot open COM9")
        return

    pattern = r"Voltage:\s*([\d.]+)V.*Current:\s*([\d.]+)A.*Temp:\s*([\d.]+)C"

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            match = re.search(pattern, line)
            if match:
                volt = float(match.group(1))
                curr = float(match.group(2))
                temp = float(match.group(3))

                # Update GUI entries
                entries[0].delete(0, tk.END)
                entries[0].insert(0, f"{temp:.2f}")

                entries[1].delete(0, tk.END)
                entries[1].insert(0, f"{volt:.2f}")

                entries[2].delete(0, tk.END)
                entries[2].insert(0, f"{curr:.2f}")

                # Check safety conditions
                root.after(0, lambda t=temp, v=volt, c=curr: check_conditions(t, v, c))

                # Run prediction
                root.after(0, lambda t=temp, v=volt, c=curr: do_prediction(t, v, c))

        except Exception as e:
            print("Serial read error:", e)


# Start reader background thread
threading.Thread(target=serial_reader, daemon=True).start()

# -----------------------------------------------------
# START GUI
# -----------------------------------------------------
root.mainloop()
