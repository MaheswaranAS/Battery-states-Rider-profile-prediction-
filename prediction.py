import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
import pandas as pd

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
# (Assuming your dataset is named dataset_500.csv)
# -----------------------------------------------------
df = pd.read_csv("dataset_500.csv")
output_labels = df.select_dtypes(include=['float64', 'int64']) \
                  .drop(['Temperature_C','Voltage_V','Current_A'], axis=1).columns.tolist()

# -----------------------------------------------------
# GUI WINDOW (Black Theme)
# -----------------------------------------------------
root = tk.Tk()
root.title("Battery SOC & SOH Prediction System")
root.geometry("600x750")
root.configure(bg="black")

style = ttk.Style()
style.theme_use("clam")

style.configure("TLabel", background="black", foreground="white", font=("Arial", 18))
style.configure("TEntry", font=("Arial", 18))
style.configure("TButton", font=("Arial", 18), foreground="black")

# -----------------------------------------------------
# INPUT LABELS & FIELDS
# -----------------------------------------------------
tk.Label(root, text="Battery Prediction GUI", bg="black", fg="cyan",
         font=("Arial", 26, "bold")).pack(pady=20)

frame = tk.Frame(root, bg="black")
frame.pack(pady=20)

labels = ["Temperature (°C)", "Voltage (V)", "Current (A)"]
entries = []

for text in labels:
    lbl = ttk.Label(frame, text=text)
    lbl.pack(pady=10)

    entry = ttk.Entry(frame, width=20, font=("Arial", 20))  # bigger input box
    entry.pack(pady=10)
    entries.append(entry)

# -----------------------------------------------------
# PLOT AREA
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
ax.set_facecolor("black")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=20)

# -----------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------
def predict():
    temp = float(entries[0].get())
    volt = float(entries[1].get())
    curr = float(entries[2].get())

    X = np.array([[temp, volt, curr]])
    X_scaled = scaler.transform(X)

    if is_keras:
        pred = model.predict(X_scaled)[0]
    else:
        pred = model.predict(X_scaled)[0]

    # Clear previous plot
    ax.clear()
    ax.set_facecolor("black")

    # Only 2 outputs → plotting 2 bars
    bars = ax.bar(range(len(pred)), pred, color="cyan")

    ax.set_title("Predicted Outputs", color="white", fontsize=18)
    ax.tick_params(colors="white")
    ax.set_xticks(range(len(pred)))
    ax.set_xticklabels(output_labels, rotation=0, color="white", fontsize=14)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f"{height:.1f}", ha="center", color="white", fontsize=14)

    canvas.draw()

# -----------------------------------------------------
# BUTTON
# -----------------------------------------------------
btn = ttk.Button(root, text="Predict & Plot", command=predict)
btn.pack(pady=30)

# -----------------------------------------------------
# START APP
# -----------------------------------------------------
root.mainloop()
