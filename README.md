ML-Based Battery State Estimation and Rider Behavior Profiling for Electric Vehicles
📌 Overview

This project presents an intelligent onboard monitoring system for electric vehicles (EVs) that combines battery state prediction and rider behavior analysis using machine learning. The system operates entirely at the edge without cloud dependency and is designed specifically for electric two-wheelers and low-cost EV platforms.

The solution estimates State of Charge (SoC), predicts State of Health (SoH), classifies rider driving behavior (low / normal / rash), and computes a behavior-aware remaining driving range in real time.

🎯 Key Features

Real-time monitoring of EV battery and rider behavior

Voltage-based SoC estimation for fast embedded operation

Machine Learning–based SoH prediction (Gradient Boosting)

Rider behavior classification: Low / Normal / Rash driving

Dynamic, behavior-aware remaining range estimation

Fully onboard (edge-based) processing – no cloud required

UART communication for reliable data transfer

Live visualization through a GUI dashboard

Low-cost and scalable design for EV two-wheelers

🧠 Technology Used

Programming Language: Python

Machine Learning: Gradient Boosting (scikit-learn)

Embedded Platform: Arduino Uno

Sensors:

Voltage sensor

ACS712 current sensor

Temperature sensor (LM35)

Accelerometer

IR speed sensor

Motor: BLDC motor with ESC

Communication: UART (Serial)

Visualization: Tkinter GUI

🔋 How It Works

Sensors continuously acquire voltage, current, temperature, speed, throttle, and acceleration data from the EV.

SoC is estimated using a voltage-to-percentage mapping suitable for real-time operation.

SoH is predicted using a trained machine learning model based on voltage, current, and temperature.

Rider behavior is analyzed using sudden changes in throttle, acceleration, current draw, and voltage drop.

Driving behavior is classified as low, normal, or rash.

Remaining driving range is calculated dynamically using SoC and rider behavior.

All data and predictions are displayed in real time via a GUI.

📂 Project Structure
├── data/
│   └── training_dataset.csv
├── models/
│   ├── gb_classifier.pkl
│   ├── gb_scaler.pkl
│   └── soh_model.pkl
├── src/
│   ├── train_model.py
│   ├── realtime_gui.py
│   └── serial_reader.py
├── README.md
└── requirements.txt

▶️ How to Run

Clone the repository:

git clone https://github.com/your-username/ev-battery-rider-ml.git
cd ev-battery-rider-ml


Install dependencies:

pip install -r requirements.txt


Connect the Arduino and ensure the correct COM port is set.

Run the real-time monitoring GUI:

python src/realtime_gui.py

📊 Machine Learning Details

Model Used: Gradient Boosting

Learning Rate: 0.1 (default)

Estimators: 100 decision trees

Features: Voltage, Current, Temperature, Speed, Throttle, Acceleration

Output:

SoH (%)

Driving behavior class

🔬 Applications

Electric two-wheelers (e-bikes, scooters)

EV safety monitoring

Battery health tracking

Rider behavior analytics

EV fleet management

🚀 Future Scope

Integration of deep learning models (LSTM) for time-series prediction

Mobile app or dashboard integration

Adaptive motor control based on rider behavior

Cloud integration for fleet-level analytics

👤 Author

Maheswaran A S
Final Year – Electrical and Electronics Engineering

📜 License

This project is intended for academic and research purposes.
Feel free to fork, modify, and learn from it.
