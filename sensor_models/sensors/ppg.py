import numpy as np
import os
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

PPG_FS = 1            # PhysioNet heart rate ~1 Hz
ADC_BITS = 12
OUTPUT_DIR = "sensor_output"

MAX_SAMPLES = 500000


# -----------------------------
# Data Loading
# -----------------------------

def load_heartrate_directory(directory):
    print("Loading heart rate directory...")
    all_data = []

    for file in os.listdir(directory):
        if file.endswith(".txt"):
            path = os.path.join(directory, file)
            print("Loading", file)

            # Heart rate files are comma-separated
            data = np.loadtxt(path, delimiter=",")

            # columns: time, bpm
            all_data.append(data[:, 1])

    data = np.concatenate(all_data)

    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]

    print("Total HR samples loaded:", len(data))
    return data



# -----------------------------
# Sensor Model
# -----------------------------

def apply_ppg_sensor_model(hr):
    # Normalize BPM (40–180 typical range)
    hr_norm = (hr - 40) / (180 - 40)
    hr_norm = np.clip(hr_norm, 0, 1)

    # Gain error
    gain = 1 + np.random.uniform(-0.02, 0.02)
    hr_norm *= gain

    # Offset
    hr_norm += 0.01

    # Noise
    hr_norm += np.random.normal(0, 0.005, len(hr_norm))

    # Map to ±1 for ADC
    hr_norm = hr_norm * 2 - 1

    return hr_norm


def adc_quantize(signal):
    levels = 2 ** ADC_BITS
    signal = np.clip(signal, -1, 1)

    digital = np.round((signal + 1) / 2 * (levels - 1))
    digital = digital - levels // 2

    return digital.astype(np.int16)


# -----------------------------
# Validation Plot
# -----------------------------

def save_validation_plot(raw, digital, fs, filename):
    duration = 60  # seconds (HR is slow)
    n = duration * fs

    raw = raw[:n]
    digital = digital[:n]

    digital_norm = digital / np.max(np.abs(digital)) * np.max(np.abs(raw))
    time = np.arange(len(raw)) / fs

    plt.figure(figsize=(10,4))
    plt.plot(time, raw, label="Analog HR")
    plt.plot(time, digital_norm, label="ADC (normalized)", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("PPG Validation (First 60s)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()

    print("Validation plot saved to:", path)


# -----------------------------
# Main
# -----------------------------

def process_ppg(directory):
    raw = load_heartrate_directory(directory)

    analog = apply_ppg_sensor_model(raw)
    digital = adc_quantize(analog)

    save_validation_plot(
        raw,
        digital,
        PPG_FS,
        "ppg_validation.png"
    )

    return digital
