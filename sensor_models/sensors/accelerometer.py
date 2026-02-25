"""
accelerometer.py

This module simulates a 3-axis accelerometer sensor interface using
the PhysioNet motion dataset.

Functionality:
- Loads multi-file accelerometer data from directory
- Applies sensor-level modeling (gain error, offset, Gaussian noise)
- Normalizes to ±4g range
- Simulates 12-bit ADC quantization
- Saves validation plot (analog vs digital comparison)
- Outputs digital acceleration stream (simulated I2C data)

This represents:
Physical sensor → Analog front end → ADC → Digital output
"""
import numpy as np
import os
import matplotlib.pyplot as plt


# Configuration

ACCEL_FS = 50            # Hz
ACCEL_RANGE_G = 4        # ±4g
ADC_BITS = 12
OUTPUT_DIR = "sensor_output"

# Development limit (set to None to use full dataset)
MAX_SAMPLES = 500000


# Data Loading

def load_motion_directory(directory):
    print("Loading motion directory...")
    all_data = []

    for file in os.listdir(directory):
        if file.endswith(".txt"):
            path = os.path.join(directory, file)
            print("Loading", file)

            data = np.loadtxt(path)

            # columns: time, x, y, z
            all_data.append(data[:, 1:4])

    data = np.vstack(all_data)

    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]

    print("Total samples loaded:", len(data))
    return data


# Sensor Model

def apply_accel_sensor_model(accel):
    # Gain error
    gain = 1 + np.random.uniform(-0.02, 0.02)
    accel = accel * gain

    # Offset
    accel += 0.02

    # Noise
    accel += np.random.normal(0, 0.01, accel.shape)

    # Normalize to ADC voltage range
    accel_norm = accel / ACCEL_RANGE_G

    return accel_norm


def adc_quantize(signal):
    levels = 2 ** ADC_BITS
    signal = np.clip(signal, -1, 1)

    digital = np.round((signal + 1) / 2 * (levels - 1))
    digital = digital - levels // 2

    return digital.astype(np.int16)


# Validation Plot

def save_validation_plot(raw, digital, fs, filename):
    duration = 10  # seconds
    n = duration * fs

    raw = raw[:n]
    digital = digital[:n]

    # Normalize digital back to analog scale
    digital_norm = digital / np.max(np.abs(digital)) * np.max(np.abs(raw))

    time = np.arange(len(raw)) / fs

    plt.figure(figsize=(10,4))
    plt.plot(time, raw, label="Analog (modeled)")
    plt.plot(time, digital_norm, label="ADC (normalized)", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.title("Accelerometer Validation (First 10s)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()

    print("Validation plot saved to:", path)


# Main Processing

def process_accelerometer(directory):
    raw = load_motion_directory(directory)

    analog = apply_accel_sensor_model(raw)
    digital = adc_quantize(analog)

    # Save validation for X axis
    save_validation_plot(
        raw[:, 0],
        digital[:, 0],
        ACCEL_FS,
        "accel_validation.png"
    )

    # Save full digital stream for ML use
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "accelerometer_digital_stream.csv")

    np.savetxt(
        "sensor_output/accel_digital.csv",
        digital,
        delimiter=","
    )

    print("Accelerometer digital stream saved to:", csv_path)

    return digital
