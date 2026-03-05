# RTL

This directory contains the main SystemVerilog RTL for `sleep_soc`, including the sensor-processing pipeline and feature extraction blocks.

# Inside this Directory
* ppg_fifo_reader.sv
  * Reads PPG samples from the sensor FIFO over I2C and outputs timestamped samples + status/error flags.
* ppg_beat_detect_rr_calc.sv
  * Detects beats from the PPG stream and computes RR intervals and beat-quality signals.
* motion_preprocess.sv
  * Preprocesses accelerometer samples into a per-epoch motion-energy feature.
* cos_lut_timer.sv
  * Generates a cosine time feature from the seconds counter using a small LUT (configurable period/scale).
* accel_reader.sv
  * Polls/configures the accelerometer over I2C and outputs X/Y/Z samples with a valid strobe.
* rmssd_engine.sv
  * Computes epoch RMSSD from accepted RR intervals.
* feature_engine.sv
  * Latches per-epoch features (time, motion, delta-HR, RMSSD) and outputs a single `feat_valid_o` strobe.
* signal_quality.sv
  * Aggregates beat quality, motion intensity, and FIFO/I2C faults to gate ML updates (`ml_update_gate_o`).
