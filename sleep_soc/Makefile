# sleep_soc/Makefile
#
# Goals:
#   - One-command build + run for your SystemVerilog SoC sim (tb_soc)
#   - One-command run for cocotb unit tests (test_mmio, timer_mmio)
#   - Simple firmware build (RISC-V GCC -> .elf/.bin/.hex)
#
# Usage:
#   make sim-soc                     # compile + run tb_soc (reads firmware/build/firmware.hex)
#   make test-mmio                   # cocotb for test_mmio
#   make test-timer                  # cocotb for timer_mmio
#   make firmware-sleepwake          # firmware from firmware/tests/test_sleepwake.c
#   make firmware-gpio               # firmware from firmware/tests/test_gpio.c
#   make firmware-sleepwake-periodic # firmware from firmware/tests/test_sleepwake_periodic.c
#   make sim-reset-ctrl              # sv tb for reset controller
#   make clean                       # remove build outputs
#
# NOTE:
#   This Makefile expects a small helper script at:
#     tools/bin2hex.py
#   which converts firmware/build/firmware.bin -> firmware/build/firmware.hex
#   as 32-bit little-endian words (one word per line), suitable for $readmemh.

SHELL := /bin/bash

# ----------------------------
# Tools
IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3

# ------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------

# Cocotb VPI library
COCOTB_LIBDIR := $(shell $(PYTHON) -c "import cocotb, os; print(os.path.join(os.path.dirname(cocotb.__file__), 'libs'))" 2>/dev/null)

# Python interpreter for cocotb
PYGPI_PYTHON_BIN := $(shell $(PYTHON) -c "import sys; print(sys.executable)")

# Build directory
BUILD_DIR := sim/build

# Ensure build directory exists
$(shell mkdir -p $(BUILD_DIR))

WAVE_DIR := sim/waves

# ------------------------------------------------------------
# Common cocotb environment (v2 naming)
# ------------------------------------------------------------
export PYGPI_PYTHON_BIN := $(PYGPI_PYTHON_BIN)
export PYTHONPATH       := $(PWD)/sim/tb
export COCOTB_LOG_LEVEL ?= INFO

# ============================================================
# MMIO TEST
# ============================================================

# ----------------------------
PYTHON       ?= python3
IVERILOG     ?= iverilog
VVP          ?= vvp

# RISC-V toolchain
RISCV_PREFIX ?= riscv64-unknown-elf-
RISCV_GCC    ?= $(RISCV_PREFIX)gcc
RISCV_OBJCOPY?= $(RISCV_PREFIX)objcopy

# ----------------------------
# Paths
# ----------------------------
RTL_DIR      := rtl
TB_DIR       := sim/tb
SIM_BUILD    := sim/build
FW_DIR       := firmware
FW_BUILD     := $(FW_DIR)/build
FW_TESTS     := $(FW_DIR)/tests
TOOLS_DIR    := tools

BIN2HEX_PY   := $(TOOLS_DIR)/bin2hex.py

# Ensure output dirs exist
$(SIM_BUILD) $(FW_BUILD):
	@mkdir -p $@

# ----------------------------
# Common RTL sources for SoC
# ----------------------------
SOC_RTL_SRCS := \
  $(RTL_DIR)/soc_top.v \
  $(RTL_DIR)/simple_sram.v \
  $(RTL_DIR)/third_party/picorv32.v \
  $(RTL_DIR)/gpio_mmio.v \
  $(RTL_DIR)/test_mmio.v \
  $(RTL_DIR)/timer_mmio.v \
  $(RTL_DIR)/ml_stub_mmio.v \
  $(RTL_DIR)/irq_ctrl_mmio.v \
  $(RTL_DIR)/pwrctrl_mmio.v \
  $(RTL_DIR)/host_i2c_target.sv \
  $(RTL_DIR)/host_i2c_bridge_regs.sv

SOC_TB_SRCS  := $(TB_DIR)/tb_soc.sv
SOC_VVP      := sim_soc.vvp
IRQ_TB_SRCS  := $(TB_DIR)/tb_irq.sv
IRQ_VVP      := sim_irq.vvp

# Per-firmware output paths
FW_HEX_MAIN      := $(FW_DIR)/build/main/firmware.hex
FW_HEX_SLEEPWAKE := $(FW_DIR)/build/test_sleepwake/firmware.hex

# Keep generic names for targets that still use FW_SRC override pattern
FW_HEX       := $(FW_BUILD)/firmware.hex
FW_ELF       := $(FW_BUILD)/firmware.elf
FW_BIN       := $(FW_BUILD)/firmware.bin

# ----------------------------
# Cocotb configuration helper
# ----------------------------
# cocotb's Icarus VPI shared library directory:
COCOTB_LIBDIR := $(shell $(PYTHON) -c "import cocotb, os; print(os.path.join(os.path.dirname(cocotb.__file__), 'libs'))")

# Use the same python interpreter cocotb will embed
export PYGPI_PYTHON_BIN := $(shell $(PYTHON) -c "import sys; print(sys.executable)")

# Make sure cocotb can import your tests
export PYTHONPATH := $(abspath $(TB_DIR))

define RUN_COCOTB_ICARUS
	$(VVP) -M "$(COCOTB_LIBDIR)" -m libcocotbvpi_icarus $1
endef

# ----------------------------
# Firmware build rules
# ----------------------------
# The canonical firmware for tb_soc is firmware/build/firmware.hex.
# Choose which C file populates it by calling:
#   make firmware-sleepwake
#   make firmware-gpio
#   make firmware-sleepwake-periodic
#   make firmware-main         # builds firmware/main.c -> firmware/build/main/firmware.hex (used by sim-irq)

FW_SRC ?= $(FW_TESTS)/test_sleepwake.c

.PHONY: firmware
firmware: $(FW_HEX)

$(FW_HEX): $(FW_DIR)/start.S $(FW_DIR)/linker.ld $(FW_SRC) $(BIN2HEX_PY) | $(FW_BUILD)
	@echo "[FW] Building $(FW_SRC) -> $(FW_HEX)"
	$(RISCV_GCC) \
	  -march=rv32imc -mabi=ilp32 \
	  -nostdlib -ffreestanding -O2 \
	  -Wl,-T,$(FW_DIR)/linker.ld \
	  -o $(FW_ELF) \
	  $(FW_DIR)/start.S \
	  $(FW_SRC)
	$(RISCV_OBJCOPY) -O binary $(FW_ELF) $(FW_BIN)
	@echo "[FW] Converting $(FW_BIN) -> $(FW_HEX) (32-bit little-endian words)"
	@$(PYTHON) $(BIN2HEX_PY) $(FW_BIN) $(FW_HEX)

.PHONY: firmware-sleepwake firmware-gpio firmware-sleepwake-periodic firmware-main

$(FW_HEX_MAIN): $(FW_DIR)/start.S $(FW_DIR)/linker.ld $(FW_DIR)/main.c $(BIN2HEX_PY)
	@mkdir -p $(dir $@)
	@echo "[FW] Building firmware/main.c -> $@"
	$(RISCV_GCC) \
	  -march=rv32imc -mabi=ilp32 \
	  -nostdlib -ffreestanding -O2 \
	  -Wl,-T,$(FW_DIR)/linker.ld \
	  -o $(FW_DIR)/build/main/firmware.elf \
	  $(FW_DIR)/start.S \
	  $(FW_DIR)/main.c
	$(RISCV_OBJCOPY) -O binary $(FW_DIR)/build/main/firmware.elf $(FW_DIR)/build/main/firmware.bin
	@$(PYTHON) $(BIN2HEX_PY) $(FW_DIR)/build/main/firmware.bin $@

$(FW_HEX_SLEEPWAKE): $(FW_DIR)/start.S $(FW_DIR)/linker.ld $(FW_TESTS)/test_sleepwake.c $(BIN2HEX_PY)
	@mkdir -p $(dir $@)
	@echo "[FW] Building firmware/tests/test_sleepwake.c -> $@"
	$(RISCV_GCC) \
	  -march=rv32imc -mabi=ilp32 \
	  -nostdlib -ffreestanding -O2 \
	  -Wl,-T,$(FW_DIR)/linker.ld \
	  -o $(FW_DIR)/build/test_sleepwake/firmware.elf \
	  $(FW_DIR)/start.S \
	  $(FW_TESTS)/test_sleepwake.c
	$(RISCV_OBJCOPY) -O binary $(FW_DIR)/build/test_sleepwake/firmware.elf $(FW_DIR)/build/test_sleepwake/firmware.bin
	@$(PYTHON) $(BIN2HEX_PY) $(FW_DIR)/build/test_sleepwake/firmware.bin $@

firmware-main: $(FW_HEX_MAIN)
firmware-sleepwake: $(FW_HEX_SLEEPWAKE)

firmware-gpio:
	@$(MAKE) FW_SRC=$(FW_TESTS)/test_gpio.c firmware

firmware-sleepwake-periodic:
	@$(MAKE) FW_SRC=$(FW_TESTS)/test_sleepwake_periodic.c firmware

# ----------------------------
# SoC simulation (SystemVerilog tb)
# ----------------------------
.PHONY: build-soc sim-soc
build-soc: $(SOC_VVP)

$(SOC_VVP): $(SOC_TB_SRCS) $(SOC_RTL_SRCS)
	@echo "[SIM] Building $@"
	$(IVERILOG) -g2012 -s tb_soc -o $@ \
	  $(SOC_TB_SRCS) \
	  $(SOC_RTL_SRCS)

# Run tb_soc. Builds test_sleepwake firmware into its own directory.
sim-soc: $(SOC_VVP) $(FW_HEX_SLEEPWAKE) | $(SIM_BUILD)
	@echo "[SIM] Running tb_soc (will read $(FW_HEX_SLEEPWAKE))"
	$(VVP) $(SOC_VVP)

# ----------------------------
# IRQ testbench (tb_irq)
# ----------------------------
.PHONY: build-irq sim-irq
build-irq: $(IRQ_VVP)

$(IRQ_VVP): $(IRQ_TB_SRCS) $(SOC_RTL_SRCS)
	@echo "[SIM] Building $@"
	$(IVERILOG) -g2012 -s tb_irq -o $@ \
	  $(IRQ_TB_SRCS) \
	  $(SOC_RTL_SRCS)

sim-irq: $(IRQ_VVP) $(FW_HEX_MAIN) | $(SIM_BUILD)
	@echo "[SIM] Running tb_irq (will read $(FW_HEX_MAIN))"
	$(VVP) $(IRQ_VVP)

# ----------------------------
# Cocotb: test_mmio
# ----------------------------
MMIO_TOP     := sim_test_mmio
MMIO_TOP_V   := $(TB_DIR)/sim_test_mmio.v
MMIO_DUT_V   := $(RTL_DIR)/test_mmio.v
MMIO_VVP     := sim_test_mmio.vvp
MMIO_PYMOD   := test_test_mmio

$(MMIO_VVP): $(MMIO_TOP_V) $(MMIO_DUT_V)
	@echo "[COCOTB] Building $@"
	$(IVERILOG) -g2012 -o $@ \
	  $(MMIO_TOP_V) \
	  $(MMIO_DUT_V)

.PHONY: test-mmio
test-mmio: $(MMIO_VVP)
	@echo "[COCOTB] Running $(MMIO_PYMOD) on $(MMIO_TOP)"
	@export COCOTB_TOPLEVEL=$(MMIO_TOP); \
	export COCOTB_TEST_MODULES=$(MMIO_PYMOD); \
	$(call RUN_COCOTB_ICARUS,$(MMIO_VVP))

# ----------------------------
# Cocotb: timer_mmio
# ----------------------------
TIMER_TOP    := sim_test_timer_mmio
TIMER_TOP_V  := $(TB_DIR)/sim_test_timer_mmio.v
TIMER_DUT_V  := $(RTL_DIR)/timer_mmio.v
TIMER_VVP    := sim_test_timer_mmio.vvp

# Set this to your actual python module name (without .py)
# Example: if file is sim/tb/test_timer_mmio.py -> TIMER_PYMOD := test_timer_mmio
TIMER_PYMOD  ?= test_timer_mmio

$(TIMER_VVP): $(TIMER_TOP_V) $(TIMER_DUT_V)
	@echo "[COCOTB] Building $@"
	$(IVERILOG) -g2012 -o $@ \
	  $(TIMER_TOP_V) \
	  $(TIMER_DUT_V)

.PHONY: test-timer
test-timer: $(TIMER_VVP)
	@echo "== Running test-timer =="
	COCOTB_TOPLEVEL=$(TIMER_TOP) \
	COCOTB_TEST_MODULES=$(TIMER_PYMOD) \
	$(VVP) -M "$(COCOTB_LIBDIR)" -m libcocotbvpi_icarus $(TIMER_VVP)


# ============================================================
# RESET CONTROLLER TB (reset_ctrl_tb)
# ============================================================

RESET_CTRL_TOP     := reset_ctrl_tb
RESET_CTRL_VVP     := $(BUILD_DIR)/$(RESET_CTRL_TOP).vvp
RESET_CTRL_SOURCES := sim/tb/reset_ctrl_tb.sv rtl/reset_ctrl.v
RESET_CTRL_VCD     := $(WAVE_DIR)/reset_ctrl.vcd

$(RESET_CTRL_VVP): $(RESET_CTRL_SOURCES) | $(BUILD_DIR)
	@echo "[SIM] Building $@"
	$(IVERILOG) -g2012 -Wall -o $@ $^

.PHONY: sim-reset-ctrl
sim-reset-ctrl: $(RESET_CTRL_VVP)
	@echo "== Running $(RESET_CTRL_TOP) (VCD -> $(RESET_CTRL_VCD)) =="
	@mkdir -p $(WAVE_DIR)
	@cd $(WAVE_DIR) && $(VVP) $(abspath $(RESET_CTRL_VVP))

.PHONY: clean-reset-ctrl
clean-reset-ctrl:
	rm -f $(RESET_CTRL_VVP) $(RESET_CTRL_VCD)
# ============================================================
# MOTION PREPROCESS
# ============================================================

MOTION_PREPROCESS_TOP     := motion_preprocess_tb
MOTION_PREPROCESS_VVP     := $(BUILD_DIR)/$(MOTION_PREPROCESS_TOP).vvp
MOTION_PREPROCESS_SOURCES := sim/tb/motion_preprocess_tb.sv rtl/sensor_processing/motion_preprocess.sv
MOTION_PREPROCESS_VCD     := $(WAVE_DIR)/motion_preprocess_tb.vcd

$(MOTION_PREPROCESS_VVP): $(MOTION_PREPROCESS_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: sim-motion-preprocess
sim-motion-preprocess: $(MOTION_PREPROCESS_VVP)
	@echo "== Running $(MOTION_PREPROCESS_TOP) (VCD -> $(MOTION_PREPROCESS_VCD)) =="
	@mkdir -p $(WAVE_DIR)
	@cd $(WAVE_DIR) && $(VVP) $(abspath $(MOTION_PREPROCESS_VVP))

# ============================================================
# PPG FIFO READER
# ============================================================

PPG_FIFO_READER_TOP     := ppg_fifo_reader_tb
PPG_FIFO_READER_VVP     := $(BUILD_DIR)/$(PPG_FIFO_READER_TOP).vvp
PPG_FIFO_READER_SOURCES := sim/tb/ppg_fifo_reader_tb.sv rtl/sensor_processing/ppg_fifo_reader.sv
PPG_FIFO_READER_VCD     := $(WAVE_DIR)/ppg_fifo_reader_tb.vcd

$(PPG_FIFO_READER_VVP): $(PPG_FIFO_READER_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: sim-ppg-fifo-reader
sim-ppg-fifo-reader: $(PPG_FIFO_READER_VVP)
	@echo "== Running $(PPG_FIFO_READER_TOP) (VCD -> $(PPG_FIFO_READER_VCD)) =="
	@mkdir -p $(WAVE_DIR)
	@cd $(WAVE_DIR) && $(VVP) $(abspath $(PPG_FIFO_READER_VVP))

# ============================================================
# PPG BEAT DETECT + RR CALC
# ============================================================

PPG_BEAT_RR_TOP     := ppg_beat_detect_rr_calc_tb
PPG_BEAT_RR_VVP     := $(BUILD_DIR)/$(PPG_BEAT_RR_TOP).vvp
PPG_BEAT_RR_SOURCES := sim/tb/ppg_beat_detect_rr_calc_tb.sv rtl/sensor_processing/ppg_beat_detect_rr_calc.sv rtl/sensor_processing/ewma_baseline.sv
PPG_BEAT_RR_VCD     := $(WAVE_DIR)/ppg_beat_detect_rr_calc_tb.vcd

$(PPG_BEAT_RR_VVP): $(PPG_BEAT_RR_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: sim-ppg-beat-rr
sim-ppg-beat-rr: $(PPG_BEAT_RR_VVP)
	@echo "== Running $(PPG_BEAT_RR_TOP) (VCD -> $(PPG_BEAT_RR_VCD)) =="
	@mkdir -p $(WAVE_DIR)
	@cd $(WAVE_DIR) && $(VVP) $(abspath $(PPG_BEAT_RR_VVP))

# ============================================================
# ACCEL READER
# ============================================================

ACCEL_READER_TOP     := accel_reader_tb
ACCEL_READER_VVP     := $(BUILD_DIR)/$(ACCEL_READER_TOP).vvp
ACCEL_READER_SOURCES := sim/tb/accel_reader_tb.sv rtl/sensor_processing/accel_reader.sv
ACCEL_READER_VCD     := $(WAVE_DIR)/accel_reader_tb.vcd

$(ACCEL_READER_VVP): $(ACCEL_READER_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: sim-accel-reader
sim-accel-reader: $(ACCEL_READER_VVP)
	@echo "== Running $(ACCEL_READER_TOP) (VCD -> $(ACCEL_READER_VCD)) =="
	@mkdir -p $(WAVE_DIR)
	@cd $(WAVE_DIR) && $(VVP) $(abspath $(ACCEL_READER_VVP))

# ============================================================
# RMSSD ENGINE
# ============================================================

RMSSD_ENGINE_TOP     := rmssd_engine_tb
RMSSD_ENGINE_VVP     := $(BUILD_DIR)/$(RMSSD_ENGINE_TOP).vvp
RMSSD_ENGINE_SOURCES := sim/tb/rmssd_engine_tb.sv rtl/sensor_processing/rmssd_engine.sv rtl/sensor_processing/ewma_baseline.sv
RMSSD_ENGINE_VCD     := $(WAVE_DIR)/rmssd_engine_tb.vcd

$(RMSSD_ENGINE_VVP): $(RMSSD_ENGINE_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: sim-rmssd-engine
sim-rmssd-engine: $(RMSSD_ENGINE_VVP)
	@echo "== Running $(RMSSD_ENGINE_TOP) (VCD -> $(RMSSD_ENGINE_VCD)) =="
	@mkdir -p $(WAVE_DIR)
	@cd $(WAVE_DIR) && $(VVP) $(abspath $(RMSSD_ENGINE_VVP))

# ============================================================
# SIGNAL QUALITY
# ============================================================

SIGNAL_QUALITY_TOP     := signal_quality_tb
SIGNAL_QUALITY_VVP     := $(BUILD_DIR)/$(SIGNAL_QUALITY_TOP).vvp
SIGNAL_QUALITY_SOURCES := sim/tb/signal_quality_tb.sv rtl/sensor_processing/signal_quality.sv
SIGNAL_QUALITY_VCD     := $(WAVE_DIR)/signal_quality_tb.vcd

$(SIGNAL_QUALITY_VVP): $(SIGNAL_QUALITY_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: sim-signal-quality
sim-signal-quality: $(SIGNAL_QUALITY_VVP)
	@echo "== Running $(SIGNAL_QUALITY_TOP) (VCD -> $(SIGNAL_QUALITY_VCD)) =="
	@mkdir -p $(WAVE_DIR)
	@cd $(WAVE_DIR) && $(VVP) $(abspath $(SIGNAL_QUALITY_VVP))

# ============================================================
# COS LUT TIMER
# ============================================================

COS_LUT_TIMER_TOP     := cos_lut_timer_tb
COS_LUT_TIMER_VVP     := $(BUILD_DIR)/$(COS_LUT_TIMER_TOP).vvp
COS_LUT_TIMER_SOURCES := sim/tb/cos_lut_timer_tb.sv rtl/cos_lut_timer.sv
COS_LUT_TIMER_VCD     := $(WAVE_DIR)/cos_lut_timer_tb.vcd

$(COS_LUT_TIMER_VVP): $(COS_LUT_TIMER_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: sim-cos-lut-timer
sim-cos-lut-timer: $(COS_LUT_TIMER_VVP)
	@echo "== Running $(COS_LUT_TIMER_TOP) (VCD -> $(COS_LUT_TIMER_VCD)) =="
	@mkdir -p $(WAVE_DIR)
	@cd $(WAVE_DIR) && $(VVP) $(abspath $(COS_LUT_TIMER_VVP))

# ============================================================
# HOST I2C TARGET
# ============================================================

HOST_I2C_TARGET_TOP     := tb_host_i2c_target
HOST_I2C_TARGET_VVP     := $(BUILD_DIR)/$(HOST_I2C_TARGET_TOP).vvp
HOST_I2C_TARGET_SOURCES := sim/tb/tb_host_i2c_target.sv \
                           rtl/host_i2c_target.sv \
                           rtl/host_i2c_bridge_regs.sv

$(HOST_I2C_TARGET_VVP): $(HOST_I2C_TARGET_SOURCES)
	$(IVERILOG) -g2012 -s $(HOST_I2C_TARGET_TOP) -o $@ $^

.PHONY: sim-host-i2c-target
sim-host-i2c-target: $(HOST_I2C_TARGET_VVP)
	@echo "== Running $(HOST_I2C_TARGET_TOP) =="
	$(VVP) $(HOST_I2C_TARGET_VVP)

# ============================================================
# SOC HOST I2C IRQ
# ============================================================

SOC_I2C_IRQ_TOP     := tb_soc_host_i2c_irq
SOC_I2C_IRQ_VVP     := $(BUILD_DIR)/$(SOC_I2C_IRQ_TOP).vvp
SOC_I2C_IRQ_SOURCES := sim/tb/tb_soc_host_i2c_irq.sv \
                       $(SOC_RTL_SRCS) \
                       rtl/host_i2c_target.sv \
                       rtl/host_i2c_bridge_regs.sv

$(SOC_I2C_IRQ_VVP): $(SOC_I2C_IRQ_SOURCES)
	$(IVERILOG) -g2012 -s $(SOC_I2C_IRQ_TOP) -o $@ $^

.PHONY: sim-soc-host-i2c-irq
sim-soc-host-i2c-irq: $(SOC_I2C_IRQ_VVP)
	@echo "== Running $(SOC_I2C_IRQ_TOP) =="
	$(VVP) $(SOC_I2C_IRQ_VVP)

# ============================================================
# Cleanup
# ============================================================

.PHONY: clean
clean:
	rm -f sim/build/*.vvp
	rm -f *.vcd
	rm -f sim/waves/*.vcd
	@echo "[COCOTB] Running $(TIMER_PYMOD) on $(TIMER_TOP)"
	@export COCOTB_TOPLEVEL=$(TIMER_TOP); \
	export COCOTB_TEST_MODULES=$(TIMER_PYMOD); \
	$(call RUN_COCOTB_ICARUS,$(TIMER_VVP))

# ----------------------------
# Convenience
# ----------------------------
.PHONY: test
test: test-mmio test-timer

.PHONY: show-tree
show-tree:
	@echo "PYTHON=$(PYTHON)"
	@echo "IVERILOG=$(IVERILOG)"
	@echo "VVP=$(VVP)"
	@echo "RISCV_GCC=$(RISCV_GCC)"
	@echo "FW_SRC=$(FW_SRC)"
	@echo "TIMER_PYMOD=$(TIMER_PYMOD)"
	@echo "BIN2HEX_PY=$(BIN2HEX_PY)"

# ----------------------------
# Clean
# ----------------------------
.PHONY: clean
clean:
	rm -f *.vvp *.vcd results.xml
	rm -rf $(SIM_BUILD)/*.vcd $(SIM_BUILD)/*.fst $(SIM_BUILD)/*.gtkw
	rm -f $(FW_BUILD)/firmware.elf $(FW_BUILD)/firmware.bin $(FW_BUILD)/firmware.hex
