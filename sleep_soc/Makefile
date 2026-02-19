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
  $(RTL_DIR)/third_party/picorv32.v \
  $(RTL_DIR)/gpio_mmio.v \
  $(RTL_DIR)/test_mmio.v \
  $(RTL_DIR)/timer_mmio.v \
  $(RTL_DIR)/ml_stub_mmio.v \
  $(RTL_DIR)/pwrctrl_mmio.v

SOC_TB_SRCS  := $(TB_DIR)/tb_soc.sv
SOC_VVP      := sim_soc.vvp

# Default firmware outputs used by soc_top
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

.PHONY: firmware-sleepwake firmware-gpio firmware-sleepwake-periodic
firmware-sleepwake:
	@$(MAKE) FW_SRC=$(FW_TESTS)/test_sleepwake.c firmware

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

# Run tb_soc. Expects firmware/build/firmware.hex to exist.
sim-soc: $(SOC_VVP) $(FW_HEX) | $(SIM_BUILD)
	@echo "[SIM] Running tb_soc (will read $(FW_HEX))"
	$(VVP) $(SOC_VVP)

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
