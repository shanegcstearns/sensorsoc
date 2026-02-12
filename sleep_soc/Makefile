# ============================================================
# Makefile for sleep_soc (Icarus + cocotb v2)
# Run from repo root: sleep_soc/
# ============================================================

SHELL := /bin/zsh

# Tools
IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3

# ------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------

# Cocotb VPI library
COCOTB_LIBDIR := $(shell $(PYTHON) -c "import cocotb, os; print(os.path.join(os.path.dirname(cocotb.__file__), 'libs'))")

# Python interpreter for cocotb
PYGPI_PYTHON_BIN := $(shell $(PYTHON) -c "import sys; print(sys.executable)")

# Build directory
BUILD_DIR := sim/build

# Ensure build directory exists
$(shell mkdir -p $(BUILD_DIR))

# ------------------------------------------------------------
# Common cocotb environment (v2 naming)
# ------------------------------------------------------------
export PYGPI_PYTHON_BIN := $(PYGPI_PYTHON_BIN)
export PYTHONPATH       := $(PWD)/sim/tb
export COCOTB_LOG_LEVEL ?= INFO

# ============================================================
# MMIO TEST
# ============================================================

MMIO_TOP     := sim_test_mmio
MMIO_MODULE  := test_test_mmio
MMIO_VVP     := $(BUILD_DIR)/$(MMIO_TOP).vvp
MMIO_SOURCES := sim/tb/sim_test_mmio.v rtl/test_mmio.v

$(MMIO_VVP): $(MMIO_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: test-mmio
test-mmio: $(MMIO_VVP)
	@echo "== Running test-mmio =="
	COCOTB_TOPLEVEL=$(MMIO_TOP) \
	COCOTB_TEST_MODULES=$(MMIO_MODULE) \
	$(VVP) -M "$(COCOTB_LIBDIR)" -m libcocotbvpi_icarus $(MMIO_VVP)

# ============================================================
# TIMER TEST
# ============================================================

TIMER_TOP     := sim_test_timer_mmio
TIMER_MODULE  := test_timer_mmio
TIMER_VVP     := $(BUILD_DIR)/$(TIMER_TOP).vvp
TIMER_SOURCES := sim/tb/sim_test_timer_mmio.v rtl/timer_mmio.v

$(TIMER_VVP): $(TIMER_SOURCES)
	$(IVERILOG) -g2012 -o $@ $^

.PHONY: test-timer
test-timer: $(TIMER_VVP)
	@echo "== Running test-timer =="
	COCOTB_TOPLEVEL=$(TIMER_TOP) \
	COCOTB_TEST_MODULES=$(TIMER_MODULE) \
	$(VVP) -M "$(COCOTB_LIBDIR)" -m libcocotbvpi_icarus $(TIMER_VVP)

# ============================================================
# Cleanup
# ============================================================

.PHONY: clean
clean:
	rm -f sim/build/*.vvp
	rm -f *.vcd
