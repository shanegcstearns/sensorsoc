# Makefile for cocotb + Icarus (vvp) on macOS
# Run from repo root: sleep_soc/

SHELL := /bin/zsh

# Tools
IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3

# Cocotb VPI library dir (computed from your active python environment)
COCOTB_LIBDIR := $(shell $(PYTHON) -c "import cocotb, os; print(os.path.join(os.path.dirname(cocotb.__file__), 'libs'))")
PYGPI_PYTHON_BIN := $(shell $(PYTHON) -c "import sys; print(sys.executable)")

# Common cocotb env (v2 naming)
export PYGPI_PYTHON_BIN := $(PYGPI_PYTHON_BIN)
export PYTHONPATH := $(PWD)/sim/tb
export COCOTB_LOG_LEVEL ?= INFO

# Build dir
BUILD_DIR := sim/build
$(shell mkdir -p $(BUILD_DIR))

# ---------- Test: test_mmio ----------
MMIO_TOP      := sim_test_mmio
MMIO_MODULE   := test_test_mmio
MMIO_VVP      := $(BUILD_DIR)/$(MMIO_TOP).vvp
MMIO_TB_V     := sim/tb/sim_test_mmio.v
MMIO_RTL_V    := rtl/test_mmio.v

.PHONY: test-mmio build-mmio

build-mmio: $(MMIO_VVP)

$(MMIO_VVP): $(MMIO_TB_V) $(MMIO_RTL_V)
	$(IVERILOG) -g2012 -o $@ $^

test-mmio: build-mmio
	@echo "== Running cocotb: $(MMIO_MODULE) on $(MMIO_TOP) =="
	COCOTB_TOPLEVEL=$(MMIO_TOP) \
	COCOTB_TEST_MODULES=$(MMIO_MODULE) \
	$(VVP) -M "$(COCOTB_LIBDIR)" -m libcocotbvpi_icarus $(MMIO_VVP)

# ---------- Convenience ----------
.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*.vvp
	rm -f *.vvp
	rm -f *.vcd sim/build/*.vcd
