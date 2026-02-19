SIM ?= verilator
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES := $(PWD)/takethree.sv

TOPLEVEL := takethree
MODULE   := ml_model_tb

#COMPILE_ARGS += -g2012

export DATASET_CSV ?= $(PWD)/processed_sleep_dataset.csv
export SPLIT ?= test
export MAX_SAMPLES ?= 5000
export MIN_ACC ?= 0.0

include $(shell cocotb-config --makefiles)/Makefile.sim
