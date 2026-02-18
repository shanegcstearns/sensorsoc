SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES := $(PWD)/takethree_hr.sv

TOPLEVEL := takethree_hr
MODULE   := takethree_hr_tb

export DATASET_CSV ?= $(PWD)/processed_sleep_dataset.csv
export SPLIT ?= test
export MAX_SAMPLES ?= 5000
export MIN_ACC ?= 0.0

include $(shell cocotb-config --makefiles)/Makefile.sim
