# Makefile

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

<<<<<<< HEAD
VERILOG_SOURCES += $(PWD)/taketwo_wrap.v $(PWD)/taketwo.v
# use VHDL_SOURCES for VHDL files

# COCOTB_TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
COCOTB_TOPLEVEL = taketwo_wrap
=======
VERILOG_SOURCES += $(PWD)/taketwo.v
# use VHDL_SOURCES for VHDL files

# COCOTB_TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
COCOTB_TOPLEVEL = taketwo
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91

# COCOTB_TEST_MODULES is the basename of the Python test file(s)
COCOTB_TEST_MODULES = ml_tb

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim