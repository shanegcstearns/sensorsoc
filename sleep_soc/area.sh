#!/bin/bash
# ------------------------------------------------------------
# Run Yosys area estimation using OpenROAD-built Yosys
# ------------------------------------------------------------

# Path to OpenROAD-built Yosys
YO_PATH=~/cse122/OpenROAD-flow-scripts/tools/yosys/yosys

# Check it exists
if [ ! -x "$YO_PATH" ]; then
    echo "ERROR: Yosys binary not found at $YO_PATH"
    exit 1
fi

# Liberty file for GF180MCU
LIB=/home/jfriday/pdk/volare/gf180mcu/versions/c6d73a35f524070e85faff4a6a9eef49553ebc2b/gf180mcuC/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/gf180mcu_fd_sc_mcu7t5v0__tt_025C_5v00.lib

# Run Yosys with inline script
"$YO_PATH" -p "
# Load technology library
read_liberty -lib $LIB

# Read all RTL
read_verilog -sv rtl/*.sv
read_verilog rtl/*.v
read_verilog rtl/third_party/*.v

# Blackbox NNgen RAM/FIFO
blackbox ram_*
blackbox *_fifo*

# Set top module
hierarchy -check -top dummy_top

# Flatten for area estimate
flatten

# Generic synthesis
synth -top dummy_top

# Prevent memory inference
memory -nomap

# Optimize and clean
opt
opt_clean

# Map flip-flops
dfflibmap -liberty $LIB

# Map combinational logic
abc -liberty $LIB

# Final cleanup
opt_clean

# Area report
stat -liberty $LIB
"