import re
import git
import os
import sys
import subprocess
import git

# I don't like this, but it's convenient.
_REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
assert (os.path.exists(_REPO_ROOT)), "REPO_ROOT path must exist"
sys.path.append(os.path.join(_REPO_ROOT, "util"))
#from utilities import runner, lint, assert_resolvable, clock_start_sequence, reset_sequence
tbpath = os.path.dirname(os.path.realpath(__file__))

import pytest

import cocotb

from cocotb.clock import Clock
from cocotb.regression import TestFactory
from cocotb.utils import get_sim_time
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, with_timeout, First
from cocotb.types import LogicArray, Range

from cocotb_test.simulator import run

#prob not axi since ML and features are on together, but ready valid might be nice tho
#from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiStreamSink, AxiStreamSource, AxiStreamBus
#from cocotbext.uart import UartSource, UartSink

#from pytest_utils.decorators import max_score, visibility, tags, leaderboard
   
import random
random.seed(42)

from functools import reduce

timescale = "1ps/1ps"

async def reset_sequence(clk_i, reset_i, cycles, FinishClkFalling=True, active_level=True):
    reset_i.setimmediatevalue(not active_level)

    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = active_level

    await ClockCycles(clk_i, cycles)

    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = not active_level

    reset_i._log.debug("Reset complete")

    # Always assign inputs on the falling edge
    if (not FinishClkFalling):
        await RisingEdge(clk_i)

async def clock_start_sequence(clk_i, period=1, unit='ns'):
    # Set the clock to Z for 10 ns. This helps separate tests.
    clk_i.value = LogicArray(['z'])
    await Timer(10, 'ns')

    # Unrealistically fast clock, but nice for mental math (1 GHz)
    c = Clock(clk_i, period, unit)

    # Start the clock (soon). Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(c.start(start_high=False))
    
def runner(simulator, timescale, tbpath, params, defs=[], testname=None, pymodule=None, jsonpath=None, jsonname="filelist.json", root=None):
    """Run the simulator on test n, with parameters params, and defines
    defs. If n is none, it will run all tests"""

    # if json path is none, assume that it is the same as tbpath
    if(jsonpath is None):
        jsonpath = tbpath

    assert (os.path.exists(jsonpath)), "jsonpath directory must exist"
    top = get_top(jsonpath, jsonname)

    # if pymodule is none, assume that the python module name is test+<name of the top module>.
    if(pymodule is None):
        pymodule = "test_" + top

    if(testname is None):
        testdir = "all"
    else:
        testdir=testname
    
    # Assume all paths in the json file are relative to the repository root.
    if(root is None):
        root = git.Repo(search_parent_directories=True).working_tree_dir

    assert (os.path.exists(root)), "root directory path must exist"

    sources = get_sources(root, tbpath)

    work_dir = os.path.join(tbpath, "run", testdir, get_param_string(params), simulator)
    build_dir = os.path.join(tbpath, "build", get_param_string(params))

    # Icarus doesn't build, it just runs.
    if simulator.startswith("icarus"):
        build_dir = work_dir

    if simulator.startswith("verilator"):
        compile_args=["-Wno-fatal", "-DVM_TRACE_FST=1", "-DVM_TRACE=1", "--timing"]
        plus_args = ["--trace", "--trace-fst"]
        if(not os.path.exists(work_dir)):
            os.makedirs(work_dir)
    else:
        compile_args=[]
        plus_args = []

    run(verilog_sources=sources,
        simulator=simulator,
        toplevel=top,
        module=pymodule,
        compile_args=compile_args,
        plus_args=plus_args,
        sim_build=build_dir,
        timescale=timescale,
        parameters=params,
        defines=defs + ["VM_TRACE_FST=1", "VM_TRACE=1"],
        work_dir=work_dir,
        waves=True,
        testcase=testname)

#What is our ML model reset case? does it have one?
@cocotb.test()
async def reset_test(dut):

    clk_i = dut.clk_i
    reset_i = dut.reset_i
    example_p = dut.example_p.value # Example

    # These are defined in utilities
    await clock_start_sequence(clk_i, 40) # 40 ns period is basically 25 MHz...
    await reset_sequence(clk_i, reset_i, 10)
    
@cocotb.test()
async def simple_test(dut):

    clk_i = dut.clk_i
    reset_i = dut.reset_i
    #example_p = dut.example_p.value # Example

    await clock_start_sequence(clk_i, 40) # 40 ns period is basically 25 MHz...
    await reset_sequence(clk_i, reset_i, 10)
    in_data = open("processed_sleep_dataset.csv", "r")
    for line in in_data:
        feats = line.strip().split(",") 
        #it's all floats for now, prob change this with shane
        dut.movement_i.value = float(feats[0])
        dut.cosine_i.value = float(feats[1])
        dut.delta_hr_i.value = float(feats[2])
        dut.hr_rmssd_i.value = float(feats[3]) 
        
        print("ML hw sim guess: ", dut.wake_o.value)
        #print("ML hw sim conf: ", dut.confidence_o.value) if we need this
        await Timer(40, units="ns")