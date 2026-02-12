import cocotb
from cocotb.clock import Clock
from cocotb.types import LogicArray
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, with_timeout
from cocotbext.axi import AxiBus, AxiMaster
from cocotb.triggers import Timer


@cocotb.test()
async def reset_test(dut):

    clk_i = dut.CLK
    reset_i = dut.RESETN
    irq = dut.irq.value # Tied to reset values, maybe it means something?

    #125 clock start
    clk_i.value = LogicArray(['z'])
    await Timer(10, 'ns')
    c = Clock(clk_i, 40, 'ns')
    # Start the clock (soon). Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(c.start(start_high=False))
    
    #125 reset sequence
    active_level = 1 # I think the model.v is active high
    await FallingEdge(clk_i)
    reset_i.value = active_level

    await ClockCycles(clk_i, 10)

    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = not active_level

    reset_i._log.debug("Reset complete")
    print(f"irq is {irq}")

@cocotb.test()
async def axis_interface_test(dut):
    clk_i = dut.CLK
    reset_i = dut.RESETN
    irq = dut.irq.value # Tied to reset values, maybe it means something?

    #125 clock start
    clk_i.value = LogicArray(['z'])
    await Timer(10, 'ns')
    c = Clock(clk_i, 40, 'ns')
    # Start the clock (soon). Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(c.start(start_high=False))
    
    #125 reset sequence
    active_level = 1 # I think the model.v is active high
    await FallingEdge(clk_i)
    reset_i.value = active_level

    await ClockCycles(clk_i, 10)

    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = not active_level

    reset_i._log.debug("Reset complete")
    print(f"irq is {irq}") #what is this???
    
    axi_master = AxiMaster(AxiBus.from_prefix(dut, "s_axi"), dut.CLK, dut.RESETN)
    data = await axi_master.read(0x0048, 4) #random test read
    print(f"Read data: {data}")