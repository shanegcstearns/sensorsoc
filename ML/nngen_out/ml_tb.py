import cocotb
from cocotb.clock import Clock
from cocotb.types import LogicArray
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, with_timeout
from cocotbext.axi import AxiLiteBus, AxiLiteMaster
from cocotb.triggers import Timer
import warnings

# Silenceing useless warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    active_level = 0
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
    active_level = 0
    await FallingEdge(clk_i)
    reset_i.value = active_level

    await ClockCycles(clk_i, 10)

    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = not active_level

    reset_i._log.debug("Reset complete")
    print(f"irq is {irq}") #what is this???
    
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "saxi"),
        dut.CLK,
        dut.RESETN,
        reset_active_level=False
    )

    def le32(b: bytes) -> int:
        return int.from_bytes(b, byteorder="little", signed=False)
    def u32(x: int) -> bytes:
        return int(x & 0xFFFFFFFF).to_bytes(4, byteorder="little")
    print("")  
    print("$$$$$$$$$$$$$$$")            
    print("TESTING STARTUP")                
    print("$$$$$$$$$$$$$$$")  

    # reading headers & busy
    print("\n*****Reading header registers*****")
    print("Expecting empty\n")

    hdr0 = le32(await axi_master.read(0x00, 4))
    print(f"Header0 (0x00): 0x{hdr0:08X} ({hdr0})")
    await ClockCycles(clk_i, 20)

    hdr1 = le32(await axi_master.read(0x04, 4))
    print(f"Header1 (0x04): 0x{hdr1:08X} ({hdr1})")
    await ClockCycles(clk_i, 20)

    hdr2 = le32(await axi_master.read(0x08, 4))
    print(f"Header2 (0x08): 0x{hdr2:08X} ({hdr2})")
    await ClockCycles(clk_i, 20)

    hdr3 = le32(await axi_master.read(0x0C, 4))
    print(f"Header3 (0x0C): 0x{hdr3:08X} ({hdr3})")
    await ClockCycles(clk_i, 20)

    busy = le32(await axi_master.read(0x14, 4))
    print(f"Busy BEFORE start (0x14): {busy}")
    await ClockCycles(clk_i, 20)

    # check busy signal before and after start

    await ClockCycles(clk_i, 20)

    print("Writing START=1 to address 0x10")
    await axi_master.write(0x10, u32(1))

    await ClockCycles(clk_i, 20)

    busy2 = le32(await axi_master.read(0x14, 4))
    print(f"Busy AFTER start: {busy2}")

    if busy2 == 1:
        print(">>> Accelerator started successfully (Busy=1)")
    else:
        print(">>> Accelerator did NOT start (Busy still 0)")

    print("\n$$$$$$$$$$$$$$$$$")
    print("END STARTUP TESTS")
    print("$$$$$$$$$$$$$$$$$\n")

    print("\n$$$$$$$")
    print("IO TEST")
    print("$$$$$$$\n")

    # writing strange values for feature data to hopefully read something other than zero out of register 36

    await axi_master.write(0x8C, u32(2))
    await ClockCycles(clk_i, 20)
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)
    await axi_master.write(0x8C, u32(4))
    await ClockCycles(clk_i, 20)
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)
    await axi_master.write(0x8C, u32(8))
    await ClockCycles(clk_i, 20)
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)
    await axi_master.write(0x8C, u32(16))
    await ClockCycles(clk_i, 20) 
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)   
    await axi_master.write(0x8C, u32(32))
    await ClockCycles(clk_i, 20)
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)
    await axi_master.write(0x8C, u32(64))
    await ClockCycles(clk_i, 20)
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)
    await axi_master.write(0x8C, u32(128))
    await ClockCycles(clk_i, 20)
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)
    await axi_master.write(0x8C, u32(256))
    await ClockCycles(clk_i, 20)
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)
    await axi_master.write(0x8C, u32(512))
    await ClockCycles(clk_i, 20)
    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)
    await axi_master.write(0x8C, u32(1024))
    await ClockCycles(clk_i, 20)

    state = le32(await axi_master.read(0x88, 4))
    await ClockCycles(clk_i, 20)