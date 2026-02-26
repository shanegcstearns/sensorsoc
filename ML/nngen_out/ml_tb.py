<<<<<<< HEAD
import os
import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ClockCycles
from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiBus, AxiRam


SCALE = 8192

def le32(b: bytes) -> int: # interpret 4 bytes as little-endian unsigned int
    return int.from_bytes(b, "little", signed=False)

def u32(x: int) -> bytes: # convert an int to 4 bytes in little-endian order (unsigned)
    return int(x & 0xFFFFFFFF).to_bytes(4, "little")

def u16_4(x: list[int]) -> bytes: # convert 4 int16s to byte struct (feature conversion)
    return struct.pack("<4h", *x) 


async def reset_dut(dut, cycles=10):
    cocotb.start_soon(Clock(dut.CLK, 40, unit="ns").start())
    await FallingEdge(dut.CLK)
    dut.RESETN.value = 0
    await ClockCycles(dut.CLK, cycles)
    await FallingEdge(dut.CLK)
    dut.RESETN.value = 1
    await ClockCycles(dut.CLK, 2)


@cocotb.test()
async def load_weights_and_infer_once(dut):
    await reset_dut(dut)
    clk_i = dut.CLK

    axil = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "saxi"),
        dut.CLK,
        dut.RESETN,
        reset_active_level=False,
    )

    axi_ram = AxiRam(
        AxiBus.from_prefix(dut, "maxi"),
        dut.CLK,
        dut.RESETN,
        reset_active_level=False,
        size=1 << 20,
    )

    # read base addresses from registers
    global_off = le32(await axil.read(0x80, 4))  # 128
    out_base   = le32(await axil.read(0x88, 4))  # 136 
    x_base     = le32(await axil.read(0x8C, 4))  # 140 
    var_base   = le32(await axil.read(0x90, 4))  # 144 

    # convert to offset matching addresses, and mask to 32 bits
    out_addr = (global_off + out_base) & 0xFFFFFFFF
    x_addr   = (global_off + x_base) & 0xFFFFFFFF
    var_addr = (global_off + var_base) & 0xFFFFFFFF

    #print offsets
    dut._log.info(f"global_off=0x{global_off:08X}")
    dut._log.info(f"x_addr   =0x{x_addr:08X}  (reg 0x8C + offset)")
    dut._log.info(f"out_addr =0x{out_addr:08X}  (reg 0x88 + offset)")
    dut._log.info(f"var_addr =0x{var_addr:08X}  (reg 0x90 + offset)")

    # load weights from bin
    bin_path = "taketwo_params.bin"
    with open(bin_path, "rb") as f:
        param_bytes = f.read()

    dut._log.info(f"Writing {len(param_bytes)} bytes of weights to var_addr=0x{var_addr:08X}")
    axi_ram.write(var_addr, param_bytes)

    dut._log.info("Reading weights back...")
    rb = axi_ram.read(var_addr, len(param_bytes))
    assert rb == param_bytes, "Comparison from written weights"

    # start accelerator
    #dut._log.info("Writing START=1 to reg 0x10")
    #await axil.write(0x10, u32(1))

    # writing input vector into memory (real test data from csv)
    with open("processed_sleep_dataset.csv") as f:
        ds = f.readlines()
    
    for line in ds:
        print(line)
        feats = line.split(",")
        print(feats)
        await axil.write(0x10, u32(1))
        feats[0] = max(-32768, min(int((float(feats[0])/8.0) * SCALE), 32767))    #movement
        feats[1] = max(-32768, min(int((float(feats[1])) * SCALE), 32767))        #cosine
        feats[2] = max(-32768, min(int((float(feats[2])/4.0) * SCALE), 32767))    #delta hr
        feats[3] = max(-32768, min(int((float(feats[0])*20.0) * SCALE), 32767))   #rmssd
        axi_ram.write(x_addr, u16_4(feats))
        # wait for busy to clear
        while(True):
            busy = le32(await axil.read(0x14, 4))
            if busy == 0:
                break
            await ClockCycles(clk_i, 10)

        # read logits from output address
        out_bytes = axi_ram.read(out_addr, 4)
        log0, log1 = struct.unpack("<2h", out_bytes)
        dut._log.info(f"logits int16: [{log0}, {log1}] (raw bytes={out_bytes.hex()})")
=======
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
>>>>>>> e6977a12a8c62aa40b661d9f693abea2a9057d91
