# import os
# import cocotb
# from cocotb.clock import Clock
# from cocotb.triggers import FallingEdge, ClockCycles
# from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiBus, AxiRam


# def le32(b: bytes) -> int:
#     return int.from_bytes(b, "little", signed=False)


# async def reset_dut(dut, cycles=10):
#     cocotb.start_soon(Clock(dut.CLK, 40, unit="ns").start())
#     await FallingEdge(dut.CLK)
#     dut.RESETN.value = 0
#     await ClockCycles(dut.CLK, cycles)
#     await FallingEdge(dut.CLK)
#     dut.RESETN.value = 1
#     await ClockCycles(dut.CLK, 2)


# @cocotb.test()
# async def load_weights_and_readback(dut):
#     await reset_dut(dut)
#     clk_i = dut.CLK
#     axil = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "saxi"), dut.CLK, dut.RESETN, reset_active_level=False)

#     # IMPORTANT: use the wrapper top so maxi_*id exists
#     axi_ram = AxiRam(AxiBus.from_prefix(dut, "maxi"), dut.CLK, dut.RESETN, reset_active_level=False, size=1 << 20)

#     global_off = le32(await axil.read(0x80, 4))  # reg 128
#     var_base   = le32(await axil.read(0x90, 4))  # reg 144
#     var_addr   = (global_off + var_base) & 0xFFFFFFFF

#     bin_path = os.path.join("taketwo_params.bin")
#     with open(bin_path, "rb") as f:
#         param_bytes = f.read()

#     dut._log.info(f"Writing {len(param_bytes)} bytes to var_addr=0x{var_addr:08X}")
#     axi_ram.write(var_addr, param_bytes)

#     dut._log.info("Reading back...")
#     rb = axi_ram.read(var_addr, len(param_bytes))

#     assert rb == param_bytes, "Readback mismatch!"
#     dut._log.info("OK: weights write+readback matched exactly")

#     # Print a few chunks so you can eyeball
#     # for i in range(0, min(256, len(param_bytes)), 16):
#     #     dut._log.info(f"param[{i:04d}:{i+16:04d}] = {rb[i:i+16].hex()}")
        
        
#     print("Writing START=1 to address 0x10")
#     await axil.write(0x10, {0x00,0x00,0x00,0x01})
    
#     await ClockCycles(clk_i, 20)
    
#     busy = le32(await axil.read(0x14, 4))
#     print(f"Check busy signal: {busy}")
        
#     await axil.write(0x8C,{0x00,0x5D,0x9A,0x00})
#     await axil.read(0x14, 4)
#     await ClockCycles(clk_i, 100)
    
#     read_out_test = await axil.read(0x88, 4) 
#     print(f"Read output test value: 0x{read_out_test}")
#     await axil.read(0x14, 4)
#     await ClockCycles(clk_i, 1000)
#     await axil.read(0x88, 4) 
import os
import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ClockCycles
from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiBus, AxiRam


def le32(b: bytes) -> int:
    return int.from_bytes(b, "little", signed=False)

def u32(x: int) -> bytes:
    return int(x & 0xFFFFFFFF).to_bytes(4, "little")


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

    # --- Read addresses from NNgen regs ---
    global_off = le32(await axil.read(0x80, 4))  # reg 128
    out_base   = le32(await axil.read(0x88, 4))  # reg 136 (address of output buffer)
    x_base     = le32(await axil.read(0x8C, 4))  # reg 140 (address of placeholder x)
    var_base   = le32(await axil.read(0x90, 4))  # reg 144 (address of variables blob)

    out_addr = (global_off + out_base) & 0xFFFFFFFF
    x_addr   = (global_off + x_base) & 0xFFFFFFFF
    var_addr = (global_off + var_base) & 0xFFFFFFFF

    dut._log.info(f"global_off=0x{global_off:08X}")
    dut._log.info(f"x_addr   =0x{x_addr:08X}  (reg 0x8C + offset)")
    dut._log.info(f"out_addr =0x{out_addr:08X}  (reg 0x88 + offset)")
    dut._log.info(f"var_addr =0x{var_addr:08X}  (reg 0x90 + offset)")

    # --- Load weights blob into maxi memory at var_addr ---
    bin_path = "taketwo_params.bin"  # same dir as Makefile run dir
    with open(bin_path, "rb") as f:
        param_bytes = f.read()

    dut._log.info(f"Writing {len(param_bytes)} bytes of weights to var_addr=0x{var_addr:08X}")
    axi_ram.write(var_addr, param_bytes)

    dut._log.info("Reading weights back...")
    rb = axi_ram.read(var_addr, len(param_bytes))
    assert rb == param_bytes, "Readback mismatch!"
    dut._log.info("OK: weights write+readback matched exactly")

    # --- Write one input vector x (4x int16) into maxi memory at x_addr ---
    # Example from your ng.eval printout: x_int = [128, -64, 26, -256]
    x_vals = [00, -93, -154, 0]
    x_bytes = struct.pack("<4h", *x_vals)  # 8 bytes
    axi_ram.write(x_addr, x_bytes)
    dut._log.info(f"Wrote x={x_vals} to x_addr=0x{x_addr:08X}")


    # pick a safe scratch region (must be 4-byte aligned; 64-byte aligned is nice)
    TEMP_ADDR = 0x2000

    # Program address regs explicitly (these are RW regs)
    await axil.write(0x80, u32(0))         # global offset = 0
    await axil.write(0x84, u32(TEMP_ADDR)) # temporal storages base
    await axil.write(0x88, u32(0x0000))    # output base
    await axil.write(0x8C, u32(0x0040))    # x base
    await axil.write(0x90, u32(0x0080))    # variables base

    axi_ram.write(0x0000, b"\x00\x00\x00\x00")

    # --- Start ---
    dut._log.info("Writing START=1 to reg 0x10")
    await axil.write(0x10, u32(1))

    # await axil.write(0x18, u32(1))  # reg 24: Reset
    # await axil.write(0x18, u32(0))

    # --- Wait for busy to clear ---
    # Busy is reg 0x14
    for _ in range(2000):
        busy = le32(await axil.read(0x14, 4))
        if busy == 0:
            break
        await ClockCycles(clk_i, 10)
    dut._log.info(f"Busy now = {busy}")
    assert busy == 0, "Timeout waiting for accelerator to finish"

    # --- Read logits from maxi memory at out_addr ---
    out_bytes = axi_ram.read(out_addr, 4)  # 2x int16 = 4 bytes
    log0, log1 = struct.unpack("<2h", out_bytes)
    dut._log.info(f"logits int16: [{log0}, {log1}] (raw bytes={out_bytes.hex()})")

    # Optional: also show the pointer regs so you can sanity-check
    dut._log.info(f"reg 0x88 (out_base) = 0x{out_base:08X}")
    dut._log.info(f"reg 0x8C (x_base)   = 0x{x_base:08X}")