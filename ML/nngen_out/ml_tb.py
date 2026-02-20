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


    global_off = le32(await axil.read(0x80, 4))  # 128
    out_base   = le32(await axil.read(0x88, 4))  # 136 
    x_base     = le32(await axil.read(0x8C, 4))  # 140 
    var_base   = le32(await axil.read(0x90, 4))  # 144 

    out_addr = (global_off + out_base) & 0xFFFFFFFF
    x_addr   = (global_off + x_base) & 0xFFFFFFFF
    var_addr = (global_off + var_base) & 0xFFFFFFFF

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
    assert rb == param_bytes, "Readback mismatch!"
    dut._log.info("OK: weights write+readback matched exactly")

    # writing input vector into memory (real test data from csv)

    x_vals = [00, -93, -154, 0]
    x_bytes = struct.pack("<4h", *x_vals) 
    axi_ram.write(x_addr, x_bytes)
    dut._log.info(f"Wrote x={x_vals} to x_addr=0x{x_addr:08X}")

    # start accelerator
    dut._log.info("Writing START=1 to reg 0x10")
    await axil.write(0x10, u32(1))

    # await axil.write(0x18, u32(1))  # reg 24: Reset
    # await axil.write(0x18, u32(0))

    # wait for busy to clear
    for _ in range(2000):
        busy = le32(await axil.read(0x14, 4))
        if busy == 0:
            break
        await ClockCycles(clk_i, 10)
    dut._log.info(f"Busy now = {busy}")
    assert busy == 0, "Timeout waiting for accelerator to finish"

    # read logits from output address
    out_bytes = axi_ram.read(out_addr, 4)
    log0, log1 = struct.unpack("<2h", out_bytes)
    dut._log.info(f"logits int16: [{log0}, {log1}] (raw bytes={out_bytes.hex()})")
