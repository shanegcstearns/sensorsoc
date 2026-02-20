import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ClockCycles
from cocotbext.axi import AxiLiteBus, AxiLiteMaster, AxiBus, AxiRam


def le32(b: bytes) -> int:
    return int.from_bytes(b, "little", signed=False)


async def reset_dut(dut, cycles=10):
    cocotb.start_soon(Clock(dut.CLK, 40, unit="ns").start())
    await FallingEdge(dut.CLK)
    dut.RESETN.value = 0
    await ClockCycles(dut.CLK, cycles)
    await FallingEdge(dut.CLK)
    dut.RESETN.value = 1
    await ClockCycles(dut.CLK, 2)


@cocotb.test()
async def load_weights_and_readback(dut):
    await reset_dut(dut)

    axil = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "saxi"), dut.CLK, dut.RESETN, reset_active_level=False)

    # IMPORTANT: use the wrapper top so maxi_*id exists
    axi_ram = AxiRam(AxiBus.from_prefix(dut, "maxi"), dut.CLK, dut.RESETN, reset_active_level=False, size=1 << 20)

    global_off = le32(await axil.read(0x80, 4))  # reg 128
    var_base   = le32(await axil.read(0x90, 4))  # reg 144
    var_addr   = (global_off + var_base) & 0xFFFFFFFF

    bin_path = os.path.join("taketwo_params.bin")
    with open(bin_path, "rb") as f:
        param_bytes = f.read()

    dut._log.info(f"Writing {len(param_bytes)} bytes to var_addr=0x{var_addr:08X}")
    axi_ram.write(var_addr, param_bytes)

    dut._log.info("Reading back...")
    rb = axi_ram.read(var_addr, len(param_bytes))

    assert rb == param_bytes, "Readback mismatch!"
    dut._log.info("OK: weights write+readback matched exactly")

    # Print a few chunks so you can eyeball
    for i in range(0, min(256, len(param_bytes)), 16):
        dut._log.info(f"param[{i:04d}:{i+16:04d}] = {rb[i:i+16].hex()}")