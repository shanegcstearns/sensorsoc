import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

def le32(x: int) -> int:
    return x & 0xFFFFFFFF

async def reset_dut(dut, cycles=10):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.resetn.value = 0

    dut.mem_valid.value = 0
    dut.mem_addr.value  = 0
    dut.mem_wdata.value = 0
    dut.mem_wstrb.value = 0

    await ClockCycles(dut.clk, cycles)
    dut.resetn.value = 1
    await ClockCycles(dut.clk, 2)

async def mmio_write32(dut, addr: int, data: int, timeout_cycles=2000):
    dut.mem_addr.value  = le32(addr)
    dut.mem_wdata.value = le32(data)
    dut.mem_wstrb.value = 0xF
    dut.mem_valid.value = 1

    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.mem_ready.value) == 1:
            break
    else:
        raise TimeoutError(f"MMIO write timeout @ 0x{addr:08X}")

    # IMPORTANT: drop mem_valid so bridge can leave ST_RESP
    dut.mem_valid.value = 0
    dut.mem_wstrb.value = 0
    await RisingEdge(dut.clk)

async def mmio_read32(dut, addr: int, timeout_cycles=2000) -> int:
    dut.mem_addr.value  = le32(addr)
    dut.mem_wdata.value = 0
    dut.mem_wstrb.value = 0
    dut.mem_valid.value = 1

    rdata = None
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.mem_ready.value) == 1:
            rdata = int(dut.mem_rdata.value)
            break
    else:
        raise TimeoutError(f"MMIO read timeout @ 0x{addr:08X}")

    dut.mem_valid.value = 0
    await RisingEdge(dut.clk)
    return le32(rdata)

@cocotb.test()
async def test_axil_bridge_basic_reads(dut):
    await reset_dut(dut)

    ML_BASE = 0x0300_4000

    # taketwo “RAM AXI interface” regs (offsets)
    REG_GLOBAL_OFF = ML_BASE + 0x80
    REG_OUT_BASE   = ML_BASE + 0x88
    REG_X_BASE     = ML_BASE + 0x8C
    REG_VAR_BASE   = ML_BASE + 0x90

    global_off = await mmio_read32(dut, REG_GLOBAL_OFF)
    out_base   = await mmio_read32(dut, REG_OUT_BASE)
    x_base     = await mmio_read32(dut, REG_X_BASE)
    var_base   = await mmio_read32(dut, REG_VAR_BASE)

    dut._log.info(f"global_off=0x{global_off:08X}")
    dut._log.info(f"out_base  =0x{out_base:08X}")
    dut._log.info(f"x_base    =0x{x_base:08X}")
    dut._log.info(f"var_base  =0x{var_base:08X}")

    # just prove reads respond with something stable-looking
    assert global_off != 0xDEADBEEF
    assert out_base   != 0xDEADBEEF
    assert x_base     != 0xDEADBEEF
    assert var_base   != 0xDEADBEEF

@cocotb.test()
async def test_axil_bridge_start_busy_nohang(dut):
    await reset_dut(dut)

    ML_BASE = 0x0300_4000
    REG_START = ML_BASE + 0x10
    REG_BUSY  = ML_BASE + 0x14

    await mmio_write32(dut, REG_START, 1)
    busy = await mmio_read32(dut, REG_BUSY)

    dut._log.info(f"BUSY readback = {busy}")
    assert busy in (0, 1)