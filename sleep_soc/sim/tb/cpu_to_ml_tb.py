import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Requires: pip install cocotbext-axi
from cocotbext.axi import AxiBus, AxiRam



# Small helpers
def le32(x: int) -> int:
    return x & 0xFFFFFFFF


async def reset_dut(dut, cycles=10):
    # If  sim wrapper already has an always #5 clock, comment this out:
    # cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.resetn.value = 0

    # Clear fake CPU bus
    dut.mem_valid.value = 0
    dut.mem_addr.value  = 0
    dut.mem_wdata.value = 0
    dut.mem_wstrb.value = 0

    await ClockCycles(dut.clk, cycles)
    dut.resetn.value = 1
    await ClockCycles(dut.clk, 2)


async def mmio_write32(dut, addr: int, data: int, timeout_cycles=2000):
    """
    Drive the fake CPU mem_* bus for a write and wait for mem_ready.
    IMPORTANT:
       ml_axil_bridge_mmio has a ST_RESP state that waits for mem_valid to drop.
      So after mem_ready, we MUST deassert mem_valid (and clear mem_wstrb) for at least 1 cycle.
    """
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

    dut.mem_valid.value = 0
    dut.mem_wstrb.value = 0
    await RisingEdge(dut.clk)


async def mmio_read32(dut, addr: int, timeout_cycles=2000) -> int:

    #Drive the fake CPU mem_* bus for a read and wait for mem_ready.
    #Returns mem_rdata.
    
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


async def poll_reg_eq(dut, addr: int, expected: int, max_tries=2000, gap_cycles=1):
    
    #Poll a register until it equals expected, with a hard timeout.
    
    for _ in range(max_tries):
        val = await mmio_read32(dut, addr)
        if val == expected:
            return val
        await ClockCycles(dut.clk, gap_cycles)
    raise TimeoutError(f"Timeout polling 0x{addr:08X} for {expected}, last={val}")



# Tests
@cocotb.test()
async def test_axil_bridge_basic_reads(dut):
    """
    What this test proves:
      - The fake CPU mem_* -> ml_axil_bridge_mmio FSM works for READs
      - The bridge successfully completes an AXI-Lite read via taketwo_wrap
      - The design is not stuck in ST_RESP / ST_R_R, and mem_ready toggles

    What it does NOT prove:
      - AXI master (maxi_*) behavior
      - START/run completion
      - Any ML correctness
    """
    await reset_dut(dut)

    ML_BASE = 0x0300_4000

    # Your taketwo “RAM AXI interface” regs (offsets)
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

    # “Smoke” checks: you got a stable response (not a sentinel).
    assert global_off != 0xDEADBEEF
    assert out_base   != 0xDEADBEEF
    assert x_base     != 0xDEADBEEF
    assert var_base   != 0xDEADBEEF


@cocotb.test()
async def test_full_flow_start_busy_with_axi_ram(dut):
    """
    Full flow test using cocotbext.axi.AxiRam for the AXI master (maxi_*) interface.

    This test:
      1) Attaches an AXI RAM slave to taketwo's maxi_* bus (so it won't hang)
      2) Programs taketwo's address registers over AXI-Lite (through your bridge)
      3) Preloads RAM at x_base/var_base
      4) Clears output RAM at out_base
      5) Writes START
      6) Polls BUSY until it clears
      7) Checks that output RAM changed

    Notes:
      -  may need to adjust the RAM size and base addresses to match what taketwo expects.
      - If taketwo uses bursts/widths not supported by compiled design, you'll see errors.
    """
    await reset_dut(dut)

    ML_BASE   = 0x0300_4000
    REG_START = ML_BASE + 0x10
    REG_BUSY  = ML_BASE + 0x14

    REG_GLOBAL_OFF = ML_BASE + 0x80
    REG_OUT_BASE   = ML_BASE + 0x88
    REG_X_BASE     = ML_BASE + 0x8C
    REG_VAR_BASE   = ML_BASE + 0x90

    # 1) Hook AxiRam to taketwo's AXI master interface (maxi_*)
    #
    # wrapper instantiates cpu_to_ml as "dut", so the maxi_* signals live at:
    #   sim_cpu_to_ml.dut.maxi_awaddr, etc.
    #
    axi_bus = AxiBus.from_prefix(dut.dut, "maxi")
    ram = AxiRam(axi_bus, dut.clk, dut.resetn, size=1 << 16)  # 64KB RAM
    # AxiRam starts its coroutines automatically on construction.

    # 2) Choose addresses inside the RAM window
    global_off = 0x00000100
    out_base   = 0x00000200
    x_base     = 0x00000300
    var_base   = 0x00000400

    # Program address-map regs via bridge (fake CPU -> mem_* -> AXI-Lite)
    await mmio_write32(dut, REG_GLOBAL_OFF, global_off)
    await mmio_write32(dut, REG_OUT_BASE,   out_base)
    await mmio_write32(dut, REG_X_BASE,     x_base)
    await mmio_write32(dut, REG_VAR_BASE,   var_base)

    # Optional: read back to confirm write path works end-to-end
    rb_out_base = await mmio_read32(dut, REG_OUT_BASE)
    assert rb_out_base == out_base, f"OUT_BASE readback mismatch: got 0x{rb_out_base:08X}"


    # 3) Preload RAM.
    # cocotbext.axi.AxiRam uses byte addressing. Write bytes via .write().
    def words_to_bytes(words):
        out = bytearray()
        for w in words:
            w = le32(w)
            out += bytes([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF])
        return out

    # Example input vectors / vars (replace with real formats later)
    x_words   = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    var_words = [0xAAAA0001, 0xAAAA0002, 0xAAAA0003, 0xAAAA0004]

    await ram.write(x_base,   words_to_bytes(x_words))
    await ram.write(var_base, words_to_bytes(var_words))

    # Clear output region (16 bytes)
    await ram.write(out_base, bytes([0] * 16))

    out_before = await ram.read(out_base, 16)
    dut._log.info(f"out_before={out_before.hex()}")


    # 4) START + 5) poll BUSY
    await mmio_write32(dut, REG_START, 1)

    # BUSY is typically 1 while running, 0 when done.
    # If design's semantics differ, change this.
    # First confirm it goes busy (optional)
    busy1 = await mmio_read32(dut, REG_BUSY)
    dut._log.info(f"BUSY after START = {busy1}")

    # Poll until BUSY clears (timeout-protected)
    # Increase max_tries if accelerator takes longer in sim.
    await poll_reg_eq(dut, REG_BUSY, expected=0, max_tries=5000, gap_cycles=2)
    dut._log.info("BUSY cleared")


    # 6) Verify output changed
    out_after = await ram.read(out_base, 16)
    dut._log.info(f"out_after ={out_after.hex()}")

    assert out_after != out_before, "Output RAM did not change; expected taketwo to write results"