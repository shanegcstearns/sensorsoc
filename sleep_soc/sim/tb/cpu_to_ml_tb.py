import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer

from cocotbext.axi import AxiBus, AxiRam



# Helpers
def u32(x: int) -> int:
    return x & 0xFFFFFFFF


async def reset_dut(dut, cycles=10):
    """
    Start the clock, apply reset, and clear the fake-MMIO inputs.

    Why:
      - cpu_to_ml module is driven by cocotb through mem_* signals.
      - The bridge FSM requires mem_valid to drop between transactions.
      - We want no X propagation on mem_* at time 0.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())  # 100 MHz
    dut.resetn.value = 0

    dut.mem_valid.value = 0
    dut.mem_addr.value = 0
    dut.mem_wdata.value = 0
    dut.mem_wstrb.value = 0

    await ClockCycles(dut.clk, cycles)
    dut.resetn.value = 1
    await ClockCycles(dut.clk, 2)


async def mmio_write32(dut, addr: int, data: int, timeout_cycles=2000):
    """
    Perform one 32-bit write over the fake CPU MMIO interface.

    Protocol expectation (matches bridge):
      - Drive mem_valid high with addr/wdata/wstrb
      - Wait for mem_ready
      - MUST drop mem_valid so bridge can exit ST_RESP
    """
    dut.mem_addr.value = u32(addr)
    dut.mem_wdata.value = u32(data)
    dut.mem_wstrb.value = 0xF
    dut.mem_valid.value = 1

    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.mem_ready.value) == 1:
            break
    else:
        raise TimeoutError(f"MMIO write timeout @ 0x{addr:08X}")

    # drop valid so bridge can return to IDLE
    dut.mem_valid.value = 0
    dut.mem_wstrb.value = 0
    await RisingEdge(dut.clk)


async def mmio_read32(dut, addr: int, timeout_cycles=2000) -> int:
    """
    Perform one 32-bit read over the fake CPU MMIO interface.
    """
    dut.mem_addr.value = u32(addr)
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
    return u32(rdata)


def words_to_bytes(words):
    """
    Convert list of 32-bit words to little-endian byte array.
    """
    b = bytearray()
    for w in words:
        w = u32(w)
        b += bytes([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF])
    return bytes(b)


async def monitor_axi_activity(dut, counters, stop_after_cycles=20000):
    """
    Count real AXI master activity (maxi_* handshakes) to prove taketwo
    actually attempted RAM reads/writes.

    counters dict keys:
      - ar : read address handshakes
      - r  : read data handshakes
      - aw : write address handshakes
      - w  : write data handshakes
      - b  : write response handshakes
    """
    for _ in range(stop_after_cycles):
        await RisingEdge(dut.clk)

        # hierarchical path: sim_cpu_to_ml.dut.<signals> (cpu_to_ml instance)
        arvalid = int(dut.dut.maxi_arvalid.value)
        arready = int(dut.dut.maxi_arready.value)
        rvalid  = int(dut.dut.maxi_rvalid.value)
        rready  = int(dut.dut.maxi_rready.value)

        awvalid = int(dut.dut.maxi_awvalid.value)
        awready = int(dut.dut.maxi_awready.value)
        wvalid  = int(dut.dut.maxi_wvalid.value)
        wready  = int(dut.dut.maxi_wready.value)
        bvalid  = int(dut.dut.maxi_bvalid.value)
        bready  = int(dut.dut.maxi_bready.value)

        if arvalid and arready:
            counters["ar"] += 1
        if rvalid and rready:
            counters["r"] += 1
        if awvalid and awready:
            counters["aw"] += 1
        if wvalid and wready:
            counters["w"] += 1
        if bvalid and bready:
            counters["b"] += 1


# ----------------------------
# Tests
# ----------------------------

@cocotb.test()
async def test_axil_bridge_basic_reads(dut):
    """
    What this test proves:

    1) fake CPU MMIO interface can successfully issue READ transactions.
    2) ml_axil_bridge_mmio converts those into AXI-Lite reads.
    3) taketwo_wrap responds (AXI-Lite slave is alive, address decode works).
    4) The bridge FSM returns mem_ready and mem_rdata correctly (no hanging in ST_RESP).
    """
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

    # Minimal sanity: we got deterministic values and not an obvious poison pattern.
    assert global_off != 0xDEADBEEF
    assert out_base   != 0xDEADBEEF
    assert x_base     != 0xDEADBEEF
    assert var_base   != 0xDEADBEEF


@cocotb.test()
async def test_full_flow_start_busy_with_axi_ram(dut):
    """
    Full flow test:

    Goal:
      Prove that taketwo can run "far enough" to interact with its AXI master port
      without hanging, by providing it an AXI RAM slave model (AxiRam).

    Steps:
      1) Reset
      2) Read base-address registers via AXI-Lite (through your bridge)
      3) Instantiate AxiRam as a slave on maxi_* (taketwo's AXI master interface)
      4) Preload input/var memory (directly into the RAM model)
      5) Write START over AXI-Lite
      6) Confirm:
          - BUSY becomes 1 at least once
          - Some AXI traffic occurred (reads and/or writes)
          - Eventually BUSY clears OR IRQ asserts (within a timeout)
          - Output memory changed from a known pattern
    """
    await reset_dut(dut)

    ML_BASE = 0x0300_4000
    REG_START = ML_BASE + 0x10
    REG_BUSY  = ML_BASE + 0x14

    REG_GLOBAL_OFF = ML_BASE + 0x80
    REG_OUT_BASE   = ML_BASE + 0x88
    REG_X_BASE     = ML_BASE + 0x8C
    REG_VAR_BASE   = ML_BASE + 0x90

    global_off = await mmio_read32(dut, REG_GLOBAL_OFF)
    out_base   = await mmio_read32(dut, REG_OUT_BASE)
    x_base     = await mmio_read32(dut, REG_X_BASE)
    var_base   = await mmio_read32(dut, REG_VAR_BASE)

    # Build AXI bus from internal signals inside cpu_to_ml instance
    # (sim_cpu_to_ml has instance "dut" of cpu_to_ml => dut.dut.*)
    axi_bus = AxiBus.from_prefix(dut.dut, "maxi")

    # Create RAM model: 64KB is usually enough for bring-up.
    # NOTE: AxiRam provides an AXI *slave* that responds to taketwo's AXI *master*.
    ram = AxiRam(axi_bus, dut.clk, dut.resetn, size=1 << 16)

    # Fill output region with a known pattern so we can detect modification later.
    # IMPORTANT: AxiRam.write/read are NOT awaitable.
    out_pattern = bytes([0xA5]) * 64
    ram.write(out_base, out_pattern)

    # Create small dummy input arrays. Exact semantics don’t matter for this test —
    # we mainly want to exercise the AXI master port without deadlock.
    x_words   = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    var_words = [0x00000001, 0x00000002, 0x00000003, 0x00000004]

    ram.write(x_base,   words_to_bytes(x_words))
    ram.write(var_base, words_to_bytes(var_words))

    # Monitor AXI traffic to prove taketwo actually touched RAM
    counters = {"ar": 0, "r": 0, "aw": 0, "w": 0, "b": 0}
    mon_task = cocotb.start_soon(monitor_axi_activity(dut, counters, stop_after_cycles=50000))

    # Start accelerator
    await mmio_write32(dut, REG_START, 1)

    # Observe BUSY becoming 1 at least once (short window)
    saw_busy_1 = False
    for _ in range(2000):
        busy = await mmio_read32(dut, REG_BUSY)
        if busy == 1:
            saw_busy_1 = True
            break
    dut._log.info(f"BUSY seen as 1? {saw_busy_1}")
    assert saw_busy_1, "Expected BUSY to go high after START"

    # Wait for completion: either BUSY clears OR ml_irq asserts
    done = False
    for _ in range(20000):
        if int(dut.ml_irq.value) == 1:
            done = True
            dut._log.info("Done condition: ml_irq asserted")
            break

        busy = await mmio_read32(dut, REG_BUSY)
        if busy == 0:
            done = True
            dut._log.info("Done condition: BUSY cleared")
            break

    # Stop monitor (let it naturally end; it will exit after its cycle count)
    # (No hard cancel necessary.)
    assert done, "Timed out waiting for DONE (BUSY clear or IRQ)"

    dut._log.info(
        f"AXI activity counters: AR={counters['ar']} R={counters['r']} "
        f"AW={counters['aw']} W={counters['w']} B={counters['b']}"
    )

    # At minimum, for any real compute, you'd expect some reads.
    # Some designs may only read or only write depending on configuration.
    assert (counters["ar"] + counters["aw"]) > 0, "No AXI address activity detected on maxi_*"

    # Check whether output region changed from the pattern
    out_after = ram.read(out_base, 64)
    changed = out_after != out_pattern
    dut._log.info(f"Output changed? {changed}")
    assert changed, "Output buffer did not change (taketwo may not have written results)"