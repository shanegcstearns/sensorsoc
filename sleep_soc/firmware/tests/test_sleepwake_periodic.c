#include <stdint.h>

// ----------------------
// Base addresses
// ----------------------
#define TIMER_BASE     0x03002000u
#define PWR_BASE       0x03001000u
#define TEST_BASE      0x0300F000u

// ----------------------
// timer_mmio offsets
// ----------------------
#define T_CTRL         (TIMER_BASE + 0x0u)  // bit0 enable, bit1 periodic
#define T_RELOAD       (TIMER_BASE + 0x4u)
#define T_COUNT        (TIMER_BASE + 0x8u)
#define T_EVENT        (TIMER_BASE + 0xCu)  // W1C bit0, read shows latched

// ----------------------
// pwrctrl_mmio offsets
// ----------------------
#define P_CTRL         (PWR_BASE + 0x0u)    // bit0 sleep_req
#define P_WAKE_STATUS  (PWR_BASE + 0x4u)    // R/W1C latched wake flags
#define P_WAKE_REASON  (PWR_BASE + 0x8u)    // RO snapshot on wake

// ----------------------
// test_mmio offsets
// ----------------------
#define TEST_STATUS    (TEST_BASE + 0x0u)
#define TEST_CODE      (TEST_BASE + 0x4u)

#define TEST_PASS      0xCAFEBABEu
#define TEST_FAIL      0xDEADBEEFu

// Wake bits per your soc_top: bit0=timer, bit1=ml
#define WAKE_TIMER_BIT (1u << 0)

static inline void mmio_write32(uint32_t addr, uint32_t data) {
    *(volatile uint32_t *)addr = data;
}

static inline uint32_t mmio_read32(uint32_t addr) {
    return *(volatile uint32_t *)addr;
}

static inline void delay_cycles(volatile uint32_t n) {
    while (n--) __asm__ volatile ("nop");
}

static void pass(void) {
    mmio_write32(TEST_STATUS, TEST_PASS);
    while (1) { }
}

static void fail(uint32_t code) {
    mmio_write32(TEST_CODE, code);
    mmio_write32(TEST_STATUS, TEST_FAIL);
    while (1) { }
}

int main(void) {
    // ----------------------
    // Clean slate
    // ----------------------
    mmio_write32(P_CTRL, 0u);                 // clear sleep request
    mmio_write32(P_WAKE_STATUS, 0xFFFFFFFFu); // W1C: clear all latched wake bits
    mmio_write32(T_EVENT, 1u);                // clear timer event latch (W1C)

    // Disable timer before programming (safe)
    mmio_write32(T_CTRL, 0u);

    // ----------------------
    // Program periodic timer
    // ----------------------
    // Make it quick in sim
    mmio_write32(T_RELOAD, 20u);
    mmio_write32(T_COUNT,  5u);

    // enable=1, periodic=1  => 0b11
    mmio_write32(T_CTRL, 0x3u);

    // ----------------------
    // Request sleep
    // ----------------------
    mmio_write32(P_CTRL, 1u);

    // Allow always-on FSM to see cpu idle at least once
    // (your soc gates after it observes mem_valid==0 while cpu_awake)
    delay_cycles(200);

    // If we got here, we are running after the timer should have fired.
    // Now verify the *cause* info latched properly.
    uint32_t reason = mmio_read32(P_WAKE_REASON);
    uint32_t status = mmio_read32(P_WAKE_STATUS);
    uint32_t tev    = mmio_read32(T_EVENT) & 1u;

    // 0x20: timer event never latched
    if (!tev) fail(0x00000020u);

    // 0x21: wake_status never latched timer bit
    if ((status & WAKE_TIMER_BIT) == 0u) fail(0x00000021u);

    // 0x22: wake_reason did not capture timer bit on wake transition
    if ((reason & WAKE_TIMER_BIT) == 0u) fail(0x00000022u);

    // ----------------------
    // Verify W1C clearing works
    // ----------------------
    mmio_write32(T_EVENT, 1u);
    if ((mmio_read32(T_EVENT) & 1u) != 0u) fail(0x00000023u);

    mmio_write32(P_WAKE_STATUS, WAKE_TIMER_BIT);
    if ((mmio_read32(P_WAKE_STATUS) & WAKE_TIMER_BIT) != 0u) fail(0x00000024u);

    // Clear sleep request so we don't immediately re-sleep later
    mmio_write32(P_CTRL, 0u);

    pass();
}
