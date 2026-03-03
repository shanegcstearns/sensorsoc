#include <stdint.h>

// TEST MMIO
#define TEST_STATUS  (*(volatile uint32_t*)0x0300F000u)
#define TEST_CODE    (*(volatile uint32_t*)0x0300F004u)

// TIMER MMIO (base 0x03002000)
#define TIMER_CTRL    (*(volatile uint32_t*)0x03002000u)  // bit0=enable, bit1=periodic
#define TIMER_RELOAD  (*(volatile uint32_t*)0x03002004u)
#define TIMER_COUNT   (*(volatile uint32_t*)0x03002008u)
#define TIMER_EVENT   (*(volatile uint32_t*)0x0300200Cu)  // W1C bit0

// ML MMIO (base 0x03003000)
#define ML_CTRL       (*(volatile uint32_t*)0x03003000u)  // bit0=trigger
#define ML_SCORE      (*(volatile uint32_t*)0x03003004u)  // RO
#define ML_EVENT      (*(volatile uint32_t*)0x03003008u)  // W1C bit0

// GPIO MMIO (base 0x03000000)
#define GPIO_OUT      (*(volatile uint32_t*)0x03000000u)
#define GPIO_TRIG     (1u << 0)

// POWER MMIO (base 0x03001000)
#define PWR_CTRL      (*(volatile uint32_t*)0x03001000u)  // bit0=sleep_req
#define PWR_WAKE_STATUS (*(volatile uint32_t*)0x03001004u)
#define PWR_WAKE_REASON (*(volatile uint32_t*)0x03001008u)

// IRQ controller MMIO (base 0x03005000)
#define IRQC_PENDING  (*(volatile uint32_t*)0x03005000u)  // W1C pending bits
#define IRQC_MASK     (*(volatile uint32_t*)0x03005004u)  // source enable bits
#define IRQC_WAKE_EN  (*(volatile uint32_t*)0x03005008u)  // wake enable bits

int main(void)
{
    // Known reset state
    TIMER_EVENT = 0x1;
    ML_EVENT    = 0x1;
    PWR_WAKE_STATUS = 0xFFFFFFFF;
    IRQC_PENDING = 0xFFFFFFFF;

    // Enable timer (bit0) + ml (bit1) in IRQ routing and wake policy.
    IRQC_MASK    = 0x3;
    IRQC_WAKE_EN = 0x3;

    // Set timer reload to 10000 cycles (short for simulation)
    TIMER_RELOAD = 10000;
    TIMER_COUNT  = 10000;

    // Enable timer periodic (bit0=enable, bit1=periodic)
    TIMER_CTRL = 0x3;

    // Sleep until timer event.
    PWR_CTRL = 0x1;
    while ((TIMER_EVENT & 0x1u) == 0u) { }

    // After wake: clear the timer event
    TIMER_EVENT = 0x1;
    IRQC_PENDING = 0x1;

    // Trigger ML work while awake.
    GPIO_OUT = GPIO_TRIG;
    ML_CTRL  = 0x1;
    GPIO_OUT = 0;

    // Sleep until ML event.
    PWR_CTRL = 0x1;
    while ((ML_EVENT & 0x1u) == 0u) { }

    TEST_CODE = ML_SCORE;
    ML_EVENT = 0x1;
    IRQC_PENDING = 0x2;

    // Clear wake status.
    PWR_WAKE_STATUS = 0xFFFFFFFF;

    // Signal PASS.
    TEST_STATUS = 0xCAFEBABE;

    while (1);
}
