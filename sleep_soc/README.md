# sleep_soc
Our overall simulation design in RTL, with simulation and testbenches in sim/tb. Full end-to-end sim coming

# Using the Makefile
 * ML test - command: `make test-ml`
   * Runs the first 1000 lines of the ml training data through the RTL and outputs it in Confusion Matrix form
 * Globaltimer test - command: `make sim-globaltimer`
  *  
 * Top pipeline sim (sim top) - command: `make sim-top-pipeline`
   * From repo root: `make -C sleep_soc sim-top-pipeline` (or `cd sleep_soc && make sim-top-pipeline`)
   * Runs `sim/tb/top_pipeline_tb.sv` with the full sensor-processing pipeline (`rtl/top.sv`)
   * Outputs a waveform at `sim/waves/top_pipeline_tb.vcd`
* SoC Watchdog Timer Sleep/Wake test - command: `make sim-soc FW_VARIANT=test_sleepwake_periodic`
  * Runs `sim/tb/tb_soc.sv` with chosen firmware loaded, and ran with CPU and MMIO instances (`rtl/soc_top.v`)
  * Outputs a waveform at `sleep_soc/soc.vcd`
* Watchdog-like Timer test - command: `make test-timer`
  * Runs cocotb unit test for timer (`test_timer_mmio.py`)
* Reset Controller test - command: `make sim-reset-ctrl`
  * 
* CPU to ML test - command: `make test-cpu-ml`
  * Cocotb test (`cpu_to_ml_tb.py`) that verifies Axi-Lite bridge mmio module, connection to ml wrapper, and ability to read and write
* X test - command: ``
  *
* X test - command: ``
  *
* X test - command: ``
  *
* X test - command: ``
  * 
