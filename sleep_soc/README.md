# sleep_soc
Our overall simulation design in RTL, with simulation and testbenches in sim/tb. Full end-to-end sim coming

# Using the Makefile
 * ML test - command: `make test-ml`
   * Runs the first 1000 lines of the ml training data through the RTL and outputs it in Confusion Matrix form
 * Globaltimer test - command: `make test-timer`
  *  
 * Top pipeline sim (sim top) - command: `make sim-top-pipeline`
   * From repo root: `make -C sleep_soc sim-top-pipeline` (or `cd sleep_soc && make sim-top-pipeline`)
   * Runs `sim/tb/top_pipeline_tb.sv` with the full sensor-processing pipeline (`rtl/top.sv`)
   * Outputs a waveform at `sim/waves/top_pipeline_tb.vcd`
* X test - command: ``
  * 
* X test - command: ``
  *
* X test - command: ``
  *
* X test - command: ``
  *
* X test - command: ``
  *
* X test - command: ``
  *
* X test - command: ``
  *
* X test - command: ``
  * 
