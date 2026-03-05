# ML Componets
This directory is used to produce RTL using NNGen. 

* taketwo.py
  * Python version of the ML model, outputs .onnx and .onnx.data files that will be fed into writeverilog.py
* writeverilog.py
  * NNGen script that produces our ML model in verilog (taketwo.v), as well as the weights and biases (taketwo_params.bin)
* processed_sleep_dataset.csv
  * Our primary training dataset
* compiled_sleep_dataset.csv
  * An old dataset used for training early models
  * Formatted different than processed_sleep_dataset.csv
  * Unused Currently
* shortencsv.py
  * Used to downsample and process csv data
* nngen_out
  * Subdirectory for our writeverilog.py to output NNGen files
     * taketwo.v
       * Our primary ML model in RTL
     * taketwo_params.bin
       * Binary containg training outputs (weights and biases), these are loaded into the ML model

