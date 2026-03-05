# Sleep Sensor SoC
A UCSC Chip Design Capstone project By: Ananya Manduva, Jackson Friday, Nathan Nakamoto, Nithin Duvvuru, Rishi Govindan, and Shane Stearns  
   
Our goal for the project is to create a custom ASIC SoC that intakes Accelerometer and PPG sensor data, processes, and creates features for a lightweight, three layer MLP model which determines whether the user is in an good stage to wake up (NREM/light sleep) or a bad time to wake (REM/deep sleep). This output is given to a small PicoRV32 core on the chip which then sends out a GPIO alarm signal to prompt wake up. The chip is set to sleep until a wachdog timer, set prior, fires off, indicating it's time to start checking sleep states.

# Directories and Files
* sleep_soc - our main project directory including all files needed for an end to end sim
* ML - Directory for MLP model training and verilog synthesis
* sensor_models - simulation sources for sensors for testing
