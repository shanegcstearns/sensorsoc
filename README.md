# Sleep Sensor SoC
A UCSC Chip Design Capstone project By: Ananya Manduva, Jackson Friday, Nathan Nakamoto, Nithin Duvvuru, Rishi Govindan, and Shane Stearns  
   
Our goal for the project is to create a custom ASIC SoC that intakes Accelerometer and PPG sensor data, processes, and creates features for a lightweight, three layer MLP model which determines whether the user is in an good stage to wake up (NREM/light sleep) or a bad time to wake (REM/deep sleep). This output is given to a small PicoRV32 core on the chip which then sends out a GPIO alarm signal to prompt wake up. The chip is set to sleep until a wachdog timer, set prior, fires off, indicating it's time to start checking sleep states.

# Directories
* sleep_soc
   * Main project directory including all source and test files needed for end-to-end sim
* ML
   * Used to produce RTL using NNGen
* sensor_models
   * Used to model accelerometer and PPG sensors for testing using sample dataset
 
  
  
  
  


# Dataset Used for ML Training and Sensor Modeling:   
Walch, Olivia. "Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography" (version 1.0.0). PhysioNet (2019). RRID:SCR_007345.   https://doi.org/10.13026/hmhs-py35  
