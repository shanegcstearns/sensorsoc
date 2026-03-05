# sensorsoc
sleep_soc - our main directory including all files needed for an end to end sim
  * taketwo.v
     * nngen-generated ML accelerator for predicting sleep transitions
     * features AXI subordinate and manager to manage reads and writes
     * control register address map:
        * Start Register(W): 16
        * Busy Register(R): 20
     * RAM AXI interface:
        * Global Register Address Offset: 128
        * Output Base Address: 136
        * Input Address: 140
        * Output Address: 144
  * taketwo_wrap.v
     * wrapper used to tie maxi_arid and maxi_awid to low for testing
  * logit_to_confidence.sv
     * converts logit values to a confidence score over time
     * inputs:
        * enable - set high when we want to consider waking. For example, person wants to wake up around 8:00AM, so we set enable high at 7:30AM 
        * logit0 - "bad time to wake"
        * logit1 - "good time to wake"
        * clock, reset (negative polarity)
     * outputs:
        * confidence - outputs a value from 0 - 255 as a confidence score. Higher means wake, lower means keep sleeping
  * axi_interface.sv
     * axi interface to write feature data into ML accelerator
     * inputs: 4 16-bit busses for signed feature values, AXI manager signals
     * outputs: AXI manager signals
 
