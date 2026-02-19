import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import pandas as pd
import numpy as np

# Match your preprocessing
SETUP_MAX_SAMPLES = int(os.getenv("SETUP_MAX_SAMPLES", "600"))  # 10*60*1
EPS = 1e-8
SEED = 16

def to_signed(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    val &= mask
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val

async def reset(dut, cycles=5):
    dut.rstn.value = 0
    dut.in_valid.value = 0
    dut.out_ready.value = 0
    dut.hr_q8.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rstn.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)

async def send_sample(dut, hr_q8: int, hr_rmssd_q8: int, movement: int, cosine: int):
    dut.hr_q8.value = int(hr_q8)
    dut.hr_rmssd_q8.value = int(hr_rmssd_q8)
    dut.movement_q8.value = int(movement)
    dut.cosine_q8.value = int(cosine)
    dut.in_valid.value = 1
    while True:
        await RisingEdge(dut.clk)
        if int(dut.in_ready.value) == 1:
            break
    dut.in_valid.value = 0

async def recv_sample(dut):
    dut.out_ready.value = 1
    while True:
        await RisingEdge(dut.clk)
        if int(dut.out_valid.value) == 1:
            pred = int(dut.state_out.value)
            l0 = to_signed(int(dut.logit0_q16.value), 32)
            l1 = to_signed(int(dut.logit1_q16.value), 32)
            await RisingEdge(dut.clk)
            dut.out_ready.value = 0
            return pred, l0, l1

def conf_counts(y_true, y_pred):
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    return tn, fp, fn, tp

def build_samples_from_compiled_csv(path: str, scale_hr, scale_rmssd, scale_movement, scale_cosine):
    """
    Reads the preprocessed CSV 
    Produces list of tuples: (hr_q8, label)
    """
    df = pd.read_csv(path)

    # convert to Q8
    df["hr_q8"] = np.round(df["delta_hr"] * scale_hr).astype(int)
    df["hr_q8"] = df["hr_q8"].clip(-32768, 32767)

    df["hr_rmssd_q8"] = np.round(df["hr_rmssd"] * scale_rmssd).astype(int)
    df["hr_rmssd_q8"] = df["hr_rmssd_q8"].clip(-32768, 32767)
    
    df["movement_q8"] = np.round(df["movement"] * scale_movement).astype(int)
    df["movement_q8"] = df["movement_q8"].clip(-32768, 32767)
    
    df["cosine_q8"] = np.round(df["cosine"] * scale_cosine).astype(int)
    df["cosine_q8"] = df["cosine_q8"].clip(-32768, 32767)

    # optionally cap to keep simulation fast
    max_samples = int(os.getenv("MAX_SAMPLES", "5000"))
    if max_samples > 0 and len(df) > max_samples:
        df = df.iloc[:max_samples]

    samples = list(zip(df["hr_q8"].astype(int).tolist(), df["hr_rmssd_q8"].astype(int).tolist(), df["movement_q8"].astype(int).tolist(), df["cosine_q8"].astype(int).tolist(), df["label"].astype(int).tolist()))
    return samples

@cocotb.test()
async def reset_test(dut): #simple reset test behavior test
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    assert dut.state_out.value == 0, f"Expected state_out=0 after reset, got {dut.state_out.value}"
    
    
# @cocotb.test()
# async def sleep_test(dut): #long pause between tests to check for any state retention issues
    
# @cocotb.test()
# async def overload_test(dut): #send samples as fast as possible to check for any timing issues or bottlenecks

@cocotb.test()
async def test_accuracy_compiled_csv(dut): #main test we should be running, simulating python test
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    dataset = os.getenv("DATASET_CSV", "processed_sleep_dataset.csv")
    if not os.path.exists(dataset):
        raise RuntimeError(f"DATASET_CSV not found: {dataset}")

    samples = build_samples_from_compiled_csv(dataset, dut.SCALE_Q0.value, dut.SCALE_Q1.value, dut.SCALE_Q2.value, dut.SCALE_Q3.value)
    dut._log.info(f"Prepared {len(samples)} samples from {dataset} (SPLIT={os.getenv('SPLIT','test')})")

    y_true, y_pred = [], []

    for k, (hr_q8, hr_rmssd_q8, movement_q8, cosine_q8, label) in enumerate(samples):
        await send_sample(dut, hr_q8, hr_rmssd_q8, movement_q8, cosine_q8)
        pred, l0, l1 = await recv_sample(dut)

        y_true.append(label)
        y_pred.append(pred)

        if k < 5:
            dut._log.info(f"ex{k}: hr_q8={hr_q8} hr_rmssd_q8={hr_rmssd_q8} movement_q8={movement_q8} cosine_q8={cosine_q8} label={label} pred={pred} logit0={l0} logit1={l1}")

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    acc = correct / len(y_true) if y_true else 0.0
    tn, fp, fn, tp = conf_counts(y_true, y_pred)

    dut._log.info(f"Accuracy: {acc*100:.2f}% ({correct}/{len(y_true)})")
    dut._log.info(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    min_acc = float(os.getenv("MIN_ACC", "0.0"))
    assert acc >= min_acc, f"acc={acc:.3f} < MIN_ACC={min_acc:.3f}"
