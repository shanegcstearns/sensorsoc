import os
import nngen as ng

ONNX_PATH = "taketwo_nngen.onnx"
OUT_DIR = "nngen_out"
PROJECT = "taketwo"

os.makedirs(OUT_DIR, exist_ok=True)

# Import ONNX (NNgen 1.3.4 expects filename)
ret = ng.from_onnx(
    ONNX_PATH,
    value_dtypes={"x": ng.int16, "logits": ng.int32},
    default_placeholder_dtype=ng.int16,
    default_variable_dtype=ng.int16,
    default_constant_dtype=ng.int16,
    default_operator_dtype=ng.int16,
    default_scale_dtype=ng.int16,
    default_bias_dtype=ng.int32,
    verbose=True,
)

outputs = ret[0]  # OrderedDict
out_nodes = list(outputs.values())
top = out_nodes[0]  # single output: logits matmul

print("Imported ONNX into NNgen")

# Quantize (if your script got here before, this is working in your setup)
# If you ever get a scale-factor argument error again, comment this out.
try:
    if hasattr(ng, "quantize"):
        # Some NNgen builds require input_scale_factors; yours seems to not error now
        ng.quantize(out_nodes)
    elif hasattr(top, "quantize"):
        top.quantize()
    print("Quantization complete")
except TypeError as e:
    print("Quantize skipped (API requires extra args):", e)

# --- Robust Verilog export: use to_veriloggen() then write file ourselves ---
out_v = os.path.join(OUT_DIR, f"{PROJECT}.v")

m = ng.to_veriloggen(top, PROJECT)       # veriloggen Module
verilog_text = m.to_verilog()   # string of Verilog HDL

with open(out_v, "w") as f:
    f.write(verilog_text)

print("Wrote Verilog to:", out_v)
