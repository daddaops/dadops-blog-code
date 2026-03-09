"""Arithmetic intensity analysis: decode vs prefill on an A100 GPU."""

model_params = 7e9              # 7 billion parameters
bytes_per_param = 2             # FP16 = 2 bytes each
model_size_bytes = model_params * bytes_per_param   # 14 GB

# DECODE: generate one token
# Load all weights, perform ~2 FLOPs per parameter
decode_flops = 2 * model_params           # ~14 GFLOPs
decode_bytes = model_size_bytes           # ~14 GB loaded from memory
decode_intensity = decode_flops / decode_bytes
print(f"Decode:  {decode_intensity:.1f} FLOP/byte")

# PREFILL: process a 1000-token prompt
# Load weights ONCE, but do N times more work
N = 1000
prefill_flops = N * 2 * model_params     # ~14 TFLOPs
prefill_bytes = model_size_bytes          # ~14 GB (same weights, loaded once)
prefill_intensity = prefill_flops / prefill_bytes
print(f"Prefill: {prefill_intensity:.0f} FLOP/byte")

# A100 GPU specs
a100_tflops = 312e12    # 312 TFLOPS peak compute
a100_bandwidth = 2e12   # 2 TB/s memory bandwidth

# Utilization = min(1, intensity * bandwidth / compute)
decode_util = (decode_intensity * a100_bandwidth) / a100_tflops
print(f"\nDecode GPU utilization:  {decode_util:.1%}")
print(f"Prefill GPU utilization: ~100%")
