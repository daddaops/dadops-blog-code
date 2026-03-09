"""Float decomposition: decompose a float into sign, exponent, and mantissa bits."""
import struct

def float_to_bits(value, fmt="float32"):
    """Decompose a float into sign, exponent, and mantissa bits."""
    if fmt == "float32":
        packed = struct.pack('>f', value)
        bits = format(struct.unpack('>I', packed)[0], '032b')
        sign = bits[0]
        exponent = bits[1:9]
        mantissa = bits[9:]
        bias = 127
    elif fmt == "float16":
        # Use numpy for float16 bit manipulation
        import numpy as np
        f16 = np.float16(value)
        raw = f16.view(np.uint16)
        bits = format(raw, '016b')
        sign = bits[0]
        exponent = bits[1:6]
        mantissa = bits[6:]
        bias = 15

    exp_val = int(exponent, 2) - bias
    mant_val = 1.0 + sum(int(b) * 2**(-(i+1))
                         for i, b in enumerate(mantissa))

    print(f"Value:    {value}")
    print(f"Format:   {fmt}")
    print(f"Sign:     {sign} ({'−' if sign == '1' else '+'})")
    print(f"Exponent: {exponent} (2^{exp_val})")
    print(f"Mantissa: {mantissa} ({mant_val:.6f})")
    print(f"Decoded:  {'-' if sign == '1' else ''}2^{exp_val} × {mant_val:.6f}"
          f" = {(-1)**int(sign) * 2**exp_val * mant_val:.6f}")

float_to_bits(13.5)
