import numpy as np
import matplotlib.pyplot as plt


def adc_nu(x, R, B):
  
    x = np.asarray(x, dtype=float)
    L = 2**B
    delta = R / L
    # Round to nearest code and saturate to [0, L-1]
    codes = np.round(x / delta)
    codes = np.clip(codes, 0, L-1).astype(int)
    y = codes * delta
    return y, delta, codes

# --- (b) Test: ramp from -5 V to 15 V, step 1 V
x = np.arange(-5, 16, 1.0)
R = 10.0
B = 3

y, delta, codes = adc_nu(x, R, B)
print(f"Delta = {delta:.3f} V")

# --- (c) Plot input (line) and quantized output (stem) on the same axes
plt.figure()
plt.plot(x, label="Input x (V)")
plt.stem(range(len(y)), y, label="Quantized y (V)")  
plt.title(f"Unipolar ADC Quantization (R={R} V, B={B} bits, Δ={delta:.3f} V)")
plt.xlabel("Sample index")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



