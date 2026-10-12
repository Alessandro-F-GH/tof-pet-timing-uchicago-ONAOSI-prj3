import numpy as np

mask = np.load("waveform_analysis/results/FBK/benchmark_48V_R1_R2/timing/wide/onishi_cnn/model/sample_mask.npy")

print("Original samples:", len(mask))
print("Retained samples:", mask.sum())
print("Removed samples:", len(mask) - mask.sum())