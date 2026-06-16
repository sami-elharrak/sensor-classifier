import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

N = 1000
temperature = rng.normal(loc=25.0, scale=2.0, size=N)
vibration   = rng.normal(loc=0.5,  scale=0.05, size=N)
pressure    = rng.normal(loc=101.3, scale=0.8, size=N)

fault_indices = rng.choice(N, size=100, replace=False)

temperature[fault_indices] += rng.normal(loc=15.0, scale=2.0, size=100)
vibration[fault_indices]   += rng.normal(loc=1.5,  scale=0.2, size=100)
pressure[fault_indices]    -= rng.normal(loc=8.0,  scale=1.0, size=100)

labels = np.zeros(N, dtype=int)
labels[fault_indices] = 1

df = pd.DataFrame({
    "temperature": temperature,
    "vibration":   vibration,
    "pressure":    pressure,
    "label":       labels
})

df.to_csv("data/sensor_readings.csv", index=False)

print(f"Saved {N} readings — {labels.sum()} faults, {N - labels.sum()} normal.")
print(df.head(10))