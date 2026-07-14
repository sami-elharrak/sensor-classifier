import numpy as np
import pandas as pd

rng = np.random.default_rng(seed = 42)

N = 1000

temperature = rng.normal(loc = 30,scale = 2,size = N)
vibration = rng.normal(loc = 0.6,scale = 0.03,size = N)
pressure    = rng.normal(loc=101.3, scale=0.8, size=N)

faults = rng.choice(N,size = 100,replace=False)

temperature[faults] = rng.normal(loc = 50,scale = 5,size = len(faults))
vibration[faults] = rng.normal(loc = 1.2,scale = 0.1,size = len(faults))
pressure[faults] = rng.normal(loc = 90,scale = 2,size = len(faults))

labels = np.zeros(N, dtype=int)
labels[faults] = 1

data = pd.DataFrame({
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'label': labels
})
data.to_csv("data/synthetic_sensor_data.csv", index=False)

print(f"Saved {N} readings — {labels.sum()} faults, {N - labels.sum()} normal.")
print(data.head(10))