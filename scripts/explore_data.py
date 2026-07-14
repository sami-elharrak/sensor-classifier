import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/synthetic_sensor_data.csv")
print(df.head())
print(df.describe())
print(df.shape)

normal_readings = df[df["label"] == 0]
faulty_readings = df[df["label"] == 1]

sensors = ["temperature", "vibration", "pressure"]

for sensor in sensors:
    plt.hist(normal_readings[sensor], alpha=0.5, label="Normal")
    plt.hist(faulty_readings[sensor], alpha=0.5, label="Faulty")
    plt.xlabel(sensor.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {sensor.capitalize()} Readings")
    plt.legend()
    plt.savefig(f"outputs/{sensor}_distribution.png")
    plt.show()

