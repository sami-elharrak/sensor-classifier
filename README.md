# Sensor Fault Classifier

A machine learning pipeline that detects faults in sensor data using a Random Forest classifier.

## Overview
This project simulates an industrial sensor system with temperature, vibration, and pressure readings. 
Synthetic data was generated with realistic normal readings and injected faults to replicate real-world 
sensor failures. The trained model classifies each reading as either normal or faulty.

## Tech Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
1. Clone the repository
2. Install dependencies: `pip install numpy pandas scikit-learn matplotlib`
3. Generate data: `python scripts/fake_sensor_data.py`
4. Explore data: `python scripts/explore_data.py`
5. Train model: `python scripts/train.py`
6. Evaluate model: `python scripts/evaluate.py`

## Results
The Random Forest classifier (100 trees, 80/20 train/test split) achieved 100% accuracy on the 
synthetic dataset. The confusion matrix confirms zero misclassifications across 200 test samples.
Realistic accuracy on real-world sensor data would be lower due to noise and class overlap.
