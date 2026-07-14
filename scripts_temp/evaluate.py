import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('data/synthetic_sensor_data.csv')
X = df[["temperature", "vibration", "pressure"]]
y = df["label"]

model = joblib.load('models/random_forest_model.pkl')
y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()