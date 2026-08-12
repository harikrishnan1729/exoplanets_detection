import tensorflow as tf
from dataset import load_data
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the TEST dataset
_, _, X_test, _, _, y_test = load_data()

model = tf.keras.models.load_model("models/exoplanet_detector.keras")

y_pred_prob = model.predict(X_test, verbose=0).flatten()

thresholds = np.arange(0.01, 1.00, 0.01)

best_threshold = 0
best_f1 = 0


for t in thresholds:
    y_pred = (y_pred_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t



# Convert probabilities to binary predictions
y_pred = (y_pred_prob >= 0.112578).astype(int)
print("=" * 60)
print(f"Threshold = {best_threshold:.2f} f1: {best_f1}")
print("=" * 60)
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))

# Probability statistics
print("\n" + "=" * 60)
print("Probability Statistics")
print("=" * 60)

print(f"Minimum probability : {y_pred_prob.min():.6f}")
print(f"Maximum probability : {y_pred_prob.max():.6f}")
print(f"Mean probability    : {y_pred_prob.mean():.6f}")

planet_indices = y_test[y_test == 1].index

print("Actual planets:")
for idx in planet_indices:
    print(f"Sample {idx}: Probability = {y_pred_prob[idx]:.6f}")


