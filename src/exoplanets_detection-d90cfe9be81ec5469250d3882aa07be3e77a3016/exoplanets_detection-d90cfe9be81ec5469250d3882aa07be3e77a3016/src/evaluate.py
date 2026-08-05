import tensorflow as tf
from dataset import load_data
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)

# Load validation data
# _, X_val, _, y_val = load_data()
X_train, X_val, y_train, y_val, X_test, y_test = load_data()

# Load trained model
model = tf.keras.models.load_model("models/exoplanet_detector.keras")

# Predict probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to 0 or 1
# print(y_pred_prob)
y_pred = (y_pred_prob >=  0.006).astype(int)

print("\nActual Planets:")
for i, (actual, probability) in enumerate(zip(y_test, y_pred_prob)):
    if actual == 1:
        print(
            f"Sample {i}: "
            f"Predicted probability = {float(probability):.6f}"
        )

# for t in np.arange(0.1, 0.9, 0.05):
#     pred = (y_pred_prob >= t).astype(int)
#     f1 = f1_score(y_test, pred)
#     print(f"{t:.2f}  F1={f1:.3f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Min :", y_pred_prob.min())
print("Mean:", y_pred_prob.mean())
print("Max :", y_pred_prob.max())