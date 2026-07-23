import tensorflow as tf
from dataset import load_data

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# Load validation data
# _, X_val, _, y_val = load_data()
_, X_test, _, y_test = load_data()

# Load trained model
model = tf.keras.models.load_model("models/exoplanet_detector.keras")

# Predict probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to 0 or 1
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

