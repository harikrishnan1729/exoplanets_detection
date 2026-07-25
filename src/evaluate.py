import tensorflow as tf
from dataset import load_data
from sklearn.metrics import confusion_matrix, classification_report

# Load the TEST set
_, _, X_test, _, _, y_test = load_data()

# Load trained model
model = tf.keras.models.load_model("models/exoplanet_detector.keras")

# Predict probabilities
y_pred_prob = model.predict(X_test, verbose=0).flatten()

# Convert probabilities to 0 or 1
threshold = 0.1
y_pred = (y_pred_prob >= threshold).astype(int)

thresholds = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

for t in thresholds:
    y_pred = (y_pred_prob >= t).astype(int)

    print(f"\nThreshold = {t}")
    print(confusion_matrix(y_test, y_pred))

print("_____________________________________________-nigga")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Minimum probability:", y_pred_prob.min())
print("Maximum probability:", y_pred_prob.max())
print("First 20 probabilities:")

print(y_pred_prob[:20].flatten())

