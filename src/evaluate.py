import tensorflow as tf
from dataset import load_data
from sklearn.metrics import confusion_matrix, classification_report

# Load the TEST dataset
_, _, X_test, _, _, y_test = load_data()

# Load the trained model
model = tf.keras.models.load_model("models/exoplanet_detector.keras")

# Predict probabilities
y_pred_prob = model.predict(X_test, verbose=0).flatten()

# Thresholds to test
threshold = 0.5


# Convert probabilities to binary predictions
y_pred = (y_pred_prob >= threshold).astype(int)
print("=" * 60)
print(f"Threshold = {threshold:.2f}")
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

print("\nFirst 20 probabilities:")
print(y_pred_prob[:20])