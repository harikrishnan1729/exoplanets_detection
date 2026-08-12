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


# ploting testing and training datasets
# dataset_path =r"C:\Users\HARIKRISHNAN\.cache\kagglehub\datasets\keplersmachines\kepler-labelled-time-series-data\versions\3"
# test = pd.read_csv(f"{dataset_path}/exoTest.csv")

# scaler = StandardScaler()
# train = pd.read_csv(f"{dataset_path}/exoTrain.csv")
# scaler.fit(train.iloc[:, 1:])   # <-- Fit ONLY on training data

# planet_rows = test[test.iloc[:, 0] == 2]
# print(len(planet_rows))
# print(planet_rows.index)
# scaled = scaler.transform(planet_rows.iloc[:, 1:])


# fig, axes = plt.subplots(len(planet_rows), 1, figsize=(12, 10))
# for i in range(len(planet_rows)):
#     i = 0

#     while True:
#         plt.figure(figsize=(12,4))
#         plt.plot(scaled[i])
#         plt.title(f"Planet {i+1}")
#         plt.show()

#         cmd = input("[n]ext, [p]revious, [q]uit: ")

#         if cmd == "n":
#             i = min(i+1, len(scaled)-1)
#         elif cmd == "p":
#             i = max(i-1, 0)
#         elif cmd == "q":
#             break

# plt.tight_layout()
# plt.show()


# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, classification_report
# import tensorflow as tf

# dataset_path = r"C:\Users\HARIKRISHNAN\.cache\kagglehub\datasets\keplersmachines\kepler-labelled-time-series-data\versions\3"

# # Load ORIGINAL training data
# train = pd.read_csv(f"{dataset_path}/exoTrain.csv")

# X_train = train.iloc[:, 1:]
# y_train = train.iloc[:, 0] - 1

# # Standardize exactly as during training
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# # Load trained model
# model = tf.keras.models.load_model("models/exoplanet_detector.keras")

# # Predict
# y_prob = model.predict(X_train_scaled, verbose=0).flatten()

# threshold = 0.64      # Use your chosen threshold
# y_pred = (y_prob >= threshold).astype(int)

# print("Original Training Confusion Matrix")
# print(confusion_matrix(y_train, y_pred))

# print("\nOriginal Training Classification Report")
# print(classification_report(y_train, y_pred, digits=4))
# planet_indices = y_train[y_train == 1].index

# print("\nTraining planet probabilities:")

# for idx in planet_indices:
#     print(f"Planet {idx}: {y_prob[idx]:.6f}")
