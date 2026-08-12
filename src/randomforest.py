from dataset import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV

# Load data
x_train, x_val, x_test, y_train, y_val, y_test = load_data()

print("Training samples:", x_train.shape[0])
print("Validation samples:", x_val.shape[0])
print("Test samples:", x_test.shape[0])

# Create Random Forest
model = RandomForestClassifier(
    n_estimators=320,
    random_state=100,
    n_jobs=-1,
    class_weight="balanced",
    min_samples_split=2,
    min_samples_leaf=2,
    max_depth=None
)
model.fit(x_train, y_train)
y_prob = model.predict_proba(x_test)[:, 1]

threshold = 0.05

y_test_pred = (y_prob >= threshold).astype(int)
print("\nTest results:")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, zero_division=0))

y_prob = model.predict_proba(x_test)[:, 1]

# print("Planet probabilities:")
# print(y_prob)

# for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
#     y_prob = model.predict_proba(x_test)[:, 1]

# # Probabilities of actual planets only
#     planet_probabilities = y_prob[y_test == 1]
#     print("Probabilities of actual planets:")
#     print(planet_probabilities)


# Get probabilities on the 20% validation set
# val_probabilities = model.predict_proba(x_val)[:, 1]

# # Try thresholds
# for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:

#     predictions = (val_probabilities >=  0.05).astype(int)
#     print(confusion_matrix(y_val, predictions))

#     print(classification_report(
#         y_val,
#         predictions,
#         zero_division=0
#     ))
#     break
