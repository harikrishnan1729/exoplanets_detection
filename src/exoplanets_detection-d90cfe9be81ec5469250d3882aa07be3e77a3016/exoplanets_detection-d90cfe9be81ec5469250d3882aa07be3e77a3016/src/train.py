from dataset import load_data
import tensorflow as tf
from model import create_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# x_train, x_val, y_train, y_val = load_data()
X_train, X_val, y_train, y_val, X_test, y_test = load_data()


# x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.2,
#                                                    random_state = 42,
#                                                    stratify = y)

print("training samples: ", X_train.shape[0])
print("validation samples: ", X_test.shape[0])

model = create_model(X_train.shape[1])
tf.keras.metrics.Precision(name="precision")
early_stopping = EarlyStopping(
    monitor="val_loss",
    mode = "max",
    patience=10,
    restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    validation_data = (X_val, y_val),
    epochs = 23,
    batch_size = 30,
    verbose = 1,
    # callbacks=[early_stopping]
    class_weight={
        0: 1.0,   # Non-planet
        1: 40.0   # Planet
    }

)

model.save("models/exoplanet_detector.keras")
print("Model saved successfully!")

loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test)

print(f"Loss      : {loss:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"AUC       : {auc:.4f}")
