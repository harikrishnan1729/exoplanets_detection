from dataset import load_data
from model import create_model
from sklearn.model_selection import train_test_split

# x_train, x_val, y_train, y_val = load_data()
x_train, x_test, y_train, y_test = load_data()

# x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.2,
#                                                    random_state = 42,
#                                                    stratify = y)

print("training samples: ", x_train.shape[0])
print("validation samples: ", x_test.shape[0])

model = create_model(x_train.shape[1])

history = model.fit(
    x_train,
    y_train,
    # validation_data = (x_val, y_val),
    epochs = 20,
    batch_size = 32,
    verbose = 1,
    class_weight = {
        0:1,
        1:145
    }
)

model.save("models/exoplanet_detector.keras")
print("Model saved successfully!")