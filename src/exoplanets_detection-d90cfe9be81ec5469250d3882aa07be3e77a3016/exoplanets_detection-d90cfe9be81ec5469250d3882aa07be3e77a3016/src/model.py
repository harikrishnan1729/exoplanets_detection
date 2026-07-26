import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

def create_model(input_shape):
    model =  tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),

        tf.keras.layers.Dense(512, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(256, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Dense(1, activation="sigmoid")
        
    ])
    
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = [
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    return model