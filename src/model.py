import tensorflow as tf

def create_model(input_shape):
    model =  tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        tf.keras.layers.Conv1D(
            filters = 32,
            kernel_size = 5,
            activation = "relu"
        ),
        tf.keras.layers.Conv1D(
            filters = 64,
            kernel_size = 5,
            activation = "relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
        
    ])
    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            gamma = 2.0,
            apply_class_balancing = True
        ),
        metrics = ["accuracy"]
    )
    return model
