import keras
from keras import layers
from tensorflow import keras
from tensorflow.keras import layers


def build_bilstm(input_shape=(128, 9), num_classes=6, units=64, learning_rate=1e-3):
    """Build and compile a single-layer Bidirectional LSTM classifier."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(units, dropout=0.2))(inputs)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="Single_Layer_BiLSTM")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
