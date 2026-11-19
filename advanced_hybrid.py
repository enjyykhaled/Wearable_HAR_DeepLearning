import keras
from keras import layers
from models.custom_layers import TemporalAttentionLayer


def build_cnn_bilstm_attention(
    input_shape=(128, 9),
    num_classes=6,
    conv_filters=64,
    kernel_size=5,
    lstm_units=64,
    attention_dim=64,
    dropout=0.2,
    lr=1e-3,
):
    """Advanced Hybrid Model: 1D-CNN + Bidirectional LSTM + Custom Temporal Attention."""
    inputs = layers.Input(shape=input_shape)

    # Local high-frequency spatial feature extraction via 1D-CNN
    x = layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    # Sequential temporal modeling via Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout))(x)

    # Custom hand-written Temporal Attention layer
    x = TemporalAttentionLayer(attention_dim=attention_dim)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="Advanced_CNN_BiLSTM_Attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model