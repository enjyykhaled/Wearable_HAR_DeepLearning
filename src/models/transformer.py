import tensorflow as tf
import keras
from keras import layers


def build_transformer(
    input_shape=(128, 9),
    num_classes=6,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    dropout=0.2,
    lr=1e-3,
):
    """Encoder-only Time-Series Transformer with learnable positional embeddings."""
    inputs = layers.Input(shape=input_shape)

    # Project 9 sensor channels to d_model dimension
    x = layers.Dense(d_model)(inputs)

    # Learnable Positional Encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(positions)
    x = x + pos_embedding

    # Multi-Head Self-Attention Block
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(dropout)(attn_out))

    # Position-wise Feed-Forward Network
    ffn = keras.Sequential(
        [
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ]
    )
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="TimeSeries_Transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model