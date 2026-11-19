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
    """Build an encoder-only Transformer for time-series classification."""
    # Create an input for sensor readings across time steps.
    inputs = layers.Input(shape=input_shape)

    # Project each sensor reading into the Transformer feature dimension.
    x = layers.Dense(d_model)(inputs)

    # Add a learnable embedding for each time step to represent position.
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding = layers.Embedding(
        input_dim=input_shape[0],
        output_dim=d_model,
    )(positions)
    x = x + pos_embedding

    # Let each time step attend to information from the other time steps.
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(
        x + layers.Dropout(dropout)(attn_out)
    )

    # Process each time step with a feed-forward network.
    ffn = keras.Sequential(
        [
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ]
    )
    # Add the feed-forward output to the input and normalize it.
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))

    # Pool the time-step features and predict a class for the sequence.
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Compile the model for multi-class classification.
    model = keras.Model(inputs, outputs, name="TimeSeries_Transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
