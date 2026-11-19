import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable()
class TemporalAttentionLayer(layers.Layer):
    """Custom Temporal Bahdanau-style Attention Layer implemented from scratch."""

    def __init__(self, attention_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="att_weight",
            shape=(feature_dim, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(self.attention_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="att_context_vector",
            shape=(self.attention_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_logits = tf.tensordot(score, self.u, axes=1)  # (batch, time_steps, 1)
        attention_weights = tf.nn.softmax(attention_logits, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"attention_dim": self.attention_dim})
        return config
