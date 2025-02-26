import tensorflow as tf

class StaggeredConv1D(tf.keras.layers.Layer):
    """Layers which use two CNNs to scan alternately."""
    
    def __init__(self, filters=32, kernel_size=3, **kwargs):
        """Initialize underlying layers."""
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1d_a = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', **kwargs)
        self.conv1d_b = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', **kwargs)

    def call(self, inputs, training=None, mask=None):
        """Forward pass."""

        if mask is not None:
            inputs = tf.identity(inputs)

        conv_a = self.conv1d_a(inputs)[:, ::2, :]
        conv_b = self.conv1d_b(inputs)[:, 1::2, :]

        min_length = tf.minimum(tf.shape(conv_a)[1], tf.shape(conv_b)[1])
        conv_a = tf.cond(tf.shape(conv_a)[1] > min_length, lambda: conv_a[:, :min_length, :], lambda: conv_a)
        conv_b = tf.cond(tf.shape(conv_b)[1] > min_length, lambda: conv_b[:, :min_length, :], lambda: conv_b)

        staggered = tf.reshape(tf.stack([conv_a, conv_b], axis=-2), 
                            (tf.shape(inputs)[0], tf.shape(inputs)[1], self.filters))

        return tf.cast(staggered, tf.float32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)  

    def compute_mask(self, inputs, mask=None):
        return None  

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config
