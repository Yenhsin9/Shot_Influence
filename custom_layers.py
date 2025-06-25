import tensorflow as tf
from tensorflow.keras.initializers import HeNormal
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import csv
class StaggeredConv1D(tf.keras.layers.Layer):
    """Layers which use two CNNs to scan alternately and provide intermediate outputs for visualization."""
    
    def __init__(self, filters=32, kernel_size=3, **kwargs):
        """Initialize underlying layers."""
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.supports_masking = True
        self.conv1d_a = tf.keras.layers.Conv1D(
            self.filters, 
            self.kernel_size, 
            padding='same', 
            **kwargs
        )

        self.conv1d_b = tf.keras.layers.Conv1D(
            self.filters, 
            self.kernel_size, 
            padding='same', 
            **kwargs
        )
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass with optional intermediate output for visualization."""
        if mask is not None:
            mask = tf.cast(mask, tf.float32)  
            mask = tf.expand_dims(mask, axis=-1)  
            inputs = inputs * mask  

        inputs_a = inputs[:, ::2, :] 
        inputs_b = inputs[:, 1::2, :]  
        
        conv_a = self.conv1d_a(inputs_a) 
        conv_b = self.conv1d_b(inputs_b) 

        max_length = tf.maximum(tf.shape(conv_a)[1], tf.shape(conv_b)[1])

        padded_conv_a = tf.pad(conv_a, [[0, 0], [0, max_length - tf.shape(conv_a)[1]], [0, 0]])
        padded_conv_b = tf.pad(conv_b, [[0, 0], [0, max_length - tf.shape(conv_b)[1]], [0, 0]])

        staggered = tf.reshape(tf.stack([padded_conv_a, padded_conv_b], axis=-2), 
                            (tf.shape(inputs)[0], max_length * 2, self.filters))

        
        return tf.cast(staggered, tf.float32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)  

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config
    

# def save_to_csvb(self, x):
#         """Save tensor to CSV file."""
#         filename="inputInCNNConB.csv"
#         try:
#             np_array = x.numpy()  # 轉為 numpy array
#             print(f"Saving to {filename}, shape:", np_array.shape)
#             with open(filename, mode='w', newline='') as file:
#                 writer = csv.writer(file)
#                 for sample in np_array:
#                     for row in sample:
#                         writer.writerow(row)
#         except Exception as e:
#             print(f"Error saving to {filename}:", e)
#         return x

# tmp = staggered
#         tmp = tf.keras.layers.Lambda(
#             lambda x: tf.py_function(self.save_to_csvfinal, [x], tf.float32)
#         )(tmp)