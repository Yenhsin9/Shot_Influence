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
        # 假設 inputs 的形狀為 (batch_size, timesteps, 48)
        # 玩家標籤位於索引 45 (第 46 個特徵通道，1 for A, 2 for B, 0 for padding)
        features = tf.concat([inputs[:, :, :45], inputs[:, :, 46:]], axis=-1)  # 移除標籤通道，形狀 (batch_size, timesteps, 47)
        labels = inputs[:, :, 45]  # 提取玩家標籤，形狀 (batch_size, timesteps)

        # 創建 A 和 B 的遮罩
        mask_a = tf.cast(tf.equal(labels, 1), tf.float32)  # Player A 的遮罩
        mask_b = tf.cast(tf.equal(labels, 2), tf.float32)  # Player B 的遮罩
        mask_a = tf.expand_dims(mask_a, axis=-1)  # (batch_size, timesteps, 1)
        mask_b = tf.expand_dims(mask_b, axis=-1)

        # 對 A 和 B 分別應用卷積
        inputs_a = features * mask_a  # 僅保留 A 的時間步
        inputs_b = features * mask_b  # 僅保留 B 的時間步
        conv_a = self.conv1d_a(inputs_a)  # 對 A 應用 conv1d_a
        conv_b = self.conv1d_b(inputs_b)  # 對 B 應用 conv1d_b

        # 合併結果，保持原始順序
        staggered = conv_a + conv_b  # 因為 mask_a 和 mask_b 互斥，直接相加

        # 應用 padding 遮罩（標籤為 0 的時間步）
        padding_mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        padding_mask = tf.expand_dims(padding_mask, axis=-1)
        staggered = staggered * padding_mask  # 將 padding 時間步設為 0

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