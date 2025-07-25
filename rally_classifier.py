import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout, Embedding,LSTM
from keras_transformer import get_encoders
from keras_transformer.gelu import gelu
from keras_ordered_neurons import ONLSTM

def get_positional_encoding(seq_len, d_model):
    """Generate sine-cosine positional encodings for even d_model."""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # Shape: (d_model/2,)
    pos_enc = np.zeros((seq_len, d_model))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pos_enc, dtype=tf.float32)

def transformer(
    shot_sequence_shape: Tuple[int, int],
    embed_types_size: int,
    embed_area_size: int,
    rally_info_shape: int = None,
    transformer_kwargs: Dict[str, Any] = {'encoder_num': 2, 'head_num': 1, 'hidden_dim': 32, 'feed_forward_activation': gelu},
    dense_kwargs: Dict[str, Any] = {}
) -> tf.keras.Model:

    # ===== Inputs =====
    input_shots = tf.keras.Input(shape=shot_sequence_shape, name='shots_input')  # (batch, 72, 2)
    input_shot_type = tf.keras.Input(shape=(shot_sequence_shape[0],), name='shot_types_input')  # (batch, 72)
    input_player = tf.keras.Input(shape=(shot_sequence_shape[0],), name='player_input')  # (batch, 72)
    input_time = tf.keras.Input(shape=(shot_sequence_shape[0],), name='time_proportion_input')  # (batch, 72)
    input_hit_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='hit_area_input')  # (batch, 72)
    input_player_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='player_area_input')  # (batch, 72)
    input_opponent_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='opponent_area_input')  # (batch, 72)
    input_mask = tf.keras.Input(shape=(shot_sequence_shape[0],), name='mask_input')  # (batch, 72)
    
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=(rally_info_shape,), name='rally_input')  # (batch, 2)
    else:
        input_rally = None

    # ===== Embedding Layers =====
    embed_type_layer = tf.keras.layers.Embedding(embed_types_size, 8, mask_zero=True, embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=42))
    embed_area_layer = tf.keras.layers.Embedding(embed_area_size, 8, mask_zero=True, embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=42))

    embedded_type = embed_type_layer(input_shot_type)
    embedded_player = embed_type_layer(input_player)
    embedded_hit_area = embed_area_layer(input_hit_area)
    embedded_player_area = embed_area_layer(input_player_area)
    embedded_opponent_area = embed_area_layer(input_opponent_area)

    time_expanded = tf.keras.layers.Reshape((shot_sequence_shape[0], 1))(input_time)

    # ===== Concatenate all shot-level inputs =====
    shot_concat = tf.keras.layers.Concatenate(axis=-1)([
        input_shots,
        embedded_type,
        embedded_player,
        time_expanded,
        embedded_hit_area,
        embedded_player_area,
        embedded_opponent_area
    ])
    print("shot_concat shape:", shot_concat.shape)  # (None, 72, 43)

    # ===== Positional Encoding =====
    x = tf.keras.layers.Masking()(shot_concat)

    pos_enc = get_positional_encoding(seq_len=shot_sequence_shape[0], d_model=44)  # Shape: (72, 44)
    pos_enc = tf.expand_dims(pos_enc, axis=0)  # Shape: (1, 72, 44)
    x = tf.keras.layers.Dense(44)(x)  # Project shot_concat to (None, 72, 44)
    x = tf.keras.layers.Add()([x, pos_enc])  # Shape: (None, 72, 44)
    print("After Positional Encoding shape:", x.shape)
    x = tf.keras.layers.Dense(transformer_kwargs['hidden_dim'])(x)  # Shape: (None, 72, 32)

    # ===== Transformer Encoder =====
    x = get_encoders(input_layer=x, **transformer_kwargs)  # Shape: (None, 72, 32)
    print("After Transformer shape:", x.shape)

    # ===== Apply Mask =====
    mask_expanded = tf.keras.layers.Reshape((shot_sequence_shape[0], 1))(input_mask)
    x = tf.keras.layers.Multiply()([x, mask_expanded])

    # ===== Global Pooling =====
    rally_represent = tf.keras.layers.GlobalMaxPooling1D()(x)

    # ===== Concatenate Rally Info =====
    if input_rally is not None:
        rally_represent = tf.keras.layers.Concatenate()([rally_represent, input_rally])

    # ===== Output Layer =====
    output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42), **dense_kwargs)(rally_represent)

    # ===== Build Model =====
    inputs = [
        input_shots,
        input_shot_type,
        input_player,
        input_time,
        input_hit_area,
        input_player_area,
        input_opponent_area,
        input_rally,
        input_mask
    ]
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def lstm(
    shot_sequence_shape: Tuple[int, int],
    embed_types_size: int,
    embed_area_size: int,
    rally_info_shape: int = None,
    lstm_kwargs: Dict[str, Any] = {'units': 32},  
    dense_kwargs: Dict[str, Any] = {}
) -> tf.keras.Model:
    """Create an LSTM rally classifier model aligned with the transformer model."""

    # ===== Inputs =====
    input_shots = tf.keras.Input(shape=shot_sequence_shape, name='shots_input')  # (batch, 72, 2)
    input_shot_type = tf.keras.Input(shape=(shot_sequence_shape[0],), name='shot_types_input')  # (batch, 72)
    input_player = tf.keras.Input(shape=(shot_sequence_shape[0],), name='player_input')  # (batch, 72)
    input_time = tf.keras.Input(shape=(shot_sequence_shape[0],), name='time_proportion_input')  # (batch, 72)
    input_hit_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='hit_area_input')  # (batch, 72)
    input_player_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='player_area_input')  # (batch, 72)
    input_opponent_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='opponent_area_input')  # (batch, 72)
    input_mask = tf.keras.Input(shape=(shot_sequence_shape[0],), name='mask_input')  # (batch, 72)
    
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=(rally_info_shape,), name='rally_input')  # (batch, rally_info_shape)
    else:
        input_rally = None

    # ===== Embedding Layers =====
    embed_type_layer = Embedding(embed_types_size, 8, mask_zero=True, embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=42))
    embed_area_layer = Embedding(embed_area_size, 8, mask_zero=True, embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=42))

    embedded_type = embed_type_layer(input_shot_type)  # (batch, 72, 8)
    embedded_player = embed_type_layer(input_player)  # (batch, 72, 8)
    embedded_hit_area = embed_area_layer(input_hit_area)  # (batch, 72, 8)
    embedded_player_area = embed_area_layer(input_player_area)  # (batch, 72, 8)
    embedded_opponent_area = embed_area_layer(input_opponent_area)  # (batch, 72, 8)

    time_expanded = tf.keras.layers.Reshape((shot_sequence_shape[0], 1))(input_time)

    # ===== Concatenate all shot-level inputs =====
    shot_concat = tf.keras.layers.Concatenate(axis=-1)([
        input_shots,
        embedded_type,
        embedded_player,
        time_expanded,
        embedded_hit_area,
        embedded_player_area,
        embedded_opponent_area
    ])  # (batch, 72, 43)
    print("shot_concat shape:", shot_concat.shape)

    # ===== Positional Encoding =====
    x = tf.keras.layers.Masking()(shot_concat)  # Apply masking for padded sequences
    pos_enc = get_positional_encoding(seq_len=shot_sequence_shape[0], d_model=44)  # Shape: (72, 44)
    pos_enc = tf.expand_dims(pos_enc, axis=0)  # Shape: (1, 72, 44)
    x = Dense(44, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42))(x)  # Project to (batch, 72, 44)
    x = tf.keras.layers.Add()([x, pos_enc])  # Add positional encoding, (batch, 72, 44)
    print("After Positional Encoding shape:", x.shape)

    # ===== LSTM Layer =====
    x = Dense(lstm_kwargs['units'], kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42))(x)  # Project to LSTM units
    x = LSTM(units=lstm_kwargs['units'], return_sequences=False, trainable=True)(x)  # (batch, units)
    print("After LSTM shape:", x.shape)

    # ===== Concatenate Rally Info =====
    if input_rally is not None:
        x = tf.keras.layers.Concatenate()([x, input_rally])  # (batch, units + rally_info_shape)
        print("After Rally Concat shape:", x.shape)

    # ===== Output Layer =====
    output = Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42), **dense_kwargs)(x)  # (batch, 1)

    # ===== Build Model =====
    inputs = [
        input_shots,
        input_shot_type,
        input_player,
        input_time,
        input_hit_area,
        input_player_area,
        input_opponent_area,
        input_rally,
        input_mask
    ]
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model