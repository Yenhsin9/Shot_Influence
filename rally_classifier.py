"""Rally classifier models."""
from typing import Tuple, Dict, Any
import tensorflow as tf
import keras_self_attention
from prosenet.model import ProSeNet
from keras_ordered_neurons import ONLSTM
from keras_pos_embd import TrigPosEmbedding
from keras_transformer import get_encoders
from keras_transformer.gelu import gelu
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout, Embedding
from custom_layers import StaggeredConv1D
import tensorflow.keras.activations as activations
import numpy as np
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2
import csv
def preprocess_inputs(shot_sequence_shape, rally_info_shape,
                      embed_types_size=None, embed_area_size=None, embed_player_size=None):
    """Preprocess input encoding: Input + Embedding + Concatenation + Masking"""
    # ✅ Input Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape, name='Shots_input')  # (None, seq_len, 2)
    input_shot_types = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Shot_types_input')
    input_time_proportion = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Time_proportion_input')
    input_hit_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Hit_area_input')
    input_player_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Player_area_input')
    input_opponent_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Opponent_area_input')
    input_player_id = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Player_input')
    input_rally = tf.keras.Input(shape=(rally_info_shape,), name='Rally_input')
    input_masks = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Mask_input')

    # ✅ Player Embedding
    if embed_player_size is not None:
        player_embedding = tf.keras.layers.Embedding(input_dim=embed_player_size, output_dim=15, mask_zero=True, name='Player_embedding',embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1))
        embedded_player = player_embedding(input_player_id)  # (None, 66, 15)

    # ✅ Location Embedding (Hit Area, Player Area, Opponent Area)
    if embed_area_size is not None:
        area_embedding = tf.keras.layers.Embedding(input_dim=embed_area_size, output_dim=10, mask_zero=True, name='Area_embedding',embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1))
        embedded_hit_area = area_embedding(input_hit_area)
        embedded_player_area = area_embedding(input_player_area)
        embedded_opponent_area = area_embedding(input_opponent_area)
    else:
        embedded_hit_area, embedded_player_area, embedded_opponent_area = input_hit_area, input_player_area, input_opponent_area

    # ✅ Shot Type Embedding with Time Influence
    if embed_types_size is not None:
        shot_embedding = tf.keras.layers.Embedding(input_dim=embed_types_size, output_dim=15, mask_zero=True, name='Shot_types_embedding',embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1))
        embedded_shot_types = shot_embedding(input_shot_types)

        mu_n = Dense(15, activation='linear', 
             kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1), 
             name='Mu_latent')(embedded_shot_types)
        theta_n = Dense(15, activation='linear', 
                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1), 
                        name='Theta_latent')(embedded_shot_types)

        # ✅ Time Proportion Enhancement
        tiled_time_proportion = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1), 
            output_shape=(62, 1)
        )(input_time_proportion)
        tiled_time_proportion = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 15]))(tiled_time_proportion)

        # μn * τn
        time_mu_proportion = tf.keras.layers.Multiply(name='Time_proportion_multiply')([mu_n, tiled_time_proportion])
        # θn + μn * τn
        temporal_score = tf.keras.layers.Add(name='Time_proportion_add')([theta_n, time_mu_proportion])
        # δn = sigmoid(θn + μn * τn)
        temporal_score = tf.keras.layers.Activation('sigmoid', name='Time_activation')(temporal_score)

        enhanced_shot_features = tf.keras.layers.Multiply(name='Shots_time_multiply')([temporal_score, embedded_shot_types])
    else:
        enhanced_shot_features = input_shot_types
 
    # ✅ **Concatenate Player Embedding, Location Embedding, and Enhanced Shot Features (Shot Encoder Output)**
    shot_encoder_output = tf.keras.layers.Concatenate(name='Shot_encoder_output')([
        embedded_hit_area, embedded_player_area, embedded_opponent_area, enhanced_shot_features, embedded_player,input_shots
    ])

    return [input_shots,input_shot_types,input_player_id,input_time_proportion, input_hit_area, input_player_area, input_opponent_area, 
              input_rally,input_masks], shot_encoder_output


# Proposed modal: CNN + Position + Mutihead Attention
def proposed_model(shot_sequence_shape: Tuple[int, int], 
                   embed_types_size: int = None,
                   embed_area_size: int = None,
                   embed_player_size: int = None, 
                   rally_info_shape: int = None,
                   cnn_kwargs: Dict[str, Any] = {'filters': 32, 'kernel_size': 3},
                   transformer_kwargs: Dict[str, Any] = {},
                   dense_kwargs: Dict[str, Any] = {}) -> tf.keras.Model:

    # ✅ Get Processed Inputs and Shot Encoder Output
    inputs, shot_encoder_output = preprocess_inputs(
        shot_sequence_shape, rally_info_shape,
        embed_types_size=embed_types_size, embed_area_size=embed_area_size, embed_player_size=embed_player_size
    )

    # ✅ CNN Feature Extraction with Masking
    layer_cnn = StaggeredConv1D(name='Local_pattern_extraction', **cnn_kwargs)
    pattern_sequence = layer_cnn(shot_encoder_output, mask=inputs[-1])
    pattern_sequence = tf.keras.layers.Dropout(0.3)(pattern_sequence) 

    # ✅ Positional Encoding
    seq_len = shot_sequence_shape[0]
    pos_encoding = Embedding(input_dim=seq_len, output_dim=pattern_sequence.shape[-1])(tf.range(seq_len))
    pattern_sequence_with_pos = pattern_sequence + pos_encoding

    # ✅ Transformer Encoder
    num_heads = transformer_kwargs['num_heads']
    seq_len = pattern_sequence_with_pos.shape[1]

    #  (batch_size, 1, seq_len)
    mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), 
        output_shape=lambda s: (s[0], 1, s[1]), 
        name='Expand_mask_dim_1'
    )(inputs[-1])

    # (batch_size, 1, 1, seq_len)
    mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), 
        output_shape=lambda s: (s[0], 1, 1, s[2]), 
        name='Expand_mask_dim_2'
    )(mask)

    # Tile to (batch_size, num_heads, seq_len, seq_len)
    mask = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, num_heads, 1, 1]), 
        output_shape=lambda s: (s[0], num_heads, s[2], s[3]), 
        name='Tile_mask_for_heads'
    )(mask)

    mha = MultiHeadAttention(num_heads=num_heads, key_dim=transformer_kwargs['key_dim'])
    attn_output, attn_weights = mha(
        pattern_sequence_with_pos, 
        pattern_sequence_with_pos, 
        attention_mask=mask, 
        return_attention_scores=True
    )
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output + pattern_sequence_with_pos)

    # ✅ Feed Forward Network
    ffn = Dense(transformer_kwargs['inner_dim'], activation=activations.gelu)(attn_output)  
    ffn_output = Dense(transformer_kwargs['ff_dim'])(ffn)  
    transformer_output = LayerNormalization(epsilon=1e-6)(ffn_output + attn_output)

    # ✅ Max Pooling
    rally_representation = tf.keras.layers.GlobalMaxPooling1D()(transformer_output)

    # ✅ Concatenate with Rally Information
    layer_concat_rally = tf.keras.layers.Concatenate(name='Seq_rally_merging')([rally_representation, inputs[-2]])
    # ✅ Final Dense Layer
    output_win_prob = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(layer_concat_rally)

    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict


