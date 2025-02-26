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

def preprocess_inputs(shot_sequence_shape, rally_info_shape,embed_types_size=None, embed_area_size=None ):
    """ premanage shot encodind data： input + Embedding + Concat + Masking """
    
     # **7 input**
    input_hit_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Hit_area_input')
    input_player_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Player_area_input')
    input_opponent_area = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Opponent_area_input')
    input_shots = tf.keras.Input(shape=shot_sequence_shape, name='Shots_input')   # (None, 66, 3)
    input_shot_types = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Shot_types_input')
    input_time_proportion = tf.keras.Input(shape=(shot_sequence_shape[0],), name='Time_proportion_input')
    input_rally = tf.keras.Input(shape=(rally_info_shape,), name='Rally_input')
    

    # **Hit Area Embedding**
    if embed_area_size is not None:
        area_embedding = tf.keras.layers.Embedding(input_dim=embed_area_size, output_dim=10, mask_zero=True, name='Area_embedding')
        embeded_hit_area = area_embedding(input_hit_area)
        embeded_player_area = area_embedding(input_player_area)
        embeded_opponent_area = area_embedding(input_opponent_area)
    else:
        embeded_hit_area, embeded_player_area, embeded_opponent_area = input_hit_area, input_player_area, input_opponent_area

    # **Shot Type Embedding**
    if embed_types_size is not None:
        shot_embedding = tf.keras.layers.Embedding(input_dim=embed_types_size, output_dim=15, mask_zero=True, name='Shot_types_embedding')
        embeded_shot_types = shot_embedding(input_shot_types)

        shot_mu_embedding = tf.keras.layers.Embedding(input_dim=embed_types_size, output_dim=15, mask_zero=True, name='Time_influence_occurrence')
        shot_theta_embedding = tf.keras.layers.Embedding(input_dim=embed_types_size, output_dim=15, mask_zero=True, name='Time_influence_shot')

        # **Time Proportion Enhancement**
        tiled_time_proportion = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_time_proportion)
        tiled_time_proportion = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 15]))(tiled_time_proportion)


        time_mu_proportion = tf.keras.layers.Multiply(name='Time_proportion_multiply')([shot_mu_embedding(input_shot_types), tiled_time_proportion])
        temporal_score = tf.keras.layers.Add(name='Time_proportion_add')([shot_theta_embedding(input_shot_types), time_mu_proportion])
        temporal_score = tf.keras.layers.Activation('sigmoid', name='Time_activation')(temporal_score)

        embeded_activity = tf.keras.layers.Multiply(name='Shots_time_multiply')([temporal_score, embeded_shot_types])
        enhanced_shot_features = tf.keras.layers.Concatenate(name='Shots_features_merging')([embeded_activity, input_shots])
    else:
        enhanced_shot_features = input_shots

    # **concat Hit Area and Shot Features**
    shots_concat_areas = tf.keras.layers.Concatenate(name='Shots_areas_merging')(
        [embeded_hit_area, embeded_player_area, embeded_opponent_area, enhanced_shot_features])

    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    masked_sequence = layer_masking(shots_concat_areas)

    return [input_hit_area, input_player_area, input_opponent_area, input_shots, 
            input_shot_types, input_time_proportion, input_rally], masked_sequence

# Proposed modal: CNN + Position + Mutihead Attention
def proposed_model(shot_sequence_shape: Tuple[int, int], 
                            embed_types_size: int = None,
                            embed_area_size: int = None,
                            rally_info_shape: int = None,
                            cnn_kwargs: Dict[str, Any] = {'filters': 32, 'kernel_size': 3},
                            transformer_kwargs: Dict[str, Any] = {},
                            dense_kwargs: Dict[str, Any] = {}) -> tf.keras.Model:

    # ✅ 預處理輸入，獲取 `masked_sequence` & `inputs`
    inputs, masked_sequence = preprocess_inputs(shot_sequence_shape, rally_info_shape,
                                                embed_types_size=embed_types_size, embed_area_size=embed_area_size)

    # ✅ CNN 短期學習 (Short-Term Feature Extraction)
    layer_cnn = StaggeredConv1D(name='Local_pattern_extraction', **cnn_kwargs)
    pattern_sequence = layer_cnn(masked_sequence)  # (None, 66, feature_dim)

    # ✅ Positional Encoding
    seq_len = shot_sequence_shape[0]
    pos_encoding = Embedding(input_dim=seq_len, output_dim=pattern_sequence.shape[-1])(tf.range(seq_len))
    pattern_sequence_with_pos = pattern_sequence + pos_encoding  # 讓 Transformer 知道順序

    # ✅ Transformer Encoder Self-Attention
    mha = MultiHeadAttention(num_heads=transformer_kwargs['num_heads'], key_dim=transformer_kwargs['key_dim'])
    attn_output = mha(pattern_sequence_with_pos, pattern_sequence_with_pos)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output + pattern_sequence_with_pos)  # 殘差連接
    ffn = Dense(transformer_kwargs['ff_dim'], activation='relu')(attn_output)
    ffn_output = Dense(pattern_sequence.shape[-1])(ffn)  # 轉回原始維度
    transformer_output = LayerNormalization(epsilon=1e-6)(ffn_output + attn_output)  # 第二次殘差連接

    # ✅ 最終 Concatenate Rally Information
    layer_concat_rally = tf.keras.layers.Concatenate(name='Seq_rally_merging')([transformer_output[:, -1, :], inputs[-1]])

    # ✅ 勝率預測 (Win Probability Prediction)
    output_win_prob = Dense(units=1, activation='sigmoid', **dense_kwargs)(layer_concat_rally)

    # ✅ 建立模型
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict
