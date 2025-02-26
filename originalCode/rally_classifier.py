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

def transformer_rally_model(shot_sequence_shape: Tuple[int, int], 
                            embed_types_size: int = None,
                            embed_area_size: int = None,
                            rally_info_shape: int = None,
                            cnn_kwargs: Dict[str, Any] = {'filters': 32, 'kernel_size': 3},
                            transformer_kwargs: Dict[str, Any] = {'num_heads': 4, 'key_dim': 64, 'ff_dim': 128},
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

def bad_net(shot_sequence_shape: Tuple[int, int], 
            embed_types_size: int = None,
            embed_area_size: int = None,
            rally_info_shape: int = None,
            cnn_kwargs: Dict[str, Any] = {'filters': 32, 'kernel_size': 3},
            rnn_kwargs: Dict[str, Any] = {'units': 32},
            attention_kwargs: Dict[str, Any] = {},
            dense_kwargs: Dict[str, Any] = {}) -> Tuple[tf.keras.Model, tf.keras.Model]:


    inputs, masked_sequence = preprocess_inputs(shot_sequence_shape, rally_info_shape,embed_types_size, embed_area_size)

    # **CNN**
    layer_cnn = StaggeredConv1D(name='Local_pattern_extraction', **cnn_kwargs)
    pattern_sequence = layer_cnn(masked_sequence)

    # **RNN**
    layer_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(return_sequences=True, **rnn_kwargs),
                                              name='Bidirectional_recurrent_layer')
    hidden_states = layer_rnn(pattern_sequence)

    # **CNN + RNN**
    layer_concat_cnn_rnn = tf.keras.layers.Concatenate(name='Patterns_states_merging')
    patterns_states = layer_concat_cnn_rnn([pattern_sequence, hidden_states])

    # **Attention**
    layer_attention = keras_self_attention.SeqWeightedAttention(return_attention=True, **attention_kwargs)
    rally_represent, contributions = layer_attention(patterns_states)

    # **input rally feature**
    layer_concat_rally = tf.keras.layers.Concatenate(name='Seq_rally_merging')([rally_represent, inputs[-1]])

    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid', **dense_kwargs)
    output_win_prob = layer_dense(layer_concat_rally)

    # **create modal**
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    model_attention = tf.keras.Model(inputs=inputs, outputs=contributions)
    
    return model_predict, model_attention


def rnn(shot_sequence_shape: Tuple[int, int],
        rally_info_shape: int = None,
        rnn_structure: str = 'lstm',
        bidirectional_rnn: bool = True,
        rnn_kwargs: Dict[str, Any] = {'units': 32},
        dense_kwargs: Dict[str, Any] = {}
        ) -> tf.keras.Model:
    """Create a RNN-based rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')

    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    rnn = rnns.get(rnn_structure, list(rnns.values())[0])
    layer_rnn = rnn(name='Recurrent_layer', **rnn_kwargs)
    if bidirectional_rnn:
        layer_rnn = tf.keras.layers.Bidirectional(
            layer_rnn, name='Bidirectional_recurrent_layer')
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=(rally_info_shape,),
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    rally_represent = layer_rnn(masked_sequence)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict


def deepmoji(shot_sequence_shape: Tuple[int, int],
             rally_info_shape: int = None,
             rnn_kwargs: Dict[str, Any] = {'units': 32},
             attention_kwargs: Dict[str, Any] = {},
             dense_kwargs: Dict[str, Any] = {}
             ) -> tf.keras.Model:
    """Create DeepMoji rally classifier model."""
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    layer_rnn1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(return_sequences=True, **rnn_kwargs),
        name='Bidirectional_recurrent_layer1')
    layer_rnn2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(return_sequences=True, **rnn_kwargs),
        name='Bidirectional_recurrent_layer2')
    layer_concat_input_rnn = tf.keras.layers.Concatenate(
        name='Input_rnn_merging')
    layer_attention = keras_self_attention.SeqWeightedAttention(
        return_attention=True, **attention_kwargs)
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    hidden_states = layer_rnn2(layer_rnn1(masked_sequence))
    input_states = layer_concat_input_rnn([masked_sequence,
                                           hidden_states])
    rally_represent, contributions = layer_attention(input_states)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    model_attention = tf.keras.Model(inputs=inputs, outputs=contributions)
    return model_predict, model_attention


def prosenet(shot_sequence_shape: Tuple[int, int],
             prosenet_kwargs: Dict[str, Any] = {'k': 100},
             rnn_kwargs: Dict[str, Any] = {'layer_type' : 'lstm',
                                           'layer_args' : {},
                                           'layers' : [32, 32],
                                           'bidirectional' : True}
             ) -> tf.keras.Model:
    """Create a ProSeNet rally classifier model."""
    model_predict = ProSeNet(input_shape=shot_sequence_shape, nclasses=2,
                             rnn_args=rnn_kwargs, **prosenet_kwargs)
    return model_predict


# def onlstm(shot_sequence_shape: Tuple[int, int], 
#                embed_types_size: int = None,
#                embed_area_size: int = None,
#                rally_info_shape: int = None,
#                cnn_kwargs: Dict[str, Any] = {'filters': 32, 'kernel_size': 3},
#                onlstm_kwargs: Dict[str, Any] = {'units': 32, 'chunk_size': 4},
#                attention_kwargs: Dict[str, Any] = {},
#                dense_kwargs: Dict[str, Any] = {}) -> Tuple[tf.keras.Model, tf.keras.Model]:
    
    
#     inputs, masked_sequence = preprocess_inputs(shot_sequence_shape, rally_info_shape,
#                                                 embed_types_size=embed_types_size, embed_area_size=embed_area_size)

#     #CNN
#     layer_cnn = StaggeredConv1D(name='Local_pattern_extraction', **cnn_kwargs)
#     pattern_sequence = layer_cnn(masked_sequence)

#     # ON-LSTM 
#     layer_onlstm = ONLSTM(name='ONLSTM', return_sequences=True, **onlstm_kwargs)
#     hidden_states = layer_onlstm(pattern_sequence)  # ✅ 這行應該會觸發 `build()`

#     # CNN + ON-LSTM 
#     layer_concat_cnn_rnn = tf.keras.layers.Concatenate(name='Patterns_states_merging')
#     patterns_states = layer_concat_cnn_rnn([pattern_sequence, hidden_states])
#     # print(f"🔥 Attention 前 patterns_states shape: {patterns_states.shape}")

#     # Self-Attention 
#     layer_attention = keras_self_attention.SeqWeightedAttention(return_attention=True)
#     rally_represent = layer_attention(patterns_states)  # ✅ 確保 shape = (None, 66, 64)

#     rally_represent = tf.keras.layers.Reshape((66, 64))(rally_represent)  # ✅ 確保 seq_len=66
#     input_rally = tf.keras.layers.RepeatVector(66)(inputs[-1])  # ✅ 讓 `input_rally` 變成 (None, 66, 2)

#     # combine rally information
#     layer_concat_rally = tf.keras.layers.Concatenate(axis=-1, name='Seq_rally_merging')([rally_represent, input_rally])

#     layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid', **dense_kwargs)
#     output_win_prob = layer_dense(layer_concat_rally)
#     model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
#     model_attention = tf.keras.Model(inputs=inputs, outputs=patterns_states)

#     return model_predict, model_attention

def onlstm(shot_sequence_shape: Tuple[int, int],
           rally_info_shape: int = None,
           onlstm_kwargs: Dict[str, Any] = {'units': 32, 'chunk_size': 4},
           dense_kwargs: Dict[str, Any] = {}
           ) -> tf.keras.Model:
    """Create an ON-LSTM rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    layer_onlstm = ONLSTM(name='ONLSTM', **onlstm_kwargs)
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    rally_represent = layer_onlstm(masked_sequence)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict

def transformer(shot_sequence_shape: Tuple[int, int],
                rally_info_shape: int = None,
                transformer_kwargs: Dict[str, Any] = {'encoder_num': 2,
                                                      'head_num': 2,
                                                      'hidden_dim': 32,
                                                      'feed_forward_activation': gelu},
                dense_kwargs: Dict[str, Any] = {}
           ) -> tf.keras.Model:
    """Create an ON-LSTM rally classifier model."""
    rnns = {'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM}
    # Layers
    input_shots = tf.keras.Input(shape=shot_sequence_shape,
                                 name='Shots_input')
    layer_masking = tf.keras.layers.Masking(name='Sequence_masking')
    layer_pos_embed = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD)
    layer_pooling = tf.keras.layers.GlobalMaxPooling1D()
    if rally_info_shape is not None:
        input_rally = tf.keras.Input(shape=rally_info_shape,
                                     name='Rally_input')
        layer_concat_rally = tf.keras.layers.Concatenate(
            name='Seq_rally_merging')
    else:
        input_rally = None
        layer_concat_rally = None
    layer_dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                        **dense_kwargs)
    # Forward pass
    inputs = [input_shots]
    masked_sequence = layer_masking(input_shots)
    pos_embed_seq = layer_pos_embed(masked_sequence)
    encoder_result = get_encoders(input_layer=pos_embed_seq, **transformer_kwargs)
    rally_represent = layer_pooling(encoder_result)
    if rally_info_shape is not None:
        inputs.append(input_rally)
        rally_represent = layer_concat_rally([rally_represent, input_rally])
    output_win_prob = layer_dense(rally_represent)
    model_predict = tf.keras.Model(inputs=inputs, outputs=output_win_prob)
    return model_predict

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
