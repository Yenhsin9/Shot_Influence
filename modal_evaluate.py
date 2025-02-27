import time
import numpy as np
import pandas as pd
import tensorflow as tf
import rally_classifier as rc
import train
import os
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

# ✅ Setting test data
timestr = time.strftime("%Y%m%d-%H%M%S")
dataset = pd.read_csv('new_data/dataset.csv')

encode_columns = []
shot_predictors = ['is_target_turn', 'aroundhead', 'backhand', 'time_proportion']
rally_predictors = ['roundscore_diff', 'continuous_score']
target = 'is_target_win'

# ✅ Calculate the maximum sequence length and pad
seq_len = dataset.groupby(["match_id", "rally_id", "set_id"]).size().max()
seq_len += 1 if seq_len % 2 == 1 else 2

# ✅ One-Hot Encoding
encoded = pd.get_dummies(dataset, columns=encode_columns)
codes_type, uniques_type = pd.factorize(encoded['type'])
encoded['type'] = codes_type + 1  # Reserve 0 for padding

# ✅ 80/10/10 Split Dataset
train_data, val_test_data = train_test_split(encoded, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

print(f"📊 Training set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")

# ✅ Prepare training, validation, and test data
(train_shots, train_shot_types), (train_rallies, train_target, train_rally_id) = train.prepare_data(
    train_data, 
    [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], 
    [rally_predictors, target, 'rally_id'], 
    pad_to=seq_len
)

(test_shots, test_shot_types), (test_rallies, test_target, test_rally_id) = train.prepare_data(
    test_data, 
    [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], 
    [rally_predictors, target, 'rally_id'], 
    pad_to=seq_len
)

seq_len = train_shots.shape[1]

test_hit_area_encoded = test_shot_types[:, :, 0].copy()
test_player_area_encoded = test_shot_types[:, :, 1].copy()
test_opponent_area_encoded = test_shot_types[:, :, 2].copy()
test_shot_types = test_shot_types[:, :, 3].copy()
test_time_proportion = test_shots[:, :, 3].copy()
test_shots = np.delete(test_shots, 3, axis=2)

shot_predictors.remove('time_proportion')

# ✅ Setting model parameters
regularizer = tf.keras.regularizers.l2(0.01)
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['AUC', 'binary_accuracy']
epochs = 100

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./history/', histogram_freq=1)

n_shot_types = len(uniques_type) + 1
n_area_types = encoded['player_location_area'].nunique() + 1
cnn_kwargs = {'filters': 32, 'kernel_size': 3, 'kernel_regularizer': regularizer, 'activation': 'relu'}
transformer_kwargs = {'num_heads': 4, 'key_dim': 64, 'ff_dim': 128}
dense_kwargs = {'kernel_regularizer': regularizer}

batch_size = 32
MODEL_NAME = 'proposedModal'
MODEL_PATH = "./model/proposedModal/20250227-113326/final_model.weights.h5"

# ✅ Prevent TensorFlow from taking up too much GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass 

# ✅ Load `proposed_model` to evaluate
prediction_model = rc.proposed_model((seq_len, len(shot_predictors)),
                                     embed_types_size=n_shot_types,
                                     embed_area_size=n_area_types,
                                     rally_info_shape=len(rally_predictors),
                                     cnn_kwargs=cnn_kwargs,
                                     transformer_kwargs=transformer_kwargs,
                                     dense_kwargs=dense_kwargs)

prediction_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# ✅ Load `proposedModal` trained weights
if os.path.exists(MODEL_PATH):
    prediction_model.load_weights(MODEL_PATH)
    print(f"✅ Model weights loaded successfully: {MODEL_PATH}")
else:
    print(f"❌ Model weights `{MODEL_PATH}` Does not exist, please execute `training.py` to train the model first!")
    exit()

# ✅ Preparing test data
test_x = [test_hit_area_encoded, test_player_area_encoded, test_opponent_area_encoded,
          test_shots, test_shot_types, test_time_proportion, test_rallies]

# ✅ Conducting test evaluation
test_results = prediction_model.evaluate(test_x, test_target)
y_pred = prediction_model.predict(test_x)
br_score = brier_score_loss(test_target, y_pred)

# ✅ Recording test results
log_file = f'./record/{MODEL_NAME}.csv'
auc = str(test_results[1])
acc = str(test_results[2])

with open(log_file, 'a') as log:
    log.write(timestr + ', ' + auc + ', ' + acc + ', ' + str(br_score) + '\n')

print(f"✅ Testing completed!AUC: {auc}, Accuracy: {acc}, Brier Score: {br_score}")
