import time
import numpy as np
import pandas as pd
import tensorflow as tf
import rally_classifier as rc
import train
import os
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import json

# ✅ Setting test data
test_data_path = "./new_data/test_data.csv"
if not os.path.exists(test_data_path):
    print(f"❌ Test dataset `{test_data_path}` not found. Please execute `training.py` first!")
    exit()

test_data = pd.read_csv(test_data_path)
print("✅ Test dataset loaded successfully!")

# Load parameters from JSON file
param_path = "./hyperParameter/model_params.json"
if not os.path.exists(param_path):
    print(f"❌ `{param_path}` not found! Please rerun `training.py` first.")
    exit()

with open(param_path, "r") as f:
    param_dict = json.load(f)

# Load dataset-dependent parameters
n_shot_types = param_dict["n_shot_types"]
n_area_types = param_dict["n_area_types"]
n_player_types = param_dict["n_player_types"]

print(f"✅ Model parameters loaded: n_shot_types={n_shot_types}, n_area_types={n_area_types},n_player_types={n_player_types}")

timestr = time.strftime("%Y%m%d-%H%M%S")

encode_columns = []
shot_predictors = ['is_target_turn', 'aroundhead', 'backhand', 'time_proportion']
rally_predictors = ['roundscore_diff', 'continuous_score']
target = 'is_target_win'

# ✅ Calculate the maximum sequence length and pad
seq_len = test_data.groupby(["match_id", "rally_id", "set_id"]).size().max()
seq_len += 1 if seq_len % 2 == 1 else 2

# ✅ One-Hot Encoding
encoded = pd.get_dummies(test_data, columns=encode_columns)
codes_type, uniques_type = pd.factorize(encoded['type'])
encoded['type'] = codes_type + 1  # Reserve 0 for padding

(test_shots, test_shot_types), (test_rallies, test_target, test_rally_id,test_players) = train.prepare_data(
    test_data, 
    [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], 
    [rally_predictors, target, 'rally_id', 'player_id'],  
    pad_to=seq_len
)

seq_len = test_shots.shape[1]

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

cnn_kwargs = {'filters': 32, 'kernel_size': 3, 'kernel_regularizer': regularizer, 'activation': 'relu'}
transformer_kwargs = {
    'num_heads': 1,  # Reduce heads to match paper
    'key_dim': 32,  # Reduce key_dim to 32
    'ff_dim': 32,  # Reduce FFN dimension to 32
    'inner_dim': 64  # Add `dinner` as inner FFN dimension
}
dense_kwargs = {'kernel_regularizer': regularizer}
batch_size = 32
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)  
MODEL_NAME = 'proposedModal'
MODEL_PATH = "./model/proposedModal/20250227-192107/final_model.weights.h5"

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
                                     embed_player_size=n_player_types,
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
test_x = [test_players,test_hit_area_encoded, test_player_area_encoded, test_opponent_area_encoded,
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
