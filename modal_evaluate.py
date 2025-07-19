import pandas as pd
import numpy as np
import tensorflow as tf
import rally_classifier as rc
import train
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
import os
import csv
from tensorflow.keras.models import load_model

# Load Data
test_data = pd.read_csv('./data/test.csv')

print(f"Test data shape: {test_data.shape}")

# Data Preprocessing
shot_predictors = ['type', 'backhand', 'aroundhead', 
                   'hit_area', 'player_location_area', 'opponent_location_area', 'player']
rally_predictors = ['score_diff', 'consecutive_points']  
target = 'is_target_win'

seq_len = 72

# Encode 'type' columns
codes_test, uniques_test = pd.factorize(test_data['type'])
test_data['type'] = codes_test + 1  
# Create a category correspondence table for train_data
type_mapping = {name: idx+1 for idx, name in enumerate(uniques_test.tolist())}

# Prepare Data
(test_shots), (test_rallies, test_target,test_rally_id),test_masks= train.prepare_data(
    test_data, 
    [shot_predictors], 
    [rally_predictors,target,'rally_id'],
    pad_to=seq_len
)

seq_len = test_shots.shape[1]
test_player_id = test_shots[:, :, 6].copy()
test_shot_type = test_shots[:, :, 0].copy()
test_hit_area = test_shots[:, :, 3].copy()
test_player_area = test_shots[:, :, 4].copy()
test_opponent_area = test_shots[:, :, 5].copy()
test_time_proportion = test_shots[:, :, 7].copy()
indices_to_delete = [0, 3, 4, 5, 6, 7]  
test_shots = np.delete(test_shots, indices_to_delete, axis=2)

# Model Hyperparameters
batch_size = 64
drop_rate = 0.3604
l2_lambda = 0.00086
cnn_kwargs = {'filters': 32, 'kernel_size': 2, 'kernel_regularizer': tf.keras.regularizers.l2(3.390804752248029e-05)}
transformer_kwargs = {
    'num_heads': 1,
    'key_dim': 32,
    'ff_dim': 32,
    'inner_dim':32,
}
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000185, clipnorm=1.0)
MODEL_NAME = 'proposedModal'
MODEL_PATH = "best_model_fold_3"

#input data
test_x = [test_shots, test_shot_type, test_player_id,test_time_proportion,test_hit_area ,test_player_area, test_opponent_area,test_rallies,test_masks]

# Build model
model = rc.proposed_model(
    (seq_len, test_shots.shape[2]),
    embed_types_size=len(type_mapping) + 1,
    embed_area_size=max(
        test_data['player_location_area'].nunique(), test_data['opponent_location_area'].nunique(),
        test_data['hit_area'].nunique(),
    ) + 1,
    rally_info_shape=len(rally_predictors),
    cnn_kwargs=cnn_kwargs,
    transformer_kwargs=transformer_kwargs,
    dropout_rate=drop_rate,
    l2_lambda=l2_lambda,
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.MeanSquaredError(name='brier_score')
    ])

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model weights loaded successfully: {MODEL_PATH}")
else:
    print(f"❌ Model weights `{MODEL_PATH}` Does not exist, please execute `training.py` to train the model first!")
    exit()

test_results = model.evaluate(test_x, test_target)
loss, auc, brier = test_results
print(f"Test Loss: {loss:.4f} | AUC: {auc:.4f} | Brier: {brier:.4f}")
