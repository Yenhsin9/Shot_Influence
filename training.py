import time
import numpy as np
import pandas as pd
import tensorflow as tf
import rally_classifier as rc
import train
import util
import os
import glob
from sklearn.model_selection import train_test_split

timestr = time.strftime("%Y%m%d-%H%M%S")

dataset = pd.read_csv('new_data/dataset.csv')

encode_columns = []
shot_predictors = ['is_target_turn', 'aroundhead', 'backhand', 'time_proportion']
rally_predictors = ['roundscore_diff', 'continuous_score']
target = 'is_target_win'

# ✅ Find largest seq len for padding
seq_len = dataset.groupby(["match_id", "rally_id", "set_id"]).size().max()
seq_len += 1 if seq_len % 2 == 1 else 2

# ✅ One-Hot Encoding
encoded = pd.get_dummies(dataset, columns=encode_columns)
codes_type, uniques_type = pd.factorize(encoded['type'])
encoded['type'] = codes_type + 1  # Reserve 0 for padding

shot_predictors = [c for c in encoded.columns if any(c.startswith(f'{p}_') for p in shot_predictors) or c in shot_predictors]
# ✅ train,val,test 80/10/10 
train_data, val_test_data = train_test_split(encoded, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

print(f"📊 Training set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")
# ✅ save test data
test_data.to_csv("./new_data/test_data.csv", index=False)
print("✅ Test dataset saved successfully!")

(train_shots, train_shot_types), (train_rallies, train_target, train_rally_id) = train.prepare_data(
    train_data, 
    [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], 
    [rally_predictors, target, 'rally_id'], 
    pad_to=seq_len
)

(val_shots, val_shot_types), (val_rallies, val_target, val_rally_id) = train.prepare_data(
    val_data, 
    [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], 
    [rally_predictors, target, 'rally_id'], 
    pad_to=seq_len
)

seq_len = train_shots.shape[1]

train_hit_area_encoded = train_shot_types[:, :, 0].copy()
train_player_area_encoded = train_shot_types[:, :, 1].copy()
train_opponent_area_encoded = train_shot_types[:, :, 2].copy()
train_shot_types = train_shot_types[:, :, 3].copy()
train_time_proportion = train_shots[:, :, 3].copy()
train_shots = np.delete(train_shots, 3, axis=2)

val_hit_area_encoded = val_shot_types[:, :, 0].copy()
val_player_area_encoded = val_shot_types[:, :, 1].copy()
val_opponent_area_encoded = val_shot_types[:, :, 2].copy()
val_shot_types = val_shot_types[:, :, 3].copy()
val_time_proportion = val_shots[:, :, 3].copy()
val_shots = np.delete(val_shots, 3, axis=2)

shot_predictors.remove('time_proportion')

# ✅ Set model hyperparameters
regularizer = tf.keras.regularizers.l2(0.01)
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['AUC', 'binary_accuracy']
epochs = 100

MODEL_DIR = "./model/"
MODEL_NAME = "proposedModal"  
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
os.makedirs(MODEL_PATH, exist_ok=True)

n_shot_types = len(uniques_type) + 1
n_area_types = encoded['player_location_area'].nunique() + 1
cnn_kwargs = {'filters': 32, 'kernel_size': 3, 'kernel_regularizer': regularizer, 'activation': 'relu'}
transformer_kwargs = {'num_heads': 4, 'key_dim': 64, 'ff_dim': 128}
dense_kwargs = {'kernel_regularizer': regularizer}
batch_size = 32

import json
param_dict = {
    "n_shot_types": n_shot_types,  # Dataset-dependent
    "n_area_types": n_area_types,  # Dataset-dependent
}
# Save parameters to a JSON file
param_path = "./hyperParameter/model_params.json"
with open(param_path, "w") as f:
    json.dump(param_dict, f)

print(f"✅ Model parameters saved to {param_path}")

# ✅ Avoid TensorFlow taking up too much GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# ✅ Build Proposed model
prediction_model = rc.proposed_model(
    (seq_len, len(shot_predictors)),
    embed_types_size=n_shot_types,
    embed_area_size=n_area_types,
    rally_info_shape=len(rally_predictors),
    cnn_kwargs=cnn_kwargs,
    transformer_kwargs=transformer_kwargs,
    dense_kwargs=dense_kwargs
)

prediction_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

train_x = [train_hit_area_encoded, train_player_area_encoded, train_opponent_area_encoded, train_shots, train_shot_types, train_time_proportion, train_rallies]
val_x = [val_hit_area_encoded, val_player_area_encoded, val_opponent_area_encoded, val_shots, val_shot_types, val_time_proportion, val_rallies]

#checkpoint
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_PATH, "weights_epoch_{epoch:02d}.weights.h5"),  
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(min_delta=0.001, patience=15, restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='./history/', histogram_freq=1),
    checkpoint_callback 
]

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./history/', histogram_freq=1)
# ✅ find all `.weights.h5` file
checkpoint_files = glob.glob(os.path.join(MODEL_PATH, "weights_epoch_*.weights.h5"))
latest_checkpoint = None
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_epoch_")[-1].split(".")[0]))
    print(f"🔄 Load new weight: {latest_checkpoint}")
    prediction_model.load_weights(latest_checkpoint)

    initial_epoch = int(latest_checkpoint.split("_epoch_")[-1].split(".")[0])
    print(f"🚀 Training start from epoch {initial_epoch + 1} ...")
else:
    print("⚠️ No existing model weights found, start from epoch 1.")
    initial_epoch = 0  

print("🔥🔥🔥 Start training！")
history = prediction_model.fit(train_x, train_target,
                               validation_data=(val_x, val_target),
                               epochs=epochs, 
                               initial_epoch=initial_epoch,  
                               batch_size=batch_size,
                               callbacks=callbacks)

# set `model_file`
model_file = os.path.join("./model/", MODEL_NAME, timestr, "final_model.weights.h5") 

# ✅ Make sure the archive directory exists
model_dir = os.path.dirname(model_file)
os.makedirs(model_dir, exist_ok=True)

# ✅ Storing model weights
prediction_model.save_weights(model_file)
print(f"✅ Model weights are stored in: {model_file}")

