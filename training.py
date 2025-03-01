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
shot_predictors = ['player_id','time_proportion', 'backhand', 'aroundhead','hit_area', 'player_location_area', 'opponent_location_area',"type"]
rally_predictors = ['roundscore_diff', 'continuous_score']
target = 'is_target_win'

# ✅ Find largest seq len for padding
seq_len = dataset.groupby("rally_id").size().max()
seq_len += 1 if seq_len % 2 == 1 else 2

# ✅ One-Hot Encoding
encoded = pd.get_dummies(dataset, columns=encode_columns)
codes_type, uniques_type = pd.factorize(encoded['type'])
encoded['type'] = codes_type + 1  # Reserve 0 for padding

shot_predictors = [c for c in encoded.columns if any(c.startswith(f'{p}_') for p in shot_predictors) or c in shot_predictors]

# 80/10/10
unique_rallies = encoded["rally_id"].drop_duplicates()

train_rally, val_test_rally = train_test_split(unique_rallies, test_size=0.2, random_state=42)

val_rally, test_rally = train_test_split(val_test_rally, test_size=0.5, random_state=42)

print(f"📊 Train Rally: {len(train_rally)}, Val Rally: {len(val_rally)}, Test Rally: {len(test_rally)}")

train_data = encoded.merge(train_rally, on= "rally_id")
val_data = encoded.merge(val_rally, on="rally_id")
test_data = encoded.merge(test_rally, on= "rally_id")

print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# ✅ save data
test_data.to_csv("./split_data/test_data.csv", index=False)
train_data.to_csv("./split_data/train_data.csv", index=False)
val_data.to_csv("./split_data/val_data.csv", index=False)
print("✅ dataset saved successfully!")

(train_shots), (train_rallies, train_target),train_rally_id = train.prepare_data(
    train_data, 
    [shot_predictors], 
    [rally_predictors, target],  
    pad_to=seq_len
)

(val_shots), (val_rallies, val_target),val_rally_id = train.prepare_data(
    val_data, 
    [shot_predictors], 
    [rally_predictors, target],  
    pad_to=seq_len
)

seq_len = train_shots.shape[1]

train_player_id = train_shots[:, :, 0].copy()
train_time_proportion = train_shots[:, :, 1].copy()
train_hit_area = train_shots[:, :, 4].copy()
train_player_area = train_shots[:, :, 5].copy()
train_opponent_area = train_shots[:, :, 6].copy()
train_shot_type = train_shots[:, :, 7].copy()
indices_to_delete = [0, 1, 4, 5, 6, 7]  
train_shots = np.delete(train_shots, indices_to_delete, axis=2)

val_player_id = val_shots[:, :, 0].copy()
val_time_proportion = val_shots[:, :, 1].copy()
val_hit_area = val_shots[:, :, 4].copy()
val_player_area = val_shots[:, :, 5].copy()
val_opponent_area = val_shots[:, :, 6].copy()
val_shot_type = val_shots[:, :, 7].copy()
indices_to_delete = [0, 1, 4, 5, 6, 7]  
val_shots = np.delete(val_shots, indices_to_delete, axis=2)

# ✅ Set model hyperparameters
regularizer = tf.keras.regularizers.l2(0.1)
loss = 'binary_crossentropy'
metrics = ['AUC', 'binary_accuracy']
epochs = 100

MODEL_DIR = "./model/"
MODEL_NAME = "proposedModal"  
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
os.makedirs(MODEL_PATH, exist_ok=True)

n_shot_types = len(uniques_type) + 1
n_area_types = encoded['player_location_area'].nunique() + 1
n_player_types = encoded['player_id'].nunique() + 1  
cnn_kwargs = {'filters': 32, 'kernel_size': 3, 'kernel_regularizer': regularizer, 'activation': 'relu'}
transformer_kwargs = {
    'num_heads': 1,  # Reduce heads to match paper
    'key_dim': 32,  # Reduce key_dim to 32
    'ff_dim': 32,  # Reduce FFN dimension to 32
    'inner_dim': 64  # Add `dinner` as inner FFN dimension
}
dense_kwargs = {'kernel_regularizer': regularizer}
batch_size = 32
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  

import json
param_dict = {
    "n_shot_types": n_shot_types, 
    "n_area_types": n_area_types,  
    "n_player_types": n_player_types 
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
    (seq_len, train_shots.shape[2]),  #train_shots.shape[1] left only 2 features 'backhand', 'aroundhead'
    embed_types_size=n_shot_types,
    embed_area_size=n_area_types,
    embed_player_size=n_player_types,
    rally_info_shape=len(rally_predictors),
    cnn_kwargs=cnn_kwargs,
    transformer_kwargs=transformer_kwargs,
    dense_kwargs=dense_kwargs
)

prediction_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

train_x = [train_shots, train_shot_type, train_player_id,train_time_proportion,train_hit_area ,train_player_area, train_opponent_area,train_rallies]

val_x = [val_shots, val_shot_type,val_player_id,val_time_proportion,val_hit_area ,val_player_area,val_opponent_area, val_rallies]

#checkpoint
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_PATH, "weights_epoch_{epoch:02d}.weights.h5"),  
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)

callbacks = tf.keras.callbacks.EarlyStopping(min_delta=0.001, patience=15, restore_best_weights=True),
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

# 取得模型對訓練集的預測結果
y_pred_train = prediction_model.predict(train_x)

# 建立 DataFrame，存儲預測與實際值
rally_predictions_train = pd.DataFrame({
    "rally_id": train_rally_id,      # 訓練集的 `rally_id`
    "predicted_win_prob": y_pred_train.flatten(),  # 預測的 `win probability`
    "actual_win": train_target.flatten()      # 真實的 `is_target_win`
})

# 存成 CSV
rally_predictions_train.to_csv("train_rally_predictions.csv", index=False)
print("✅ 訓練期間預測與真實勝率對比結果已存成 CSV：train_rally_predictions.csv")

# set `model_file`
model_file = os.path.join("./model/", MODEL_NAME, timestr, "final_model.weights.h5") 

# ✅ Make sure the archive directory exists
model_dir = os.path.dirname(model_file)
os.makedirs(model_dir, exist_ok=True)

# ✅ Storing model weights
prediction_model.save_weights(model_file)
print(f"✅ Model weights are stored in: {model_file}")

