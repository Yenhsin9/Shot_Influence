import time
import numpy as np
import pandas as pd
import tensorflow as tf
import rally_classifier as rc
import train
import util
from sklearn.metrics import brier_score_loss
import glob
import os
timestr = time.strftime("%Y%m%d-%H%M%S")
dataset = pd.read_csv('new_data/dataset.csv')
from sklearn.model_selection import train_test_split

test_mask = (dataset['match_id'] == 30) | (dataset['match_id'] == 34)
val_ratio = 0.3
encode_columns = []
shot_predictors = ['is_target_turn', 'aroundhead', 'backhand', 'time_proportion']
rally_predictors = ['roundscore_diff', 'continuous_score']
target = 'is_target_win'

seq_len = dataset.groupby(["match_id", "rally_id","set_id"]).size().max()
seq_len += 1 if seq_len % 2 == 1 else 2
encoded = pd.get_dummies(dataset, columns=encode_columns)
codes_type, uniques_type = pd.factorize(encoded['type'])
encoded['type'] = codes_type + 1  # Reserve code 0 for paddings
shot_predictors = [c for c in encoded.columns if any(c.startswith(f'{p}_')for p in shot_predictors) or c in shot_predictors]
train_data, val_data, test_data = train.split_data(encoded, val_ratio=val_ratio, test_mask=test_mask)

(train_shots, train_shot_types), (train_rallies, train_target, train_rally_id) = train.prepare_data(train_data, [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], [rally_predictors, target, 'rally_id'], pad_to=seq_len)

(val_shots, val_shot_types), (val_rallies, val_target, val_rally_id) = train.prepare_data(val_data, [shot_predictors, ['hit_area', 'player_location_area', 'opponent_location_area', 'type']], [rally_predictors, target, 'rally_id'], pad_to=seq_len)
seq_len = train_shots.shape[1] #padding num

train_hit_area_encoded = train_shot_types[:, :, 0].copy()
train_player_area_encoded = train_shot_types[:, :, 1].copy()
train_opponent_area_encoded = train_shot_types[:, :, 2].copy()
train_shot_types = train_shot_types[:, :, 3].copy()            # (batch_size, seq_len, 4) → (batch_size, seq_len)  redefine train_shot_types
train_time_proportion = train_shots[:, :, 3].copy()             # time proportion
train_shots = np.delete(train_shots, 3, axis=2)
val_hit_area_encoded = val_shot_types[:, :, 0].copy()
val_player_area_encoded = val_shot_types[:, :, 1].copy()
val_opponent_area_encoded = val_shot_types[:, :, 2].copy()
val_shot_types = val_shot_types[:, :, 3].copy()
val_time_proportion = val_shots[:, :, 3].copy()             # time proportion
val_shots = np.delete(val_shots, 3, axis=2)

shot_predictors.remove('time_proportion')


regularizer = tf.keras.regularizers.l2(0.01)
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['AUC', 'binary_accuracy']
epochs = 100

MODEL_DIR = "./model/"
MODEL_NAME = "onlstm"  
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
os.makedirs(MODEL_PATH, exist_ok=True)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_PATH, "weights_epoch_{epoch:02d}.weights.h5"),  
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(min_delta=0.0005, patience=15, restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='./history/', histogram_freq=1),
    checkpoint_callback 
]

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./history/', histogram_freq=1)

n_shot_types = len(uniques_type) + 1
n_area_types = encoded['player_location_area'].nunique() + 1
cnn_kwargs = {'filters': 32, 'kernel_size': 3, 'kernel_regularizer': regularizer,
              'activation': 'relu'}
rnn_kwargs = {'units': 32, 'kernel_regularizer': regularizer}
onlstm_kwargs = {
    'units': 32,        
    'chunk_size': 4,     
    'dropout': 0.3,      
    'recurrent_dropout': 0.3  
}
transformer_kwargs={'num_heads': 4, 'key_dim': 64, 'ff_dim': 128}

dense_kwargs = {'kernel_regularizer': regularizer}

batch_size = 32

# Avoid tensorflow use full memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# prediction_model, attention_model = rc.bad_net((seq_len, len(shot_predictors)),
#                                                embed_types_size=n_shot_types,
#                                                embed_area_size=n_area_types,
#                                                rally_info_shape=len(rally_predictors),
#                                                cnn_kwargs=cnn_kwargs,
#                                                rnn_kwargs=rnn_kwargs,
#                                                dense_kwargs=dense_kwargs)
prediction_model= rc.proposed_model((seq_len, len(shot_predictors)),
                                               embed_types_size=n_shot_types,
                                               embed_area_size=n_area_types,
                                               rally_info_shape=len(rally_predictors),
                                               cnn_kwargs=cnn_kwargs,
                                               transformer_kwargs=transformer_kwargs,
                                               dense_kwargs=dense_kwargs,)  


prediction_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

train_x = [train_hit_area_encoded, train_player_area_encoded, train_opponent_area_encoded, train_shots, train_shot_types, train_time_proportion, train_rallies]
val_x = [val_hit_area_encoded, val_player_area_encoded, val_opponent_area_encoded, val_shots, val_shot_types, val_time_proportion, val_rallies]


train_target = np.expand_dims(train_target, axis=-1)
val_target = np.expand_dims(val_target, axis=-1)
# ✅ 找到所有 `.weights.h5` 檔案
checkpoint_files = glob.glob(os.path.join(MODEL_PATH, "weights_epoch_*.weights.h5"))

# ✅ 確保 `latest_checkpoint` 變數存在，避免報錯
latest_checkpoint = None

if checkpoint_files:
    # ✅ 找出 `epoch` 最大的 `.weights.h5`
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_epoch_")[-1].split(".")[0]))
    print(f"🔄 載入最新權重: {latest_checkpoint}")
    prediction_model.load_weights(latest_checkpoint)

    # ✅ 解析 `epoch` 數字，確保 `fit()` 可以繼續訓練
    initial_epoch = int(latest_checkpoint.split("_epoch_")[-1].split(".")[0])
    print(f"🚀 訓練將從 epoch {initial_epoch + 1} 繼續...")
else:
    print("⚠️ 沒有找到現有的模型權重，從頭開始訓練。")
    initial_epoch = 0  # ✅ 如果沒有找到存檔，從 `epoch 0` 開始
print("🔥🔥🔥 即將開始訓練！")
import numpy as np

print("🛠 檢查 `train_x` 是否包含 NaN 或 無限大數值...")
for i, x in enumerate(train_x):
    print(f"train_x[{i}] shape: {x.shape}, NaN 數量: {np.isnan(x).sum()}, 無限值數量: {np.isinf(x).sum()}")

print("🛠 檢查 `train_target` 是否包含 NaN 或 無限大數值...")
print(f"train_target shape: {train_target.shape}, NaN 數量: {np.isnan(train_target).sum()}, 無限值數量: {np.isinf(train_target).sum()}")

history = prediction_model.fit(train_x, train_target,
                               validation_data=(val_x, val_target),
                               epochs=epochs, 
                               initial_epoch=initial_epoch, 
                               batch_size=batch_size,
                               callbacks=[callbacks])

# 設定 `model_file`
model_file = os.path.join("./model/", MODEL_NAME, timestr, "final_model.weights.h5") 

# ✅ 確保存檔目錄存在
model_dir = os.path.dirname(model_file)
os.makedirs(model_dir, exist_ok=True)

# ✅ 儲存模型權重
prediction_model.save_weights(model_file)
print(f"✅ 模型權重已儲存至: {model_file}")

