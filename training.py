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
from tensorflow.keras.callbacks import Callback

# Timestamp for model saving
timestr = time.strftime("%Y%m%d-%H%M%S")

# Load data
train_data = pd.read_csv('./data/train.csv')
val_given_data = pd.read_csv('./data/val_given.csv')
val_gt_data = pd.read_csv('./data/val_gt.csv')

print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_given_data.shape}, GT: {val_gt_data.shape}")

# ✅ Data preprocessing parameters
shot_predictors = ['player', 'type', 'backhand', 'aroundhead', 
                   'hit_area', 'player_location_area', 'opponent_location_area']
rally_predictors = ['roundscore_diff', 'consecutive_points']  
target = 'is_target_win'

# ✅ Find largest seq len for padding
seq_len = train_data.groupby('rally_id').size().max()
seq_len += 1 if seq_len % 2 == 1 else 2 

# ✅ Encode 'type' column
codes_type, uniques_type = pd.factorize(train_data['type'])
train_data['type'] = codes_type + 1  
val_given_data['type'] = val_given_data['type'].apply(
    lambda x: (uniques_type.tolist().index(x) + 1) if x in uniques_type else 0
)

(train_shots), (train_rallies, train_target,train_rally_id)= train.prepare_data(
    train_data, 
    [shot_predictors], 
    [rally_predictors,target,'rally_id'],
    pad_to=seq_len
)

(val_shots), (val_rallies, val_target,val_rally_id) = train.prepare_data(
    val_given_data, 
    [shot_predictors], 
    [rally_predictors,target,'rally_id'],
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
regularizer = tf.keras.regularizers.l2(0.01)
loss = 'binary_crossentropy'
metrics = ['AUC', 'binary_accuracy']
epochs = 20

MODEL_DIR = "./model/"
MODEL_NAME = "proposedModal"  
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
os.makedirs(MODEL_PATH, exist_ok=True)

n_shot_types = len(uniques_type) + 1
n_area_types = max(train_data['player_location_area'].nunique(), 
                   train_data['opponent_location_area'].nunique()) + 1  
n_player_types = train_data['player'].nunique() + 1  
cnn_kwargs = {'filters': 32, 'kernel_size': 3, 'kernel_regularizer': regularizer, 'activation': 'relu'}
transformer_kwargs = {
    'num_heads': 1,  # Reduce heads to match paper
    'key_dim': 32,  # Reduce key_dim to 32
    'ff_dim': 32,  # Reduce FFN dimension to 32
    'inner_dim': 64  # Add `dinner` as inner FFN dimension
}
dense_kwargs = {'kernel_regularizer': regularizer}
batch_size = 32
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)  

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

class EpochDebugCallback(Callback):
    def __init__(self, train_data, train_labels, val_data, val_labels, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.train_labels = tf.reshape(train_labels, (-1, 1))  # 調整標籤數據形狀
        self.val_data = val_data
        self.val_labels = tf.reshape(val_labels, (-1, 1))  # 同樣調整驗證數據形狀
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n🌀 Epoch {epoch + 1} 結束，評估模型...")

        # 使用 GradientTape 手動計算梯度
        with tf.GradientTape() as tape:
            predictions = self.model(self.train_data, training=True)
            loss = self.model.compiled_loss(self.train_labels, predictions)

        # 計算梯度
        gradients = tape.gradient(loss, self.model.trainable_weights)
        
        # 輸出每層的梯度平均值
        for weight, grad in zip(self.model.trainable_weights, gradients):
            if grad is not None:
                tf.print(f'🔍 Layer: {weight.name}, Gradient mean: {tf.reduce_mean(tf.abs(grad))}')
            else:
                tf.print(f'⚠️ Layer: {weight.name}, Gradient is None')

        # 訓練集預測值和 loss
        train_pred = self.model.predict(self.train_data, batch_size=self.batch_size)
        train_loss = tf.keras.losses.binary_crossentropy(self.train_labels, train_pred)
        print(f"📊 訓練集預測值範圍: {train_pred.min():.4f} - {train_pred.max():.4f}, 預測均值: {train_pred.mean():.4f}")
        print(f"🧮 訓練集手動計算的 binary_crossentropy loss: {tf.reduce_mean(train_loss).numpy():.4f}")

        # 驗證集預測值和 loss
        val_pred = self.model.predict(self.val_data, batch_size=self.batch_size)
        val_loss = tf.keras.losses.binary_crossentropy(self.val_labels, val_pred)
        print(f"📊 驗證集預測值範圍: {val_pred.min():.4f} - {val_pred.max():.4f}, 預測均值: {val_pred.mean():.4f}")
        print(f"🧮 驗證集手動計算的 binary_crossentropy loss: {tf.reduce_mean(val_loss).numpy():.4f}")
        print(f"📝 Keras 記錄的 Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")


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
                               callbacks=[EpochDebugCallback(train_x,train_target,val_x,val_target)])

# 取得模型對訓練集的預測結果
y_pred_train = prediction_model.predict(train_x)
y_pred_val = prediction_model.predict(val_x)

# 查看模型預測值的分佈情況
import matplotlib.pyplot as plt

plt.hist(y_pred_train.flatten(), bins=50)
plt.xlabel("預測值 (Predicted Probability)")
plt.ylabel("頻率 (Frequency)")
plt.title("模型初始預測值分佈")
plt.show()

# 如果預測值大部分集中在 0 或 1，會看到分佈圖兩端有很高的頻率

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

