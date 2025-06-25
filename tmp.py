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
train_data = pd.read_csv('./data/train.csv')
val_data = pd.read_csv('./data/val.csv')

print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")

# Data Preprocessing
shot_predictors = ['type', 'backhand', 'aroundhead', 
                   'hit_area', 'player_location_area', 'opponent_location_area', 'player']
rally_predictors = ['roundscore_diff', 'consecutive_points']  
target = 'is_target_win'

seq_len = train_data.groupby('rally_id').size().max()
seq_len += 1 if seq_len % 2 == 1 else 2 

# Encode 'type' columns
codes_train, uniques_train = pd.factorize(train_data['type'])
train_data['type'] = codes_train + 1  
# Create a category correspondence table for train_data
type_mapping = {name: idx+1 for idx, name in enumerate(uniques_train.tolist())}

# Let the new categories of val_data also get unique codes
# ensuring that the same categories get the same number
next_id = len(type_mapping) + 1 
val_data['type'] = val_data['type'].apply(
    lambda x: type_mapping.setdefault(x, next_id + len(type_mapping))
)

print("Category correspondence table:", type_mapping)

# Prepare Data
(train_shots), (train_rallies, train_target,train_rally_id),train_masks= train.prepare_data(
    train_data, 
    [shot_predictors], 
    [rally_predictors,target,'rally_id'],
    pad_to=seq_len
)

(val_shots), (val_rallies, val_target,val_rally_id),val_masks = train.prepare_data(
    val_data, 
    [shot_predictors], 
    [rally_predictors,target,'rally_id'],
    pad_to=seq_len
)

seq_len = train_shots.shape[1]
train_player_id = train_shots[:, :, 6].copy()
train_shot_type = train_shots[:, :, 0].copy()
train_hit_area = train_shots[:, :, 3].copy()
train_player_area = train_shots[:, :, 4].copy()
train_opponent_area = train_shots[:, :, 5].copy()
train_time_proportion = train_shots[:, :, 7].copy()
indices_to_delete = [0, 3, 4, 5, 6, 7]  
train_shots = np.delete(train_shots, indices_to_delete, axis=2)

val_player_id = val_shots[:, :, 6].copy()
val_shot_type = val_shots[:, :, 0].copy()
val_hit_area = val_shots[:, :, 3].copy()
val_player_area = val_shots[:, :, 4].copy()
val_opponent_area = val_shots[:, :, 5].copy()
val_time_proportion = val_shots[:, :, 7].copy()
indices_to_delete = [0, 3, 4, 5, 6, 7]  
val_shots = np.delete(val_shots, indices_to_delete, axis=2)

# Model Hyperparameters
cnn_kwargs = {'filters': 16, 'kernel_size': 3, 'kernel_regularizer': tf.keras.regularizers.l2(0.01)}
transformer_kwargs = {
    'num_heads': 1,  
    'key_dim':16,  
    'ff_dim': 16,  
    'inner_dim': 32  
}
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)

#input data
train_x = [train_shots, train_shot_type, train_player_id,train_time_proportion,train_hit_area ,train_player_area, train_opponent_area,train_rallies,train_masks]
val_x = [val_shots, val_shot_type,val_player_id,val_time_proportion,val_hit_area ,val_player_area,val_opponent_area, val_rallies,val_masks]

# Build the Model
model = rc.proposed_model(
    (seq_len, train_shots.shape[2]),
    embed_types_size=len(type_mapping) + 1,
    embed_area_size=max(train_data['player_location_area'].nunique(), train_data['opponent_location_area'].nunique(),train_data['hit_area'].nunique()
                        ,val_data['player_location_area'].nunique(), val_data['opponent_location_area'].nunique(),val_data['hit_area'].nunique()) + 1,  
    embed_player_size=27,
    rally_info_shape=len(rally_predictors),
    cnn_kwargs=cnn_kwargs,
    transformer_kwargs=transformer_kwargs,
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.MeanSquaredError(name='brier_score')
    ])

# Add Callbacks
model_path = 'best_model.keras'
if os.path.exists(model_path):
    os.remove(model_path)  

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, save_weights_only=False),
]

# Training Model
history = model.fit(
    train_x, train_target,
    validation_data=(val_x, val_target),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=callbacks,
    shuffle=True
)

# Plotting Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')

# 獲取 AUC 數據
train_auc = history.history['auc']  # 訓練集 AUC
val_auc = history.history['val_auc']  # 驗證集 AUC
train_br = history.history['brier_score']
val_br = history.history['val_brier_score']

epochs = range(1, len(train_auc) + 1)  # Epoch 數

# 繪製 AUC 曲線
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_auc, label='Train AUC', marker='o', linestyle='-')
plt.plot(epochs, val_auc, label='Validation AUC', marker='s', linestyle='--')

# 圖表標題 & 標籤
plt.title('Train vs Validation AUC Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.grid()

plt.show()

plt.figure()
plt.plot(epochs, train_br, label='Train Brier Score')
plt.plot(epochs, val_br,   label='Validation Brier Score')
plt.xlabel('Epoch')
plt.ylabel('Brier Score')
plt.title('Brier Score over Epochs')
plt.legend()
plt.grid(True)
plt.show()
# import tensorflow as tf
# from tensorflow.keras.callbacks import Callback
# import numpy as np
# import csv

# class ShotEncoderOutputCallback(Callback):
#     def __init__(self, train_data, output_file='train_shot_encoder_output.csv', batch_limit=5):
#         super().__init__()
#         self.train_data = train_data
#         self.output_file = output_file
#         self.batch_limit = batch_limit  # 只保存前幾個 batch 以免文件過大

#     def on_batch_end(self, batch, logs=None):
#         if batch >= self.batch_limit:
#             return
        
#         # 獲取中間層輸出（Shot Encoder Output）
#         Shots_input = self.model.get_layer("Tile_mask_for_heads").output
#         intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=Shots_input)
        
#         encoder_output_data = intermediate_model.predict(self.train_data)

#         # 將輸出寫入 CSV 文件
#         with open(self.output_file, mode='a', newline='') as file:  # 使用 'a' 追加模式
#             writer = csv.writer(file)
#             for row in encoder_output_data:
#                 # 將 row 轉成列表，保證是可迭代的
#                 if isinstance(row, (float, int, np.float32, np.int32)):
#                     writer.writerow([row])  # 包裝成列表
#                 else:
#                     writer.writerow(row)
        
#         print(f"Saved shot encoder output to {self.output_file} for batch {batch + 1}")


