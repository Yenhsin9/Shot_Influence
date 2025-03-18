import pandas as pd
import numpy as np
import tensorflow as tf
import rally_classifier as rc
import train
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
import os
import csv

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
cnn_kwargs = {'filters': 32, 'kernel_size': 3, 'kernel_regularizer': tf.keras.regularizers.l2(0.01)}
transformer_kwargs = {
    'num_heads': 1,  
    'key_dim':32,  
    'ff_dim': 32,  
    'inner_dim': 64  
}
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

#input data
train_x = [train_shots, train_shot_type, train_player_id,train_time_proportion,train_hit_area ,train_player_area, train_opponent_area,train_rallies,train_masks]
val_x = [val_shots, val_shot_type,val_player_id,val_time_proportion,val_hit_area ,val_player_area,val_opponent_area, val_rallies,val_masks]

# Build the Model
model = rc.proposed_model(
    (seq_len, train_shots.shape[2]),
    embed_types_size=len(type_mapping) + 1,
    embed_area_size=max(train_data['player_location_area'].nunique(), train_data['opponent_location_area'].nunique(),train_data['hit_area'].nunique()
                        ,val_data['player_location_area'].nunique(), val_data['opponent_location_area'].nunique(),val_data['hit_area'].nunique()) + 1,  
    embed_player_size=43,  
    rally_info_shape=len(rally_predictors),
    cnn_kwargs=cnn_kwargs,
    transformer_kwargs=transformer_kwargs,
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC', 'binary_accuracy'])

# Add Callbacks
model_path = 'best_model.keras'
if os.path.exists(model_path):
    os.remove(model_path)  

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, save_weights_only=False),
    # shot_encoder_callback
]

# Training Model
history = model.fit(
    train_x, train_target,
    validation_data=(val_x, val_target),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=callbacks,
    #shuffle=False
)


# Plotting Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

# 取得預測結果並儲存
train_predictions = model.predict(train_x)
predictions_df = pd.DataFrame({
    'rally_id': train_rally_id,
    'true_label': train_target,
    'predicted_win_probability': train_predictions.flatten()
})
predictions_csv = 'predictions2.csv'
predictions_df.to_csv(predictions_csv, index=False)
print(f"✅ 已儲存預測結果至 {predictions_csv}")

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


# shot_encoder_callback = ShotEncoderOutputCallback(
#     train_data=[
#         train_x
#     ],
#     output_file='+Tile_mask_for_heads.csv',
#     batch_limit=5  # 只保存前 5 個 batch
# )

# import pandas as pd
# # 假設 train_shots 形狀是 (batch_size, seq_len, 8)
# batch_size, seq_len, feature_dim = train_shots.shape
# flat_shots = train_shots.reshape(-1, feature_dim)
# df_shots = pd.DataFrame(flat_shots)
# # 將 DataFrame 保存為 CSV 文件，每一行表示 8 個特徵
# df_shots.to_csv("train_shots.csv", index=False,header=False)
# print("train_shots 已成功保存到 train_shots.csv")
