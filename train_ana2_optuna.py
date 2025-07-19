
import pandas as pd
import numpy as np
import tensorflow as tf
import rally_classifier as rc
import train
import os
import optuna
from tensorflow.keras.callbacks import EarlyStopping
import draw_plot

def run_trial(trial):
    # 固定參數
    num_folds = 1
    fold = 1
    shot_predictors = ['type', 'backhand', 'aroundhead', 'hit_area', 'player_location_area', 'opponent_location_area', 'player']
    rally_predictors = ['score_diff', 'consecutive_points']
    target = 'is_target_win'
    epochs = 100

    # 搜尋超參數
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    filters = trial.suggest_categorical("filters", [16, 32, 64])
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    kernel_reg = trial.suggest_float("kernel_regularizer", 1e-6, 1e-2, log=True)
    key_dim = filters
    ff_dim = filters
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4])
    inner_dim = trial.suggest_categorical("inner_dim", [32, 64, 128])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    l2_lambda = trial.suggest_float("l2_lambda", 1e-6, 1e-3, log=True)
    cnn_kwargs = {
        "filters": filters,
        "kernel_size": kernel_size,
        "kernel_regularizer": tf.keras.regularizers.l2(kernel_reg)
    }

    transformer_kwargs = {
        "num_heads": num_heads,
        "key_dim": key_dim,
        "ff_dim": ff_dim,
        "inner_dim": inner_dim
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    # 載入資料
    train_data = pd.read_csv(f'./data/train_fold_{fold}.csv')
    val_data = pd.read_csv(f'./data/val_fold_{fold}.csv')
    all_train_data = pd.concat([train_data], ignore_index=True)
    type_mapping = {t: i+1 for i, t in enumerate(all_train_data['type'].unique())}
    type_mapping['unknown'] = 0

    train_data['type'] = train_data['type'].map(type_mapping).fillna(0).astype(int)
    val_data['type'] = val_data['type'].map(type_mapping).fillna(0).astype(int)

    seq_len = max(
        pd.read_csv(f'./data/train_fold_{fold}.csv').groupby('rally_id').size().max()
        for fold in range(1, num_folds + 1)
    )
    seq_len += 1 if seq_len % 2 == 1 else 2

    (train_shots), (train_rallies, train_target, train_rally_id), train_masks = train.prepare_data(
        train_data, [shot_predictors], [rally_predictors, target, 'rally_id'], pad_to=seq_len
    )
    (val_shots), (val_rallies, val_target, val_rally_id), val_masks = train.prepare_data(
        val_data, [shot_predictors], [rally_predictors, target, 'rally_id'], pad_to=seq_len
    )

    # Extract features
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
    val_shots = np.delete(val_shots, indices_to_delete, axis=2)

    # Prepare input data
    train_x = [train_shots, train_shot_type, train_player_id, train_time_proportion, train_hit_area,
               train_player_area, train_opponent_area, train_rallies, train_masks]
    val_x = [val_shots, val_shot_type, val_player_id, val_time_proportion, val_hit_area,
             val_player_area, val_opponent_area, val_rallies, val_masks]

    model = rc.proposed_model(
        (seq_len, train_shots.shape[2]),
        embed_types_size=len(type_mapping) + 1,
        embed_area_size = max(
            train_data['player_location_area'].max(),
            train_data['opponent_location_area'].max(),
            train_data['hit_area'].max(),
            val_data['player_location_area'].max(),
            val_data['opponent_location_area'].max(),
            val_data['hit_area'].max()
        ) + 1,
        rally_info_shape=len(rally_predictors),
        cnn_kwargs=cnn_kwargs,
        transformer_kwargs=transformer_kwargs,
        dropout_rate=dropout_rate,
        l2_lambda=l2_lambda,
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.MeanSquaredError(name='brier_score')]
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ]

    history = model.fit(
        train_x, train_target,
        validation_data=(val_x, val_target),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    val_auc = history.history['val_auc']
    val_loss = history.history['val_loss']

    avg_auc = np.mean(val_auc[-5:])
    avg_loss = np.mean(val_loss[-5:])
    std_auc = np.std(val_auc)
    std_loss = np.std(val_loss)

    # 刪除 best_model.keras 檔案
    if os.path.exists("best_model_fold_1.keras"):
        os.remove("best_model_fold_1.keras")

    return  avg_loss, std_loss

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(run_trial, n_trials=30)

    print("\n✅ 最佳參數組合與結果:")
    best_trials = study.best_trials  # 獲取所有 Pareto 前沿試驗
    for i, trial in enumerate(best_trials):
        print(f"\n試驗 {i + 1}:")
        for k, v in trial.params.items():
            print(f"{k}: {v}")
        print(f"目標值: {trial.values}")  # 顯示每個試驗的目標值 (e.g., [avg_auc, avg_loss])

    # 可選：打印總試驗數
    print(f"\n總共找到 {len(best_trials)} 個最佳試驗。")

