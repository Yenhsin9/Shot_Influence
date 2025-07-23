import pandas as pd
import numpy as np
import tensorflow as tf
import rally_classifier as rc
import train
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import draw_plot

num_folds = 5

# Compute type_mapping using all train folds
all_train_data = pd.concat([pd.read_csv(f'./data/train_fold_{fold}.csv') for fold in range(1, num_folds + 1)], ignore_index=True)
type_mapping = {t: i+1 for i, t in enumerate(all_train_data['type'].unique())}
type_mapping['unknown'] = 0
print("Category correspondence table:", type_mapping)

# Determine seq_len based on all train folds
seq_len = max(
    pd.read_csv(f'./data/train_fold_{fold}.csv').groupby('rally_id').size().max()
    for fold in range(1, num_folds + 1)
)
seq_len += 1 if seq_len % 2 == 1 else 2
print(f"Sequence length: {seq_len}")

# Initialize lists to store metrics for all folds
all_train_loss = []
all_val_loss = []
all_train_auc = []
all_val_auc = []
all_train_brier = []
all_val_brier = []
all_train_acc = []
all_val_acc = []
final_val_aucs = []  # To store final validation AUC for each fold

# 5-fold cross-validation loop
for fold in range(1, num_folds + 1):

    fold_train_loss = []
    fold_val_loss = []
    fold_train_auc = []
    fold_val_auc = []
    fold_train_brier = []
    fold_val_brier = []
    fold_train_acc = []
    fold_val_acc = []

    # Define hyperparameters and configurations
    shot_predictors = ['type', 'backhand', 'aroundhead', 'hit_area', 'player_location_area', 'opponent_location_area', 'player']
    rally_predictors = ['score_diff', 'consecutive_points']
    target = 'is_target_win'

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
    print(f"\nTraining Fold {fold}...")
    epochs = 100
    # Load fold data
    train_data = pd.read_csv(f'./data/train_fold_{fold}.csv')
    val_data = pd.read_csv(f'./data/val_fold_{fold}.csv')
    print(f"Fold {fold} - Train data shape: {train_data.shape}")
    print(f"Fold {fold} - Validation data shape: {val_data.shape}")

    # Apply type mapping
    train_data['type'] = train_data['type'].map(type_mapping).fillna(0).astype(int)
    val_data['type'] = val_data['type'].map(type_mapping).fillna(0).astype(int)

    # Prepare data
    (train_shots), (train_rallies, train_target, train_rally_id), train_masks = train.prepare_data(
        train_data,
        [shot_predictors],
        [rally_predictors, target, 'rally_id'],
        pad_to=seq_len
    )
    (val_shots), (val_rallies, val_target, val_rally_id), val_masks = train.prepare_data(
        val_data,
        [shot_predictors],
        [rally_predictors, target, 'rally_id'],
        pad_to=seq_len
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

    # Build model
    model = rc.proposed_model(
        (seq_len, train_shots.shape[2]),
        embed_types_size=len(type_mapping) + 1,
        embed_area_size=max(
            train_data['player_location_area'].nunique(), train_data['opponent_location_area'].nunique(),
            train_data['hit_area'].nunique(), val_data['player_location_area'].nunique(),
            val_data['opponent_location_area'].nunique(), val_data['hit_area'].nunique()
        ) + 1,
        rally_info_shape=len(rally_predictors),
        cnn_kwargs=cnn_kwargs,
        transformer_kwargs=transformer_kwargs,
        dropout_rate=drop_rate,
        l2_lambda=l2_lambda,
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.MeanSquaredError(name='brier_score'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy')
        ]
    )

    model_path = f'best_model_fold_{fold}'

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True,  save_weights_only=False),  
    ]

    history = model.fit(
        train_x, train_target,
        validation_data=(val_x, val_target),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )
    # Store metrics for this fold
    fold_train_loss.extend(history.history['loss'])
    fold_val_loss.extend(history.history['val_loss'])
    fold_train_auc.extend(history.history['auc'])
    fold_val_auc.extend(history.history['val_auc'])
    fold_train_brier.extend(history.history['brier_score'])
    fold_val_brier.extend(history.history['val_brier_score'])
    fold_train_acc.extend(history.history['accuracy'])          
    fold_val_acc.extend(history.history['val_accuracy'])

    # Append fold metrics to the global lists
    all_train_loss.append(fold_train_loss)
    all_val_loss.append(fold_val_loss)
    all_train_auc.append(fold_train_auc)
    all_val_auc.append(fold_val_auc)
    all_train_brier.append(fold_train_brier)
    all_val_brier.append(fold_val_brier)
    all_train_acc.append(fold_train_acc)
    all_val_acc.append(fold_val_acc)
    
    # Print final validation metrics
    print(f"Fold {fold} - Best Val Loss: {min(fold_val_loss):.4f}, "
          f"Best Val AUC: {max(fold_val_auc):.4f}, "
          f"Best Val Brier Score: {min(fold_val_brier):.4f}, "
          f"Best Val Accuracy: {max(fold_val_acc):.4f}")

    # Plotting Training and Validation Loss
    draw_plot.draw_plot(
        fold_train_auc, fold_val_auc, fold_train_brier, fold_val_brier, fold_train_loss, fold_val_loss,
        fold_train_acc,fold_val_acc,
    )


 # 儲存 Train Metrics
pd.DataFrame(all_train_loss).T.to_csv(f'./plot_data/all_train_loss.csv', index=False)
pd.DataFrame(all_train_auc).T.to_csv(f'./plot_data/all_train_auc.csv', index=False)
pd.DataFrame(all_train_brier).T.to_csv(f'./plot_data/all_train_brier.csv', index=False)
pd.DataFrame(all_train_acc).T.to_csv(f'./plot_data/all_train_acc.csv', index=False)

# 儲存 Validation Metrics
pd.DataFrame(all_val_loss).T.to_csv(f'./plot_data/all_val_loss.csv', index=False)
pd.DataFrame(all_val_auc).T.to_csv(f'./plot_data/all_val_auc.csv', index=False)
pd.DataFrame(all_val_brier).T.to_csv(f'./plot_data/all_val_brier.csv', index=False)
pd.DataFrame(all_val_acc).T.to_csv(f'./plot_data/all_val_acc.csv', index=False)

# 提取每個 fold 的最佳 AUC
best_aucs_per_fold = [max(fold_auc) for fold_auc in all_val_auc]
best_loss_per_fold = [min(fold_loss) for fold_loss in all_val_loss]
best_brier_per_fold = [min(fold_brier) for fold_brier in all_val_brier]
best_acc_per_fold = [max(fold_acc) for fold_acc in all_val_acc]

# 找到最佳 fold
if best_aucs_per_fold:
    best_fold_idx = np.argmax(best_aucs_per_fold)
    best_fold = best_fold_idx + 1  # 因為 fold 從 1 開始
    best_auc = best_aucs_per_fold[best_fold_idx]
    print(f"最佳模型來自 Fold {best_fold}，訓練 AUC: {best_auc:.4f}")

# Compute average validation metrics
avg_val_loss = np.mean(best_loss_per_fold)
avg_val_auc = np.mean(best_aucs_per_fold)
std_val_auc = np.std(best_aucs_per_fold)
avg_val_brier = np.mean(best_brier_per_fold)
avg_acc = np.mean(best_acc_per_fold)
print("\nAverage Validation Metrics Across Folds:")
print(f"Average Val Loss: {avg_val_loss:.4f}")
print(f"Average Val AUC: {avg_val_auc:.4f} (±{std_val_auc:.4f})")
print(f"Average Val Brier Score: {avg_val_brier:.4f}")
print(f"Average Val Accuracy: {avg_acc:.4f}")
