import pandas as pd
import numpy as np
# 讀取 CSV
train_data = pd.read_csv("train_data.csv")
val_data = pd.read_csv("val_data.csv")
test_data = pd.read_csv("test_data.csv")

# 檢查 `rally_id` 是否有交叉
train_rallies = set(train_data[["match_id", "set_id", "rally_id"]].itertuples(index=False, name=None))
val_rallies = set(val_data[["match_id", "set_id", "rally_id"]].itertuples(index=False, name=None))
test_rallies = set(test_data[["match_id", "set_id", "rally_id"]].itertuples(index=False, name=None))

train_val_overlap = train_rallies.intersection(val_rallies)
train_test_overlap = train_rallies.intersection(test_rallies)
val_test_overlap = val_rallies.intersection(test_rallies)

print(f"🚀 Train & Val overlap: {len(train_val_overlap)}")
print(f"🚀 Train & Test overlap: {len(train_test_overlap)}")
print(f"🚀 Val & Test overlap: {len(val_test_overlap)}")

# 重新計算 is_target_win（只取每個 rally 最後一個 shot）
train_target_dist = train_data.groupby(["match_id", "set_id", "rally_id"]).last()["getpoint_player"].map(lambda x: 1 if x == "B" else 0).value_counts(normalize=True)
val_target_dist = val_data.groupby(["match_id", "set_id", "rally_id"]).last()["getpoint_player"].map(lambda x: 1 if x == "B" else 0).value_counts(normalize=True)
test_target_dist = test_data.groupby(["match_id", "set_id", "rally_id"]).last()["getpoint_player"].map(lambda x: 1 if x == "B" else 0).value_counts(normalize=True)

print("🚀 Train is_target_win 分布:")
print(train_target_dist)

print("🚀 Val is_target_win 分布:")
print(val_target_dist)

print("🚀 Test is_target_win 分布:")
print(test_target_dist)
