"""Utility functions."""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import util


def split_data(dataset: pd.DataFrame,
               val_ratio: int = 0.2,
               test_mask: List[bool] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into training, validation and testing set."""
    test = dataset[test_mask]
    non_test = dataset[~test_mask]
    groups = non_test['rally_id'].unique()
    ngroup = len(groups)
    val_groups = np.random.choice(groups, int(ngroup * val_ratio))
    val_mask = non_test['rally_id'].isin(val_groups)
    val = non_test[val_mask]
    train = non_test[~val_mask]
    return train, val, test


def prepare_data(dataset: pd.DataFrame,
                 shot_attributes: Union[str, List[str], List[Union[str, List[str]]]],
                 rally_attributes: Union[str, List[str], List[Union[str, List[str]]]],
                 pad_to: int,
                 min_len: int = 1
                 ) -> Tuple[Union[np.ndarray, List[np.ndarray]],
                            Union[np.ndarray, List[np.ndarray]]]:
    """Convert dataset to appropriate format for training."""
    shots = []
    rallies = []
    masks=[]

    shot_attributes_f = util.flatten(shot_attributes)
    rally_attributes_f = util.flatten(rally_attributes)

    pre_setid = None
    consecutive_points = 0
    last_getpoint_player = None
    pre_diff = 0
    for rally_id, rally in dataset.groupby('rally_id'):
        if min_len > 0 and len(rally) < min_len:
            continue
        
        # ====== (Time Proportion) ======
        rally['time_proportion'] = np.linspace(0, 1, len(rally))

        tmp_getpoint_player = rally['is_target_win'].iloc[-1]
        # ====== (Score Difference) ======
        setid = rally['set'].iloc[-1]
        score_A = rally['roundscore_A'].iloc[-1]
        score_B = rally['roundscore_B'].iloc[-1]
        tmpdiff = score_A-score_B
            
         # (Consecutive Points)
        if setid != pre_setid: 
            roundscore_diff = 0
            consecutive_points = 0
        else:
            roundscore_diff = pre_diff
            if last_getpoint_player == 1:
                consecutive_points += 1
            else:
                consecutive_points = 0  

        pre_diff = tmpdiff
        last_getpoint_player = tmp_getpoint_player
        pre_setid = setid
        rally['consecutive_points'] = consecutive_points
        rally['roundscore_diff'] = roundscore_diff

        if 'time_proportion' not in shot_attributes_f:
            shot_attributes_f.append('time_proportion')
        if 'roundscore_diff' not in rally_attributes_f:
            rally_attributes_f.append('roundscore_diff')
        if 'consecutive_points' not in rally_attributes_f:
            rally_attributes_f.append('consecutive_points')

        shots.append(rally[shot_attributes_f].values.astype('float32'))
        rally_features = rally[rally_attributes_f].values[-1].astype('float32')
        rallies.append(rally_features)

        # ====== (Padding) ======
        pad = ((0, pad_to - len(rally)), (0, 0))
        shots[-1] = np.pad(shots[-1], pad, mode='constant', constant_values=0)
        # creating mask
        # 對整個 batch 的數據生成遮罩
        mask = np.any(shots[-1] != 0, axis=-1).astype(np.float32)
        masks.append(mask)

    shots = np.asarray(shots)
    rallies = np.asarray(rallies)
    masks = np.asarray(masks)
    # Split back to input specification
    shot_attributes_len = util.list_len(shot_attributes)
    if len(shot_attributes_len) > 1:
        shots = np.split(shots, np.cumsum(shot_attributes_len)[:-1], axis=-1)
        # Reduce dimension when there is one attribute only
        for i in range(len(shots)):
            if shot_attributes_len[i] == 1:
                shots[i] = shots[i][:, :, 0]
    rally_attributes_len = util.list_len(rally_attributes)
    if len(rally_attributes_len) > 1:
        rallies = np.split(rallies, np.cumsum(rally_attributes_len)[:-1], axis=-1)
        for i in range(len(rallies)):
            if rally_attributes_len[i] == 1:
                rallies[i] = rallies[i][:, 0]
    return (shots, rallies,masks)
