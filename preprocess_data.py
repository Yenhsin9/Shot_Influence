import os
import re
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split, KFold

VAL_ID = [27,28,29,30,31,32,33,34,35]  
TEST_ID = [36,37,38,39,40,41,42,43,44]  

class PreDataProcessor:
    def __init__(self, path: str):
        self.match = pd.read_csv(f"{path}match.csv")
        # convert players to categorical values (anonymize)
        self.show_unique_players()
        #1/0
        self.player_map = {name: idx + 1 for idx, name in enumerate(self.unique_players)}  # 建立從 1 開始的 map

        self.match['winner'] = self.match['winner'].map(self.player_map)
        self.match['loser'] = self.match['loser'].map(self.player_map)

        self.homography = pd.read_csv(f"{path}homography.csv")
        #self.homography = self.homography.drop(columns=['video', 'db'])
        self.homography = self.homography.drop(columns=['video'])
        self.homography['set'] = self.match['set']
        self.homography['duration'] = self.match['duration']
        self.homography['winner'] = self.match['winner']
        self.homography['loser'] = self.match['loser']
        self.homography.to_csv(f"{path}match_metadata.csv", index=False)
        self.homography_matrix = pd.read_csv(f"{path}match_metadata.csv", converters={'homography_matrix':lambda x: np.array(ast.literal_eval(x))})

        all_matches = self.read_metadata(directory=f"{path}set")
        cleaned_matches = self.engineer_match(all_matches)
        cleaned_matches.to_csv(f"{path}shot_metadata.csv", index=False)

    def read_metadata(self, directory):
        all_matches = []  #all match files
        for idx in range(len(self.match)):
            match_idx = self.match['id'][idx]
            match_name = self.match['video'][idx]
            winner = self.match['winner'][idx]
            loser = self.match['loser'][idx]
            current_homography = self.homography_matrix[self.homography_matrix['id'] == match_idx]['homography_matrix'].to_numpy()[0]

            match_path = os.path.join(directory, match_name)
            csv_paths = [os.path.join(match_path, f) for f in os.listdir(match_path) if f.endswith('.csv')]
            
            one_match = []
            for csv_path in csv_paths:
                data = pd.read_csv(csv_path)
                set_id = int(re.findall(r'\d+', os.path.basename(csv_path))[0])  

                for rally_id, rally in data.groupby('rally'):
                    getpoint_player = rally['getpoint_player'].iloc[-1] 

                    mapping = {'A': winner, 'B': loser}
                    rally['player'] = rally['player'].map(mapping)
                    rally['getpoint_player'] = rally['getpoint_player'].map(mapping)
                    rally['is_target_win'] = 1 if getpoint_player=='A' else 0
                    rally['set'] = set_id
                    rally['match_id'] = match_idx
                    rally['winner'] = winner
                    rally['loser'] = loser

                    one_match.append(rally)

            match = pd.concat(one_match, ignore_index=True, sort=False).assign(match_id=match_idx)

            # project screen coordinate to real coordinate
            for i in range(len(match)):
                # project ball coordinates
                p = np.array([match['landing_x'][i], match['landing_y'][i], 1])
                p_real = current_homography.dot(p)
                p_real /= p_real[2]
                match.loc[i, ['landing_x', 'landing_y']] = round(p_real[0], 1), round(p_real[1], 2)

                # project player coordinates
                p = np.array([match['player_location_x'][i], match['player_location_y'][i], 1])
                p_real = current_homography.dot(p)
                p_real /= p_real[2]
                match.loc[i, ['player_location_x', 'player_location_y']] = round(p_real[0], 1), round(p_real[1], 2)

                # project opponent coordinates
                p = np.array([match['opponent_location_x'][i], match['opponent_location_y'][i], 1])
                p_real = current_homography.dot(p)
                p_real /= p_real[2]
                match.loc[i, ['opponent_location_x', 'opponent_location_y']] = round(p_real[0], 1), round(p_real[1], 2)

            all_matches.append(match)

        all_matches = pd.concat(all_matches, ignore_index=True, sort=False)
        return all_matches

    def engineer_match(self, matches):
        matches['rally_id'] = matches.groupby(['match_id', 'set', 'rally']).ngroup()
        print("Original: ")
        self.print_current_size(matches)

        # Drop flaw rally
        if 'flaw' in matches.columns:
            flaw_rally = matches[matches['flaw'].notna()]['rally_id']
            matches = matches[~matches['rally_id'].isin(flaw_rally)]
            matches = matches.reset_index(drop=True)
        print("After Dropping flaw: ")
        self.print_current_size(matches)

        # Drop unknown ball type
        unknown_rally = matches[matches['type'] == '未知球種']['rally_id']
        matches = matches[~matches['rally_id'].isin(unknown_rally)]
        matches = matches.reset_index(drop=True)
        print("After dropping unknown ball type: ")
        self.print_current_size(matches)

        # Drop hit_area at outside
        outside_area = [10, 11, 12, 13, 14, 15, 16]
        matches.loc[matches['server'] == 1, 'hit_area'] = 7
        for area in outside_area:
            outside_rallies = matches.loc[matches['hit_area'] == area, 'rally_id']
            matches = matches[~matches['rally_id'].isin(outside_rallies)]
            matches = matches.reset_index(drop=True)
        # Deal with hit_area convert hit_area to integer
        matches = self.drop_na_rally(matches, columns=['hit_area'])
        matches['hit_area'] = matches['hit_area'].astype(float).astype(int)
        print("After converting hit_area: ")
        self.print_current_size(matches)

        # Convert landing_area, player location, and opponent location outside to 10 and to integer
        matches = self.drop_na_rally(matches, columns=['landing_area'])
        matches = self.drop_na_rally(matches, columns=['player_location_area'])
        matches = self.drop_na_rally(matches, columns=['opponent_location_area'])
        for area in outside_area:
            matches.loc[matches['landing_area'] == area, 'landing_area'] = 10
            matches.loc[matches['player_location_area'] == area, 'player_location_area'] = 10
            matches.loc[matches['opponent_location_area'] == area, 'opponent_location_area'] = 10
        matches['landing_area'] = matches['landing_area'].astype(float).astype(int)
        matches['player_location_area'] = matches['player_location_area'].astype(float).astype(int)
        matches['opponent_location_area'] = matches['opponent_location_area'].astype(float).astype(int)
        print("After converting landing_area: ")
        self.print_current_size(matches)

        # Deal with ball type. Convert ball types to general version (10 types)
        # Convert 小平球 to 平球 because of old version
        matches['type'] = matches['type'].replace('小平球', '平球')
        combined_types = {'切球': '切球', '過度切球': '切球', '點扣': '殺球', '殺球': '殺球', '平球': '平球', '後場抽平球': '平球', '擋小球': '接殺防守',
                    '防守回挑': '接殺防守', '防守回抽': '接殺防守', '放小球': '網前球', '勾球': '網前球', '推球': '推撲球', '撲球': '推撲球'}
        shot_type_transform = {
            '發短球': 'short service',
            '長球': 'clear',
            '推撲球': 'push/rush',
            '殺球': 'smash',
            '接殺防守': 'defensive shot',
            '平球': 'drive',
            '網前球': 'net shot',
            '挑球': 'lob',
            '切球': 'drop',
            '發長球': 'long service',
        }
        matches.loc[:, 'type'] = matches['type'].replace(combined_types)
        matches.loc[:, 'type'] = matches['type'].replace(shot_type_transform)
        print("After converting ball type: ")
        self.print_current_size(matches)

        # Fill zero value in backhand
        matches.loc[:, 'backhand'] = matches['backhand'].fillna(value=0)
        matches['backhand'] = matches['backhand'].astype(float).astype(int)

        # Fill zero value in aroundhead
        matches.loc[:, 'aroundhead'] = matches['aroundhead'].fillna(value=0)

        # Convert ball round type to integer
        matches.loc[:, 'ball_round'] = matches['ball_round'].astype(float).astype(int)

        # Translate lose reasons from Chinese to English (foul is treated as not pass over the net)
        reason_transform = {'出界': 'out', '落點判斷失誤': 'misjudged', '掛網': 'touched the net', '未過網': 'not pass over the net', '對手落地致勝': "opponent's ball landed", '犯規': 'not pass over the net'}
        
        matches.loc[:, 'lose_reason'] = matches['lose_reason'].replace(reason_transform)

        mean_x, std_x = 175., 82.
        mean_y, std_y = 467., 192.
        matches['landing_x'] = (matches['landing_x']-mean_x) / std_x
        matches['landing_y'] = (matches['landing_y']-mean_y) / std_y

        # Remove some unrelated fields
        matches = matches.drop(columns=['hit_height','hit_x','hit_y', 'win_reason', 'flaw', 'db'])

        return matches

    def compute_statistics(self):
        self.show_unique_players()
        self.compute_matchup_counts()

    def show_unique_players(self):
        # show players
        column_players = self.match[['winner', 'loser']].values.ravel()
        self.unique_players = pd.unique(column_players).tolist()
        print("Show unique player: ")
        print(self.unique_players, len(self.unique_players))

    def compute_matchup_counts(self):
        # compute matchup counts of each player
        column_values = self.match[['winner', 'loser']].values
        players = []
        for column_value in column_values:
            players.append(column_value)

        player_matrix = [[0] * len(self.unique_players) for _ in range(len(self.unique_players))]
        for player in players:
            player_index_row, player_index_col = self.unique_players.index(player[0]), self.unique_players.index(player[1])
            player_matrix[player_index_row][player_index_col] += 1
            player_matrix[player_index_col][player_index_row] += 1
        player_matrix = pd.DataFrame(player_matrix, index=self.unique_players, columns=self.unique_players)
        
        plot = sns.heatmap(player_matrix, annot=True, linewidths=0.5, cbar=False)
        plt.xticks(rotation=30, ha='right')
        plot.get_figure().savefig("./figures/player_matrix.png", dpi=300, bbox_inches='tight')
        plot.clear()

    def drop_na_rally(self, df, columns=[]):
        """Drop rallies which contain na value in columns."""
        df = df.copy()
        for column in columns:
            rallies = df[df[column].isna()]['rally_id']
            df = df[~df['rally_id'].isin(rallies)]
        df = df.reset_index(drop=True)
        return df

    def print_current_size(self, all_match):
        print("Current size")
        print('\tUnique rally: {}\t Total rows: {}'.format(all_match['rally_id'].nunique(), len(all_match)))


class CoachAITrainTestSplit:
    def __init__(self, path):
        self.metadata = pd.read_csv(os.path.join(path, 'shot_metadata.csv'))
        self.matches = pd.read_csv(os.path.join(path, 'match_metadata.csv'))
        self.given_strokes_num = 4
        self.path = path

        type_codes, type_uniques = pd.factorize(self.metadata['type'])
        self.metadata['type'] = type_codes + 1
        
        test_index = []
        train_val_index=[]

        for match_id in self.metadata['match_id'].unique():
            match = self.metadata[self.metadata['match_id']==match_id]

            rally_index = match['rally_id'].unique()
            np.random.shuffle(rally_index) 
            train_num = int(len(rally_index) * 0.6)
            valid_num = int(len(rally_index) * 0.2)
            train_val_num = train_num + valid_num

            train_val_index.extend(rally_index[:train_val_num])
            test_index.extend(rally_index[train_val_num:])

        train_val_index = np.array(train_val_index)
        test_index = np.array(test_index)

        assert len(np.intersect1d(train_val_index, test_index)) == 0, "Overlap detected between train_val_rally_ids and test_rally_ids!"
        test_rally_data = self.metadata[self.metadata['rally_id'].isin(test_index)].reset_index(drop=True)
        test_rally_data.to_csv(os.path.join(path, 'test.csv'), index=False)

        # 初始化 KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=22)
        fold_datasets = []
        
        # 基于 rally_ids 进行普通 K-Fold 分割
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_index)):
            train_rally_ids = train_val_index[train_idx]
            val_rally_ids = train_val_index[val_idx]
        
            # 提取训练和验证数据
            train_rally_data = self.metadata[self.metadata['rally_id'].isin(train_rally_ids)].reset_index(drop=True)
            valid_rally_data = self.metadata[self.metadata['rally_id'].isin(val_rally_ids)].reset_index(drop=True)
            
            # 检查数据泄漏
            assert len(np.intersect1d(train_rally_ids, val_rally_ids)) == 0, "Overlap detected between train_rally_ids and val_rally_ids!"
            
            # 检查 player_id 分布
            print(f"Fold {fold + 1} Train player distribution:\n", train_rally_data['player'].value_counts(normalize=True).sort_index())
            print(f"Fold {fold + 1} Validation player distribution:\n", valid_rally_data['player'].value_counts(normalize=True).sort_index())

            train_rally_data.to_csv(os.path.join(path, f'train_fold_{fold+1}.csv'), index=False)
            valid_rally_data.to_csv(os.path.join(path, f'val_fold_{fold+1}.csv'), index=False)

        # 檢查驗證集和測試集中的玩家是否出現在訓練集中
        for fold in range(1, 6):
            train_data = pd.read_csv(os.path.join(path, f'train_fold_{fold}.csv'))
            val_data = pd.read_csv(os.path.join(path, f'val_fold_{fold}.csv'))
            train_players = set(train_data['player'].unique())
            val_players = set(val_data['player'].unique())
            print(f'========== Fold {fold} Val not in Train =========')
            print(val_players - train_players)
        test_players = set(test_rally_data['player'].unique())
        print('========== Test not in Train =========')
        print(test_players - train_players)

        # 後續預處理
        self.preprocess_files()

    def preprocess_files(self):
        for fold in range(1, 6):
            train_data = pd.read_csv(os.path.join(self.path, f'train_fold_{fold}.csv'))
            val_data = pd.read_csv(os.path.join(self.path, f'val_fold_{fold}.csv'))
            test_data = pd.read_csv(os.path.join(self.path, 'test.csv'))

            for data, name in [(train_data, f'train_fold_{fold}'), (val_data, f'val_fold_{fold}'), (test_data, 'test')]:
                data['rally_length'] = data.groupby(['match_id', 'rally_id'])['rally_id'].transform('count')
                data = data[data['rally_length'] >= self.given_strokes_num + 1]
                data = data.drop(['server'], axis=1, errors='ignore')
                data.to_csv(os.path.join(self.path, f'{name}.csv'), index=False)

        # 打印每個集合的唯一比賽數量
        for fold in range(1, 6):
            train_data = pd.read_csv(os.path.join(self.path, f'train_fold_{fold}.csv'))
            val_data = pd.read_csv(os.path.join(self.path, f'val_fold_{fold}.csv'))
            print(f'Fold {fold} - Train rallies: {train_data["rally_id"].nunique()}, Val rallies: {val_data["rally_id"].nunique()}')
        test_data = pd.read_csv(os.path.join(self.path, 'test.csv'))
        print(f'Test rallies: {test_data["rally_id"].nunique()}')


if __name__ == "__main__":
    path = "./data/"
    data_processor = PreDataProcessor(path=path)
    data_processor.compute_statistics()

    data_splitter = CoachAITrainTestSplit(path=path)