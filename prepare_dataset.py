import pandas as pd
def To_Second(time):
    return (pd.to_datetime(time, format="%H:%M:%S").dt.hour * 3600 + \
    pd.to_datetime(time, format="%H:%M:%S").dt.minute * 60 + \
    pd.to_datetime(time, format="%H:%M:%S").dt.second)

def process_dataset(dataset, output_file):

    # roundscore_diff
    dataset['roundscore_diff'] = dataset['roundscore_B'] - dataset['roundscore_A']
    
    # continuous_score 
    dataset["continuous_score"] = 0
    continuous_count = 0

    for i in range(len(dataset)):
        if dataset.loc[i, "rally_id"] ==1:
            continuous_count = 0
        if dataset.loc[i, "getpoint_player"] == "B":
            continuous_count += 1
        elif dataset.loc[i, "getpoint_player"] == "A":
            continuous_count = 0
        dataset.at[i, "continuous_score"] = continuous_count
    
    #is_target_win 
    dataset['is_target_win'] = (dataset['getpoint_player'] == 'B').astype(float)

    # is_target_turn 
    dataset["is_target_turn"] = (dataset["player"] == "B").astype(float)

    #time proportion
    dataset["time"] = pd.to_datetime(dataset["time"], format="%H:%M:%S").dt.hour * 3600 + \
                  pd.to_datetime(dataset["time"], format="%H:%M:%S").dt.minute * 60 + \
                  pd.to_datetime(dataset["time"], format="%H:%M:%S").dt.second

    dataset["time_proportion"] = dataset.groupby(["match_id", "rally_id","set_id"])["time"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)

    dataset.fillna(0, inplace=True)  # fill nan to 0
    dataset.to_csv(output_file, index=False)
    print(f"✅ transfer successfully!")

if __name__ == "__main__":
    input_path = "new_data/olddataset.csv"
    output_path = "new_data/dataset.csv"
    process_dataset(input_path, output_path)
