import os
import pandas as pd
from prepare_dataset import process_dataset

def load_all_sets(set_folder):
    all_data = []
    match_id_counter = 1  

    for match_folder in sorted(os.listdir(set_folder)):
        match_path = os.path.join(set_folder, match_folder)

        if os.path.isdir(match_path): 
            match_id = match_id_counter  
            match_id_counter += 1  
            set_id=0
            for file in sorted(os.listdir(match_path)):
                if file.endswith(".csv"):
                    file_path = os.path.join(match_path, file)
                    set_id+=1
                    #read each csv in match floder
                    df = pd.read_csv(file_path)
                    df["match_id"] = match_id  
                    df["set_id"]=set_id
                    df = df.rename(columns={"rally":"rally_id"})
                    all_data.append(df)

    if all_data:
        dataset = pd.concat(all_data, ignore_index=True)
        return dataset
    else:
        print("no csv file")
        return None


set_folder_path = "/Users/yenhsin/Desktop/Shot-Influence/set" 
output_folder = "/Users/yenhsin/Desktop/Shot-Influence/new_data"
output_path = os.path.join(output_folder, "dataset.csv")

# make sure output floder exist
os.makedirs(output_folder, exist_ok=True)

# read all dataset
dataset = load_all_sets(set_folder_path)

if dataset is not None:
    output_path = "new_data/dataset.csv"
    processed_dataset = process_dataset(dataset,output_path)