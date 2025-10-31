import os
import shutil
import pandas as pd


file_path = '/Users/luanr/pycharm/FedAlzheimer/FedAlzheimer/ADNI1_Complete_1Yr_3T_10_11_2024 (1).csv'
df = pd.read_csv(file_path)

image_dir = '/Users/luanr/pycharm/FedAlzheimer/ADNI 4'

def organize_files_by_group():
    for index, row in df.iterrows():
        image_id = row['Image Data ID']
        group = row['Group']   
        
        group_folder = os.path.join(image_dir, group)
        os.makedirs(group_folder, exist_ok=True)
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if image_id in file and file.endswith('.nii'):
                    source = os.path.join(root, file)
                    
                    destination = os.path.join(group_folder, file)
                    
                    shutil.move(source, destination)
                    print(f"Movido: {file} -> {destination}")
                    break

organize_files_by_group()
