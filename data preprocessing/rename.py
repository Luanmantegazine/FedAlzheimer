import os
import pandas as pd

file_path = '/Users/luanr/pycharm/TF/ADNI1_Complete_1Yr_3T_10_11_2024 (1).csv'  
df = pd.read_csv(file_path)

image_dir = '/Users/luanr/pycharm/TF/ADNI'  

def rename_files():
    for index, row in df.iterrows():
        image_id = row['Image Data ID']
        subject = row['Subject']
        modality = row['Modality']
        description = row['Description'].replace('; ', '_').replace(' ', '_') 
        acq_date = pd.to_datetime(row['Acq Date']).strftime('%Y%m%d%H%M%S') 
        
        new_file_name = f"{image_id}.nii"
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if image_id in file and file.endswith('.nii'):
                    source = os.path.join(root, file)
                    
                    if file != new_file_name:
                        new_file_path = os.path.join(root, new_file_name)
                        os.rename(source, new_file_path)
                        print(f"Renomeado: {file} -> {new_file_name}")
                    else:
                        print(f"O arquivo {file} já está no formato correto.")
                    break
rename_files()
