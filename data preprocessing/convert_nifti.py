import os
import subprocess

nifti_dir = '/Users/luanr/pycharm/TF/ADNI/MCI'
output_dir = '/Users/luanr/pycharm/TF/ADNI/MCI'

for file in os.listdir(nifti_dir):
    if file.endswith('.nii') or file.endswith('.nii.gz'):
        input_path = os.path.join(nifti_dir, file)
        output_file = f"{os.path.splitext(file)[0]}.jpg"
        output_path = os.path.join(output_dir, output_file)
        subprocess.run(['med2image', '-i', input_path, '-d', output_path, '-o', "sample", '-t', 'jpg', '-s', 'm'])
