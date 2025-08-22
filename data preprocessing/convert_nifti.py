import os
from med2image.med2image.med2image import med2image_nii

nifti_dir = '/Users/luanr/pycharm/FedAlzheimer/FedAlzheimer/ADNI 5/CN'
output_dir_base = '/Users/luanr/pycharm/FedAlzheimer/FedAlzheimer/ADNI 5/CN_jpg_meio'

os.makedirs(output_dir_base, exist_ok=True)
print(f"Lendo arquivos de: {nifti_dir}")
print(f"Salvando imagens em: {output_dir_base}\n")


for file in os.listdir(nifti_dir):
    if file.endswith('.nii') or file.endswith('.nii.gz'):
        print(f"Processando arquivo: {file}...")

        input_path = os.path.join(nifti_dir, file)
        output_file_stem = os.path.splitext(os.path.splitext(file)[0])[0]

        patient_output_dir = os.path.join(output_dir_base, output_file_stem)
        os.makedirs(patient_output_dir, exist_ok=True)

        try:

            converter = med2image_nii(
                inputFile       = input_path,
                outputDir       = patient_output_dir,
                outputFileStem  = f'{output_file_stem}.jpg',
                outputFileType  = 'jpg',
                sliceToConvert  = 'm'  # 'm' para a fatia do meio (middle)
            )

            converter.run()

            print(f"  -> Sucesso! Imagem salva em '{patient_output_dir}'")

        except Exception as e:
            print(f"  -> ERRO ao converter o arquivo {file}: {e}")

print("\nProcesso concluído.")