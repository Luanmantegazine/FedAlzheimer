import os
import shutil


caminho_pasta_origem = '/Users/luanr/pycharm/FedAlzheimer/FedAlzheimer/ADNI 5/AD_jpg_meio'
caminho_pasta_destino = '/Users/luanr/pycharm/FedAlzheimer/FedAlzheimer/ADNI 5/AD'

if not os.path.exists(caminho_pasta_destino):
    os.makedirs(caminho_pasta_destino)

for pasta in os.listdir(caminho_pasta_origem):
    caminho_completo_pasta = os.path.join(caminho_pasta_origem, pasta)

    if os.path.isdir(caminho_completo_pasta):
        imagens = [f for f in os.listdir(caminho_completo_pasta) if
                   f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if len(imagens) == 1:
            caminho_imagem = os.path.join(caminho_completo_pasta, imagens[0])
            novo_nome = f"{pasta}_{imagens[0]}"
            caminho_imagem_novo = os.path.join(caminho_pasta_destino, novo_nome)

            shutil.move(caminho_imagem, caminho_imagem_novo)
            print(f"Imagem {imagens[0]} movida e renomeada para {novo_nome}")
        else:
            print(f"Pasta {pasta} não contém exatamente uma imagem.")
