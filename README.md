# Alzheimer’s MRI Classification via Federated + Split Learning

Este repositório contém uma implementação de aprendizado federado (Federated Learning, FL) combinado com aprendizado dividido (Split Learning, SL) para classificação de estágios da Doença de Alzheimer (AD) a partir de imagens de ressonância magnética (MRI). Comparamos duas arquiteturas padrão de visão computacional — ResNet-50 e DenseNet-169 — sob diferentes métodos de agregação (FedAvg, FedAvgM, FedProx) e cenários non-IID.

---

## Sumário

1. [Visão Geral](#visão-geral)  
2. [Características Principais](#características-principais)  
3. [Requisitos](#requisitos)  
4. [Estrutura do Projeto](#estrutura-do-projeto)  
5. [Pré-processamento de Dados](#pré-processamento-de-dados)  
6. [Arquitetura e Split Learning](#arquitetura-e-split-learning)  
7. [Configuração do Federated Learning](#configuração-do-federated-learning)  
8. [Métodos de Agregação](#métodos-de-agregação)  
9. [Treinamento & Avaliação](#treinamento--avaliação)  
10. [Resultados & Métricas](#resultados--métricas)  
11. [Como Executar](#como-executar)  
12. [Referências](#referências)  
13. [Licença](#licença)  

---

## Visão Geral

A Doença de Alzheimer (AD) é um distúrbio neurodegenerativo incurável caracterizado por declínio cognitivo progressivo. Modelos de deep learning em grandes bases de MRI podem ajudar no diagnóstico precoce, mas rotular manualmente centenas de milhares de imagens é custoso e há barreiras de privacidade entre instituições.

Este projeto combina:

- **Split Learning**: divisão do modelo em duas “metades” (cliente & servidor), protegendo os dados brutos e balanceando a carga computacional.  
- **Federated Learning**: agregação de pesos treinados localmente por múltiplos clientes sem compartilhar dados.  

Comparamos ResNet-50 e DenseNet-169, variando:
- Número de clientes: **3**, **5** e **7**  
- Tamanhos de batch: **64** vs **32**  
- Agregadores: **FedAvg**, **FedAvgM** (momento no servidor) e **FedProx**  

---

## Características Principais

- **Privacidade por design**: dados de MRI nunca saem do cliente.  
- **Carga balanceada**: cliente executa apenas primeiras camadas, servidor finaliza o forward/backward.  
- **Comparativo arquitetural**: ResNet-50 vs DenseNet-169 em ambientes federados.  
- **Análise de cenários non-IID**: impacto de batch size, número de clientes e método de agregação.  
- **Resultados** documentados em métricas de Acurácia, Recall, Precisão e AUC.

---

## Requisitos

- Python 3.8+  
- PyTorch 1.10+  
- TorchVision  
- NumPy, Pandas  
- scikit-learn  
- tqdm  

Instale com:
```bash
pip install -r requirements.txt
python FedAlzheimer.py
