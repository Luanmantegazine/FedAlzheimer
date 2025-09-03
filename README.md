# Alzheimer's MRI Classification via Federated and Split Learning

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

This repository contains the official implementation for training and evaluating a novel approach that combines Federated Learning (FL) and Split Learning (SL) for classifying Alzheimer's Disease (AD) stages from Magnetic Resonance Imaging (MRI) scans. We provide a comparative analysis of two state-of-the-art computer vision architectures—**ResNet-50** and **DenseNet-169**—under various aggregation methods (`FedAvg`, `FedAvgM`, `FedProx`) in non-IID data distribution scenarios.

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Results](#-results)
- [License](#-license)
- [Citation](#-citation)

## 🌎 Overview

Alzheimer's Disease (AD) is an incurable neurodegenerative disorder characterized by progressive cognitive decline. While deep learning models trained on large MRI databases show promise for early diagnosis, their development is often hindered by two major challenges: the high cost of manual data annotation and strict data privacy regulations that prevent data sharing between institutions.

This project addresses these challenges by integrating:

1.  **Split Learning (SL)**: The deep learning model is split into two parts. The first few layers run on the client's machine, while the rest of the model runs on a central server. This balances the computational load and ensures that raw data never leaves the client's premises.
2.  **Federated Learning (FL)**: Multiple clients train their local model "splits" and share only the model updates (weights) with the server for aggregation. This allows for collaborative training without centralized data storage.

Our experiments systematically evaluate the performance of ResNet-50 and DenseNet-169 by varying:
- **Number of Clients**: 3, 5, and 7
- **Batch Sizes**: 32 vs. 64
- **Aggregation Algorithms**: `FedAvg`, `FedAvgM` (server-side momentum), and `FedProx`

## ✨ Key Features

-   **Privacy-Preserving by Design**: Raw MRI data remains entirely within the client's local environment.
-   **Balanced Computational Load**: Clients handle only the initial layers, reducing the hardware requirements for participating institutions.
-   **Comprehensive Architectural Comparison**: In-depth analysis of ResNet-50 vs. DenseNet-169 in a federated setting.
-   **Robust Non-IID Analysis**: Investigates the impact of batch size, client count, and aggregation methods on model stability and performance.
-   **Detailed Performance Metrics**: Results are documented using Accuracy, Recall, Precision, and AUC.

## 🛠️ Methodology

The core of our approach is the synergy between Split and Federated Learning. The neural network is "split" at an early layer. Each client institution computes the forward pass up to the split point on its local data. The resulting activations (the "smashed data") are sent to the server. The server completes the forward pass, computes the loss, and sends the gradients back to the split point. The client then completes the backpropagation locally. The server uses Federated Learning algorithms to aggregate the weight updates from the server-side portions of the models from all participating clients.

![A diagram illustrating the workflow would be highly beneficial here.]

## 💾 Dataset

The models were trained and evaluated on pre-processed T1-weighted MRI scans from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** database. The dataset is structured for a 3-class classification task: Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD).

The pre-processed dataset used in this study can be downloaded from here:
[**Download Dataset (Google Drive)**](https://drive.google.com/drive/folders/1gpxGyE9BDlrVlPskj7qk45y3h98sAj7Y?usp=drive_link)

## 🚀 Getting Started

Follow these instructions to set up the environment and run the experiments.

### Prerequisites

-   Python 3.8+
-   PyTorch 1.10+
-   TorchVision
-   NumPy & Pandas
-   scikit-learn
-   tqdm

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Luanmantegazine/your-repo-name.git](https://github.com/Luanmantegazine/your-repo-name.git)
    cd your-repo-name
    ```

2.  It is recommended to create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required packages. (It's a good practice to create a `requirements.txt` file).
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run a training experiment, execute the main script with the desired parameters. For example:

```bash
python train.py \
    --model densenet169 \
    --aggregator fedprox \
    --clients 7 \
    --batch_size 32 \
    --epochs 100