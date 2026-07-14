# BBB Permeability Prediction using 3D Graph Neural Networks and Ensemble Learning

Prediction of Blood–Brain Barrier (BBB) permeability (logBB) using a hybrid machine learning framework that combines **3D Graph Neural Networks (SchNet)** and **ExtraTrees** regression.

This repository contains the source code developed for my Master's thesis in Biomedical Engineering at the **University of Palermo**, focused on molecular property prediction for drug discovery.

---

## Methodological Pipeline

<p align="center">
  <img src="docs/pipeline.png" alt="Methodological Pipeline" width="1000">
</p>

The proposed workflow combines complementary molecular representations to improve continuous logBB prediction.

The pipeline consists of:

- Molecular preprocessing and descriptor extraction with **RDKit**
- Training of a **SchNet** model on 3D molecular graphs
- Training of an **ExtraTrees** model on selected molecular descriptors
- Multi-seed and cross-validation ensemble generation
- Weighted hybrid ensemble using Out-of-Fold predictions
- Final evaluation on an independent hold-out test set

---

# Overview

The Blood–Brain Barrier (BBB) represents one of the major challenges in central nervous system drug development.

Experimental determination of BBB permeability is expensive, time-consuming and often unavailable during the early stages of drug discovery. Machine learning models can therefore provide valuable support by estimating BBB permeability directly from molecular structure.

This project investigates the prediction of **continuous logBB values** through the integration of graph neural networks and descriptor-based machine learning models.

---

# Dataset

The project combines publicly available BBB permeability datasets into a single regression dataset containing:

- **1303 molecules**
- Experimental **logBB** values
- 3D molecular conformations
- RDKit molecular descriptors

The processed dataset is **not included** in this repository due to licensing and academic restrictions.

---

# Methodology

## SchNet

The graph neural network was developed using **PyTorch Geometric**.

The model exploits:

- 3D atomic coordinates
- Atomic number embeddings
- 15 engineered node features
- 9 selected global molecular descriptors
- Batch Normalization
- PReLU activation
- Dropout regularization
- Early stopping
- 5-fold cross-validation
- Multi-seed training (3 random seeds)

---

## ExtraTrees

The descriptor-based model was developed using **Scikit-learn**.

Main steps include:

- RDKit descriptor extraction
- Sequential Forward Feature Selection (SFS)
- Standardization
- Hyperparameter optimization using GridSearchCV
- Ensemble averaging across folds

---

## Hybrid Ensemble

The final prediction combines SchNet and ExtraTrees outputs through a weighted linear ensemble:

<p align="center">

**ŷ = α · ŷ<sub>SchNet</sub> + (1 − α) · ŷ<sub>ExtraTrees</sub>**

</p>

The ensemble coefficient (α) is selected **only using Out-of-Fold predictions**, preventing information leakage from the hold-out test set.

Nested cross-validation is then performed to evaluate the robustness of the hybrid strategy.

---

# Results

The hybrid ensemble consistently achieved better predictive performance than the individual models by combining graph-based structural information with handcrafted molecular descriptors.

The final evaluation was performed on an independent hold-out test set using:

- R²
- RMSE
- MAE

| Model | R² | RMSE | MAE |
|------|----:|----:|----:|
| ExtraTrees | *(insert final value)* | *(insert)* | *(insert)* |
| SchNet | *(insert final value)* | *(insert)* | *(insert)* |
| **Hybrid Ensemble** | **0.5161** | **0.6974** | **0.4254** |

---

## Hybrid Predictions

<p align="center">
<img src="docs/scatter_hybrid.png" width="700">
</p>

Comparison between experimental and predicted logBB values on the independent hold-out test set.

---

## Ensemble Weight Optimization

<p align="center">
<img src="docs/alpha_curve.png" width="700">
</p>

Selection of the optimal hybrid weight (α) using Out-of-Fold predictions.

---

## SchNet Training

<p align="center">
<img src="docs/training_curve.png" width="700">
</p>

Example of training and validation curves showing early stopping during model optimization.

---

# Repository Structure

```text
.
├── data/
│   └── README.md
│
├── docs/
│   ├── pipeline.png
│   ├── scatter_hybrid.png
│   ├── alpha_curve.png
│   └── training_curve.png
│
├── models/
│
├── results/
│
├── src/
│   ├── feature_selection/
│   │   └── extratrees_feature_selection.py
│   │
│   ├── models/
│   │   ├── schnet.py
│   │   ├── schnet_multiseed_ensemble.py
│   │   ├── extratrees_ensemble.py
│   │   ├── hybrid_extratrees_schnet.py
│   │   └── schnet_common.py
│   │
│   └── config.py
│
├── README.md
├── requirements.txt
└── RUNNING.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/bbb-permeability-prediction.git

cd bbb-permeability-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Execute the scripts in the following order:

```bash
python -m src.feature_selection.extratrees_feature_selection

python -m src.models.extratrees_ensemble

python -m src.models.schnet

python -m src.models.schnet_multiseed_ensemble

python -m src.models.hybrid_extratrees_schnet
```

---

# Technologies

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

# Future Improvements

Possible future developments include:

- evaluation on larger molecular datasets
- comparison with additional Graph Neural Network architectures
- uncertainty estimation
- explainable AI methods
- web deployment for virtual screening applications

---

# Author

**Marco Gambino**

M.Sc. Biomedical Engineering  
University of Palermo

- LinkedIn: https://www.linkedin.com/in/marco-gambino-7a3071138/
- GitHub: https://github.com/marcgambi

---

## Disclaimer

This repository is intended for research and educational purposes only.

The dataset used during the thesis is not redistributed.
