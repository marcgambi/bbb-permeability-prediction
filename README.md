# BBB Permeability Prediction using 3D Graph Neural Networks and Ensemble Learning

Prediction of Blood-Brain Barrier (BBB) permeability (logBB) using a hybrid machine learning framework combining **3D Graph Neural Networks (SchNet)** and **tree-based ensemble methods (ExtraTrees)**.

This project was developed as part of my Master's thesis in Biomedical Engineering and focuses on molecular property prediction for drug discovery.

---

## Overview

The Blood-Brain Barrier (BBB) is one of the main challenges in central nervous system drug development. Experimental determination of BBB permeability is expensive and time-consuming, making computational prediction an attractive alternative during early-stage drug discovery.

This project investigates the prediction of **continuous logBB values** through the integration of:

- 3D molecular graph representations
- Physicochemical descriptors computed with RDKit
- Graph Neural Networks
- Ensemble learning techniques

The final model combines complementary information extracted from both graph-based and descriptor-based approaches.

---

## Methodology

The proposed workflow consists of five main stages:

1. Molecular preprocessing
2. Feature extraction with RDKit
3. SchNet training on 3D molecular graphs
4. ExtraTrees regression with feature selection
5. Weighted hybrid ensemble for final prediction

```

```
Dataset
    │
    ▼
RDKit descriptors ─────────────► ExtraTrees + SFS
    │
    ▼
3D molecular graphs ───────────► SchNet
                 │
                 ▼
      Weighted Hybrid Ensemble
                 │
                 ▼
          logBB Prediction
```

---

## Models

### SchNet

The graph neural network was trained using:

- 3D atomic coordinates
- atomic number embeddings
- additional node features
- selected global molecular descriptors
- early stopping
- 5-fold cross-validation
- multi-seed ensemble

### ExtraTrees

The descriptor-based model includes:

- RDKit molecular descriptors
- Sequential Forward Feature Selection (SFS)
- GridSearchCV hyperparameter optimization
- ensemble averaging across folds

### Hybrid Ensemble

Final predictions are obtained through a weighted combination of SchNet and ExtraTrees predictions.

The ensemble weight is selected exclusively using Out-of-Fold (OOF) predictions, preventing information leakage from the test set.

---

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- Scikit-learn
- NumPy
- Pandas

---

## Repository Structure

```
.
├── src
│   ├── feature_selection
│   ├── models
│   └── config.py
│
├── data
├── models
├── results
├── docs
│
├── README.md
├── requirements.txt
└── RUNNING.md
```

---

## Dataset

The dataset used in this project combines publicly available BBB permeability data with molecular preprocessing performed during the thesis work.

To comply with licensing restrictions and academic policies, the processed dataset is **not included** in this repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/bbb-permeability-prediction.git

cd bbb-permeability-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the pipeline

The scripts should be executed in the following order:

```bash
python -m src.feature_selection.extratrees_feature_selection

python -m src.models.extratrees_ensemble

python -m src.models.schnet

python -m src.models.schnet_multiseed_ensemble

python -m src.models.hybrid_extratrees_schnet
```

---

## Results

The hybrid ensemble consistently outperformed the individual models, demonstrating that combining graph-based molecular representations with handcrafted molecular descriptors improves prediction performance.

Evaluation was performed using:

- R²
- RMSE
- MAE

on an independent hold-out test set.

---

## Future Improvements

Possible future developments include:

- evaluation on larger public datasets
- comparison with additional GNN architectures
- uncertainty estimation
- explainability techniques (e.g., GNNExplainer)
- deployment as a web application

---

## Author

**Marco Gambino**

M.Sc. Biomedical Engineering

University of Palermo

LinkedIn: https://linkedin.com/in/yourprofile

GitHub: https://github.com/yourusername

---
