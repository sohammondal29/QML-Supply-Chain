# Quantum Machine Learning for Supply Chain Backorder Prediction

A comparative study of **classical machine learning** and **quantum machine learning (QML)** approaches for predicting product backorders in supply chain systems.

This project investigates whether quantum machine learning models can provide meaningful predictive performance on a real-world supply chain classification problem, and compares their results against several widely used classical machine learning algorithms.

---

## 📌 Project Overview

Backorders occur when customer demand cannot be immediately fulfilled because the required product is unavailable in inventory. Accurately predicting potential backorders can help organizations improve:

* Inventory planning
* Stock availability
* Procurement decisions
* Demand management
* Customer fulfillment
* Supply chain efficiency

This project formulates **backorder prediction as a binary classification problem** and evaluates both classical and quantum machine learning models.

The workflow uses a reduced feature space containing **three selected features**, allowing the quantum models to operate with a small number of qubits while keeping the comparison with classical models consistent.

The project focuses on **benchmarking classical ML against QML**, rather than claiming universal quantum advantage.

---

## 🎯 Objectives

The main objectives of this project are:

1. Predict whether a product will go on backorder.
2. Select a compact feature set suitable for quantum circuit encoding.
3. Implement multiple classical machine learning models.
4. Implement quantum machine learning classifiers using Qiskit.
5. Compare classical and quantum models using common classification metrics.
6. Study the effect of quantum circuit configuration on model performance.
7. Explore the practical use of QML for supply chain prediction problems.

---

## 📊 Dataset

The repository contains separate training and testing datasets.

| Dataset  | Samples | Features |
| -------- | ------: | -------: |
| Training |  10,000 |        3 |
| Testing  |   4,000 |        3 |

The target variable is:

```text
went_on_backorder
```

This is a **binary classification** target.

### Selected Features

The three features used in the repository are:

* `national_inv` — current inventory level
* `forecast_3_month` — forecasted demand over the next three months
* `in_transit_qty` — quantity currently in transit

The testing dataset in the repository confirms these three features along with the `went_on_backorder` target.

### Dataset Files

```text
Training_Top3Features.csv
Testing_Top3Features.csv
```

The training dataset contains 10,000 samples, while the testing dataset contains 4,000 samples.

---

# 🧠 Models Implemented

The project is divided into two main categories:

```text
Classical Machine Learning
        │
        ├── CatBoost
        ├── LightGBM
        ├── XGBoost
        ├── Random Forest
        ├── ANN
        ├── SVM
        ├── KNN
        └── Decision Tree

Quantum Machine Learning
        │
        ├── Variational Quantum Classifier (VQC)
        └── Quantum Neural Network Classifier (QNNC)
```

---

# 💻 Classical Machine Learning

The `ClassicalML.ipynb` notebook implements and compares eight classical classifiers:

### Ensemble Models

* **CatBoost**
* **LightGBM**
* **XGBoost**
* **Random Forest**

### Traditional / Neural Models

* **Artificial Neural Network (ANN)**
* **Support Vector Machine (SVM)**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree**

Hyperparameter tuning is performed using **GridSearchCV with 3-fold cross-validation**, with F1 score used as the model-selection criterion.

---

# ⚛️ Quantum Machine Learning

The repository contains two quantum machine learning implementations.

## 1. Variational Quantum Classifier (VQC)

Implementation:

```text
VQC.ipynb
```

The VQC implementation uses Qiskit's `VQC` algorithm with configurable quantum circuit components.

### Quantum Components

**Feature Maps**

* Pauli Feature Map
* ZZ Feature Map

**Ansatz**

* RealAmplitudes

**Entanglement**

* Linear
* Full
* Circular

**Optimizer**

* L-BFGS-B

**Loss**

* Cross-entropy

The notebook allows the number of feature-map and ansatz repetitions to be varied during experimentation.

### VQC Configuration

The experiments use:

```text
Number of features : 3
Feature-map reps   : 1–5
Ansatz reps        : 1–5
Feature maps       : Pauli / ZZ
Entanglement       : Linear / Full / Circular
Optimizer          : L-BFGS-B
```

---

## 2. Quantum Neural Network Classifier (QNNC)

Implementation:

```text
QNNC.ipynb
```

The QNNC notebook implements a parameterized quantum neural network using **Qiskit Machine Learning** and integrates the quantum model with **PyTorch** for optimization.

The current implementation uses:

* Parameterized quantum circuits
* `SamplerQNN`
* `TorchConnector`
* `StatevectorSampler`
* Pauli feature mapping
* Real-Amplitudes ansatz
* Linear, full and circular entanglement configurations

The notebook currently uses:

```text
Qiskit       : 2.5.2
Qiskit ML    : 0.9.1
```

for the QNNC implementation.

The QNNC experiments are designed to investigate how quantum circuit structure and trainable parameters affect binary backorder classification.

---

# 📈 Classical Model Results

The classical models achieve strong performance on the 4,000-sample test set.

| Model         |   Accuracy |   F1 Score |    ROC AUC |
| ------------- | ---------: | ---------: | ---------: |
| **ANN**       | **81.55%** | **0.8112** | **0.8762** |
| XGBoost       |     81.27% |     0.8100 |     0.8738 |
| LightGBM      |     81.25% |     0.8080 |     0.8740 |
| CatBoost      |     81.17% |     0.8046 |     0.8740 |
| SVM           |     80.80% |     0.8050 |     0.8678 |
| Random Forest |     80.67% |     0.8035 |     0.8658 |
| Decision Tree |     80.10% |     0.7937 |     0.8606 |
| KNN           |     78.02% |     0.7721 |     0.8382 |

These results are generated directly by the classical notebook.

### Best Classical Model

**ANN** achieved the highest test accuracy:

```text
Accuracy : 81.55%
F1 Score : 0.8112
ROC AUC  : 0.8762
```

---

# ⚛️ VQC Results

For the VQC experiment with:

```text
Feature Map     : Pauli
Feature Map Reps: 1
Ansatz Reps     : 1
Entanglement    : Linear
```

the recorded results were:

| Metric    | Training |    Testing |
| --------- | -------: | ---------: |
| Accuracy  |   69.45% | **67.73%** |
| F1 Score  |   69.81% | **68.24%** |
| Recall    |   70.64% | **69.35%** |
| Precision |   69.00% | **67.17%** |

The VQC notebook records the final objective value at approximately `0.9496` after convergence.

---

# 🔬 Classical vs Quantum

The experiments show a clear difference between the classical and quantum approaches on this dataset.

| Category               | Best Recorded Accuracy |
| ---------------------- | ---------------------: |
| **Classical ML — ANN** |             **81.55%** |
| **Quantum ML — VQC**   |             **67.73%** |

The results demonstrate that classical models currently perform better on this particular dataset and experimental setup.

This does **not** imply that quantum machine learning is ineffective. Instead, it provides a practical benchmark for understanding the current performance of small quantum circuits on a supply chain classification task.

---

# 🔍 Key Observations

### Classical Models

* ANN achieved the highest recorded accuracy.
* XGBoost, LightGBM and CatBoost produced competitive results.
* Ensemble methods consistently achieved around 80%+ accuracy.
* Classical models benefit from mature optimization and efficient processing of tabular datasets.

### Quantum Models

* VQC successfully learns the backorder classification task with a compact three-feature representation.
* The choice of feature map, ansatz repetitions and entanglement topology affects performance.
* Quantum models operate with a very small feature dimension, making them suitable for studying QML under limited qubit requirements.
* The current results do not demonstrate quantum advantage over the classical baselines.

---

# 🏗️ Repository Structure

The current repository contains:

```text
QML-Supply-Chain/
│
├── 📓 ClassicalML.ipynb
│   └── Classical ML model training and comparison
│
├── 📓 VQC.ipynb
│   └── Variational Quantum Classifier
│
├── 📓 QNNC.ipynb
│   └── Quantum Neural Network Classifier
│
├── 📊 Training_Top3Features.csv
│   └── 10,000 training samples
│
├── 📊 Testing_Top3Features.csv
│   └── 4,000 testing samples
│
├── 📦 catboost_info.zip
│   └── CatBoost-related artifacts
│
└── 📄 README.md
```

These are the files currently present in the GitHub repository.

---

# 🛠️ Technologies Used

## Classical Machine Learning

* Python
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

## Quantum Machine Learning

* Qiskit
* Qiskit Machine Learning
* PyTorch
* NumPy
* Scikit-learn

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sohammondal29/QML-Supply-Chain.git
cd QML-Supply-Chain
```

Install the classical ML dependencies:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost
```

For the current QNNC implementation:

```bash
pip install "qiskit==2.5.2" "qiskit-machine-learning==0.9.1"
```

The QNNC notebook itself pins these versions.

---

# 🚀 Running the Project

## 1. Classical Models

Open:

```text
ClassicalML.ipynb
```

The notebook:

1. Loads the training and testing CSV files.
2. Separates features and target.
3. Defines eight classical classifiers.
4. Performs GridSearchCV.
5. Trains the best configuration for each model.
6. Generates predictions.
7. Calculates accuracy, F1 score, ROC AUC and confusion matrices.

---

## 2. Variational Quantum Classifier

Open:

```text
VQC.ipynb
```

The notebook allows experimentation with:

```text
Feature Map
    ├── Pauli
    └── ZZ

Entanglement
    ├── Linear
    ├── Full
    └── Circular

Feature Map Repetitions
    └── 1–5

Ansatz Repetitions
    └── 1–5
```

The model is optimized using L-BFGS-B.

---

## 3. Quantum Neural Network

Open:

```text
QNNC.ipynb
```

The notebook contains the QNNC implementation using Qiskit Machine Learning and PyTorch.

> **Note:** Quantum circuit simulation can be computationally expensive, especially as the dataset size, number of epochs, circuit depth, and number of experiments increase.

---

# 📊 Evaluation Metrics

The project evaluates the models using:

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many predicted backorders were actually backorders.

### Recall

Measures how many actual backorders were correctly identified.

### F1 Score

Provides a balance between precision and recall.

### ROC AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### Confusion Matrix

Provides the counts of:

```text
True Positives
True Negatives
False Positives
False Negatives
```

---

# 💡 Why Backorder Prediction?

Backorder prediction is an important supply chain problem because incorrect inventory decisions can lead to:

* Delayed customer orders
* Lost sales
* Excess inventory
* Increased operational costs
* Poor customer satisfaction

A predictive model can help identify products that are likely to experience supply shortages before the backorder occurs.

---

# 🔮 Future Work

Possible directions for extending this project include:

### Quantum Model Improvements

* Larger quantum feature spaces
* More advanced feature maps
* Deeper variational circuits
* Alternative quantum kernels
* Noise-aware experiments
* Error mitigation
* Experiments on real quantum hardware

### Data & Modeling

* Use the complete feature set instead of only three selected features.
* Investigate additional feature-selection techniques.
* Perform larger-scale hyperparameter searches.
* Compare additional classical and quantum classifiers.
* Evaluate robustness across different train/test splits.

### Quantum Hardware

The current experiments primarily focus on quantum circuit simulation. Future work could evaluate the models on available quantum hardware and study the impact of hardware noise and limited connectivity.

---

# 📚 Research Perspective

This project is intended as an experimental comparison between **classical machine learning and quantum machine learning for supply chain backorder prediction**.

The results show that classical models currently provide stronger performance for this particular tabular dataset, while the quantum experiments demonstrate how variational quantum circuits can be applied to the same classification problem.

Rather than assuming quantum advantage, the project uses classical models as practical baselines and investigates where current QML approaches stand relative to them.

---

# 🤝 Contributing

Contributions and improvements are welcome.

Potential areas include:

* New classical ML models
* New quantum classifiers
* Improved feature selection
* Alternative quantum feature maps
* Circuit optimization
* Quantum hardware experiments
* Improved evaluation methodologies
* Additional supply chain datasets

---

# 👨‍💻 Author

**Soham Mondal**

IIT Kharagpur

GitHub:
https://github.com/sohammondal29

---

# ⭐ Acknowledgements

This project makes use of open-source machine learning and quantum computing frameworks, particularly:

* Qiskit
* Qiskit Machine Learning
* PyTorch
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

---

## 📌 Summary

This project explores **backorder prediction in supply chain management using both classical and quantum machine learning**.

The current experiments demonstrate:

```text
Classical ML
      ↓
ANN → 81.55% Accuracy

Quantum ML
      ↓
VQC → 67.73% Accuracy
```

The comparison provides a practical baseline for studying the current capabilities and limitations of quantum machine learning on a real-world tabular classification problem.
