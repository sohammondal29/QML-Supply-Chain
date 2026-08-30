# QML-Supply-Chain

### Backorder Prediction with Classical & Quantum Machine Learning

A machine learning study that investigates **product backorder prediction in supply chain systems** using both conventional machine learning algorithms and quantum machine learning models.

The project uses a compact set of three supply-chain features and compares the performance of classical classifiers with a **Variational Quantum Classifier (VQC)** and a **Quantum Neural Network Classifier (QNNC)**.

---

## 🔎 Problem Statement

In a supply chain, a product can go on backorder when customer demand cannot be fulfilled immediately due to insufficient available inventory.

Being able to identify products that are likely to go on backorder can support better decisions around:

* Inventory management
* Procurement
* Stock replenishment
* Demand planning
* Order fulfillment

This project treats the problem as a **binary classification task**, where the objective is to predict the value of:

```text
went_on_backorder
```

---

## 🧩 Approach

The project follows a simple experimental pipeline:

```text
Supply Chain Dataset
        │
        ▼
Feature Selection
        │
        ▼
3 Selected Features
        │
        ├─────────────────────┐
        ▼                     ▼
Classical ML              Quantum ML
        │                     │
        ▼                     ▼
8 Classifiers            VQC + QNNC
        │                     │
        └──────────┬──────────┘
                   ▼
             Model Evaluation
                   │
                   ▼
          Performance Comparison
```

The same prediction problem is approached from both classical and quantum perspectives.

---

## 📁 What's Inside the Repository?

```text
QML-Supply-Chain/
│
├── ClassicalML.ipynb
├── VQC.ipynb
├── QNNC.ipynb
│
├── Training_Top3Features.csv
├── Testing_Top3Features.csv
│
├── catboost_info.zip
│
└── README.md
```

### Notebooks

| File                | Purpose                                        |
| ------------------- | ---------------------------------------------- |
| `ClassicalML.ipynb` | Training and comparison of classical ML models |
| `VQC.ipynb`         | Variational Quantum Classifier experiments     |
| `QNNC.ipynb`        | Quantum Neural Network Classifier experiments  |

### Data

| File                        | Description   |
| --------------------------- | ------------- |
| `Training_Top3Features.csv` | Training data |
| `Testing_Top3Features.csv`  | Testing data  |

---

# 📦 Dataset

The experiments use two prepared CSV files containing three selected features.

### Dataset split

| Split     |    Samples |
| --------- | ---------: |
| Training  |     10,000 |
| Testing   |      4,000 |
| **Total** | **14,000** |

### Input variables

The models operate on:

```text
national_inv
forecast_3_month
in_transit_qty
```

The prediction target is:

```text
went_on_backorder
```

Therefore, each sample can be represented as:

```text
X = [national_inv, forecast_3_month, in_transit_qty]

y = went_on_backorder
```

The reduced three-feature representation is particularly useful for the quantum experiments because it allows the input to be encoded using a small number of qubits.

---

# 🖥️ Classical Baseline

Before evaluating quantum models, the project establishes a classical ML baseline.

The following algorithms are implemented in `ClassicalML.ipynb`:

### Tree / Ensemble Models

* CatBoost
* XGBoost
* LightGBM
* Random Forest

### Other Classifiers

* Artificial Neural Network
* Support Vector Machine
* K-Nearest Neighbors
* Decision Tree

The classical notebook uses **GridSearchCV** to explore model hyperparameters and uses cross-validation during model selection.

This provides a reference point for evaluating how the quantum approaches perform on the same prediction task.

---

# ⚛️ Quantum Experiments

The quantum side of the project consists of two separate implementations.

## Variational Quantum Classifier

Notebook:

```text
VQC.ipynb
```

The VQC experiments investigate how different quantum circuit configurations influence classification performance.

### Circuit components

**Feature maps**

* Pauli Feature Map
* ZZ Feature Map

**Variational ansatz**

* RealAmplitudes

**Entanglement structures**

* Linear
* Full
* Circular

The circuit depth can be modified through:

```text
Feature Map Repetitions
Ansatz Repetitions
```

The optimization is performed using **L-BFGS-B**.

---

## Quantum Neural Network Classifier

Notebook:

```text
QNNC.ipynb
```

The QNNC implementation uses Qiskit Machine Learning to construct a trainable quantum neural network.

The implementation includes:

* `SamplerQNN`
* `TorchConnector`
* `StatevectorSampler`
* Parameterized feature maps
* Real-Amplitudes ansatz
* Configurable entanglement

The trainable circuit parameters are optimized through the PyTorch training pipeline.

### Current QNNC environment

```text
Qiskit                  2.5.2
Qiskit Machine Learning 0.9.1
```

> **Note:** QNNC is treated here as a quantum model implementation. This repository does not claim a separate hybrid-model benchmark.

---

# 📊 Experimental Results

## Classical Models

The strongest classical result in the experiments comes from the Artificial Neural Network.

| Model         |   Accuracy |         F1 |    ROC AUC |
| ------------- | ---------: | ---------: | ---------: |
| **ANN**       | **81.55%** | **0.8112** | **0.8762** |
| XGBoost       |     81.35% |     0.8111 |     0.8737 |
| LightGBM      |     81.25% |     0.8080 |     0.8740 |
| CatBoost      |     81.17% |     0.8046 |     0.8740 |
| SVM           |     80.80% |     0.8050 |     0.8678 |
| Decision Tree |     80.10% |     0.7937 |     0.8606 |
| KNN           |     80.07% |     0.7881 |     0.8428 |
| Random Forest |     79.55% |     0.7860 |     0.8543 |

### Best classical result

```text
ANN

Accuracy : 81.55%
F1 Score : 0.8112
ROC AUC  : 0.8762
```

---

# 🧪 VQC Results

For the recorded VQC experiment using:

```text
Feature Map      : Pauli
Feature Map Reps : 1
Ansatz Reps      : 1
Entanglement     : Linear
```

the model achieved approximately:

| Metric    |    Result |
| --------- | --------: |
| Accuracy  | **67.7%** |
| F1 Score  | **0.682** |
| Precision | **0.672** |
| Recall    | **0.694** |

The VQC experiment demonstrates that a small parameterized quantum circuit can learn the binary classification task, although its recorded performance is below the classical baselines.

---

# ⚖️ Classical vs Quantum

One of the main purposes of the repository is to make this comparison explicit.

```text
                    TEST ACCURACY

ANN                  ████████████████████  81.55%
XGBoost              ████████████████████  81.35%
LightGBM             ████████████████████  81.25%
CatBoost             ████████████████████  81.17%

VQC                  █████████████████     67.7%
```

### What this tells us

For the current dataset and experimental configuration:

**Classical models perform better than the evaluated VQC configuration.**

That is an important result in itself. The purpose of the project is not to assume that quantum models must outperform classical algorithms, but to experimentally investigate their behavior on the same classification problem.

---

# 🧠 Why Use Only Three Features?

Quantum computers currently operate under significant resource constraints, particularly when working with simulated or near-term quantum circuits.

Using three selected features provides a manageable input space:

```text
3 classical features
       ↓
3 quantum input parameters
       ↓
3-qubit circuit
```

This makes it possible to investigate quantum classifiers without immediately introducing a large number of qubits or a very deep circuit.

At the same time, keeping the feature representation consistent makes the classical-vs-quantum comparison easier to interpret.

---

# 🔬 What Was Investigated?

The quantum experiments focus on several circuit-design choices.

### 1. Feature Map

Different feature maps determine how classical data is encoded into the quantum circuit.

```text
Pauli Feature Map
        vs
ZZ Feature Map
```

### 2. Ansatz Depth

The number of repetitions controls the structure and parameter count of the variational circuit.

```text
reps = 1
reps = 2
...
reps = 5
```

### 3. Entanglement

The project investigates:

```text
Linear
Full
Circular
```

entanglement structures.

These choices can significantly affect the behavior and performance of a variational quantum classifier.

---

# 🛠️ Tech Stack

### Machine Learning

```text
Python
NumPy
Pandas
Scikit-learn
XGBoost
LightGBM
CatBoost
PyTorch
```

### Quantum Computing

```text
Qiskit
Qiskit Machine Learning
```

### Development

```text
Google Colab
Git
GitHub
```

---

# ⚙️ Setup

Clone the repository:

```bash
git clone https://github.com/sohammondal29/QML-Supply-Chain.git

cd QML-Supply-Chain
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

For the QNNC experiments, the notebook uses:

```bash
pip install "qiskit==2.5.2"
pip install "qiskit-machine-learning==0.9.1"
```

Open the required notebook.

---

# ▶️ Running the Experiments

## Classical ML

Open:

```text
ClassicalML.ipynb
```

Run the notebook to:

1. Load the prepared datasets.
2. Separate features and target.
3. Train the classical classifiers.
4. Perform hyperparameter search.
5. Evaluate the selected models.
6. Compare their classification metrics.

---

## VQC

Open:

```text
VQC.ipynb
```

The notebook can be used to experiment with different:

* Feature maps
* Entanglement structures
* Feature-map repetitions
* Ansatz repetitions

---

## QNNC

Open:

```text
QNNC.ipynb
```

The notebook contains the quantum neural network implementation and its PyTorch-based training procedure.

> Quantum circuit simulation can be significantly slower than conventional ML, especially when increasing the dataset size, circuit depth, number of training epochs, or number of experiments.

---

# 📐 Evaluation

The models are evaluated using standard classification metrics.

### Accuracy

Overall proportion of correct predictions.

### Precision

How many samples predicted as backorders were actually backorders.

### Recall

How many actual backorders were successfully detected.

### F1 Score

Harmonic mean of precision and recall.

### ROC-AUC

Measures discrimination between the two classes across different thresholds.

### Confusion Matrix

Used to inspect:

```text
True Positives
True Negatives
False Positives
False Negatives
```

---

# 💭 Takeaways

A few conclusions emerge from the current experiments:

### Classical ML remains highly competitive

The classical models achieve around 80%+ test accuracy, with the ANN reaching the highest recorded performance.

### Small quantum circuits can solve the task to some extent

The VQC achieves meaningful classification performance despite using only three input features and a small variational circuit.

### Quantum advantage is not demonstrated

The current experiments do **not** establish a quantum advantage. On this dataset and configuration, the classical models outperform the tested quantum approach.

### Circuit design matters

Feature-map choice, ansatz depth and entanglement topology can influence quantum model performance considerably.

---

# 🚧 Limitations

The current project has several limitations:

* Only three features are used for the quantum experiments.
* Quantum models are evaluated primarily through circuit simulation.
* The available quantum resources restrict circuit size and depth.
* The classical models benefit from highly mature optimization algorithms.
* The experiments do not demonstrate a quantum computational speedup.
* Results depend on the selected circuit configuration and training procedure.

These limitations are important when interpreting the classical-vs-quantum comparison.

---

# 🔮 Possible Extensions

Future experiments could explore:

* More sophisticated quantum feature maps
* Quantum kernel methods
* Additional QML classifiers
* Larger feature subsets
* Noise-aware simulations
* Error mitigation
* Real quantum hardware
* More extensive circuit-depth experiments
* Additional supply-chain datasets
* Statistical comparison over multiple random seeds

A particularly interesting direction would be investigating whether carefully engineered quantum models can close the gap with classical methods as the feature representation and circuit architecture become richer.

---

# 📚 Project Motivation

The broader motivation behind this work is to explore how emerging quantum computing techniques could eventually be applied to practical machine learning problems.

Supply chain management provides an interesting test case because it involves:

* Large numbers of products
* Complex demand patterns
* Inventory constraints
* Uncertain future requirements
* Cost-sensitive decisions

Backorder prediction provides a concrete classification problem through which classical and quantum approaches can be studied side by side.

---

# 👨‍💻 Author

**Soham Mondal**

IIT Kharagpur

GitHub:
https://github.com/sohammondal29

---

# ⭐ If You Find This Project Interesting

Feel free to explore the notebooks, experiment with different quantum circuit configurations, and compare the results with the classical baselines.

The repository is intended as an experimental starting point for studying **Quantum Machine Learning on real-world supply chain data**.
