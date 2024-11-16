
# **Deep Causality Graph Analysis for Tennessee Eastman Process**

## **Project Overview**
This project introduces a robust Deep Causality Graph (DCG) model to analyze the complex interactions in the Tennessee Eastman (TE) chemical process. The DCG model integrates **Gated Recurrent Units (GRU)** and **Multilayer Perceptrons (MLP)** to uncover and quantify causal relationships among process variables. It also incorporates advanced causality metrics and Variable Contribution Index (VCI) analysis for fault detection and diagnosis.

## **Key Features**
- Implementation of a Deep Causality Graph (DCG) model.
- Estimation and visualization of causal matrices.
- Calculation of Variable Contribution Index (VCI) for fault diagnostics.
- Interactive tools for exploring causal relationships.
- Fault detection and diagnosis through causal inference.

## **Directory Structure**
```
.
├── DCG_data/               # TE process dataset files
├── results/                # Output files, including visualizations and metrics
│   ├── causal_matrix.png
│   ├── causality_matrix.csv
│   ├── evaluation_results.npz
│   └── vci_plot.png
├── models/                 # Trained model checkpoints
└── src/                    # Source code for the project
```

## **Installation**
1. Clone the repository:
   ```bash
   git clone [repository-url]
   ```
2. Install required dependencies:
   ```bash
   pip install torch pandas numpy matplotlib seaborn networkx
   ```

## **Data Format**
The input TE process data must be in `.dat` format, containing time-series measurements of 52 process variables. Each file should consist of space-separated values with no headers.

## **Model Architecture**
The Deep Causality Graph (DCG) model is designed to capture both temporal and instantaneous relationships between process variables. The key components include:

### 1. **GRU Feature Extractor**  
   - **Purpose**: Encodes temporal dependencies among process variables.
   - **Structure**: Utilizes Gated Recurrent Units (GRU) to process sequential data, producing latent representations that encapsulate historical dependencies.  

### 2. **MLP Feature Extractor**  
   - **Purpose**: Captures instantaneous relationships between variables.
   - **Structure**: Implements a fully connected neural network to analyze non-sequential correlations, providing a complementary view to the GRU extractor.  

### 3. **Prediction Unit**  
   - **Purpose**: Generates probabilistic predictions of causal impacts.
   - **Structure**: Combines outputs from GRU and MLP layers to construct the causality matrix, offering a probabilistic interpretation of causal relationships.

### 4. **Group Lasso Regularization**  
   - **Purpose**: Enforces sparsity in the causality graph, reducing overfitting and improving interpretability.
   - **Mechanism**: Penalizes grouped coefficients in the learned causal matrix, isolating the most significant causal interactions.

## **Usage**
### **Training the Model**
To train the DCG model on TE process data:
```bash
python train_dcg.py --data_folder DCG_data --epochs 100
```

### **Evaluating Causal Relationships**
To compute and visualize the causality matrix:
```bash
python evaluate_causality.py --model_path models/dcg_model_final.pth
```

### **Calculating Variable Contribution Index (VCI)**
To perform VCI analysis for fault detection:
```bash
python calculate_vci.py --model_path models/dcg_model_final.pth
```

## **Results**
The DCG model effectively captures causal relationships and performs well on the TE process dataset, achieving the following:

### **Key Metrics**
- **Average Model Loss**: -2.0034
- **R² Scores**: > 0.90 for most variables.
- **Identified Fault-Related Variables**: [5, 9, 12, 15, 23, 24, 25, 26, 27, 28, 29, 31].

### **Visualizations**
- **Causality Matrix**: A comprehensive matrix highlighting causal strengths between process variables.
- **VCI Plot**: Visual representation of each variable's contribution to observed faults.

## **Contributing**
We welcome contributions to improve this project. If you wish to contribute:
- Fork the repository.
- Create a feature branch.
- Submit a pull request with detailed explanations.

Please report issues or feature requests by creating an issue in the repository.


## **Acknowledgments**
We extend our gratitude to:
- **Tennessee Eastman Process Simulation Team** for the benchmark data.
- **PyTorch Team** for the deep learning framework.
- Authors of the original DCG paper for their theoretical contributions.

## **Contact**
For questions, feedback, or collaboration, feel free to create an issue or contact us via the repository.
