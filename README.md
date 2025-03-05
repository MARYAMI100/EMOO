# EMOO
# Ensemble Multi-Objective Hyperparameter Optimization for the Classification of Imbalanced Heart Disease Data

This repository contains Python implementations of multi-objective hyperparameter optimization using the Non-dominated Sorting Genetic Algorithm II (NSGA-II) combined with Adaptive Boosting (AdaBoost) for robust classification of imbalanced heart disease datasets. 

## **Overview**
Heart diseases are a leading cause of mortality, and machine learning can aid in their early detection. However, imbalanced datasets and improper hyperparameter tuning can lead to biased models that favor the majority class, reducing sensitivity and specificity. To address this, our approach integrates NSGA-II for multi-objective hyperparameter optimization with AdaBoost for ensemble learning, while also minimizing the standard deviation (STD) of key performance metrics. By reducing variability among accuracy, sensitivity, specificity, and F1-score, our method ensures balanced performance across all metrics, improving model robustness and fairness in handling real-world imbalanced medical datasets.

## **Features**
- **Multi-objective optimization**: Simultaneously maximizes accuracy, sensitivity, specificity, and F1 score while minimizing metric standard deviation.
- **Ensemble learning**: Uses AdaBoost to combine Pareto-optimal classifiers for improved generalization.
- **Hyperparameter tuning**: Optimizes Support Vector Machines (SVM), Multi-Layer Perceptron (MLP), and Random Forest (RF) models.
- **Imbalanced dataset handling**: Uses statistical techniques to ensure fair evaluation across all classes.

## **Files**
- `SVM_NSGAII_AdaBoost.py` - Hyperparameter optimization for SVM using NSGA-II and AdaBoost.
- `MLP_NSGAII_AdaBoost.py` - Hyperparameter optimization for MLP using NSGA-II and AdaBoost.
- `RF_NSGAII_AdaBoost.py` - Hyperparameter optimization for Random Forest using NSGA-II and AdaBoost.
  
## **Installation**
### **Requirements**
Ensure you have Python 3.8+ installed. Install the required dependencies using:

```sh
pip install -r requirements.txt
