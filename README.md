# Heart Failure Survival Prediction System

A machine learning system that predicts patient survival outcomes from heart failure using clinical data, achieving 85% accuracy through ensemble methods and synthetic data augmentation. The system helps healthcare professionals identify high-risk patients early by analyzing 13 clinical features from electronic medical records, prioritizing recall to minimize missed diagnoses of at-risk patients.

---

## Problem and Goal

- **Problem:** Cardiovascular diseases cause 17 million deaths annually worldwide, with heart failure representing a critical subset requiring early intervention. Traditional clinical assessment may miss subtle patterns in patient data indicating elevated mortality risk. Existing predictive models suffer from limited training data and fail to optimize for the clinical priority of detecting at-risk patients.
- **Why It Matters:** Early identification of high-risk heart failure patients enables timely interventions and personalized treatment plans. A system prioritizing recall over precision can save lives by ensuring fewer high-risk patients are missed, even at the cost of increased false positives for clinical review.
- **Goal:** Build a machine learning system that predicts heart failure survival outcomes with 85%+ accuracy while maximizing recall to minimize missed diagnoses. Compare 21 machine learning algorithms, address the 299-sample dataset limitation through synthetic data generation, and deliver actionable insights for clinical decision support.

---

## Approach

1. **Data Preprocessing and Exploration:** Analyzed 299 heart failure patient records from Faisalabad Institute of Cardiology & Allied Hospital (Pakistan, 2015) with 13 clinical features, verified data quality (no missing values or duplicates), standardized quantitative features, and analyzed class distribution (68% survival vs 32% death events).

2. **Synthetic Data Augmentation:** Implemented Gaussian Mixture Model (GMM) with 7 components (selected via BIC) to generate 5,000 synthetic patient records, expanding dataset from 299 to 5,299 samples while preserving statistical properties and enforcing binary constraints on categorical variables.

3. **Multi-Algorithm Model Training:** Trained and evaluated 21 machine learning models including Gradient Boosting, Random Forest, AdaBoost (1,000 trees), Neural Networks (Keras Functional API), KNN, Voting Classifier, Decision Trees, and Logistic Regression using 60/20/20 train/validation/test split.

4. **Hyperparameter Tuning and Threshold Optimization:** Performed grid search across epochs, batch sizes, tree depth (4-5), estimators (200-1,000), and decision thresholds (0.25-0.30) to maximize recall while maintaining acceptable precision for mortality prediction.

5. **Model Evaluation:** Compared parametric vs non-parametric approaches and augmented vs non-augmented training strategies using accuracy, precision, recall, loss, confusion matrices, and AUC-PR curves. Non-parametric models achieved 0.98 test loss vs 2.25 for parametric; augmented models achieved 0.99 test loss vs 1.72 for non-augmented.

6. **Unsupervised Learning Exploration:** Applied K-Means, DBSCAN, and Agglomerative Clustering with PCA/SVD dimensionality reduction to detect patient subpopulations. Silhouette scores (<0.25) indicated poor cluster separation, confirming unsupervised methods were not useful for this dataset.

---

## Results

### Technical Deliverables
- **Augmented Gradient Boosting Model:** 85% test accuracy, 77% precision, 63% recall, 0.27 test loss (best overall performer across 21 models)
- **Augmented Voting Classifier:** 85% test accuracy, 75% recall, 71% precision using soft voting with decision threshold of 0.25
- **AdaBoost Ensemble:** 83% test accuracy with 1,000 trees achieving strong precision-recall balance
- **Keras Functional API Neural Network:** 2-layer MLP (16→4 neurons) reaching 90% test accuracy on augmented data
- **GMM Synthetic Data Generator:** 7-component model producing 5,000 synthetic records (17.7x dataset expansion)

### Key Outcomes
- **Average Model Performance:** 82.42% test accuracy (SD 3.17%), 56% recall (SD 10.92%), 72% precision (SD 8.63%) across all models
- **Data Augmentation Effectiveness:** Augmented models outperformed non-augmented by 42% (test loss: 0.99 vs 1.72)
- **Algorithm Performance:** Non-parametric models (tree-based) proved more effective than parametric approaches with 56% lower test loss (0.98 vs 2.25)
- **Clinical Decision Framework:** Optimized decision thresholds (0.25-0.30) to prioritize recall, reducing missed diagnoses while acknowledging models should complement rather than replace clinical judgment

---

## Tech/Methods

**Languages & Frameworks:** Python, TensorFlow/Keras, scikit-learn, pandas, NumPy

**Visualization:** Matplotlib, Seaborn, Altair

**Infrastructure:** Jupyter Notebooks, Google Colab, Git/GitHub

**Methods:** Supervised Learning (Gradient Boosting, Random Forest, AdaBoost, Neural Networks, KNN, Voting Classifier, Decision Trees, Logistic Regression), Unsupervised Learning (K-Means, DBSCAN, Agglomerative Clustering, GMM), Dimensionality Reduction (PCA, SVD), Hyperparameter tuning, Decision threshold optimization, Feature standardization, Synthetic data generation, Binary classification evaluation (accuracy, precision, recall, loss, confusion matrices, AUC-PR curves)

---

## Repo Structure

```
heart-failure-survival-prediction/
├── data/                                    # Clinical datasets
│   ├── heart_failure_clinical_records_dataset.csv  # Original 299-patient dataset
│   ├── train_data.csv                       # Training set (60%, standardized)
│   ├── val_data.csv                         # Validation set (20%, standardized)
│   ├── test_data.csv                        # Test set (20%, standardized)
│   ├── augmented_train_data.csv             # GMM-augmented training (5,299 samples)
│   └── Data Description.xlsx                # Clinical feature documentation
│
├── models/                                  # Model development notebooks
│   ├── EDA.ipynb                            # Step 1: EDA, preprocessing, baseline, augmentation
│   ├── Model Results.ipynb                  # Step 2: 21-model comparison and evaluation
│   └── clustering.ipynb                     # Step 3: Unsupervised learning exploration
│
├── slides/                                  # Presentation materials
│   ├── Heart Failure Survival Prediction - Final Presentation.pptx
│   ├── Heart Failure Survival Prediction - Final Presentation.pdf
│   └── baseline presentation/
│
└── README.md                                # Project documentation
```

---

## Prerequisites

**Software Requirements:**
- Python 3.8+
- Jupyter Notebook or Google Colab

**Required Packages:**
```bash
pip install pandas numpy matplotlib seaborn altair tensorflow scikit-learn
```

**Data:**
- Original dataset and preprocessed splits included in `data/` directory
- Source: Faisalabad Institute of Cardiology & Allied Hospital, Pakistan (2015)
- 299 patients (194 male, 105 female), ages 40-95, average 130-day follow-up

---

## How to Run

### Step 1: EDA and Data Augmentation
**Notebook:** `models/EDA.ipynb`

1. Open notebook in Jupyter or Google Colab
2. Update file paths if needed (change `/content/drive/MyDrive/...` to relative paths for local execution)
3. Run all cells sequentially

**What it does:**
- Loads original dataset and performs EDA (class distribution, correlation analysis)
- Splits data (60/20/20) and standardizes features using training set statistics
- Implements baseline model (73% test accuracy)
- Fits 7-component GMM and generates 5,000 synthetic samples
- Tests initial models (Logistic Regression, Neural Networks)
- Exports preprocessed and augmented datasets to `data/`

---

### Step 2: Comprehensive Model Comparison
**Notebook:** `models/Model Results.ipynb`

1. Ensure Step 1 completed (preprocessed data exists in `data/`)
2. Open notebook and run all cells
3. Note: May take 15-30 minutes to train 21 models

**What it does:**
- Trains tree-based models (Random Forest, Gradient Boosting, AdaBoost, Decision Trees)
- Trains neural networks (Keras Sequential, Functional API)
- Trains other algorithms (KNN, Logistic Regression, Voting Classifier, Bagging)
- Performs hyperparameter tuning (max_depth, n_estimators, epochs, batch_size, k-neighbors)
- Evaluates all models on test set with confusion matrices and PR curves
- Identifies Augmented Gradient Boosting as best performer (85% accuracy, 0.27 loss)

---

### Step 3: Unsupervised Learning (Optional)
**Notebook:** `models/clustering.ipynb`

1. Open notebook and run all cells
2. Analyzes K-Means, DBSCAN, Agglomerative Clustering with PCA/SVD

**Results:** Low silhouette scores (<0.25) indicate clusters have significant overlap, confirming unsupervised methods are not useful for this dataset.

---

### Quick Start
If preprocessed data already exists:
```bash
jupyter notebook "models/Model Results.ipynb"
```
Run all cells to reproduce 21-model comparison and performance metrics.

---

## Notes: Limitations and Next Steps

### Current Limitations:
- **Dataset Size & Generalization:** Limited to 299 patients from two Pakistani hospitals with age bias (40-95 years) and gender imbalance (194M/105F), restricting broader applicability
- **Feature Coverage:** Only 13 clinical features; real-world records contain hundreds of variables (medications, comorbidities, genetic markers)
- **Synthetic Data Quality:** GMM-generated samples may not capture true clinical distributions or edge cases
- **Accuracy Ceiling:** Models did not exceed 90% accuracy across train/validation/test sets
- **Clinical Validation:** Not validated in prospective clinical trials or real-world deployment

### Next Steps:
- **Expand Dataset:** Integrate multi-hospital datasets (MIMIC-III, eICU) to increase sample size to 10,000+ patients and test cross-population generalizability
- **Advanced Architectures:** Experiment with LSTMs for temporal data, attention mechanisms for feature importance, advanced ensemble methods
- **Feature Engineering:** Incorporate BNP levels, NYHA class, Charlson comorbidity index, medication adherence metrics
- **Explainability:** Implement SHAP values or LIME for per-patient risk factor explanations
- **Bias & Fairness Analysis:** Conduct deeper analysis of model performance across demographic subgroups
- **Production Deployment:** Containerize with Docker, build REST API, deploy on cloud infrastructure with monitoring and retraining pipelines
- **Prospective Clinical Trial:** Partner with healthcare institution to measure impact on mortality reduction, hospital readmissions, and resource utilization

---

## Credits / Data / Licenses

### Data Sources:
- **Heart Failure Clinical Records Dataset** - 299 patients, Faisalabad Institute of Cardiology & Allied Hospital, Pakistan (2015)
  - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
  - Citation: Chicco, D., Jurman, G. "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". *BMC Medical Informatics and Decision Making* 20, 16 (2020). https://doi.org/10.1186/s12911-020-1023-5
  - Usage: Academic/research purposes

### Frameworks and Tools:
- **TensorFlow/Keras:** Apache License 2.0
- **scikit-learn:** BSD 3-Clause License
- **pandas, NumPy, Matplotlib, Seaborn, Altair:** BSD/MIT Licenses

### Academic Context:
- **UC Berkeley School of Information** - DATASCI 207: Applied Machine Learning
- **Project Duration:** August 2024 - December 2024 (12-week academic project)
- **GitHub Repository:** https://github.com/JasmolSD/207_007_final_project

---

## Team Members

| Name | Email | LinkedIn |
|------|-------|----------|
| Kent Bourgoing | kent1bp@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/kent-bourgoing/) |
| Sebastian Rosales | sbsrosales11@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/s-rosales/) |
| Jasmol Singh Dhesi | jasmol_dhesi@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/jasmoldhesi/) |
| Sergey Eduardovich Nam | snamatucb@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/sharp0111/) |
| Jason Chang | chan572@ischool.berkeley.edu | [LinkedIn](https://www.linkedin.com/in/jasonchang572/) |

### Individual Contributions:
- **Kent Bourgoing:** Functional API Neural Network, AdaBoost, model result statistics, presentation slides
- **Sebastian Rosales:** Data preprocessing, GMM data augmentation, baseline model, Logistic Regression, Sequential Neural Network, decision boundary exploration
- **Jasmol Singh Dhesi:** K-Means, DBSCAN, Agglomerative Clustering, PCA, SVD
- **Sergey Eduardovich Nam:** Decision Tree, Random Forest, Gradient Boosting, project coordination
- **Jason Chang:** KNN, Majority Vote, Bagging

---

**Note:** This project was developed for academic purposes to demonstrate machine learning applications in healthcare. Models are research prototypes and should not be used for clinical decision-making without proper validation, regulatory approval, and physician oversight. Predictions should complement, not replace, healthcare professionals' clinical judgment.
