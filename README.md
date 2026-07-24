# How to Use XGBoost in Python for Cancer Prediction with High Accuracy

This project demonstrates how to use **XGBoost (Extreme Gradient Boosting)** to predict whether a breast tumor is **Benign** or **Malignant**, based on the **Breast Cancer Wisconsin (Original)** dataset. XGBoost is trained, evaluated with a Confusion Matrix and Classification Report, and validated using k-Fold Cross Validation for a robust performance estimate. Feature Importance is also analyzed to understand which cell characteristics drive the model's predictions.

## About the Dataset

The dataset contains **683 samples**, each described by 9 cell nucleus characteristics measured from a digitized image of a breast mass:

| Feature | Description |
|---|---|
| Clump Thickness | Thickness of cell clumps |
| Uniformity of Cell Size | Consistency in cell size |
| Uniformity of Cell Shape | Consistency in cell shape |
| Marginal Adhesion | How well cells stick together |
| Single Epithelial Cell Size | Size of epithelial cells |
| Bare Nuclei | Nuclei not surrounded by cytoplasm |
| Bland Chromatin | Texture of the nucleus |
| Normal Nucleoli | Size/number of nucleoli |
| Mitoses | Rate of cell division |

**Target (`Class`)**: `2` = Benign, `4` = Malignant.

## What this notebook covers

1. **Importing the libraries** — numpy, matplotlib, seaborn, pandas
2. **Importing the dataset** — loading features and target, dropping the patient ID column
3. **Exploratory Data Analysis (EDA)** — `dataset.info()` and `dataset.describe()` to inspect structure and summary statistics
4. **Class Distribution** — visualizing the balance between Benign and Malignant cases
5. **Correlation Heatmap** — visualizing relationships between features and the target
6. **Checking for missing/inconsistent values** — cleaning the `Bare Nuclei` column
7. **Encoding the target variable** — mapping classes to 0/1 for XGBoost compatibility
8. **Splitting the dataset** — into Training set and Test set
9. **Training XGBoost** — fitting an `XGBClassifier` on the Training set
10. **Confusion Matrix & Accuracy** — evaluating on the Test set, with a heatmap visualization
11. **Classification Report** — precision, recall, and F1-score per class
12. **ROC Curve & AUC Score** — evaluating class separability across thresholds
13. **Precision-Recall Curve** — a more informative view for imbalanced classes
14. **Applying k-Fold Cross Validation** — robust performance estimate (mean & standard deviation)
15. **Feature Importance** — visualizing which features most influence predictions

## Tech Stack

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (`train_test_split`, `LabelEncoder`, `cross_val_score`, `confusion_matrix`, `accuracy_score`, `classification_report`, `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`)
- XGBoost (`XGBClassifier`)

## Dataset

`Data.csv` — based on the Breast Cancer Wisconsin (Original) dataset, 683 rows × 10 columns (9 features + target class).

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/khaledAlzeer/xgboost-cancer-prediction.git
   cd xgboost-cancer-prediction
   ```
2. Install the requirements:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
   ```
3. Open the notebook in Jupyter or Google Colab and run all cells.

## Results

- **Exploratory Data Analysis** confirms the dataset's structure, class balance (Benign vs Malignant), and feature correlations before modeling.
- XGBoost achieves high accuracy in distinguishing Benign from Malignant tumors on the Test set.
- **ROC Curve / AUC** and **Precision-Recall Curve** confirm strong class separability, which is critical in a medical diagnosis context.
- **10-fold Cross Validation** confirms the model's performance is both high and stable (low standard deviation).
- **Feature Importance** analysis highlights which cell characteristics (e.g. Uniformity of Cell Size/Shape) are most predictive of malignancy.

## Key Takeaway

XGBoost is a powerful, efficient algorithm for tabular medical data, capable of achieving high accuracy with minimal preprocessing (no feature scaling required). Combined with thorough Exploratory Data Analysis, proper data cleaning, robust evaluation metrics (ROC, Precision-Recall, k-Fold Cross Validation), and Feature Importance analysis, it becomes not just accurate but also interpretable and trustworthy — essential qualities for sensitive applications like cancer diagnosis.

## License

This project is open-source and available under the MIT License.
