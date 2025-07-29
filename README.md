# Home Loan Data Analysis

## Problem Statement

For a safe and secure lending experience, it's important to analyze past loan data. In this project, we build a deep learning model that predicts whether or not a loan applicant is likely to default, using historical financial data. 

**Objective:**  
Create a deep learning model to predict loan repayment probability using historical borrower data.

**Domain:** 
Finance — Credit Risk Modeling

---

## Dataset

The dataset contains various features describing audio characteristics of songs. Some key features include:

- `credit.policy`: Customer meets company credit underwriting criteria or not.
- `purpose`: Reason for loan.
- `int.rate`: Interest rate as a proportion .
- `installment`, `log.annual.inc`, `dti`, `fico`, `days.with.cr.line`, etc.

> A full **Dataset Dictionary** is provided in the [`data/`](./data/) folder as an Word file: [`dataset_dictionary.docx`](./data/dataset_dictionary.docs)

---

## Technologies & Tools Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for clustering and PCA)

---

## Methodology

1. **Data Loading**
   - Loaded the loan dataset using pandas

2. **Exploratory Data Analysis**
   - Inspected missing values and duplicate rows
   - Examined imbalance in the target column (not.fully.paid)

3. **Data Preprocessing**
   - One-hot encoding for categorical variables
   - SMOTE for handling class imbalance

4. **Model Development**
   - Built a deep learning model using TensorFlow/Keras
   - Included dropout layers and early stopping to reduce overfitting

5. **Model Evaluation**
   - Metrics used:
      - Accuracy
      - Sensitivity (Recall)
      - ROC AUC Score
   - Plotted ROC Curve

---

## Model Architecture (Keras)

```
Sequential([
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

```

---

## Performance Metrics
- Recall (Sensitivity): ~High to avoid false negatives
- ROC AUC: Used for better evaluation under imbalance
- Accuracy: Measured but not relied on solely

---

## Installation & Usage

### Requirements
- Python 3.8+
- Jupyter Notebook
- pandas, numpy, seaborn, matplotlib
- Scikit-learn
- TensorFlow / Keras
- Imbalanced-learn (SMOTE)

### Run the notebook
> git clone https://github.com/avesh-srivastava/dl-home-loan-data-analysis.git
> cd dl-home-loan-data-analysis
> pip install -r requirements.txt
> jupyter notebook notebooks/Home_Loan_Data_Analysis.ipynb

---

## Folder Structure

```
dl-home-loan-data-analysis/
├── data/
│ └── dataset_dictionary.docx
│ └── home_loan_data.csv
├── notebooks/
│ └──Home_Loan_Data_Analysis.ipynb
├── outputs/
│ └──Home_Loan_Data_Analysis.html
├── README.md
└── requirements.txt

```

--- 

## License

This project is for educational purposes only. Feel free to reuse the code with proper attribution.