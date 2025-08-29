# Road Accident Analysis
This project studies how seatbelt usage, seating position, and vehicle type influence injury severity in road traffic accidents. Using a real-world dataset sourced from Victorian crash records, the goal is to assess whether these variables hold predictive value in understanding injury outcomes and informing safety strategies.

## Overview

The workflow involves structured data preprocessing, distributional analysis, and correlation evaluation using Normalized Mutual Information (NMI) and Pearson coefficients. Supervised learning models—including Logistic Regression, K-Nearest Neighbor, and Decision Tree—are applied to predict injury severity. Additional features such as hospital admission status are incorporated to explore improvements in model accuracy and robustness.

## File Structure

The required CSV files `person.csv` and `vehicle.csv` must be placed inside the `dataset/` directory:

```
project/
├── road_accident_analysis.ipynb
├── dataset/
│   ├── person.csv
│   └── vehicle.csv
└── README.md
```

## How to run
This project is implemented entirely in a single Jupyter Notebook: `road_accident_analysis.ipynb`.
### Prerequisites

Ensure you have Python 3.7+ installed, and the following libraries available:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `statistics`

You can install them using:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### To run the notebook:

Open the notebook in **VS Code (with Jupyter extension)** or **Jupyter Notebook**, and execute all cells in order:

1. Launch your notebook interface
   
2. Open `road_accident_analysis.ipynb`
   
3. Click **"Run All"** or run cells sequentially from top to bottom
   

The notebook will:

- Perform end-to-end data preprocessing including filtering, encoding, and aggregation

- Visualize missing data and feature distributions

- Apply correlation analysis using Normalized Mutual Information and Pearson correlation 

- Train and evaluate three machine learning models: Logistic Regression, KNN, and Decision Tree

- Compare model performance with and without hyperparameter tuning

- Investigate the impact of additional features such as `TAKEN_HOSPITAL` on prediction accuracy

- Generate and save all output plots and printed metrics for further interpretation

## Output

### Preprocessing Outputs

- Missing Data Count 
  
- Variable Distributions – Pie charts for:
  
    - INJ_LEVEL
      
    - SEATBELT
      
    - SEATING_POSITION
      
    - VEHICLE_TYPE
    
- Descriptive Statistics 
  
- Label Encoding Output 
  

### Correlation and Causal Analysis

- Normalised Mutual Information – Bar chart comparing NMI scores for features vs. INJ_LEVEL
  
- Correlation Heatmap – Pearson correlation heatmap between selected variables and injury level
  
- Class Proportion Plot – Bar chart showing relative frequency of each INJ_LEVEL class
  

### Supervised Learning Models and Evaluation

For each model (Logistic Regression, K-Nearest Neighbor, and Decision Tree):

- Best Parameters 
  
- Model Accuracy – Printed results for both with and without hyperparameter tuning
  
- Optimal Confusion Matrix
  
- Confusion Matrix Comparison (with and without grid search)
  
- Classification Report – Two printed reports per model including:
  
    - Accuracy
      
    - Macro average
      
    - Weighted average
      

### Further Analysis (TAKEN_HOSPITAL Variable)

- Missing Data Count
  
- Heatmap – Relationship between TAKEN_HOSPITAL and INJ_LEVEL
  
- Correlation Heatmap – Pearson heatmap between TAKEN_HOSPITAL and INJ_LEVEL
  
- Model Comparisons (All Three Models) – Repeated LR, KNN, and DT with TAKEN_HOSPITAL included:
  
    - Best parameters
      
    - Model accuracy
      
    - Classification reports
      
    - Confusion matrices
