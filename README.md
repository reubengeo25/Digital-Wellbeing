# Stress Level Prediction Using Machine Learning

## Overview
This project leverages **machine learning and data preprocessing techniques** to predict stress levels based on user behavior metrics such as screen time, social media usage, sleep patterns, and physical activity. The dataset undergoes **data cleaning, feature engineering, and normalization** before training classification models to categorize stress levels.

## Key Features
- **Data Preprocessing:** Handled missing values, encoded categorical variables, and applied normalization using **Pandas, NumPy, and Scikit-learn**.  
- **Feature Engineering:** Created interaction terms and optimized feature selection for better model accuracy.  
- **Machine Learning Models:** Trained **Random Forest and XGBoost classifiers**, fine-tuned hyperparameters using **RandomizedSearchCV**, and improved accuracy to **77%**.  
- **Model Evaluation:** Used **classification reports, precision-recall metrics, and correlation heatmaps** to assess model performance.  

## Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn, XGBoost, Random Forest  
- Matplotlib, Seaborn (for data visualization)  
- SMOTE (for handling class imbalance)  
- Joblib (for model persistence)  

## How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/stress-prediction-ml.git
   cd stress-prediction-ml
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the Jupyter Notebook for training:  
   ```bash
   jupyter notebook train.ipynb
   ```  

## Future Improvements
- Expand dataset with real-world behavioral data  
- Implement deep learning models (LSTMs) for sequential stress prediction  
- Deploy as a web API for real-time stress level analysis
