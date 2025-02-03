# Breast Cancer Prediction using Machine Learning

This project uses machine learning models to predict the diagnosis of breast cancer based on data from a CSV file. The dataset contains features such as the radius, perimeter, and area of cell nuclei present in breast cancer biopsies. The goal is to predict whether a tumor is benign (0) or malignant (1).

## Technologies Used
- Python
- Pandas
- Numpy
- Scikit-learn
- Jupyter Notebook
- Joblib (for model serialization)

## Project Overview
This project involves the following key steps:
1. **Data Preprocessing**: 
   - Load the dataset using Pandas.
   - Drop unnecessary columns like 'id' and 'Unnamed: 32'.
   - Map the 'diagnosis' column from 'B' and 'M' to 0 and 1 respectively.
   
2. **Feature Engineering**: 
   - The features are grouped into three categories: 
     - Mean features
     - Standard Error (SE) features
     - Worst features
   
3. **Model Training**: 
   - A Random Forest Classifier is used to predict the diagnosis of breast cancer.
   - Hyperparameters like `max_depth` and `n_estimators` are tuned using GridSearchCV to optimize the model.

4. **Model Evaluation**:
   - The model’s accuracy, precision, and recall scores are calculated.
   - A confusion matrix is used to evaluate the model’s performance.
   
5. **Model Saving**:
   - The trained model is saved using `joblib` for future use.

## Steps to Run the Code
1. Clone this repository or download the Jupyter Notebook file.
2. Make sure to have the required libraries installed. You can install them using:
   ```bash
   pip install pandas numpy scikit-learn joblib

