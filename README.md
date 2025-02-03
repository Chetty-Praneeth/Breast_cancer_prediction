# Breast Cancer Prediction using Machine Learning

This project demonstrates the application of machine learning techniques to predict the diagnosis of breast cancer using a dataset containing various features of cell nuclei. The goal is to classify the tumor as either benign (0) or malignant (1) based on the features of the cells.

## What is the Project About?

The project uses a Random Forest Classifier to predict the diagnosis of breast cancer. It involves data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation. Various models like RandomForestClassifier, Support Vector Classifier (SVC), and KNeighborsClassifier were tested for comparison.

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

## Inputs and Outputs:

### Inputs:
- `radius_mean`: Mean radius of the cell nuclei.
- `perimeter_mean`: Mean perimeter of the cell nuclei.
- `area_mean`: Mean area of the cell nuclei.
- `compactness_mean`: Mean compactness of the cell nuclei.
- `concavity_mean`: Mean concavity of the cell nuclei.
- `concave_points_mean`: Mean concave points of the cell nuclei.
- Other statistical features of the cell nuclei.

### Output:
- **Diagnosis**: The classification of the tumor as either benign (`0`) or malignant (`1`).
## Output Description  

Once the user enters the required input features in the **Streamlit UI**, the trained model processes the data and provides a prediction on whether the tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous).**  

### How It Works:
1. The user inputs relevant features such as **radius, perimeter, area, compactness, concavity, etc.**  
2. The trained **Random Forest Classifier** analyzes the input data.  
3. The model outputs a classification result:  
   - **Benign (0):** Indicates that the tumor is likely non-cancerous.  
   - **Malignant (1):** Indicates that the tumor is likely cancerous and may require further medical evaluation.  

### Example Output:
![Prediction Output](image/output.png)


## Fitness Function and Evaluation Metrics:
The model performance is evaluated using:
- **Accuracy**: Percentage of correctly predicted classifications.
- **Precision**: The ratio of correctly predicted benign or malignant tumors.
- **Recall**: The ratio of the correctly predicted malignant tumors.

## Neural Network and Model Details:
- **Model**: Random Forest Classifier.
- **Hyperparameters Tuned**:
  - `max_depth`: The maximum depth of the tree.
  - `n_estimators`: Number of trees in the forest.

## Steps to Run the Code:
1. Clone the Repository:
   ```bash
   git clone https://github.com/YourUsername/BreastCancerPrediction.git
   cd BreastCancerPrediction

2. Install Dependencies: With requirements.txt in your project folder, you can install all the necessary dependencies in one go by running:
pip install -r requirements.txt
3. Replace the data.csv file with your dataset (ensure it has the same structure as expected in the code).
4. Run the Model:
python breast_cancer_prediction.py
## Output:

The program prints the accuracy, precision, recall, and confusion matrix for model evaluation.
The model is saved as model.pkl using joblib for future predictions.
## Other Models Tested:

* SVC (Support Vector Classifier)
* KNeighborsClassifier
* These models were tested for comparison to determine the best performing algorithm.

## Future Improvements:

Experiment with other models like Logistic Regression, Neural Networks, etc.
Tune additional hyperparameters.
Implement data normalization and scaling for better performance.
Deploy the model into a web app or API for real-time prediction.
## License:

This project is licensed under the MIT License - see the LICENSE file for details.