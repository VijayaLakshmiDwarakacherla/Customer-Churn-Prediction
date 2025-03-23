# Customer Churn Prediction

## Problem Statement
Customer churn is a critical issue for businesses, leading to revenue loss and reduced market share. The goal of this project is to develop a machine learning model that predicts whether a customer will churn (discontinue service) based on their historical usage, demographic information, and subscription details. This will help businesses take proactive steps to retain high-risk customers and optimize their retention strategies.

## Dataset Description
The dataset used for this project is `Churn_Modelling.csv`, which contains customer details and subscription-related information. The key features in the dataset include:

- **CustomerId**: Unique identifier for each customer (Dropped during preprocessing).
- **Surname**: Customer's last name (Dropped during preprocessing).
- **CreditScore**: Credit score of the customer.
- **Geography**: Customer's country (e.g., France, Spain, Germany).
- **Gender**: Male or Female.
- **Age**: Customer's age.
- **Tenure**: Number of years the customer has been with the company.
- **Balance**: Customer's account balance.
- **NumOfProducts**: Number of products the customer has with the company.
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: Customer's estimated salary.
- **Exited (Target Variable)**: 1 if the customer churned, 0 otherwise.

## Technologies Used
### **Programming Language**
- **Python**: Used for data analysis, preprocessing, and machine learning.

### **Libraries and Tools**
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical operations.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-Learn**: Machine learning models and evaluation.
- **XGBoost & GradientBoosting**: Advanced ensemble models.
- **TensorFlow/Keras**: Neural networks for deep learning models.

## Exploratory Data Analysis (EDA)
- **Checked for missing values and duplicate records**.
- **Performed statistical summary of numerical and categorical features**.
- **Plotted a heatmap to visualize correlations among numerical features**.
- **Dropped unnecessary columns (`CustomerId`, `Surname`)**.

## Machine Learning Models Used
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **AdaBoost & Gradient Boosting**
- **XGBoost**
- **Neural Networks (TensorFlow & Keras)**

## Model Evaluation Metrics
- **Accuracy**: Overall correctness of the model.
- **Precision, Recall, and F1-score**: To assess model performance for class imbalance.
- **Confusion Matrix**: To analyze true positives, false positives, etc.
- **ROC Curve & AUC**: To measure classification performance.

## Data Processing Techniques
- **Feature Scaling**: StandardScaler used to normalize numerical features.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) for feature selection.
- **Hyperparameter Tuning**: GridSearchCV used for optimizing model parameters.
- **Cross-Validation**: Evaluates model generalization on unseen data.
- **Early Stopping & ModelCheckpoint**: Used for preventing overfitting in neural networks.

## Outcome
By building an effective customer churn prediction model, businesses can identify high-risk customers and take targeted actions to retain them. This project provides insights into customer behavior, enabling companies to enhance customer satisfaction, reduce churn rates, and optimize business strategies. The final trained model can be deployed to predict customer churn in real-time, helping businesses improve retention and maximize revenue.

## How to Run the Project
1. **Clone the repository**:
   ```bash
   git clone https://github.com/VijayaLakshmiDwarakacherla/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```
2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```
