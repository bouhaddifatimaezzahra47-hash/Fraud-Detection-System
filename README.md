This personal project focuses on comparing three machine learning models for credit card fraud detection:
Logistic Regression, Random Forest, and Gradient Boosting .
It covers the complete workflow, from data preprocessing and model training to performance evaluation and feature
importance analysis.
 each part of this code demonstrates how to:

- Explore data
- Clean data (missing values & duplicated values)
- Analyze data & study distribution of normal and fraudulent transactions
- Separate the features & target variable
- Analyze outliers & apply RobustScaler(after justification)
- Split the data into training & testing sets
- Balance the training data using BorderlineSMOTE
- Tuning hyperparameters & training & determine the optimal classification threshold of the three machine learning models
- Evaluate each model using Accuracy, Precision, Recall, F1-score,Confusion Matrix
- Compare the performance of all models.
- Select the best-performing model.
- Use SHAP to identify the most important features.
____________________________________________________
Built with :

- programming language : Python
- Data manipulation and analysis : Pandas and Numpy
- Machine learning models and Evaluation : Scikit-learn
- Borderline-SMOTE :Imbalanced-learn
- Data visualization : Matplotlib
- Feature Importance : SHAP
- Model loading : Joblib
- Web interface: Gradio
______________________________________________________
Dataset:

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- description of data:The dataset contains transactions made by credit cards in September 2013 by European cardholders.
                      This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807
                      transactions.The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all
                      transactions.
- Features: Time, Amount, V1...V28(the principal components obtained with PCA)
-Target : class=0--> normal
          class=1--> fraud











