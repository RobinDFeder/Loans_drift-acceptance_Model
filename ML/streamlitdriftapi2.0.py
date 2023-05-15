import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import streamlit as st

#load the original dataset 
original_data = pd.read_csv('LoanApp_Train.csv')

df = original_data
df['Loan_ID'] = df['Loan_ID'].rank(method='dense').astype(int)

original_data.isnull().sum()

original_data = original_data.dropna()

original_data.isnull().sum()

original_data.replace({"Loan_Status":{"N":0, "Y":1}},inplace=True)

original_data['Dependents'].value_counts()

original_data = original_data.replace(to_replace='3+', value=4)

# converting labels values to num values
original_data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True) 

# Split the data into k folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Separate the features and target variable
X = original_data.drop('Loan_Status', axis=1)
y = original_data['Loan_Status']

scaler = StandardScaler()

# Train a random forest classifier using k-fold cross-validation
clf = RandomForestClassifier(n_estimators=100, random_state=42)

accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    accuracy = balanced_accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calculate the average accuracy across all folds
avg_accuracy = sum(accuracies) / len(accuracies)
st.write('Average balanced accuracy:', avg_accuracy)

# Load the new dataset for detecting data drift
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    df = new_data
    df['Loan_ID'] = df['Loan_ID'].rank(method='dense').astype(int)

    new_data.isnull().sum()

    new_data = new_data.dropna()

    new_data.isnull().sum()

    new_data.replace({"Loan_Status":{"N":0, "Y":1}},inplace=True)

    new_data['Dependents'].value_counts()

    new_data = new_data.replace(to_replace='3+', value=4)

    # converting labels values to num values
    new_data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True) 

   
# # Load the new dataset for detecting data drift
# uploaded_file = st.file_uploader("Choose a CSV file")
# if uploaded_file is not None:
#     new_data = pd.read_csv(uploaded_file)
#     df = new_data
#     df['Loan_ID'] = df['Loan_ID'].rank(method='dense').astype(int)

#     new_data.isnull().sum()

#     new_data = new_data.dropna()

#     new_data.isnull().sum()

#     new_data.replace({"Loan_Status":{"N":0, "Y":1}},inplace=True)

#     new_data['Dependents'].value_counts()

#     new_data = new_data.replace(to_replace='3+', value=4)

#     # converting labels values to num values
#     new_data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
#                       'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True) 

    # Separate the features and target variable
    X_new = new_data.drop('Loan_Status', axis=1)
    y_new = new_data['Loan_Status']

    # Scale the features
    X_new_scaled = scaler.transform(X_new)

    # Make predictions on the new dataset
    y_pred_new = clf.predict(X_new_scaled)

    # Calculate the balanced accuracy score on the new dataset
    balanced_accuracy_new = balanced_accuracy_score(y_new, y_pred_new)

    # Compare the predictions with the original target values to detect data drift
    if avg_accuracy - balanced_accuracy_new > 0.1:
        st.write('Data drift detected!')
    else:
        st.write('No data drift detected.')

    # if avg_accuracy - balanced_accuracy_new > 0.1:
    #     st.write('Data drift detected!')
    # else:
    #     st.write('No data drift detected.')

    st.title('Loan Eligibility Checker')

    Loan_ID = st.text_input('Loan ID')
    Gender = st.text_input('Gender')
    Married = st.text_input('Marital Status')
    Dependents = st.text_input('Dependents')
    Education = st.text_input('Education')
    Self_Employed = st.text_input('Self Eployed')
    Applicant_Income = st.text_input('Applicant Income')
    Co_Applicant_Income = st.text_input('Co-Applicant income')
    Loan_Amount = st.text_input('Loan Amount')
    Loan_Amount_Term = st.text_input('Loan Amount Term')
    Credit_History = st.text_input('Credit History')
    Property_Area = st.text_input('Property Area')

    #Prediction code 
    
    # if st.button('Submit'):
    #     prediction = model.predict([[Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Applicant_Income,
    #                                  Co_Applicant_Income, Loan_Amount, Loan_Amount_Term, Credit_History,
    #                                  Property_Area]])

    # # Input Variables
    # Loan_ID = st.text_input('Loan ID')
    # Gender = st.selectbox('Gender', ['Male', 'Female'])
    # Married = st.selectbox('Marital Status', ['No', 'Yes'])
    # Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    # Education = st.selectbox('Education', ['Not Graduate', 'Graduate'])
    # Self_Employed = st.selectbox('Self Employed', ['No', 'Yes'])
    # Applicant_Income = st.number_input('Applicant Income', min_value=0)
    # Co_Applicant_Income = st.number_input('Co-Applicant income', min_value=0)
    # Loan_Amount = st.number_input('Loan Amount', min_value=0)
    # Loan_Amount_Term = st.number_input('Loan Amount Term', min_value=0)
    # Credit_History = st.selectbox('Credit History', ['0', '1'])
    # Property_Area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])

    # # Prediction code
    # if st.button('Check Eligibility'):
    #     input_data = pd.DataFrame({
    #         'Loan_ID': [Loan_ID],
    #         'Gender': [Gender],
    #         'Married': [Married],
    #         'Dependents': [Dependents],
    #         'Education': [Education],
    #         'Self_Employed': [Self_Employed],
    #         'Applicant_Income': [Applicant_Income],
    #         'Co_Applicant_Income': [Co_Applicant_Income],
    #         'Loan_Amount': [Loan_Amount],
    #         'Loan_Amount_Term': [Loan_Amount_Term],
    #         'Credit_History': [Credit_History],
    #         'Property_Area': [Property_Area]
    #     })

        # Preprocess the input data
    if st.button('Check Eligibility'):
        input_data = pd.DataFrame ([[Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Applicant_Income,
                                     Co_Applicant_Income, Loan_Amount, Loan_Amount_Term, Credit_History,
                                     Property_Area]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make predictions
        prediction = clf.predict(input_data_scaled)

        if prediction[0] == 1:
            st.success('Congratulations! You are eligible for a loan.')
        else:
            st.error('Sorry, you are not eligible for a loan.')