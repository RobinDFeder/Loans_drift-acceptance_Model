import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained machine learning model
model = pickle.load(open('/Users/robinfeder/Desktop/Thesis_Project/ML/trained_model.pkl', 'rb'))

def main():
    st.title('Loan Eligibility Prediction')

    # Create a button to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with loan data", type="csv")

    # If a file was uploaded, display its contents
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write('File contents:')
        st.write(data)

        # Use the pre-trained model to make predictions on the uploaded data
        predictions = model.predict(data)
        st.write('Predictions:')
        st.write(predictions)

        # Check for drift by comparing the predictions to a baseline
        # (Assuming that we have a baseline saved in a CSV file named "baseline.csv")
        baseline = pd.read_csv("baseline.csv")
        drift = np.mean(predictions != baseline['Loan_Status']) > 0.05
        if drift:
            st.warning('Warning: The model appears to have drifted!')
        else:
            st.success('The model is performing well.')

        # If there is no drift, allow the user to input new loan data for prediction
        if not drift:
            st.write('\n\nEnter the following information for the loan applicant:')
            Loan_ID = st.text_input('Loan ID')
            Gender = st.selectbox('Gender', ['Male', 'Female'])
            Married = st.selectbox('Marital Status', ['Yes', 'No'])
            Dependents = st.selectbox('Number of Dependents', ['0', '1', '2', '3+'])
            Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
            Self_Employed = st.selectbox('Self Employed', ['Yes', 'No'])
            Applicant_Income = st.text_input('Applicant Income')
            Co_Applicant_Income = st.text_input('Co-Applicant income')
            Loan_Amount = st.text_input('Loan Amount')
            Loan_Amount_Term = st.text_input('Loan Amount Term')
            Credit_History = st.selectbox('Credit History', ['0', '1'])
            Property_Area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])

            # Use the pre-trained model to make a prediction on the new data
            if st.button('Submit'):
                new_data = [[Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Applicant_Income,
                             Co_Applicant_Income, Loan_Amount, Loan_Amount_Term, Credit_History,
                             Property_Area]]
                prediction = model.predict(new_data)[0]
                if prediction == 1:
                    st.success('Congratulations! You are eligible for a loan.')
                else:
                    st.error('Sorry, you are not eligible for a loan.')


if __name__ == '__main__':
    main()
