import pickle
import random
import streamlit as st
import sklearn as sc


model = pickle.load(open('/Users/robinfeder/Desktop/Thesis_Project/ML/trained_model.pkl', 'rb'))


def main():
    st.title('Thesis Project')

    # input Variables
    Loan_ID = st.text_input('Loan ID')
    # Loan_ID = str(random.randint(1000,9999))
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

    # Prediction code
    if st.button('Submit'):
        prediction = model.predict([[Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Applicant_Income,
                                     Co_Applicant_Income, Loan_Amount, Loan_Amount_Term, Credit_History,
                                     Property_Area]])
        output = prediction[0]
        if output == 1:
            st.success('Congratulations! You are eligible for a loan.')
        else:
            st.error('Sorry, you are not eligible for a loan.')


if __name__ == '__main__':
    main()
