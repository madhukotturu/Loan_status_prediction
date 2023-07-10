import json
import numpy as np
import pandas as pd
import pickle

with open('ft_cols(1).json', 'r') as I:
    ftrs = json.load(I) ['features']
def get_data_colmns():
    return {'Features' : ftrs}


with open('Loan_apprvl_status.pickle','rb')as f:
    logistic_model = pickle.load(f)

    
def predict_loan_status(Gender, Married, Dependents, Education, Self_Employed,
                        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
                        Credit_History, Property_Area):


    input_val = np.zeros(len(logistic_model.coef_[0]))

    
    input_val[0] = Gender
    input_val[1] = Married
    input_val[2] = Dependents
    input_val[3] = Education
    input_val[4] = Self_Employed


    
    input_val[5] = ApplicantIncome
    input_val[6] = CoapplicantIncome
    input_val[7] = LoanAmount
    input_val[8] = Loan_Amount_Term
    input_val[9] = Credit_History
    input_val[10] = Property_Area


    
    prediction = logistic_model.predict([input_val])


    
    return prediction[0]
    
    
predicted_status = predict_loan_status(0,0,0,0,1,293,0,86,9,0,1)
print(predicted_status)

