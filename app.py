import os
import pickle
import pandas as pd
import shap
from flask import Flask, render_template, request, redirect, url_for, session
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session handling

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Dummy login credentials
USERNAME = 'admin'
PASSWORD = 'password'

# Home / login page
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('index.html', error=error, logged_in=False)

# Dashboard page
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', logged_in=True, prediction_text=None, shap_plot_url=None, word_explanation=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Collect form data
    form = request.form
    data = {key: int(value) if value.isdigit() else value for key, value in form.items()}
    df = pd.DataFrame([data])

    # Feature engineering
    df['Total_Satisfaction'] = (
        df['EnvironmentSatisfaction'] +
        df['JobInvolvement'] +
        df['JobSatisfaction'] +
        df['RelationshipSatisfaction'] +
        df['WorkLifeBalance']
    ) / 5
    df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)
    df.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance','Total_Satisfaction'], axis=1, inplace=True)
    df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0); df.drop('Age', axis=1, inplace=True)
    df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0); df.drop('DailyRate', axis=1, inplace=True)
    df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0); df.drop('Department', axis=1, inplace=True)
    df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0); df.drop('DistanceFromHome', axis=1, inplace=True)
    df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0); df.drop('JobRole', axis=1, inplace=True)
    df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0); df.drop('HourlyRate', axis=1, inplace=True)
    df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0); df.drop('MonthlyIncome', axis=1, inplace=True)
    df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0); df.drop('NumCompaniesWorked', axis=1, inplace=True)
    df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0); df.drop('TotalWorkingYears', axis=1, inplace=True)
    df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0); df.drop('YearsAtCompany', axis=1, inplace=True)
    df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0); df.drop('YearsInCurrentRole', axis=1, inplace=True)
    df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0); df.drop('YearsSinceLastPromotion', axis=1, inplace=True)
    df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0); df.drop('YearsWithCurrManager', axis=1, inplace=True)

    # Make prediction
    prediction = model.predict(df)[0]
    prediction_text = 'Employee Might Leave The Job' if prediction == 1 else 'Employee Might Not Leave The Job'

    # SHAP plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    
    shap_plot_path = 'static/shap_plot.png'
    plt.figure()
    shap.summary_plot(shap_values, df, show=False)
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()

    shap_plot_url = '/' + shap_plot_path
    word_explanation = "SHAP plot shows feature contributions to the prediction."

    return render_template('index.html',
                           logged_in=True,
                           prediction_text=prediction_text,
                           shap_plot_url=shap_plot_url,
                           word_explanation=word_explanation)

# Logout
@app.route('/logout')
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))

if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
