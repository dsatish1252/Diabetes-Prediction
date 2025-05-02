import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load models and scaler
scaler = joblib.load('path/to/scaler.pkl')
log_clf = joblib.load('path/to/logistic_model.pkl')
rf_clf = joblib.load('path/to/random_forest_model.pkl')
gb_clf = joblib.load('path/to/gradient_boost_model.pkl')
knn_clf = joblib.load('path/to/knn_model.pkl')


# Load dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Session state
if 'nav' not in st.session_state:
    st.session_state.nav = 'Prediction'
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Sidebar with dropdown
with st.sidebar:
    st.title("🔍 Select Page")
    nav_choice = st.selectbox("Go to", ["Prediction", "EDA", "About"], index=["Prediction", "EDA", "About"].index(st.session_state.nav))
    st.session_state.nav = nav_choice

    if st.session_state.nav == 'Prediction':
        st.markdown("---")
        gender = st.selectbox('Gender', ['Male', 'Female'], help="Select your gender")
        age = st.number_input('Age', 18, 100, 30, help="Enter your age (18 to 100)")
        hypertension = st.selectbox('Hypertension', [0, 1], help="0 = No, 1 = Yes")
        heart_disease = st.selectbox('Heart Disease', [0, 1], help="0 = No, 1 = Yes")
        smoking_history = st.selectbox('Smoking History',
                                    ['never', 'No Info', 'former', 'current', 'not current', 'ever'],
                                    help="Select your smoking history status")
        bmi = st.slider('BMI', 10.0, 50.0, 25.0, help="Body Mass Index (10.0 to 50.0)")
        hba1c = st.slider('HbA1c Level', 4.0, 14.0, 5.5, help="HbA1c Level (4.0 to 14.0)")
        blood_glucose = st.slider('Blood Glucose Level', 50.0, 300.0, 100.0, help="Blood Glucose (50.0 to 300.0)")



        if st.button("Predict Now"):
            gender_map = {'Male': 0, 'Female': 1}
            smoke_map = {'never': 0, 'No Info': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5}
            features = [gender_map[gender], age, hypertension, heart_disease,
                        smoke_map[smoking_history], bmi, hba1c, blood_glucose]
            scaled = scaler.transform(np.array(features).reshape(1, -1))
            probs = (log_clf.predict_proba(scaled) + rf_clf.predict_proba(scaled) +
                     gb_clf.predict_proba(scaled) + knn_clf.predict_proba(scaled)) / 4
            pred = np.argmax(probs, axis=1)[0]
            st.session_state.predicted = True
            st.session_state.prediction_result = pred

# Main content area
if st.session_state.nav == 'Prediction':
    st.title("🩺 Diabetes Prediction")
    if st.session_state.predicted:
        if st.session_state.prediction_result == 0:
            st.success("✅ You are not likely to have diabetes.")
        else:
            st.error("⚠️ You are likely to have diabetes. Please consult a doctor.")
    else:
        st.info("Please enter details in the sidebar and click 'Predict Now'.")

elif st.session_state.nav == 'EDA':
    st.title("📊 Exploratory Data Analysis")

    st.subheader("🔍 Sample Data")
    st.write(df.head())

    st.subheader("📌 Basic Statistics")
    st.write(df.describe())

    st.subheader("📈 Age Distribution")
    st.plotly_chart(px.histogram(df, x='age', nbins=20))

    st.subheader("👫 Gender Distribution")
    st.plotly_chart(px.pie(df, names='gender'))

    st.subheader("⚖️ BMI Distribution")
    st.plotly_chart(px.histogram(df, x='bmi', nbins=20))

    st.subheader("🧪 HbA1c vs Blood Glucose Level")
    st.plotly_chart(px.scatter(df, x='HbA1c_level', y='blood_glucose_level', color='diabetes'))

    st.subheader("🔗 Correlation Heatmap")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(fig)



elif st.session_state.nav == 'About':
    st.subheader('About the Diabetes Prediction Web App')

    st.markdown("""
    ### Overview
    This web application allows you to predict the likelihood of diabetes based on multiple input features such as age, BMI, HbA1c levels, blood glucose levels, and medical history.

    ### Functionality
    - **Prediction:** You can input your medical information and the app will predict if you are likely to have diabetes.
    - **Exploratory Data Analysis (EDA):** The app provides various visualizations to help users understand the relationships and distributions in the diabetes dataset.

    ### Creator Information
    - **Creator:** Satish Dasu
    - **Email:** isatishdasu@gmail.com
    - **LinkedIn:** [Satish Dasu LinkedIn](https://www.linkedin.com/in/satish-dasu-3599b2265/)
    - **GitHub:** [Satish Dasu GitHub](https://github.com/dsatish1252)
    - **Course:** IT Automation with Python (Coursera, Google)
    - **Full Stack Web Developer & Machine Learning Enthusiast**

    ### Technology Stack
    - **Backend:** Python, scikit-learn
    - **Frontend:** Streamlit, Plotly, Matplotlib, Seaborn
    - **Machine Learning Models:** Logistic Regression, Random Forest, Gradient Boosting, KNN

    ### Input Feature Descriptions:
    - **Age:** Your age (Range: 18 to 100 years)
    - **Gender:** The gender of the patient (Male/Female)
    - **Hypertension:** Indicates whether the patient has hypertension (0 = No, 1 = Yes)
    - **Heart Disease:** Indicates whether the patient has heart disease (0 = No, 1 = Yes)
    - **Smoking History:** The smoking history of the patient (Options: never, No Info, former, current, not current, ever)
    - **BMI:** Body Mass Index (Range: 10 to 50)
    - **HbA1c Level:** The HbA1c level (Range: 4.0 to 14.0%)
    - **Blood Glucose Level:** The blood glucose level (Range: 50 to 300 mg/dL)
    """)
