import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --------------------------
# 🌟 Streamlit UI
# --------------------------

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🔮 Customer Churn Prediction</h1>
    <p style='text-align: center;'>Use this smart predictor to identify customers likely to churn.</p>
    <hr style='border: 1px solid #f0f0f0;'>
    """,
    unsafe_allow_html=True
)

# Input form
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, value=30)
    credit_score = st.number_input('💳 Credit Score', value=650)

with col2:
    balance = st.number_input('💰 Balance', value=0.0)
    estimated_salary = st.number_input('📈 Estimated Salary', value=50000.0)
    tenure = st.slider('⌛ Tenure (years)', 0, 10, value=3)
    num_of_products = st.slider('📦 Number of Products', 1, 4, value=1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', ['No', 'Yes'])
    is_active_member = st.selectbox('✅ Active Member?', ['No', 'Yes'])

# Convert string inputs to numeric
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Merge encoded features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale features
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Result
st.markdown("### 🔎 Prediction Result")
st.progress(int(prediction_proba * 100))

st.markdown(f"Churn Probability: **{prediction_proba * 100:.2f}%**")

if prediction_proba > 0.5:
    st.markdown(f"""
    <div style='background-color:#ffdddd; padding:15px; border-radius:10px; text-align:center;'>
        <h3 style='color:red;'>⚠️ High Risk of Churn</h3>
        <p>Predicted Churn Probability: <strong>{prediction_proba * 100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style='background-color:#ddffdd; padding:15px; border-radius:10px; text-align:center;'>
        <h3 style='color:green;'>✅ Low Risk of Churn</h3>
        <p>Predicted Churn Probability: <strong>{prediction_proba * 100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<hr><p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
