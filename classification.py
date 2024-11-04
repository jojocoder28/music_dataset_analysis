import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('./Saavnify_data_cleaned.csv')
# Load the saved model, scaler, and encoder
satisfaction_model = joblib.load('./models/pod_variety_satisfaction_model.joblib')
# scaler = joblib.load('./models/scaler.joblib')
scaler = StandardScaler()
label_encoder_y = joblib.load('./models/label_class.joblib')
categorical_features = ['Age',
    'primary_location', 'Gender', 'Saavnify_usage_period', 'Saavnify_listening_device',
    'Saavnify_subscription_plan', 'premium_sub_willingness', 'preffered_premium_plan',
    'preferred_listening_content', 'fav_music_genre', 'music_time_slot',
    'music_Influencial_mood', 'music_lis_frequency', 'music_expl_method',
    'pod_lis_frequency', 'fav_pod_genre', 'preffered_pod_format',
    'pod_host_preference', 'preffered_pod_duration', "music_recc_rating"
]
label_encoders = {}
data_encoded = data.copy()
for feature in categorical_features:
    label_encoder = LabelEncoder()
    data_encoded[feature] = label_encoder.fit_transform(data[feature])
    label_encoders[feature] = label_encoder
# Load data to get unique values for input options

# Define input features
satisfaction_feature_names = [
    'Age', 'primary_location', 'Gender', 'Saavnify_usage_period', 'Saavnify_listening_device',
    'Saavnify_subscription_plan', 'premium_sub_willingness', 'preffered_premium_plan',
    'preferred_listening_content', 'fav_music_genre', 'music_time_slot',
    'music_Influencial_mood', 'music_lis_frequency', 'music_expl_method',
    'pod_lis_frequency', 'fav_pod_genre', 'preffered_pod_format',
    'pod_host_preference', 'preffered_pod_duration', 'music_recc_rating'
]
X = data_encoded[satisfaction_feature_names]
X_scaled = scaler.fit_transform(X)


# Streamlit app interface
st.title("Pod Variety Satisfaction Prediction")

# Input fields
input_data = {}
for feature in satisfaction_feature_names:
    if feature == 'Age':
        input_data[feature] = st.selectbox(feature, options=data[feature].unique())
    else:
        input_data[feature] = st.selectbox(feature, options=data[feature].unique())

# Preprocess and predict function
def predict_pod_variety_satisfaction(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for feature in input_df.select_dtypes(include=['object']).columns:
        input_df[feature] = label_encoders[feature].transform(input_df[feature])
    
    # Scale data
    input_scaled = scaler.transform(input_df)
    
    # Predict
    satisfaction = satisfaction_model.predict(input_scaled)
    satisfaction_encoded = label_encoder_y.inverse_transform(satisfaction)
    return satisfaction_encoded

# Prediction
if st.button("Predict Pod Variety Satisfaction"):
    satisfaction = predict_pod_variety_satisfaction(input_data)
    st.write(f"Predicted Pod Variety Satisfaction: {satisfaction[0]}")
