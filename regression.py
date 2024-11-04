import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your original data
data = pd.read_csv('./Saavnify_data_cleaned.csv')

# Define feature names from the training phase (excluding target)
music_rating_feature_names = [
    'Age',
    'primary_location',
    'Gender',
    'Saavnify_usage_period',
    'Saavnify_listening_device',
    'Saavnify_subscription_plan',
    'premium_sub_willingness',
    'preffered_premium_plan',
    'preferred_listening_content',
    'fav_music_genre',
    'music_time_slot',
    'music_Influencial_mood',
    'music_lis_frequency',
    'music_expl_method',
    'pod_lis_frequency',
    'fav_pod_genre',
    'preffered_pod_format',
    'pod_host_preference',
    'preffered_pod_duration'
]

# Create a function to train the model
def train_model(data):
    # Separate features and target
    X = data[music_rating_feature_names]
    y = data["music_recc_rating"]

    # Handle categorical features
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X.loc[:, column] = le.fit_transform(X[column])  # Use .loc to avoid SettingWithCopyWarning
        label_encoders[column] = le

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a Random Forest Regressor
    reg_model = RandomForestRegressor(random_state=42)  # Set random_state for reproducibility
    reg_model.fit(X_scaled, y)

    return reg_model, scaler, label_encoders

# Train the model when the script runs
reg_model, scaler, label_encoders = train_model(data)

# Function to predict music recommendation rating
def predict_music_rating(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for column in label_encoders:
        if column in input_df.columns:
            input_df[column] = label_encoders[column].transform(input_df[column])
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Predict using the trained model
    return reg_model.predict(input_scaled)

# Create the Streamlit app for regression
st.title("Music Recommendation Rating Prediction")

# Create input fields for the user
age = st.selectbox("Age", options=data['Age'].unique())
primary_location = st.selectbox("Primary Location", options=data['primary_location'].unique())
gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])
saavnify_usage_period = st.selectbox("Saavnify Usage Period", options=data['Saavnify_usage_period'].unique())
saavnify_listening_device = st.selectbox("Saavnify Listening Device", options=data['Saavnify_listening_device'].unique())
subscription_plan = st.selectbox("Saavnify Subscription Plan", options=data['Saavnify_subscription_plan'].unique())
premium_sub_willingness = st.selectbox("Willingness to Pay for Premium Subscription", options=data['premium_sub_willingness'].unique())  # Assuming binary (Yes/No)
preferred_premium_plan = st.selectbox("Preferred Premium Plan", options=data['preffered_premium_plan'].unique())
preferred_listening_content = st.selectbox("Preferred Listening Content", options=data['preferred_listening_content'].unique())
fav_music_genre = st.selectbox("Favorite Music Genre", options=data['fav_music_genre'].unique())
music_time_slot = st.selectbox("Preferred Music Time Slot", options=data['music_time_slot'].unique())
music_influencial_mood = st.selectbox("Influential Mood While Listening", options=data['music_Influencial_mood'].unique())
music_lis_frequency = st.selectbox("Listening Frequency (times per week)", options=data["music_lis_frequency"].unique())
music_expl_method = st.selectbox("Music Exploration Method", options=data['music_expl_method'].unique())
pod_lis_frequency = st.selectbox("Podcast Listening Frequency (times per week)", options=data["pod_lis_frequency"].unique())
fav_pod_genre = st.selectbox("Favorite Podcast Genre", options=data['fav_pod_genre'].unique())
preferred_pod_format = st.selectbox("Preferred Podcast Format", options=data['preffered_pod_format'].unique())
pod_host_preference = st.selectbox("Preferred Podcast Host", options=data['pod_host_preference'].unique())
preferred_pod_duration = st.selectbox("Preferred Podcast Duration (minutes)", options=data["preffered_pod_duration"].unique())

# Prepare the input data for prediction
input_data = {
    "Age" : age,
    'primary_location': primary_location,
    'Gender': gender,
    'Saavnify_usage_period': saavnify_usage_period,
    'Saavnify_listening_device': saavnify_listening_device,
    'Saavnify_subscription_plan': subscription_plan,
    'premium_sub_willingness': premium_sub_willingness,
    'preffered_premium_plan': preferred_premium_plan,
    'preferred_listening_content': preferred_listening_content,
    'fav_music_genre': fav_music_genre,
    'music_time_slot': music_time_slot,
    'music_Influencial_mood': music_influencial_mood,
    'music_lis_frequency': music_lis_frequency,
    'music_expl_method': music_expl_method,
    'pod_lis_frequency': pod_lis_frequency,
    'fav_pod_genre': fav_pod_genre,
    'preffered_pod_format': preferred_pod_format,
    'pod_host_preference': pod_host_preference,
    'preffered_pod_duration': preferred_pod_duration
}

if st.button("Predict Music Rating"):
    # st.write("Input data collected:")
    # st.write(input_data)

    try:
        music_rating = predict_music_rating(input_data)
        st.write(f"Predicted Music Recommendation Rating: {music_rating[0]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
