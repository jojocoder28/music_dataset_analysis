import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import google.generativeai as genai

# Load data and fill NaN values
data = pd.read_csv("Saavnify_data_cleaned.csv")
data.fillna(value="None", inplace=True)

# Feature engineering to create user profile
features = [
    'fav_music_genre', 'music_time_slot', 'music_Influencial_mood', 
    'music_lis_frequency', 'fav_pod_genre', 'preffered_pod_format', 
    'pod_host_preference', 'preffered_pod_duration'
]
data['user_profile'] = data[features].agg(' '.join, axis=1)

# Configure Gemini API
genai.configure(api_key="AIzaSyC1oWuMicF5Vm9SRGoBZdD_MY5bBskX1nA")  # Replace with your actual Gemini API key

# TF-IDF Vectorizer for retrieval
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['user_profile'])

def retrieve_similar_profiles(profile_details, top_n=5):
    """
    Retrieves the top N similar profiles based on cosine similarity.
    """
    profile_vec = tfidf.transform([profile_details])
    cosine_similarities = cosine_similarity(profile_vec, tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
    return data.iloc[similar_indices][['user_profile']]

# Sample function to get Gemini-generated recommendations
def get_gemini_recommendation(profile_details):
    prompt = f"Suggest songs or podcasts for a user with profile: {profile_details}. Suggest mostly bollywood songs"
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Saavnify Music and Podcast Recommendation System")

# Input options for users to select their preferences
user_genre = st.selectbox("Favorite Music Genre", data['fav_music_genre'].unique())
user_mood = st.selectbox("Music Influential Mood", data['music_Influencial_mood'].unique())
user_time_slot = st.selectbox("Preferred Listening Time Slot", data['music_time_slot'].unique())

# Create a temporary profile from selected preferences
profile_details = f"{user_genre} {user_mood} {user_time_slot}"

# Display Recommendations
if st.button("Get Recommendations"):
    # Retrieve similar profiles for RAG
    similar_profiles = retrieve_similar_profiles(profile_details)
    
    # Combine retrieved profiles into a single prompt for Gemini
    retrieved_text = " ".join(similar_profiles['user_profile'])
    gemini_prompt = f"Based on similar profiles: {retrieved_text}, suggest personalized songs or podcasts for a user profile: {profile_details}."
    
    # Generate recommendations with Gemini
    recommendation = get_gemini_recommendation(gemini_prompt)
    st.write("Gemini Recommendations:")
    st.write(recommendation)
