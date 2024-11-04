import streamlit as st

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Music Rating Prediction", "Pod Variety Satisfaction Prediction", "Music Recommendation"))

if page == "Music Rating Prediction":
    import regression 
elif page == "Music Recommendation":
    import recommendation
else:
    import classification
