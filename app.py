# app.py
import sklearn
print("scikit-learn version:", sklearn.version)
import streamlit as st
from preprocessing import load_data, preprocess_data
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid_engine import HybridRecommender
from utils import get_unseen_movies

# Load and preprocess data
ratings, movies = load_data('ratings.csv', 'movies.csv')
data = preprocess_data(ratings, movies)

# Initialize models
content_model = ContentBasedRecommender(data)
collab_model = CollaborativeRecommender(data)
hybrid_model = HybridRecommender(content_model, collab_model, data)

# Streamlit UI
st.title("Hybrid Movie Recommendation System")
user_id = st.number_input("Enter your User ID", min_value=1, step=1)
movie_input = st.text_input("Enter a Movie Title You Like", "Toy Story (1995)")
top_n = st.slider("Number of Recommendations", 5, 20, 10)

if st.button("Get Recommendations"):
    recs = hybrid_model.recommend(user_id, movie_input, top_n)
    if recs.empty:
        st.warning("No recommendations found. Check movie title.")
    else:
        st.success("Recommendations:")
        st.dataframe(recs)
