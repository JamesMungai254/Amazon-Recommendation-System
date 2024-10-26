import os
# os.system("pip install numpy==1.26.4")

import streamlit as st
import pandas as pd
from joblib import load
from surprise import Dataset, Reader

# Load the model
model = load('svd_model.joblib')


# Load the data for filtering options
df = pd.read_csv('saurav9786/amazon-product-reviews/versions/1/ratings_Electronics (1).csv',names=['user_id', 'product_id','rating','timestamp']) 

df = df.sample(n=10000, random_state=42)
df = df[['user_id','product_id' ,'rating']]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id','product_id' ,'rating']], reader)

# Function to get recommendations
def get_recommendations(user_id, model, data, top_n=10):
    trainset = data.build_full_trainset()
    user_ratings = trainset.ur[trainset.to_inner_uid(user_id)]
    already_rated = {item_id for item_id, rating in user_ratings}
    
    predictions = [
        (item, model.predict(user_id, item).est)
        for item in trainset.all_items()
        if item not in already_rated
    ]
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    recommended_product_ids = [trainset.to_raw_iid(item) for item, _ in recommendations]
    return recommended_product_ids

# Streamlit UI
st.title("Amazon Product Recommender")
st.sidebar.header("Filter Options")
product_id = st.sidebar.text_input("Product ID")
min_rating = st.sidebar.slider("Minimum Rating", min_value=1, max_value=5, value=3)

# Filtered Data
filtered_df = df[df['product_id'] == product_id] if product_id else df[df['rating'] >= min_rating]
st.write("### Filtered Products")
st.write(filtered_df)

# Get recommendations for a user
user_id = st.text_input("Enter User ID for Recommendations")
if st.button("Get Recommendations"):
    try:
        recommendations = get_recommendations(user_id, model, data)
        st.write("### Recommended Products")
        st.write(recommendations)
    except ValueError:
        st.write("User ID not found in dataset")
