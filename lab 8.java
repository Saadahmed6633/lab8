//Code 1

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Step 1: Create a synthetic dataset
# Rows represent users, columns represent movies, and values represent ratings (1-5, or 0 if not rated)
data = {
    'User1': [5, 0, 0, 3],
    'User2': [4, 0, 0, 2],
    'User3': [0, 0, 4, 5],
    'User4': [0, 3, 5, 4],
    'User5': [2, 0, 0, 5],
}

ratings = pd.DataFrame(data)
ratings.set_index('User', inplace=True)

# Step 2: Compute cosine similarity between movies
movie_ratings = ratings.T  # Transpose to make movies the rows
similarity_matrix = cosine_similarity(movie_ratings)
similarity_df = pd.DataFrame(similarity_matrix, index=movie_ratings.index, columns=movie_ratings.index)

# Step 3: Define the recommender function
def recommend_movies(user, ratings_df, similarity_df, top_n=3):
    user_ratings = ratings_df.loc[user]
    rated_movies = user_ratings[user_ratings > 0].index  # Movies rated by the user


// Code 2

# Predict ratings for all unrated movies
movie_scores = {}
for movie in ratings_df.columns:
    if movie not in rated_movies:
        # Weighted sum of similarities for prediction
        relevant_similarities = similarity_df[movie][rated_movies]
        relevant_ratings = user_ratings[rated_movies]
        predicted_rating = np.dot(relevant_similarities, relevant_ratings) / relevant_similarities.sum()
        movie_scores[movie] = predicted_rating

# Sort by predicted rating and return top_n movies
recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
return [movie for movie, score in recommended_movies]

# Step 4: Test the recommender system
user_to_recommend = 'User3'
recommendations = recommend_movies(user_to_recommend, ratings, similarity_df)

print(f"Recommended movies for {user_to_recommend}: {recommendations}")



//code 3

# Predict ratings for all unrated movies
movie_scores = {}

for movie in ratings_df.columns:
    if movie not in rated_movies:
        # Weighted sum of similarities for prediction
        relevant_similarities = similarity_df[movie][rated_movies]
        relevant_ratings = user_ratings[rated_movies]
        predicted_rating = np.dot(relevant_similarities, relevant_ratings) / relevant_similarities.sum()
        movie_scores[movie] = predicted_rating

# Sort by predicted rating and return top_n movies
recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
return [movie for movie, score in recommended_movies]

# Step 4: Test the recommender system
user_to_recommend = 'User3'
recommendations = recommend_movies(user_to_recommend, ratings, similarity_df)

print(f"Recommended movies for {user_to_recommend}: {recommendations}")

