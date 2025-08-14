import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
from datetime import datetime
import random
from rapidfuzz import process
import urllib.parse

# -----------------------------
# Configuration
# -----------------------------
API_KEY = "93cba6645f461e73dcdf61d8dddad4de"
DEFAULT_POSTER = "https://via.placeholder.com/150"

# -----------------------------
# Data Loading
# -----------------------------
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')
movies_df = movies_df.merge(credits_df, on='title')

# -----------------------------
# Data Preprocessing
# -----------------------------
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'crew', 'cast', 'vote_average', 'vote_count', 'release_date']]
movies_df.dropna(inplace=True)
movies_df = movies_df.drop_duplicates(subset='title')

def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

def convert_cast(obj, limit=3):
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:limit]]
    except:
        return []

def fetch_director(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']
    except:
        return []

movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df['cast'] = movies_df['cast'].apply(convert_cast)
movies_df['crew'] = movies_df['crew'].apply(fetch_director)
movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())

for col in ['genres', 'keywords', 'cast', 'crew']:
    movies_df[col] = movies_df[col].apply(lambda lst: [item.replace(" ", "") for item in lst])

movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

new_df = movies_df[['movie_id', 'title', 'overview', 'tags', 'vote_average', 'vote_count', 'genres', 'cast', 'crew', 'release_date']].copy()
new_df['overview'] = new_df['overview'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
def extract_year(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").year
    except:
        return None

new_df['release_year'] = new_df['release_date'].apply(extract_year)

ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])
new_df['tags'] = new_df['tags'].apply(stem)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

def get_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={urllib.parse.quote(title)}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else DEFAULT_POSTER
    except:
        return DEFAULT_POSTER

def recommend(movie, genre_filter=[], actor_filter=None, director_filter=None, num_recommendations=5):
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
        sim_scores = sorted(enumerate(similarity[movie_index]), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, _ in sim_scores[1:]:
            rec = new_df.iloc[idx]
            if (not genre_filter or set(genre_filter).intersection(rec["genres"])) and \
               (not actor_filter or any(actor.lower() == actor_filter.lower() for actor in rec["cast"])) and \
               (not director_filter or any(director.lower() == director_filter.lower() for director in rec["crew"])):
                recommendations.append({
                    'title': rec['title'],
                    'overview': rec['overview'],
                    'vote_average': rec['vote_average'],
                    'genres': rec['genres'],
                    'cast': rec['cast'],
                    'crew': rec['crew'],
                    'poster_path': get_poster(rec['title'])
                })
                if len(recommendations) >= num_recommendations:
                    break
        return recommendations
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        return []


def get_trending_movies(num=5):
    try:
        df = new_df[['title', 'overview', 'vote_average', 'vote_count', 'release_date']].copy()
        df.dropna(subset=['vote_average', 'release_date'], inplace=True)  # Ensure no NaNs
        
        if df.empty:
            print("No trending movies found.")
            return []
        
        df = df.sort_values(by='vote_average', ascending=False).head(num)
        
        trending_movies = []
        for _, row in df.iterrows():
            poster_url = get_poster(row['title'])
            trending_movies.append({
                'title': row['title'],
                'overview': row['overview'],
                'vote_average': row['vote_average'],
                'vote_count': row['vote_count'],
                'release_date': row['release_date'],
                'poster_path': poster_url
            })
        return trending_movies
    except Exception as e:
        print(f"Error loading trending movies: {str(e)}")
        return []

def get_random_movie():
    return new_df.sample(1)['title'].values[0]

with open("movie_dict.pkl", "wb") as f:
    pickle.dump(new_df.to_dict(), f)
with open("similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)