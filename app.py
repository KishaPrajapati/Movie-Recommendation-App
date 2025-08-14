import streamlit as st
import pickle
import pandas as pd
import urllib.parse
import matplotlib.pyplot as plt
import movie
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Session State
# -----------------------------
session_defaults = {
    "watchlist": [],
    "watched": [],
    "user_ratings": {},
    "user_reviews": {},
    "selected_movie": ""
}

for key, value in session_defaults.items():
    st.session_state.setdefault(key, value)

# -----------------------------
# Data Loading
# -----------------------------
try:
    movies_df = pd.DataFrame(pickle.load(open("movie_dict.pkl", "rb")))
    all_titles = sorted(movies_df['title'].unique())
except:
    st.error("Failed to load movie data!")
    st.stop()

# -----------------------------
# Helper Functions
# -----------------------------
def update_watchlist(movie_title, action):
    if action == "add" and movie_title not in st.session_state.watchlist:
        st.session_state.watchlist.append(movie_title)
        st.toast(f"✅ '{movie_title}' added to watchlist!")
    elif action == "remove" and movie_title in st.session_state.watchlist:
        st.session_state.watchlist.remove(movie_title)
        st.session_state.user_ratings.pop(movie_title, None)
        st.session_state.user_reviews.pop(movie_title, None)
        st.toast(f"❌ '{movie_title}' removed from watchlist!")

# -----------------------------
# Tab Definitions
# -----------------------------
tabs = st.tabs(["🏠 Home", "🎬 Recommendations", "📌 Watchlist", "🔎 Explore Genres", "📈 Trending", "ℹ About"])

# Home Tab
with tabs[0]:
    st.title("🎥 Movie Recommender System")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if st.button("🎲 Get Random Movie", use_container_width=True):
            st.session_state.selected_movie = movie.get_random_movie()
        
        if st.session_state.selected_movie:
            st.success(f"🎬 Try watching: *{st.session_state.selected_movie}*")
            if st.button("➕ Add to Watchlist", key="add_random"):
                update_watchlist(st.session_state.selected_movie, "add")
        
        st.image("https://cdn.pixabay.com/photo/2017/02/20/18/03/cinema-2084904_960_720.jpg", use_container_width=True)
    
    with col2:
        st.subheader("Your Stats 📊")
        cols = st.columns(3)
        cols[0].metric("Watchlist", len(st.session_state.watchlist))
        cols[1].metric("Watched", len(st.session_state.watched))
        cols[2].metric("Rated", len(st.session_state.user_ratings))
        
        if st.session_state.watchlist:
            st.subheader("Recent Additions")
            for title in st.session_state.watchlist[-3:][::-1]:
                st.write(f"• {title}")

# Recommendations Tab
with tabs[1]:
    st.header("🎬 Personalized Recommendations")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.session_state.selected_movie = st.selectbox(
            "🎥 Select Movie", all_titles,
            index=all_titles.index(st.session_state.selected_movie) if st.session_state.selected_movie else 0
        )
        
        if st.button(f"➕ Add to Watchlist", key="add_current"):
            update_watchlist(st.session_state.selected_movie, "add")
        
        with st.expander("🔍 Filters"):
            genre_filter = st.multiselect("🎭 Genres", options=sorted({g for genres in movies_df["genres"] for g in genres}))
            actor_filter = st.selectbox("🎬 Actor", ["Any"] + sorted({a for cast in movies_df["cast"] for a in cast}))
            director_filter = st.selectbox("🎥 Director", ["Any"] + sorted({d for crew in movies_df["crew"] for d in crew}))
            sort_by = st.radio("Sort By", ["Relevance", "Rating", "Newest"], horizontal=True)
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Finding matches..."):
            recs = movie.recommend(
                st.session_state.selected_movie,
                genre_filter=genre_filter,
                actor_filter=actor_filter if actor_filter != "Any" else None,
                director_filter=director_filter if director_filter != "Any" else None,
                num_recommendations=5
            )
            if sort_by == "Rating":
                recs = sorted(recs, key=lambda x: x['vote_average'], reverse=True)
            elif sort_by == "Newest":
                recs = sorted(recs, key=lambda x: x.get('release_year', 0), reverse=True)
            st.session_state.recommendations = recs

    if 'recommendations' in st.session_state and st.session_state.recommendations:
        st.subheader("Recommended Movies")
        for rec in st.session_state.recommendations:
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(rec.get("poster_path", "https://via.placeholder.com/150"), width=200)
                with cols[1]:
                    st.markdown(f"**{rec['title']}**")
                    st.caption(f"⭐ {rec['vote_average']} | 🎭 {', '.join(rec['genres'][:3])}")
                    st.write(rec['overview'][:200] + "...")
                    with st.expander("More Info"):
                        st.write(f"**Genres:** {', '.join(rec['genres'])}")
                        st.write(f"**Cast:** {', '.join(rec['cast'])}")
                        st.write(f"**Director:** {', '.join(rec['crew'])}")
                        
                        
                    
                    btn_cols = st.columns(3)
                    btn_cols[0].button("➕ Watchlist", key=f"add_{rec['title']}",
                                      on_click=update_watchlist, args=(rec['title'], "add"))
                    btn_cols[1].link_button("🎥 Trailer", 
                                          f"https://www.youtube.com/results?search_query={urllib.parse.quote(rec['title'])}+trailer")
                st.divider()

# Watchlist Tab
with tabs[2]:
    st.header("📌 Your Watchlist")
    
    if not st.session_state.watchlist:
        st.info("Add movies from recommendations!")
    else:
        for idx, title in enumerate(st.session_state.watchlist):
            with st.expander(title, expanded=True):
                cols = st.columns([1, 3, 1])
                with cols[0]:
                    st.image(movie.get_poster(title), width=150)
                with cols[1]:
                    if title in st.session_state.user_ratings:
                        st.write(f"⭐ Your Rating: {st.session_state.user_ratings[title]}/5")
                    if title in st.session_state.user_reviews:
                        st.write(f"📝 Review: {st.session_state.user_reviews[title]}")
                    
                    with st.form(f"form_{idx}"):
                        rating = st.slider("Rate", 1, 5, key=f"rate_{idx}")
                        review = st.text_area("Review", key=f"review_{idx}")
                        if st.form_submit_button("💾 Save"):
                            st.session_state.user_ratings[title] = rating
                            st.session_state.user_reviews[title] = review
                            st.toast("Saved!")
                with cols[2]:
                    st.button("❌ Remove", key=f"rem_{idx}", on_click=update_watchlist, args=(title, "remove"))
                    if st.checkbox("✅ Watched", key=f"watched_{idx}"):
                        if title not in st.session_state.watched:
                            st.session_state.watched.append(title)
        
        st.subheader("Insights")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.user_ratings:
                st.bar_chart(pd.DataFrame({
                    "Movies": st.session_state.user_ratings.keys(),
                    "Ratings": st.session_state.user_ratings.values()
                }).set_index("Movies"))
        
        with col2:
            genre_counts = {}
            for title in st.session_state.watchlist:
                for genre in movies_df[movies_df['title'] == title]['genres'].iloc[0]:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            if genre_counts:
                st.bar_chart(pd.Series(genre_counts))
with tabs[3]:
    st.header("🔍 Genre Explorer")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        genre = st.selectbox("🎭 Choose Genre", sorted({g for genres in movies_df["genres"] for g in genres}))
        decade = st.slider("📅 Decade", 1950, 2020, 2010, step=10)
    
    # Filter movies based on chosen genre and decade
    filtered = movies_df[
        (movies_df.genres.apply(lambda x: genre in x)) &
        (movies_df.release_year.between(decade, decade+9))
    ]
    
    if not filtered.empty:
        st.subheader(f"{genre} Movies ({decade}s) - Top 10")
        # Sort by vote_average descending and select the top 10 movies
        top_filtered = filtered.sort_values(by='vote_average', ascending=False).head(10)
        
        for _, row in top_filtered.iterrows():
            with st.container():
                cols = st.columns([1, 3, 1])
                
                with cols[0]:
                    st.image(movie.get_poster(row.title), width=100)
                    
                with cols[1]:
                    st.markdown(f"**{row.title}** ({row.release_year})")
                    st.caption(f"⭐ {row.vote_average} | 🎬 {', '.join(row.cast[:2])}")
                    st.write(row.overview[:150] + "...")
                    
                    # Expander with additional movie details
                    with st.expander("More Info"):
                        st.write(f"**Genres:** {', '.join(row.genres)}")
                        st.write(f"**Cast:** {', '.join(row.cast)}")
                        st.write(f"**Director:** {', '.join(row.crew)}")
                        st.write(f"**Vote Count:** {row.vote_count}")
                        st.write(f"**Release Date:** {row.release_date}")
                        
                with cols[2]:
                    if st.button("➕ Watchlist", key=f"gen_{row.title}"):
                        update_watchlist(row.title, "add")
                        
                st.markdown("---")
    else:
        st.warning("No movies found for these filters")

# Trending Tab
with tabs[4]:
    st.header("📈 Trending Now")
    trending = movie.get_trending_movies(10)
    
    if trending:
        st.subheader("Top Movies")
        for idx, movie_data in enumerate(trending):
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(movie_data['poster_path'], width=200)
                
                with cols[1]:
                    st.markdown(f"### {idx+1}. {movie_data['title']}")
                    try:
                        date = datetime.strptime(movie_data['release_date'], "%Y-%m-%d").strftime("%b %Y")
                    except:
                        date = "N/A"
                    st.caption(f"📅 {date} | ⭐ {movie_data['vote_average']} ({movie_data['vote_count']} votes)")
                    st.write(movie_data['overview'][:200] + "...")
                    if st.button("➕ Watchlist", key=f"tr_{idx}"):
                        update_watchlist(movie_data['title'], "add")
                st.divider()
        
        st.subheader("Trend Analysis")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh([m['title'] for m in trending], [m['vote_average'] for m in trending])
        ax.set_xlabel("Average Rating")
        st.pyplot(fig)
    else:
        st.error("❌ Failed to load trending movies")

# About Tab
# About Tab
with tabs[5]:
    st.header("ℹ About")
    
    st.markdown("""
    ## 🎥 Movie Recommendation System
    **Features:**
    - AI-powered recommendations 🎯
    - Personalized watchlists 📌
    - Genre exploration 🎭
    - Trending analytics 📈
    """)

    try:
        # Handling missing or incorrectly formatted genres
        genre_count = {}
        for genres in movies_df['genres'].dropna():  # Ignore NaN values
            if isinstance(genres, list):  # Ensure genres is a list before iterating
                for genre in genres:
                    genre_count[genre] = genre_count.get(genre, 0) + 1
        
        # Identify most popular genre
        most_popular_genre = max(genre_count, key=genre_count.get) if genre_count else "N/A"

        # Calculate average rating (handling empty ratings list)
        avg_rating = (
            f"{sum(st.session_state.user_ratings.values()) / len(st.session_state.user_ratings):.1f}/5"
            if st.session_state.user_ratings else "N/A"
        )
        
        # Display statistics
        stats = {
            "Total Movies": len(movies_df),
            "Your Watchlist": len(st.session_state.watchlist),
            "Average Rating": avg_rating,
            "Most Popular Genre": most_popular_genre
        }

    except Exception as e:
        stats = {"Error": f"Could not load statistics: {str(e)}"}

    # Display metrics in columns
    cols = st.columns(4)
    for i, (key, value) in enumerate(stats.items()):
        cols[i % 4].metric(key, value)

    st.markdown("""
    **📌 Technical Stack:**
    - **Python** / **Streamlit** 🐍
    - **TF-IDF Vectorization** for recommendations
    - **TMDb API** Integration for movie data 🌍
    - **Pandas/Numpy** for data handling 📊
    """)
