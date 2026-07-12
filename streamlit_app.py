import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Headless nltk downloads
nltk.download('wordnet', quiet=True)

# Page Configuration
st.set_page_config(
    page_title="Zomato Analytics Portal",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #E23744;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4A4A4A;
        margin-bottom: 2rem;
    }
    .sentiment-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: bold;
        color: #E23744;
    }
</style>
""", unsafe_allow_dict=True, unsafe_allow_html=True)

# ----------------- Load Assets (Cached) -----------------
@st.cache_resource
def load_models():
    sentiment_model = joblib.load("models/final_sentiment_model.pkl")
    tfidf_vectorizer = joblib.load("models/final_tfidf_vectorizer.pkl")
    kmeans_model = joblib.load("models/final_kmeans_model.pkl")
    return sentiment_model, tfidf_vectorizer, kmeans_model

@st.cache_data
def load_datasets():
    df_meta = pd.read_csv("Zomato Restaurant names and Metadata.csv")
    df_reviews = pd.read_csv("Zomato Restaurant reviews.csv")
    
    # Simple cleaning
    df_reviews = df_reviews.drop_duplicates()
    
    def parse_cost(val):
        if pd.isna(val): return np.nan
        val_str = str(val).replace(',', '').strip()
        match = re.search(r'\d+', val_str)
        return float(match.group()) if match else np.nan

    def parse_rating(val):
        if pd.isna(val): return np.nan
        val_str = str(val).strip().lower()
        if val_str == 'like': return 4.0
        match = re.search(r'[\d\.]+', val_str)
        return float(match.group()) if match else np.nan

    df_meta['Cost_Cleaned'] = df_meta['Cost'].apply(parse_cost)
    df_meta['Cost_Cleaned'] = df_meta['Cost_Cleaned'].fillna(df_meta['Cost_Cleaned'].median())
    
    df_reviews['Rating_Cleaned'] = df_reviews['Rating'].apply(parse_rating)
    df_reviews = df_reviews.dropna(subset=['Rating_Cleaned', 'Review'])
    
    def parse_reviewer_stats(val):
        if pd.isna(val): return 0, 0
        val_str = str(val).lower()
        reviews, followers = 0, 0
        rev_match = re.search(r'(\d+)\s*review', val_str)
        if rev_match: reviews = int(rev_match.group(1))
        fol_match = re.search(r'(\d+)\s*follower', val_str)
        if fol_match: followers = int(fol_match.group(1))
        return reviews, followers

    stats_extracted = df_reviews['Metadata'].apply(parse_reviewer_stats)
    df_reviews['Reviewer_Reviews'] = [x[0] for x in stats_extracted]
    df_reviews['Reviewer_Followers'] = [x[1] for x in stats_extracted]
    df_reviews['Pictures_Cleaned'] = pd.to_numeric(df_reviews['Pictures'], errors='coerce').fillna(0).astype(int)
    df_reviews['Review_Length'] = df_reviews['Review'].apply(lambda x: len(str(x)))
    
    return df_meta, df_reviews

# Load resources
try:
    sentiment_model, tfidf_vectorizer, kmeans_model = load_models()
    df_meta, df_reviews = load_datasets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    error_msg = str(e)

# Text Preprocessing Helper
CONTRACTION_MAP = {
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
    "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "would've": "would have", "should've": "should have",
    "could've": "could have", "he'd": "he would", "she'd": "she would", "i'd": "i would",
    "they'd": "they would", "we'd": "we would", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have"
}

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    for word, replacement in CONTRACTION_MAP.items():
        text = text.replace(word, replacement)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])

# ----------------- Sidebar -----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Zomato_Logo.svg", width=180)
    st.markdown("### Capstone Project Dashboard")
    st.markdown("**Topic**: Sentiment Analysis & Restaurant Clustering")
    st.markdown("**Dataset**: Hyderabad Restaurants")
    st.write("---")
    st.write("### Model Performance Indicators")
    st.write("📊 **Sentiment Model**: Logistic Regression")
    st.write("✔️ **Test Accuracy**: 86.94%")
    st.write("📈 **F1 Weighted Score**: 0.87")
    st.write("---")
    st.caption("Developed as part of ML Capstone Submission.")

# ----------------- Main Interface -----------------
st.markdown('<div class="main-header">Zomato Restaurant Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Sentiment Predictor and Restaurant Clustering Dashboard</div>', unsafe_allow_html=True)

if not assets_loaded:
    st.error(f"Error loading models or datasets. Please run the training script first. Details: {error_msg}")
    st.stop()

# Set up tabs
tab1, tab2 = st.tabs(["💬 Sentiment Analysis Engine", "🏢 Restaurant Clustering Explorer"])

# Tab 1: Sentiment Analysis
with tab1:
    st.markdown("### Predict Review Sentiment")
    st.write("Type a customer review below to classify it as **Positive** or **Negative** using our trained Logistic Regression classifier.")
    
    review_input = st.text_area("Review text input", placeholder="E.g., The biryani was absolutely delicious, the chicken was tender and spice levels were perfect! Great place.", height=150)
    
    if st.button("Classify Sentiment"):
        if review_input.strip() == "":
            st.warning("Please enter some review text first.")
        else:
            # Clean and Vectorize
            cleaned_text = preprocess_text(review_input)
            vectors_unseen = tfidf_vectorizer.transform([cleaned_text]).toarray()
            # Padding length/pictures features with zeros (since model was trained on X_combined)
            dummy_scaled = np.zeros((1, 2))
            X_unseen_combined = np.hstack((vectors_unseen, dummy_scaled))
            
            # Predict
            pred = sentiment_model.predict(X_unseen_combined)[0]
            prob = sentiment_model.predict_proba(X_unseen_combined)[0]
            
            # Display results
            if pred == 1:
                st.success(f"🟢 **Positive Sentiment** (Confidence: {prob[1]*100:.2f}%)")
                st.balloons()
            else:
                st.error(f"🔴 **Negative/Neutral Sentiment** (Confidence: {prob[0]*100:.2f}%)")
            
            st.markdown("### Preprocessing Details:")
            st.write(f"**Original**: *\"{review_input}\"*")
            st.write(f"**Cleaned/Lemmatized**: *\"{cleaned_text}\"*")

# Tab 2: Restaurant Clustering Explorer
with tab2:
    st.markdown("### Hyderabad Restaurant Clusters")
    st.write("Restaurants are clustered into **3 distinct categories** based on cost, rating, review volume, picture metrics, and positive review percentage.")
    
    # Re-aggregate clustering features to present dashboards
    agg_revs = df_reviews.groupby('Restaurant').agg(
        Avg_Rating=('Rating_Cleaned', 'mean'),
        Reviews_Count=('Rating_Cleaned', 'count'),
        Avg_Pictures=('Pictures_Cleaned', 'mean'),
        Positive_Share=('Rating_Cleaned', lambda x: (x >= 3.5).mean())
    ).reset_index()
    
    df_res_clust = pd.merge(df_meta, agg_revs, left_on='Name', right_on='Restaurant')
    clust_features = ['Cost_Cleaned', 'Avg_Rating', 'Reviews_Count', 'Avg_Pictures', 'Positive_Share']
    
    # Scale and Fit K-Means locally for display coordinates
    scaler_clust = StandardScaler()
    X_clust_scaled = scaler_clust.fit_transform(df_res_clust[clust_features])
    df_res_clust['Cluster'] = kmeans_model.predict(X_clust_scaled)
    
    # Map Cluster Labels
    cluster_labels = {
        0: "Cluster 0: Budget & Moderate Quality",
        1: "Cluster 1: Premium & High Quality",
        2: "Cluster 2: Ultra-Visual & Socially Trendy"
    }
    df_res_clust['Cluster_Label'] = df_res_clust['Cluster'].map(cluster_labels)
    
    # PCA for 2D Visual
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_clust_scaled)
    df_res_clust['PCA1'] = coords[:, 0]
    df_res_clust['PCA2'] = coords[:, 1]
    
    # Restaurant Selector
    selected_restaurant = st.selectbox("Select a Restaurant to Explore:", sorted(df_res_clust['Name'].unique()))
    
    res_info = df_res_clust[df_res_clust['Name'] == selected_restaurant].iloc[0]
    
    # Columns layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"#### 🏢 {selected_restaurant}")
        st.markdown(f"**Segment**: `{res_info['Cluster_Label']}`")
        st.write("---")
        st.markdown(f"💵 **Cost for Two**: <span class='metric-val'>{res_info['Cost_Cleaned']:.0f} INR</span>", unsafe_allow_html=True)
        st.markdown(f"⭐ **Average Rating**: <span class='metric-val'>{res_info['Avg_Rating']:.2f} / 5</span>", unsafe_allow_html=True)
        st.markdown(f"💬 **Total Reviews**: **{res_info['Reviews_Count']:.0f}**")
        st.markdown(f"📸 **Avg Pictures / Review**: **{res_info['Avg_Pictures']:.2f}**")
        st.markdown(f"📈 **Positive Reviews Ratio**: **{res_info['Positive_Share']*100:.1f}%**")
        st.write("---")
        st.markdown(f"**Cuisines Served**: *{res_info['Cuisines']}*")
        st.markdown(f"🕒 **Operating Timings**: *{res_info['Timings']}*")
        
    with col2:
        st.markdown("#### PCA 2D Clustering Space Visualization")
        # Highlight selected restaurant
        df_res_clust['Highlight'] = "Other Restaurants"
        df_res_clust.loc[df_res_clust['Name'] == selected_restaurant, 'Highlight'] = "Selected Restaurant"
        
        fig = px.scatter(
            df_res_clust,
            x='PCA1',
            y='PCA2',
            color='Cluster_Label',
            symbol='Highlight',
            symbol_sequence=['circle', 'star'],
            hover_name='Name',
            hover_data=['Cost_Cleaned', 'Avg_Rating'],
            color_discrete_sequence=px.colors.qualitative.Set1,
            title="PCA Projection of Restaurant Clusters"
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)
        
    st.write("---")
    st.markdown("### Cluster Characteristics Summary Table")
    st.write(df_res_clust.groupby('Cluster_Label')[clust_features].mean())
