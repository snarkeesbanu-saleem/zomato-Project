import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
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
""", unsafe_allow_html=True)

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
    
    # Calculate review sentiments
    df_reviews['Sentiment'] = (df_reviews['Rating_Cleaned'] >= 3.5).astype(int)
    
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
    st.error(f"Error loading models or datasets. Details: {error_msg}")
    st.stop()

# Set up tabs
tab1, tab2, tab3 = st.tabs([
    "💬 Sentiment Prediction Engine", 
    "🏢 Restaurant Clustering Explorer", 
    "📊 Exploratory Data Dashboard (15 Charts)"
])

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
            
            # Load stopwords for clean print
            try:
                nltk_stopwords = set(nltk.corpus.stopwords.words('english'))
                cleaned_words = [w for w in cleaned_text.split() if w not in nltk_stopwords]
                cleaned_text_print = " ".join(cleaned_words)
            except Exception:
                cleaned_text_print = cleaned_text
                
            vectors_unseen = tfidf_vectorizer.transform([cleaned_text_print]).toarray()
            # Padding length/pictures features with zeros (since model was trained on X_combined)
            dummy_scaled = np.zeros((1, 2))
            X_unseen_combined = np.hstack((vectors_unseen, dummy_scaled))
            
            # Predict
            pred = sentiment_model.predict(X_unseen_combined)[0]
            prob = sentiment_model.predict_proba(X_unseen_combined)[0]
            
            # Display results
            col_res, col_gauge = st.columns([1, 1])
            with col_res:
                st.markdown("#### Prediction Output")
                if pred == 1:
                    st.success(f"🟢 **Positive Sentiment**")
                    st.markdown(f"**Confidence**: `{prob[1]*100:.2f}%` probability of positive experience.")
                    st.balloons()
                else:
                    st.error(f"🔴 **Negative/Neutral Sentiment**")
                    st.markdown(f"**Confidence**: `{prob[0]*100:.2f}%` probability of negative experience.")
                
                st.markdown("#### Preprocessing Details:")
                st.write(f"**Cleaned/Lemmatized Input**: *\"{cleaned_text_print}\"*")
                
            with col_gauge:
                st.markdown("#### Sentiment Probability Meter")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Positive Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#E23744"},
                        'steps' : [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "whitesmoke"}
                        ],
                        'threshold' : {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

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

# Tab 3: Exploratory Data Dashboard
with tab3:
    st.markdown("### Exploratory Data Analysis Dashboard")
    st.write("Select one of the 15 charts generated during exploratory analysis to view interactive charts and business insights.")
    
    chart_option = st.selectbox(
        "Choose an Exploratory Chart to Display:",
        [
            "Chart 1: Distribution of Restaurant Ratings",
            "Chart 2: Distribution of Cost for Two",
            "Chart 3: Top 15 Most Popular Cuisines",
            "Chart 4: Top 15 Most Reviewed Restaurants",
            "Chart 5: Cost Distribution for Top 10 Cuisines",
            "Chart 6: Restaurant Average Cost vs. Average Rating",
            "Chart 7: Proportion of Review Sentiments",
            "Chart 8: Monthly Review Count Timeline",
            "Chart 9: Number of Pictures Uploaded vs. Customer Rating",
            "Chart 10: Distribution of Reviewer Followers (Followers < 100)",
            "Chart 11: Average Customer Rating for Top 10 Cuisines",
            "Chart 12: Review Character Count vs. Rating",
            "Chart 13: Rating Distribution by Reviewer Experience Level",
            "Chart 14: Correlation Matrix of Key Numerical Variables",
            "Chart 15: Pair Plot of Key Metrics Colored by Sentiment"
        ]
    )
    
    st.write("---")
    
    # ----------------- Render Chart Selected -----------------
    if chart_option == "Chart 1: Distribution of Restaurant Ratings":
        fig = px.histogram(df_reviews, x='Rating_Cleaned', nbins=10, title="Distribution of Restaurant Ratings", color_discrete_sequence=['#E23744'])
        fig.update_layout(xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Ratings are highly left-skewed, showing that the majority of reviews are positive (4.0 and 5.0 stars). However, there is a clear spike at 1.0 star representing severe customer dissatisfaction.")
        
    elif chart_option == "Chart 2: Distribution of Cost for Two":
        fig = px.histogram(df_meta, x='Cost_Cleaned', nbins=20, marginal="box", title="Distribution of Cost for Two (INR)", color_discrete_sequence=['teal'])
        fig.update_layout(xaxis_title="Cost for Two (INR)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Right-skewed distribution peaking around 500 to 800 INR, representing casual and mid-range dining. Very few restaurants charge over 1500 INR.")
        
    elif chart_option == "Chart 3: Top 15 Most Popular Cuisines":
        cuisines_list = df_meta['Cuisines'].dropna().str.split(', ').explode().reset_index(drop=True)
        top_cuis = cuisines_list.value_counts().reset_index()
        top_cuis.columns = ['Cuisine', 'Count']
        fig = px.bar(top_cuis.iloc[:15], x='Count', y='Cuisine', orientation='h', title="Top 15 Most Popular Cuisines", color='Count', color_continuous_scale='Bluered')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: North Indian and Chinese are the most dominant cuisines offered, followed by Biryani, Continental, and Fast Food.")
        
    elif chart_option == "Chart 4: Top 15 Most Reviewed Restaurants":
        review_counts = df_reviews['Restaurant'].value_counts().reset_index()
        review_counts.columns = ['Restaurant', 'Reviews']
        fig = px.bar(review_counts.iloc[:15], x='Reviews', y='Restaurant', orientation='h', title="Top 15 Most Reviewed Restaurants", color='Reviews', color_continuous_scale='plasma')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Review counts are highly uniform at exactly 100 reviews for the top restaurants, suggesting a uniform data collection method in the Hyderabad review subset.")
        
    elif chart_option == "Chart 5: Cost Distribution for Top 10 Cuisines":
        df_exploded = df_meta.assign(Cuisine=df_meta['Cuisines'].str.split(', ')).explode('Cuisine').reset_index(drop=True)
        top_cuisines = df_exploded['Cuisine'].value_counts().index[:10]
        df_top_cuisines = df_exploded[df_exploded['Cuisine'].isin(top_cuisines)]
        fig = px.box(df_top_cuisines, x='Cuisine', y='Cost_Cleaned', color='Cuisine', title="Cost Distribution for Top 10 Cuisines")
        fig.update_layout(xaxis_title="Cuisine", yaxis_title="Cost for Two (INR)", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Continental and Italian cuisines have a higher median cost, whereas South Indian and Fast Food represent budget-friendly dining choices.")
        
    elif chart_option == "Chart 6: Restaurant Average Cost vs. Average Rating":
        avg_ratings = df_reviews.groupby('Restaurant')['Rating_Cleaned'].mean().reset_index()
        df_restaurant_local = pd.merge(df_meta, avg_ratings, left_on='Name', right_on='Restaurant')
        fig = px.scatter(df_restaurant_local, x='Cost_Cleaned', y='Rating_Cleaned', trendline="ols", title="Restaurant Average Cost vs. Average Rating", color='Rating_Cleaned', color_continuous_scale='deep')
        fig.update_layout(xaxis_title="Cost for Two (INR)", yaxis_title="Average Rating")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: There is a mild positive correlation, indicating that higher-cost restaurants receive slightly better reviews, likely due to premium food quality and service standards.")
        
    elif chart_option == "Chart 7: Proportion of Review Sentiments":
        sent_counts = df_reviews['Sentiment'].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        sent_counts['Sentiment'] = sent_counts['Sentiment'].map({1: 'Positive (>=3.5)', 0: 'Negative/Neutral (<3.5)'})
        fig = px.pie(sent_counts, values='Count', names='Sentiment', title="Proportion of Review Sentiments", color_discrete_sequence=['#66b3ff', '#ff9999'])
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Roughly 63.5% of Zomato reviews are positive, whereas 36.5% represent negative or neutral reviews.")
        
    elif chart_option == "Chart 8: Monthly Review Count Timeline":
        df_reviews['Time_Parsed'] = pd.to_datetime(df_reviews['Time'], errors='coerce')
        df_reviews['Year_Month'] = df_reviews['Time_Parsed'].dt.to_period('M')
        timeline = df_reviews.groupby('Year_Month').size().reset_index()
        timeline.columns = ['Year_Month', 'Count']
        timeline['Year_Month_Str'] = timeline['Year_Month'].astype(str)
        fig = px.line(timeline, x='Year_Month_Str', y='Count', title="Monthly Review Count Timeline", markers=True, color_discrete_sequence=['purple'])
        fig.update_layout(xaxis_title="Year-Month", yaxis_title="Number of Reviews")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Reviews volume peaked in late 2018 and early 2019, showing substantial platform growth during that time window.")
        
    elif chart_option == "Chart 9: Number of Pictures Uploaded vs. Customer Rating":
        fig = px.box(df_reviews[df_reviews['Pictures_Cleaned'] <= 15], x='Rating_Cleaned', y='Pictures_Cleaned', title="Number of Pictures Uploaded vs. Customer Rating", color='Rating_Cleaned')
        fig.update_layout(xaxis_title="Rating", yaxis_title="Number of Pictures")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Higher ratings (4.0 and 5.0 stars) show a higher volume of picture uploads, showing that happy diners are far more motivated to take and share photos.")
        
    elif chart_option == "Chart 10: Distribution of Reviewer Followers (Followers < 100)":
        fig = px.histogram(df_reviews[df_reviews['Reviewer_Followers'] < 100], x='Reviewer_Followers', nbins=20, title="Distribution of Reviewer Followers", color_discrete_sequence=['darkorange'])
        fig.update_layout(xaxis_title="Number of Followers", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: The vast majority of reviewers have very few followers (0 to 5), representing a casual reviewer base. Influencers represent a small minority.")
        
    elif chart_option == "Chart 11: Average Customer Rating for Top 10 Cuisines":
        df_rev_meta = pd.merge(df_reviews, df_meta, left_on='Restaurant', right_on='Name')
        df_rev_meta_exploded = df_rev_meta.assign(Cuisine=df_rev_meta['Cuisines'].str.split(', ')).explode('Cuisine').reset_index(drop=True)
        top_cuisines = df_rev_meta_exploded['Cuisine'].value_counts().index[:10]
        top_cuisines_ratings = df_rev_meta_exploded[df_rev_meta_exploded['Cuisine'].isin(top_cuisines)]
        avg_rating_cuis = top_cuisines_ratings.groupby('Cuisine')['Rating_Cleaned'].mean().reset_index()
        fig = px.bar(avg_rating_cuis, x='Cuisine', y='Rating_Cleaned', title="Average Customer Rating for Top 10 Cuisines", color='Rating_Cleaned', color_continuous_scale='Blues')
        fig.update_layout(yaxis_title="Average Rating")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Premium and experiential cuisines like Continental and Italian receive higher average ratings compared to Fast Food and Chinese.")
        
    elif chart_option == "Chart 12: Review Character Count vs. Rating":
        fig = px.box(df_reviews[df_reviews['Review_Length'] <= 1500], x='Rating_Cleaned', y='Review_Length', title="Review Character Count vs. Rating", color='Rating_Cleaned')
        fig.update_layout(xaxis_title="Rating", yaxis_title="Review Length (characters)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Lower ratings (1.0 and 2.0 stars) show longer reviews, indicating that angry customers write highly detailed logs of service failures.")
        
    elif chart_option == "Chart 13: Rating Distribution by Reviewer Experience Level":
        def get_exp_level(revs):
            if revs <= 1: return 'Newcomer'
            elif revs <= 10: return 'Casual'
            elif revs <= 50: return 'Frequent'
            else: return 'Expert'
        df_reviews['Reviewer_Experience'] = df_reviews['Reviewer_Reviews'].apply(get_exp_level)
        fig = px.box(df_reviews, x='Reviewer_Experience', y='Rating_Cleaned', category_orders={'Reviewer_Experience': ['Newcomer', 'Casual', 'Frequent', 'Expert']}, title="Rating Distribution by Reviewer Experience Level", color='Reviewer_Experience')
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Newcomers give high positive ratings (median 5.0), whereas Expert reviewers are much more critical, showing a more balanced rating spread.")
        
    elif chart_option == "Chart 14: Correlation Matrix of Key Numerical Variables":
        corr_cols = ['Rating_Cleaned', 'Reviewer_Reviews', 'Reviewer_Followers', 'Pictures_Cleaned', 'Review_Length']
        corr_matrix = df_reviews[corr_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap of Key Numerical Metrics")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: Strong positive correlation exists between reviewer reviews and followers (0.90), indicating popular reviewers continue posting volume. Very low direct linear correlation between individual metrics and ratings.")
        
    elif chart_option == "Chart 15: Pair Plot of Key Metrics Colored by Sentiment":
        st.write("Pair plots are visualized by taking a sample of reviews to ensure performance.")
        # Sample to prevent browser freeze
        df_sample = df_reviews.sample(n=500, random_state=42)
        df_sample['Sentiment_Str'] = df_sample['Sentiment'].map({1: 'Positive', 0: 'Negative/Neutral'})
        fig = px.scatter_matrix(
            df_sample,
            dimensions=['Rating_Cleaned', 'Pictures_Cleaned', 'Review_Length'],
            color='Sentiment_Str',
            color_discrete_sequence=['#66b3ff', '#ff9999'],
            title="Pair Plot of Rating, Pictures, and Review Length"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Insight**: The joint distribution visual confirms that higher-length reviews and high-picture counts align with positive reviews, though clusters overlap.")
