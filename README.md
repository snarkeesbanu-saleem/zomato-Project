# Zomato Restaurant Clustering & Sentiment Analysis Capstone Project

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/Zomato_Logo.svg" alt="Zomato Logo" width="250"/>
</p>

This repository contains the complete implementation of the Zomato Restaurant Clustering and Sentiment Analysis Capstone Project. 

The project solves two primary business objectives using the Hyderabad restaurant datasets:
1. **Supervised Sentiment Analysis**: Classifying user text reviews into Positive and Negative sentiments to monitor brand health and identify service issues.
2. **Unsupervised Restaurant Clustering**: Grouping restaurants based on metadata and review statistics to tailor customer segment targeting.

---

## 🌐 Live Interactive App
You can interact with our live Machine Learning and Analytics dashboard here:
👉 **[Launch Live Zomato Analytics Portal](https://zomato-project-j6kbbpv8kehpy2hkg4sjjc.streamlit.app/)**

---

## 🛠️ Technologies & Tools Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/XGBoost-1E88E5?style=for-the-badge" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
  <img src="https://img.shields.io/badge/NLTK-8C52FF?style=for-the-badge" alt="NLTK"/>
</p>

---

## 📊 Dataset Overview
The project integrates two core datasets:
* **Zomato Restaurant Names and Metadata**: Contains name, cuisines, average cost, Collections, and timing for 105 Hyderabad restaurants.
* **Zomato Restaurant Reviews**: Contains 10,000 customer reviews with text, star ratings, reviewer followers, and picture upload metrics.

---

## 🚀 Key Highlights & Results

### 1. Exploratory Data Analysis (EDA)
* Analyzed rating distributions, showing a high concentration of positive reviews (4-5 stars) and a small spike of severe complaints at 1 star.
* Visualized cost distribution (median dining cost of 600 INR for two).
* Checked the impact of reviewer experience and pictures on ratings.

### 2. Hypothesis Testing
* Evaluated three Welch's Independent T-Tests:
  * **Dining Cost vs. Rating**: Proved that higher cost dining correlates with higher ratings ($p = 0.0003$).
  * **Pictures vs. Rating**: Proved that reviews with pictures have significantly higher ratings than reviews without pictures ($p \approx 0.0000$).
  * **Expert vs. Casual Reviewers**: Proved that expert/popular reviewers give lower average ratings, representing a higher level of criticism ($p \approx 0.0000$).

### 3. Supervised Sentiment Classification
* Trained **Logistic Regression**, **Random Forest**, and **XGBoost** models on TF-IDF vectors (2500 features) combined with scaled reviewer metrics.
* **Logistic Regression Classifier** yielded the best performance:
  * **Accuracy**: **86.94%**
  * **Weighted F1-Score**: **0.87**
  * Strong recall for negative reviews to facilitate rapid customer support intervention.

### 4. Unsupervised Restaurant Clustering
* Implemented **K-Means Clustering** and **Hierarchical Clustering** (Ward's Linkage).
* The optimal number of clusters was determined to be **$K=3$** using Silhouette and Inertia elbow criteria:
  * **Cluster 0 (Budget & Moderate Quality)**: Mean cost ~630 INR, rating ~3.35, low pictures. (63 restaurants)
  * **Cluster 1 (Premium & High Quality)**: Mean cost ~1290 INR, rating ~4.03, moderate pictures. (35 restaurants)
  * **Cluster 2 (Ultra-Visual & Socially Trendy)**: Mean cost ~1100 INR, rating ~4.05, exceptionally high picture uploads (~3.06 pictures/review). (2 restaurants)

---

## 📁 Repository Structure
```text
├── Zomato Restaurant names and Metadata.csv   # Restaurant profiles
├── Zomato Restaurant reviews.csv             # Customer reviews
├── Zomato project.pptx                       # Project slides deck
├── Sample_ML_Submission_Template (2).ipynb   # Completed Jupyter Notebook
├── models/                                   # Saved serializations
│   ├── final_sentiment_model.pkl
│   ├── final_tfidf_vectorizer.pkl
│   └── final_kmeans_model.pkl
├── streamlit_app.py                          # Streamlit UI Source Code
├── requirements.txt                          # Streamlit Dependencies
└── README.md                                 # Project documentation
```

---

## 🛠️ Usage
To run the models locally and predict sentiment on new reviews:
```python
import joblib

# Load models
loaded_model = joblib.load("models/final_sentiment_model.pkl")
loaded_tfidf = joblib.load("models/final_tfidf_vectorizer.pkl")

# Predict sentiment
unseen_reviews = ["The food was delicious and the staff was extremely friendly!"]
features = loaded_tfidf.transform(unseen_reviews).toarray()
# Predict
pred = loaded_model.predict(features)
print("Positive" if pred[0] == 1 else "Negative")
```
