# 🎬 Movie Recommendation System

## 📌 Overview

This project is a **Content-Based Movie Recommendation System** built using Python and machine learning techniques. It analyzes movie metadata and recommends similar movies based on textual features such as genres, keywords, cast, and overview.

The system uses **TF-IDF Vectorization** and **Cosine Similarity** to compute similarity scores between movies.

---

## 🚀 Features

* Data preprocessing and cleaning
* Feature engineering from movie metadata
* TF-IDF vectorization of movie tags
* Cosine similarity computation
* Custom movie recommendation function
* Model/data serialization using Pickle

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Pickle

---

## 📂 Dataset

The project uses the `movies_metadata.csv` dataset containing movie-related information such as:

* Title
* Genres
* Overview
* Keywords
* Cast
* Crew
* Vote Average
* Vote Count

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading

* Load the dataset using Pandas
* Inspect structure and columns

### 2️⃣ Data Cleaning

* Handle missing values
* Drop unnecessary columns
* Format JSON-like columns

### 3️⃣ Feature Engineering

* Extract relevant features (genres, keywords, cast, crew)
* Combine them into a single `tags` column
* Text preprocessing (lowercasing, removing spaces)

### 4️⃣ Vectorization

* Apply **TF-IDF Vectorizer**
* Generate a feature matrix

### 5️⃣ Similarity Calculation

* Compute cosine similarity between movies

### 6️⃣ Recommendation Function

```python
def recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']

    idx = indices[title]
    sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_score.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx]
```

---

## 💾 Saved Files

The following files are generated for reuse/deployment:

* `tfidf_matrix.pkl` – TF-IDF feature matrix
* `indices.pkl` – Title-to-index mapping
* `df.pkl` – Processed dataframe
* `tfidf.pkl` – Trained TF-IDF vectorizer

---

## ▶️ How to Run

1. Install dependencies:

   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook movies.ipynb
   ```

3. Run all cells sequentially.

4. Call the recommendation function:

   ```python
   recommend("Avatar", 5)
   ```

---

## 📈 Future Improvements

* Add collaborative filtering
* Deploy as a web app (Streamlit/Flask)
* Add poster previews using an API
* Improve ranking using weighted ratings

---

## 📜 License

This project is for educational purposes.

---

## 👨‍💻 Author

Developed as part of a machine learning/data science practice project.
