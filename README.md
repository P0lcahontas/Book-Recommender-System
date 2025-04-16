# 📚 Book Recommender System

A simple content-based book recommendation web application built with **Flask**, **pandas**, and **scikit-learn**. The system uses collaborative filtering and k-nearest neighbors to recommend similar books based on user ratings.

---

## 🚀 Features

- Recommend books similar to a user-given book title.
- Filters out low-activity users and rarely-rated books to ensure quality recommendations.
- Displays book title, author, and publication year for recommended titles.
- Interactive web interface built with Flask.

---

## 🧠 How It Works

1. **Data Cleaning & Preprocessing:**
   - Three datasets (`BX-Books.csv`, `BX-Users.csv`, `BX-Book-Ratings.csv`) are loaded.
   - Low-activity users (less than 200 ratings) and books with fewer than 50 ratings are filtered out.
   - Duplicate user-book ratings are removed to ensure data integrity.

2. **Recommendation Logic:**
   - Ratings are pivoted to form a user-item matrix.
   - A **K-Nearest Neighbors** model is trained using cosine similarity.
   - Given a book title, the system finds the most similar books based on user ratings.

3. **Web Interface:**
   - User inputs a book title.
   - System displays similar books with their authors and publication years.

---

## 🛠️ Tech Stack

- **Python**
- **Flask** (for web server)
- **pandas / numpy** (for data handling)
- **scikit-learn** (KNN model)
- **scipy** (for sparse matrix optimization)
- **HTML / Jinja2** (for rendering templates)

---
