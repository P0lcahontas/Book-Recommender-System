from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# Initialize Flask app
app = Flask(__name__)

# Load data
books = pd.read_csv("BX-Books.csv", sep=';', encoding="latin-1", on_bad_lines='skip')
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=';', encoding="latin-1", on_bad_lines='skip')

# Preprocess data
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication']]
books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year'}, inplace=True)
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'}, inplace=True)

# Merge ratings with book details
rating_with_books = ratings.merge(books, on='ISBN')

# Keep books with at least 50 ratings
book_counts = rating_with_books.groupby('title')['rating'].count().reset_index()
book_counts.rename(columns={'rating': 'num_ratings'}, inplace=True)
final_rating = rating_with_books.merge(book_counts, on='title')
final_rating = final_rating[final_rating['num_ratings'] >= 50]

# Create a pivot table (books as rows, users as columns)
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating').fillna(0)

# Store book details (title, author, year) for recommendations
book_details = books.set_index('title')[['author', 'year']].to_dict(orient='index')

# Convert pivot table to sparse matrix
book_sparse = csr_matrix(book_pivot)

# Train KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_sparse)

# Function to get recommendations
def get_recommendations(book_name, n=5):
    if book_name not in book_pivot.index:
        return None  # Book not found

    book_index = book_pivot.index.get_loc(book_name)
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=n+1)
    
    recommended_books = []
    for i in range(1, len(suggestions[0])):  # Skip the first book (itself)
        suggested_title = book_pivot.index[suggestions[0][i]]
        recommended_books.append({
            'title': suggested_title,
            'author': book_details.get(suggested_title, {}).get('author', 'Unknown'),
            'year': book_details.get(suggested_title, {}).get('year', 'Unknown')
        })
    
    return recommended_books

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error_message = None

    if request.method == "POST":
        book_name = request.form.get("book_name")
        if book_name:
            recommendations = get_recommendations(book_name)
            if recommendations is None:
                error_message = "Book not found. Please try another title."

    return render_template("index.html", recommendations=recommendations, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
