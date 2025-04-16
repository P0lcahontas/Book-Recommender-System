import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template



# Initialize Flask app
app = Flask(__name__)

import warnings
warnings.filterwarnings("ignore")

books = pd.read_csv("BX-Books.csv", sep=';', encoding="latin-1", on_bad_lines='skip')
users = pd.read_csv("BX-Users.csv", sep=';', encoding="latin-1", on_bad_lines='skip')
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=';', encoding="latin-1", on_bad_lines='skip')

print(books.head(3))

books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
users.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)

books.head

x = ratings['user_id'].value_counts() > 200
y = x[x].index  #user_ids
print(y.shape)
ratings = ratings[ratings['user_id'].isin(y)]

rating_with_books = ratings.merge(books, on='ISBN')
rating_with_books.head()

number_rating = rating_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)
final_rating = rating_with_books.merge(number_rating, on='title')
final_rating.shape
final_rating = final_rating[final_rating['number_of_ratings'] >= 50]
final_rating.drop_duplicates(['user_id','title'], inplace=True)

book_pivot = final_rating.pivot_table(columns='user_id', index='title', values="rating")
book_pivot.fillna(0, inplace=True)
print(book_pivot.head())

# Store book details (title, author, year) for recommendations
bookss = books.drop_duplicates(subset=['title'], keep='first')
book_details = bookss.set_index('title')[['author', 'year']].to_dict(orient='index')

from scipy.sparse import csr_matrix
book_sparse = csr_matrix(book_pivot)

from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# book_index = 740  # Example: Choosing the 741st book
# print(f"Book input: {book_pivot.index[book_index]}") 

# distances, suggestions = model.kneighbors(book_pivot.iloc[book_index, :].values.reshape(1, -1))

# for i in range(len(suggestions)):
#   print(book_pivot.index[suggestions[i]])



# Function to get recommendations
def get_recommendations(book_name):
    if book_name not in book_pivot.index:
        return None  # Book not found

    book_index = book_pivot.index.get_loc(book_name)
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_index, :].values.reshape(1, -1))
    
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