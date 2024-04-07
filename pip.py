# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Read 'Books.csv' with specified data types
books_dtype = {
    'ISBN': str,
    'Book-Title': str,
    'Book-Author': str,
    'Year-Of-Publication': str,
    'Publisher': str,
    'Image-URL-S': str,
    'Image-URL-M': str,
    'Image-URL-L': str
}
books = pd.read_csv('Book.csv', dtype=books_dtype)

# Read 'Ratings.csv' with specified data types
ratings_dtype = {
    'User-ID': int,
    'ISBN': str,
    'Book-Rating': int
}
ratings = pd.read_csv('Rating.csv', dtype=ratings_dtype)

# Merge ratings with book details
ratings_with_name = ratings.merge(books, on='ISBN')

# Filter users with more than 150 ratings
users_ratings_count = ratings_with_name.groupby('User-ID').count()['Book-Rating']
important_users = users_ratings_count[users_ratings_count > 150].index

# Filter books with more than 50 ratings
books_ratings_count = ratings_with_name.groupby('Book-Title').count()['Book-Rating']
famous_books = books_ratings_count[books_ratings_count >= 50].index

# Filter ratings by important users and famous books
filtered_ratings = ratings_with_name[(ratings_with_name['User-ID'].isin(important_users)) &
                                     (ratings_with_name['Book-Title'].isin(famous_books))]

# Create pivot table for recommendations
pivot_table = filtered_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating', fill_value=0)

# Compute similarity scores using cosine similarity
similarity_scores = cosine_similarity(pivot_table)

def recommend(book_name):
    # Find index of book in pivot table
    index = np.where(pivot_table.index == book_name)[0][0]
    # Find similar items based on cosine similarity scores
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:11]

    # Extract book details for recommendations
    recommendations = []
    for i in similar_items:
        book_details = books[books['Book-Title'] == pivot_table.index[i[0]]].drop_duplicates('Book-Title')
        recommendations.append({
            'Book-Title': book_details['Book-Title'].iloc[0],
            'Book-Author': book_details['Book-Author'].iloc[0],
            'Image-URL-L': book_details['Image-URL-L'].iloc[0]
        })

    return recommendations

def main():
    st.title("Book Recommendation System")

    book_name = st.text_input("Enter a book name:")

    if st.button("Recommend"):
        if book_name == "":
            st.write("Please enter a book name!")
        else:
            recommendations = recommend(book_name)
            for i, book in enumerate(recommendations):
                st.write(f"Recommendation {i+1}:")
                st.write(f"Book Title: {book['Book-Title']}")
                st.write(f"Book Author: {book['Book-Author']}")
                st.image(book['Image-URL-L'])

if __name__ == "__main__()":
    main()