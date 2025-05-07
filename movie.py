import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk
from tkinter import scrolledtext

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv('C:/Users/Deepesh/Downloads/imdb_top_1000.csv')
df = df[['Series_Title', 'Overview']].dropna()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_plot'] = df['Overview'].apply(clean_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_plot'])

def recommend_from_story(input_story, df, tfidf_matrix, top_n=5):
    clean_input = clean_text(input_story)
    input_vector = vectorizer.transform([clean_input])
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = [cosine_similarities[i] for i in top_indices]
    return results[['Series_Title', 'Overview', 'similarity_score']]

def show_recommendations():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        result_box.delete("1.0", tk.END)
        result_box.insert(tk.END, "Please enter a movie storyline.")
        return
    recommendations = recommend_from_story(user_input, df, tfidf_matrix)
    result_box.delete("1.0", tk.END)
    for _, row in recommendations.iterrows():
        result_box.insert(tk.END, f"🎬 {row['Series_Title']} (Score: {row['similarity_score']:.2f})\n")
        result_box.insert(tk.END, f"Plot: {row['Overview']}\n\n")

window = tk.Tk()
window.title("Movie Recommendation System (Based on Storyline)")
window.geometry("700x600")

label = tk.Label(window, text="Enter a movie storyline:", font=("Arial", 12))
label.pack(pady=5)

entry = tk.Text(window, height=5, width=80)
entry.pack(pady=5)

button = tk.Button(window, text="Get Recommendations", command=show_recommendations, bg="lightblue")
button.pack(pady=10)

result_box = scrolledtext.ScrolledText(window, height=20, width=85)
result_box.pack(pady=5)

window.mainloop()