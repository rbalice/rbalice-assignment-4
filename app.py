from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform(documents)

n_components = 100
svd = TruncatedSVD(n_components=n_components)
lsa_matrix = svd.fit_transform(tfidf_matrix)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices
    query_vec = vectorizer.transform([query])
    query_lsa = svd.transform(query_vec)
    
    cosine_similarities = cosine_similarity(query_lsa, lsa_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    
    top_documents = [documents[i] for i in top_indices]
    top_similarities = cosine_similarities[top_indices]
    
    return top_documents, top_similarities.tolist(), top_indices.tolist() 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=3000)
