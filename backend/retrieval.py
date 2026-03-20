import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path="data/diseases.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_most_similar_disease(query, data):
    documents = [item["symptoms"] for item in data]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents + [query])

    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_index = similarity_scores.argmax()

    best_match = data[best_index]
    best_score = float(similarity_scores[0][best_index])

    return best_match, best_score