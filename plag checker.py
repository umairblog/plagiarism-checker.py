import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
   
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def calculate_similarity(text1, text2):
    
    texts = [preprocess_text(text1), preprocess_text(text2)]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def check_plagiarism(document1, document2):
   
    similarity_score = calculate_similarity(document1, document2)
    print(f"Similarity Score: {similarity_score}")
    if similarity_score > 0.7:
        print("Plagiarism suspected!")
    else:
        print("No plagiarism detected.")


if __name__ == "__main__":
    doc1 = """Natural language processing (NLP) is a field of artificial intelligence in which computers analyze, understand, and derive meaning from human language in a smart and useful way."""
    doc2 = """Natural language processing involves the study of interactions between computers and humans using the natural language, focusing on how to program computers to process and analyze large amounts of natural language data."""
    
    check_plagiarism(doc1, doc2)
