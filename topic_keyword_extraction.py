from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load transcript data from file
with open(r"C:\Users\Prasanna\OneDrive\Desktop\use_case_transcript.txt", 'r', encoding='utf-8') as file:
    transcript_text = file.read()

# Initialize RAKE
r = Rake()

# Extract keywords using RAKE
r.extract_keywords_from_text(transcript_text)

# Get the highest scored phrase as the topic name
topic_name = r.get_ranked_phrases()[0]

# Print the extracted topic name
print("Extracted Topic Name using rake:", topic_name)

###################################################################

import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")

def extract_topics_and_keywords(transcript_file):
    # Read the transcript file
    with open(transcript_file, 'r', encoding='utf-8') as file:
        transcript_text = file.read()

    # Tokenize the transcript text using spaCy
    doc = nlp(transcript_text)

    # Extract topics using spaCy's noun chunks
    topics = [chunk.text for chunk in doc.noun_chunks]

    # Extract keywords using spaCy's named entity recognition
    keywords = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'PERSON']]

    # Use TF-IDF for keyword extraction
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([transcript_text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Sort feature names based on TF-IDF scores
    keywords_tfidf = [feature_names[i] for i in tfidf_scores.argsort()[-10:][::-1]]

    return topics, keywords, keywords_tfidf

if __name__ == "__main__":
    transcript_file = "path/to/your/transcript.txt"  # Update this with the path to your transcript file
    topics, keywords, keywords_tfidf = extract_topics_and_keywords(transcript_file)

    print("Extracted Topics using rake:")
    print(topics)
    print("\nExtracted Keywords using rake (NER):")
    print(keywords)
    print("\nTop Keywords using rake (TF-IDF):")
    print(keywords_tfidf)

#########################################################################################################################

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Load XLM-RoBERTa tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

# Load transcript data from file
with open(r"C:\Users\Prasanna\Downloads\transcript.txt",'r', encoding='utf-8') as file:
    transcript_text = file.read()

# Tokenize transcript
tokens = tokenizer.encode(transcript_text, return_tensors='pt')

# Perform topic extraction using TF-IDF and LDA
# Convert tokens to strings
token_strings = [tokenizer.decode(token) for token in tokens[0]]

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = vectorizer.fit_transform(token_strings)

# Get feature names using get_feature_names_out
feature_names = vectorizer.get_feature_names_out()

# Apply Latent Dirichlet Allocation (LDA)
lda_model = LatentDirichletAllocation(n_components=5, max_iter=10, learning_method='online')
lda_topic_matrix = lda_model.fit_transform(tfidf)

# Get top words for each topic
topic_words = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[:-10 - 1:-1]
    topic_words.append([feature_names[i] for i in top_words_idx])

# Print top words for each topic
for idx, words in enumerate(topic_words):
    print(f"Topic using LDA {idx + 1}: {', '.join(words)}")