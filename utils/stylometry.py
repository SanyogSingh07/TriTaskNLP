import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string

def extract_features(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    if not words or not sentences:
        return [0.0, 0.0, 0.0]
        
    sentence_length = len(words) / len(sentences)
    
    # Vocabulary richness (Type-Token Ratio)
    unique_words = set(w.lower() for w in words if w.isalnum())
    ttr = len(unique_words) / len(words)
    
    # Punctuation usage
    punct_count = sum(1 for char in text if char in string.punctuation)
    punct_ratio = punct_count / len(words)
    
    return [sentence_length, ttr, punct_ratio]
