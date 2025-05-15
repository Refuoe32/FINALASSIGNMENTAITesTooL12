"""
Utility functions for test generation.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpus/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """
    Preprocess text for analysis.
    
    Args:
        text: String containing the text to preprocess
        
    Returns:
        String: Preprocessed text
    """
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s\.]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text, top_n=20):
    """
    Extract important keywords from text.
    
    Args:
        text: String containing the text to analyze
        top_n: Number of top keywords to extract
        
    Returns:
        list: List of keyword strings
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize
    tokens = word_tokenize(processed_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Count frequency
    word_counts = Counter(lemmatized_tokens)
    
    # Extract multi-word phrases
    bigrams = list(nltk.bigrams(lemmatized_tokens))
    bigram_counts = Counter(bigrams)
    
    # Rank keywords (single words and bigrams)
    top_words = [word for word, count in word_counts.most_common(top_n)]
    
    top_bigrams = []
    for (w1, w2), count in bigram_counts.most_common(top_n // 2):
        # Only consider bigrams with sufficient frequency
        if count > 1:
            bigram = f"{w1} {w2}"
            top_bigrams.append(bigram)
    
    # Combine single words and bigrams
    all_keywords = top_words + top_bigrams
    
    # Remove duplicates and limit to top_n
    seen = set()
    unique_keywords = []
    for keyword in all_keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords[:top_n]
