"""
Text preprocessing utilities for sentiment analysis
"""

import re
import logging
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing for sentiment analysis.
    """

    def __init__(self, language: str = 'english'):
        """
        Initialize text preprocessor.
        
        Args:
            language: Language for stopwords
        """
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, mentions, special characters, etc.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return word_tokenize(text.lower())

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]

    def extract_features(self, text: str) -> dict:
        """
        Extract linguistic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        tokens = self.tokenize(text)
        
        return {
            'text_length': len(text),
            'token_count': len(tokens),
            'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            'unique_tokens': len(set(tokens)),
            'stop_word_ratio': sum(1 for t in tokens if t in self.stop_words) / len(tokens) if tokens else 0,
            'has_exclamation': '!' in text,
            'has_question': '?' in text,
            'has_caps': any(c.isupper() for c in text),
        }
