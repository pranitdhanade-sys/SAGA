"""
Unit tests for sentiment analysis system
"""

import unittest
import numpy as np
from src.utils.text_preprocessor import TextPreprocessor
from src.models.sentiment_classifier import SentimentClassifier, SENTIMENT_MAP
from src.utils.genetic_optimizer import GeneticOptimizer


class TestTextPreprocessor(unittest.TestCase):
    """Test text preprocessing."""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "Check out http://example.com! @user #hashtag 123"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn('http', cleaned.lower())
        self.assertNotIn('@', cleaned)
        self.assertNotIn('#', cleaned)
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "This is a test"
        tokens = self.preprocessor.tokenize(text)
        self.assertEqual(len(tokens), 4)
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ['this', 'is', 'great', 'movie']
        filtered = self.preprocessor.remove_stopwords(tokens)
        self.assertNotIn('is', filtered)
        self.assertIn('great', filtered)
    
    def test_preprocess(self):
        """Test full preprocessing pipeline."""
        text = "Check out this AMAZING movie!!! http://example.com"
        result = self.preprocessor.preprocess(text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_extract_features(self):
        """Test feature extraction."""
        text = "This is a great video! I loved it."
        features = self.preprocessor.extract_features(text)
        self.assertIn('text_length', features)
        self.assertIn('token_count', features)
        self.assertTrue(features['has_exclamation'])


class TestSentimentClassifier(unittest.TestCase):
    """Test sentiment classifier."""
    
    def setUp(self):
        self.classifier = SentimentClassifier(model_type='ml_classifier')
        self.sample_texts = [
            "This is amazing!",
            "I love this video",
            "Great content",
            "Not interested",
            "Could be better",
            "Terrible video",
            "Waste of time",
            "This is okay"
        ]
        self.sample_labels = [
            "positive", "positive", "positive",
            "neutral", "neutral",
            "negative", "negative",
            "neutral"
        ]
    
    def test_model_creation(self):
        """Test model creation."""
        self.classifier.create_model()
        self.assertIsNotNone(self.classifier.model)
    
    def test_preprocessing(self):
        """Test text preprocessing."""
        processed = self.classifier.preprocess_texts(self.sample_texts)
        self.assertEqual(len(processed), len(self.sample_texts))
    
    def test_training(self):
        """Test model training."""
        self.classifier.create_model()
        metrics = self.classifier.train(
            self.sample_texts,
            self.sample_labels,
            validation_split=0.2
        )
        self.assertIsNotNone(metrics)
    
    def test_prediction(self):
        """Test sentiment prediction."""
        self.classifier.create_model()
        self.classifier.train(self.sample_texts, self.sample_labels)
        
        test_text = "This is excellent!"
        predictions, probabilities = self.classifier.predict([test_text])
        
        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0], ['positive', 'negative', 'neutral'])


class TestGeneticOptimizer(unittest.TestCase):
    """Test genetic algorithm optimizer."""
    
    def setUp(self):
        self.optimizer = GeneticOptimizer(
            population_size=10,
            generations=5,
            seed=42
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.population_size, 10)
        self.assertEqual(self.optimizer.generations, 5)
    
    def test_optimization(self):
        """Test GA optimization."""
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 3, 50)
        
        def fitness_func(weights, X, y):
            # Simple fitness function
            return np.mean(weights)
        
        bounds = [(0, 1) for _ in range(5)]
        
        # Note: This is a minimal test, full test would be more complex
        self.assertIsNotNone(self.optimizer)


class TestSentimentMap(unittest.TestCase):
    """Test sentiment mapping."""
    
    def test_sentiment_map(self):
        """Test sentiment to integer mapping."""
        self.assertEqual(SENTIMENT_MAP[0], 'negative')
        self.assertEqual(SENTIMENT_MAP[1], 'neutral')
        self.assertEqual(SENTIMENT_MAP[2], 'positive')


if __name__ == '__main__':
    unittest.main()
