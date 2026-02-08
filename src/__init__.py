"""
YouTube Sentiment Analysis System with Genetic Algorithms, ML, and Neural Networks
"""

__version__ = "1.0.0"
__author__ = "Sentiment Analysis Team"

from .models.sentiment_classifier import SentimentClassifier
from .api.youtube_scraper import YouTubeScraper
from .utils.genetic_optimizer import GeneticOptimizer

__all__ = [
    'SentimentClassifier',
    'YouTubeScraper',
    'GeneticOptimizer'
]
