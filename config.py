"""
Configuration module for sentiment analysis system
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')

# Model Configuration
MODEL_TYPE = os.getenv('MODEL_TYPE', 'neural_network')
SENTIMENT_THRESHOLD_POSITIVE = float(os.getenv('SENTIMENT_THRESHOLD_POSITIVE', '0.6'))
SENTIMENT_THRESHOLD_NEGATIVE = float(os.getenv('SENTIMENT_THRESHOLD_NEGATIVE', '0.4'))

# GA Configuration
GA_POPULATION_SIZE = int(os.getenv('GA_POPULATION_SIZE', '50'))
GA_GENERATIONS = int(os.getenv('GA_GENERATIONS', '20'))
GA_CROSSOVER_PROBABILITY = float(os.getenv('GA_CROSSOVER_PROBABILITY', '0.8'))
GA_MUTATION_PROBABILITY = float(os.getenv('GA_MUTATION_PROBABILITY', '0.2'))

# Processing
MAX_COMMENTS_PER_VIDEO = int(os.getenv('MAX_COMMENTS_PER_VIDEO', '1000'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'

# Output
EXPORT_FORMAT = os.getenv('EXPORT_FORMAT', 'json')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Neural Network
NN_VOCAB_SIZE = 5000
NN_MAX_LENGTH = 100
NN_EMBEDDING_DIM = 128
NN_EPOCHS = 20

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
