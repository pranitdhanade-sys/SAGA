# SAGA - Complete Module Reference

## 📦 Package Structure

### Root Level Files
- **main.py** - Main application orchestrator
- **config.py** - Configuration management
- **examples.py** - Usage examples
- **requirements.txt** - Python dependencies
- **setup.cfg** - Project configuration
- **.env.example** - Environment template
- **README.md** - Full documentation
- **QUICKSTART.md** - Quick start guide
- **PROJECT_SUMMARY.md** - Project completion summary

### Documentation Files
- **YouTube_Sentiment_Analysis.ipynb** - Interactive Jupyter notebook with 13 sections

### Test Suite
- **tests/test_sentiment_analysis.py** - Unit tests
  - TestTextPreprocessor
  - TestSentimentClassifier
  - TestGeneticOptimizer
  - TestSentimentMap

---

## 🔧 Module Details

### src/api/youtube_scraper.py
**YouTube Data API Integration**

Classes:
- `YouTubeScraper` - Main scraper class

Methods:
- `__init__(api_key)` - Initialize with API key
- `get_video_id_from_url(url)` - Extract video ID
- `get_comments(video_id, max_results, text_format)` - Fetch comments
- `get_video_info(video_id)` - Get video metadata
- `scrape_comments_batch(video_ids)` - Batch processing
- `filter_spam_comments(comments)` - Spam filtering

Features:
- Pagination support
- Rate limit handling
- Comment metadata extraction
- Spam detection

---

### src/utils/text_preprocessor.py
**Text Processing Pipeline**

Classes:
- `TextPreprocessor` - Main preprocessor class

Methods:
- `clean_text(text)` - Remove URLs, mentions, etc
- `tokenize(text)` - Split into tokens
- `remove_stopwords(tokens)` - Filter stopwords
- `lemmatize(tokens)` - Lemmatization
- `preprocess(text)` - Full pipeline
- `preprocess_batch(texts)` - Batch processing
- `extract_features(text)` - Feature extraction

Features:
- URL removal
- Lowercase conversion
- Stopword filtering
- Lemmatization
- Feature extraction (length, word count, etc)

---

### src/utils/genetic_optimizer.py
**Genetic Algorithm for Optimization**

Classes:
- `GeneticOptimizer` - GA optimization class

Methods:
- `__init__(population_size, generations, crossover_prob, mutation_prob)`
- `optimize_features(X, y, fitness_function, feature_bounds)` - Feature optimization
- `optimize_model_parameters(parameter_ranges, fitness_function)` - Hyperparameter tuning
- `_mutate_bounds(individual, bounds, indpb)` - Mutation operator

Features:
- DEAP framework
- Crossover and mutation
- Population-based search
- Fitness evaluation

---

### src/utils/report_generator.py
**Report Generation and Export**

Classes:
- `ReportGenerator` - Report generation class

Methods:
- `__init__(output_dir)` - Initialize generator
- `generate_csv_report(comments_data, filename)` - CSV export
- `generate_json_report(comments_data, metadata, filename)` - JSON export
- `generate_html_report(comments_data, title, filename)` - HTML export
- `generate_summary_report(comments_data, model_metrics, filename)` - Text summary
- `_generate_summary(comments_data)` - Summary statistics

Formats:
- CSV - Tabular format
- JSON - Structured data
- HTML - Interactive report
- TXT - Text summary

---

### src/models/ml_classifier.py
**Machine Learning Classifiers**

Classes:
- `MLSentimentClassifier` - Single ML model
- `EnsembleClassifier` - Ensemble of models

ML Algorithms:
1. Random Forest
2. Gradient Boosting
3. SVM (Support Vector Machine)
4. Naive Bayes
5. Logistic Regression
6. AdaBoost

Methods:
- `train(X_texts, y, test_size)` - Training
- `predict(texts)` - Prediction
- `predict_single(text)` - Single prediction
- `save_model(filepath)` - Model persistence
- `load_model(filepath)` - Model loading
- `get_feature_importance(top_n)` - Feature importance

---

### src/models/neural_network.py
**Deep Learning Models**

Classes:
- `NeuralNetworkClassifier` - Single NN model
- `HybridNeuralNetwork` - CNN-RNN hybrid

Architectures:
1. LSTM
2. GRU
3. Bidirectional LSTM
4. CNN (1D)
5. Hybrid CNN-RNN

Methods:
- `build_model(num_classes)` - Model construction
- `prepare_texts(texts, fit)` - Sequence preparation
- `train(X_texts, y, validation_split, epochs, batch_size)` - Training
- `predict(texts)` - Batch prediction
- `predict_single(text)` - Single prediction
- `save_model(filepath)` - Model saving
- `load_model(filepath)` - Model loading

Features:
- Embedding layers
- Dropout regularization
- Early stopping
- Learning rate reduction

---

### src/models/sentiment_classifier.py
**Unified Sentiment Classifier**

Classes:
- `SentimentClassifier` - Main classifier interface

Model Types:
- 'neural_network' - LSTM/GRU/CNN/etc
- 'ml_classifier' - Random Forest/SVM/etc
- 'ensemble' - Multiple models
- 'hybrid' - CNN-RNN

Methods:
- `create_model()` - Model creation
- `preprocess_texts(texts)` - Text preprocessing
- `train(X_texts, y_labels, ...)` - Model training
- `predict(texts, return_probabilities)` - Batch prediction
- `predict_single(text)` - Single prediction
- `batch_predict(comments_data)` - Batch with metadata
- `save_model(filepath)` - Model persistence
- `load_model(filepath)` - Model loading

Sentiment Classes:
- positive (2)
- neutral (1)
- negative (0)

---

### src/visualization/sentiment_visualizer.py
**Sentiment Visualization**

Classes:
- `SentimentVisualizer` - Visualization class

Visualization Methods:
- `sentiment_distribution()` - Bar chart
- `sentiment_pie_chart()` - Pie chart
- `sentiment_timeline()` - Timeline plot
- `confusion_matrix_plot()` - Confusion matrix
- `interactive_sentiment_distribution()` - Plotly chart
- `model_comparison()` - Model metrics comparison
- `word_frequency_by_sentiment()` - Word frequency analysis

Features:
- Static plots (Matplotlib/Seaborn)
- Interactive plots (Plotly)
- Multiple chart types
- Customizable colors

---

### src/models/sentiment_classifier.py
**Main Application**

Classes:
- `YouTubeSentimentAnalyzer` - Main orchestrator

Key Methods:
- `analyze_video(video_id, max_comments, ...)` - Single video analysis
- `batch_analyze_videos(video_ids)` - Multiple videos
- `train_model_on_custom_data(texts, sentiments)` - Custom training
- `predict_sentiment(text)` - Single prediction

Features:
- End-to-end pipeline
- YouTube integration
- Multi-format export
- Spam detection
- Toxicity filtering

---

## 🚀 Usage Workflow

```
1. Import Main Class
   from main import YouTubeSentimentAnalyzer

2. Initialize Analyzer
   analyzer = YouTubeSentimentAnalyzer(
       youtube_api_key='YOUR_KEY',
       model_type='neural_network'
   )

3. Train Model (Optional)
   analyzer.train_model_on_custom_data(
       texts, sentiments
   )

4. Analyze Comments
   results = analyzer.analyze_video(
       video_id='...',
       max_comments=500
   )

5. Generate Reports
   - CSV export
   - JSON export
   - HTML report
   - Text summary

6. Visualize Results
   - Distribution charts
   - Timeline analysis
   - Word clouds
   - Confusion matrix
```

---

## 📊 Dependencies

### Core
- numpy
- pandas
- scikit-learn

### Deep Learning
- tensorflow
- torch
- transformers

### NLP
- nltk
- textblob

### Optimization
- deap

### API
- google-auth-oauthlib
- google-api-python-client

### Visualization
- matplotlib
- seaborn
- plotly

### Other
- python-dotenv
- requests
- beautifulsoup4

Total: 20+ packages

---

## 🧪 Testing

Run tests:
```bash
python -m pytest tests/ -v
```

Test modules:
- TestTextPreprocessor
- TestSentimentClassifier
- TestGeneticOptimizer
- TestSentimentMap

---

## 🔄 Data Flow

```
YouTube Video URL
    ↓
Video ID Extraction
    ↓
YouTube API Query
    ↓
Raw Comments + Metadata
    ↓
Spam Filtering
    ↓
Text Preprocessing
    ↓
Feature Extraction
    ↓
Sentiment Prediction
    ↓
Toxicity Detection
    ↓
Results with Confidence
    ↓
Visualization + Export
    ↓
Reports (CSV/JSON/HTML/TXT)
```

---

## 🎯 Configuration

**config.py** contains:
- API keys and endpoints
- Model parameters
- GA settings
- Processing options
- Export formats
- Logging configuration

Edit **.env** file:
```env
YOUTUBE_API_KEY=your_key
MODEL_TYPE=neural_network
GA_POPULATION_SIZE=50
GA_GENERATIONS=20
MAX_COMMENTS_PER_VIDEO=1000
EXPORT_FORMAT=json
USE_GPU=true
```

---

## 📚 Documentation

1. **README.md** - Complete guide (500+ lines)
2. **QUICKSTART.md** - 5-minute setup
3. **PROJECT_SUMMARY.md** - Project overview
4. **YouTube_Sentiment_Analysis.ipynb** - Interactive tutorial
5. **examples.py** - Code examples
6. **Inline docstrings** - Function documentation

---

## 💡 Key Features Summary

✅ YouTube API integration  
✅ 7 ML algorithms  
✅ 5 NN architectures  
✅ Genetic algorithm optimization  
✅ Ensemble methods  
✅ Text preprocessing pipeline  
✅ Spam & toxicity detection  
✅ Multi-format reporting  
✅ Interactive visualizations  
✅ Model persistence  
✅ Comprehensive testing  
✅ Full documentation  

---

**Version**: 1.0.0  
**Status**: Production Ready ✅  
**Last Updated**: February 2026
