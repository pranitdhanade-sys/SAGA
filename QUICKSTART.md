# Quick Start Guide - YouTube Sentiment Analysis System

## 🚀 Getting Started in 5 Minutes

### Step 1: Setup Environment

```bash
cd /home/violet/Documents/SAGA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your YouTube API key
# YOUTUBE_API_KEY=your_key_here
```

### Step 3: Run Examples

```bash
# Run Python examples
python examples.py

# Or use Jupyter Notebook
jupyter notebook YouTube_Sentiment_Analysis.ipynb
```

### Step 4: Test Basic Functionality

```python
from main import YouTubeSentimentAnalyzer

# Create analyzer
analyzer = YouTubeSentimentAnalyzer(
    model_type='neural_network',
    model_architecture='lstm'
)

# Test on sample comments
test_comments = [
    "This is amazing!",
    "Not bad",
    "Terrible!"
]

# Train on sample data
sample_data = [
    ("Great video!", "positive"),
    ("Not interested", "neutral"),
    ("Worst ever!", "negative")
]

analyzer.train_model_on_custom_data(
    [t[0] for t in sample_data],
    [t[1] for t in sample_data]
)

# Make predictions
for comment in test_comments:
    result = analyzer.predict_sentiment(comment)
    print(f"{comment} -> {result['sentiment']} ({result['confidence']:.2%})")
```

## 📁 Project Structure

```
SAGA/
├── src/
│   ├── api/
│   │   └── youtube_scraper.py       # YouTube API integration
│   ├── models/
│   │   ├── sentiment_classifier.py  # Main classifier
│   │   ├── ml_classifier.py         # ML models (RF, SVM, etc)
│   │   └── neural_network.py        # DL models (LSTM, CNN, etc)
│   ├── utils/
│   │   ├── text_preprocessor.py     # Text preprocessing
│   │   ├── genetic_optimizer.py     # GA optimization
│   │   └── report_generator.py      # Report export
│   └── visualization/
│       └── sentiment_visualizer.py  # Charts & plots
├── main.py                          # Main application
├── examples.py                      # Example usage
├── config.py                        # Configuration
├── requirements.txt                 # Dependencies
├── YouTube_Sentiment_Analysis.ipynb # Interactive notebook
└── tests/                           # Unit tests
```

## 🎯 Key Features

### Models Available

**Machine Learning:**
- Random Forest
- Gradient Boosting
- SVM (Support Vector Machine)
- Naive Bayes
- Logistic Regression
- AdaBoost
- Ensemble (voting)

**Neural Networks:**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional LSTM
- CNN (Convolutional)
- Hybrid CNN-RNN

**Optimization:**
- Genetic Algorithm for hyperparameter tuning
- Feature selection with GA

### Output Formats

- CSV (tabular data)
- JSON (structured data)
- HTML (interactive report)
- TXT (text summary)

## 📊 Example Usage

### Train & Predict

```python
from src.models.sentiment_classifier import SentimentClassifier

clf = SentimentClassifier(model_type='neural_network')

# Train
clf.train(texts, labels, epochs=20)

# Predict
sentiments = clf.predict(new_texts)
```

### Visualizations

```python
from src.visualization.sentiment_visualizer import SentimentVisualizer

viz = SentimentVisualizer()

viz.sentiment_distribution(sentiments)
viz.sentiment_pie_chart(sentiments)
viz.sentiment_timeline(comments_with_timestamps)
viz.confusion_matrix_plot(y_true, y_pred)
```

### Reports

```python
from src.utils.report_generator import ReportGenerator

gen = ReportGenerator()

gen.generate_csv_report(comments)
gen.generate_html_report(comments)
gen.generate_json_report(comments)
```

## 🔧 Configuration

Edit `.env` file:

```env
# API
YOUTUBE_API_KEY=your_key

# Model
MODEL_TYPE=neural_network
SENTIMENT_THRESHOLD_POSITIVE=0.6
SENTIMENT_THRESHOLD_NEGATIVE=0.4

# GA
GA_POPULATION_SIZE=50
GA_GENERATIONS=20

# Processing
MAX_COMMENTS_PER_VIDEO=1000
BATCH_SIZE=32
USE_GPU=true
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_sentiment_analysis.py::TestSentimentClassifier -v
```

## 📚 Documentation

- **README.md** - Full documentation
- **YouTube_Sentiment_Analysis.ipynb** - Interactive tutorial
- **examples.py** - Usage examples
- **docstrings** - In-code documentation

## 🎓 Learning Path

1. **Beginner:** Run `examples.py` to see all features
2. **Intermediate:** Use Jupyter notebook for interactive learning
3. **Advanced:** Integrate GA optimization and custom models
4. **Expert:** Modify architectures and train on YouTube data

## ⚡ Performance Tips

- Use GPU: Set `USE_GPU=true`
- Batch processing for multiple videos
- Cache preprocessed data
- Use smaller models for real-time inference

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size or max comments |
| Slow training | Enable GPU or use smaller model |
| Poor accuracy | Increase training data or epochs |
| API quota exceeded | Wait or upgrade YouTube API plan |

## 📞 Support

- Check [README.md](README.md) for detailed docs
- Review [examples.py](examples.py) for code samples
- See [YouTube_Sentiment_Analysis.ipynb](YouTube_Sentiment_Analysis.ipynb) for tutorials
- Run tests: `pytest tests/`

---

**Happy sentiment analyzing! 🎬📊**
