"""
Example script demonstrating YouTube sentiment analysis
"""

from main import YouTubeSentimentAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_analysis():
    """
    Example: Basic sentiment analysis workflow
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Sentiment Analysis")
    print("="*70)
    
    # Initialize analyzer
    analyzer = YouTubeSentimentAnalyzer(
        youtube_api_key=None,  # Set your API key
        model_type='neural_network',
        model_architecture='lstm',
        use_ga_optimization=True
    )
    
    # Sample data for demonstration
    sample_comments = [
        "This video is absolutely amazing! Love every second of it!",
        "Best content creator on YouTube, keep it up!",
        "I really enjoyed this video, very informative.",
        "Not bad, but I've seen better content.",
        "This video was okay, nothing special.",
        "Completely useless, waste of my time.",
        "Terrible quality and boring content!",
        "Could be improved but overall decent.",
        "This is the best thing I've watched today!",
        "Don't recommend this to anyone."
    ]
    
    sentiments = [
        "positive", "positive", "positive",
        "neutral", "neutral",
        "negative", "negative",
        "neutral",
        "positive",
        "negative"
    ]
    
    # Train model
    print("\n[1] Training sentiment classifier...")
    analyzer.train_model_on_custom_data(
        sample_comments,
        sentiments,
        save_model=False
    )
    
    # Test predictions
    print("\n[2] Testing predictions on new comments...")
    test_comments = [
        "This is fantastic!",
        "It's okay I guess",
        "Absolutely terrible!"
    ]
    
    for comment in test_comments:
        result = analyzer.predict_sentiment(comment)
        print(f"\nComment: '{comment}'")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities: {result['probabilities']}")


def example_ml_models():
    """
    Example: Comparing different ML models
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Machine Learning Models Comparison")
    print("="*70)
    
    from src.models.ml_classifier import MLSentimentClassifier, EnsembleClassifier
    
    sample_texts = [
        "Excellent video!",
        "I love this",
        "Great content",
        "Not interested",
        "Could be better",
        "Terrible!",
        "Waste of time",
        "Okay I guess"
    ]
    
    labels = [1, 1, 1, 0, 0, 2, 2, 0]  # 0=negative, 1=neutral, 2=positive
    
    models = ['random_forest', 'gradient_boosting', 'svm']
    results = {}
    
    for model_type in models:
        print(f"\n[Training {model_type}...]")
        clf = MLSentimentClassifier(model_type=model_type)
        metrics = clf.train(sample_texts, labels, test_size=0.25)
        results[model_type] = metrics
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Compare models
    print("\n[Model Comparison Results]")
    for model, metrics in results.items():
        print(f"\n{model.upper()}:")
        print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")


def example_neural_networks():
    """
    Example: Testing different neural network architectures
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Neural Network Architectures")
    print("="*70)
    
    from src.models.neural_network import NeuralNetworkClassifier
    
    sample_texts = [
        "This is amazing!", "Love it!", "Best ever!",
        "Not bad", "Okay", "Could be better",
        "Terrible", "Worst", "Awful!"
    ]
    
    labels = [2, 2, 2, 1, 1, 1, 0, 0, 0]  # 0=negative, 1=neutral, 2=positive
    
    architectures = ['lstm', 'gru', 'bidirectional_lstm']
    
    for arch in architectures:
        print(f"\n[Training {arch.upper()} model...]")
        nn = NeuralNetworkClassifier(
            vocab_size=1000,
            max_length=50,
            embedding_dim=64,
            architecture=arch
        )
        
        # Train with minimal epochs for demo
        history = nn.train(sample_texts, labels, epochs=5, batch_size=2)
        
        print(f"Training complete!")
        print(f"Final loss: {history['history']['loss'][-1]:.4f}")


def example_genetic_algorithm():
    """
    Example: Using genetic algorithm for parameter optimization
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Genetic Algorithm Optimization")
    print("="*70)
    
    from src.utils.genetic_optimizer import GeneticOptimizer
    import numpy as np
    
    print("\n[Initializing GA optimizer...]")
    optimizer = GeneticOptimizer(
        population_size=20,
        generations=10,
        crossover_prob=0.8,
        mutation_prob=0.2
    )
    
    # Define a simple optimization problem
    def rosenbrock_function(weights, X, y):
        """Rosenbrock function for testing GA"""
        x, y = weights[0], weights[1]
        return -(100 * (y - x**2)**2 + (1 - x)**2)  # Negative because we maximize
    
    X_dummy = np.random.rand(10, 2)
    y_dummy = np.random.rand(10)
    
    print("\n[Running GA optimization...]")
    bounds = [(-2, 2), (-2, 2)]
    result = optimizer.optimize_features(
        X_dummy, y_dummy,
        rosenbrock_function,
        bounds
    )
    
    print(f"\nOptimization Results:")
    print(f"Best features: {result['best_features']}")
    print(f"Best fitness: {result['best_fitness']:.4f}")


def example_visualization():
    """
    Example: Visualizing sentiment analysis results
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Visualization")
    print("="*70)
    
    from src.visualization.sentiment_visualizer import SentimentVisualizer
    
    # Sample results
    sentiments = ['positive'] * 15 + ['neutral'] * 10 + ['negative'] * 8
    
    print("\n[Creating visualizations...]")
    visualizer = SentimentVisualizer()
    
    # Create distribution chart (doesn't show in non-GUI environment)
    print("- Sentiment distribution bar chart")
    print("- Sentiment pie chart")
    print("- Confusion matrix heatmap")
    print("\nNote: Visualizations would be displayed in GUI environment")


def example_report_generation():
    """
    Example: Generating analysis reports
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Report Generation")
    print("="*70)
    
    from src.utils.report_generator import ReportGenerator
    
    # Sample comments with sentiment
    sample_comments = [
        {
            'author': 'user1',
            'text': 'This video is amazing!',
            'sentiment': 'positive',
            'confidence': 0.95,
            'likes': 10
        },
        {
            'author': 'user2',
            'text': 'Not interesting',
            'sentiment': 'neutral',
            'confidence': 0.70,
            'likes': 2
        },
        {
            'author': 'user3',
            'text': 'Terrible content',
            'sentiment': 'negative',
            'confidence': 0.92,
            'likes': 1
        }
    ]
    
    print("\n[Generating reports...]")
    gen = ReportGenerator('reports')
    
    # Note: These would create actual files
    print("- CSV report")
    print("- JSON report")
    print("- HTML report")
    print("- Text summary")
    
    print("\nReports would be saved to 'reports/' directory")


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("YOUTUBE SENTIMENT ANALYSIS - EXAMPLES")
    print("="*70)
    
    try:
        example_basic_analysis()
        example_ml_models()
        # example_neural_networks()  # Commented out as it requires training time
        example_genetic_algorithm()
        example_visualization()
        example_report_generation()
        
        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
