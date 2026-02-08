"""
Main application for YouTube Sentiment Analysis System
"""

import logging
from typing import List, Dict, Optional
from src.api.youtube_scraper import YouTubeScraper
from src.models.sentiment_classifier import SentimentClassifier
from src.utils.text_preprocessor import TextPreprocessor
from src.utils.report_generator import ReportGenerator
from src.visualization.sentiment_visualizer import SentimentVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeSentimentAnalyzer:
    """
    Main application orchestrating sentiment analysis of YouTube comments.
    """

    def __init__(
        self,
        youtube_api_key: Optional[str] = None,
        model_type: str = 'neural_network',
        model_architecture: str = 'lstm',
        use_ga_optimization: bool = True
    ):
        """
        Initialize analyzer.
        
        Args:
            youtube_api_key: YouTube API key
            model_type: Type of sentiment model
            model_architecture: Architecture for neural networks
            use_ga_optimization: Use genetic algorithm optimization
        """
        self.youtube_scraper = YouTubeScraper(youtube_api_key) if youtube_api_key else None
        self.sentiment_classifier = SentimentClassifier(
            model_type=model_type,
            architecture=model_architecture,
            use_ga_optimization=use_ga_optimization
        )
        self.visualizer = SentimentVisualizer()
        self.report_generator = ReportGenerator()
        self.preprocessor = TextPreprocessor()

    def analyze_video(
        self,
        video_id_or_url: str,
        max_comments: int = 500,
        model_type: str = 'neural_network',
        train_model: bool = True,
        export_format: str = 'json'
    ) -> Dict:
        """
        Analyze sentiment of comments from a YouTube video.
        
        Args:
            video_id_or_url: YouTube video URL or ID
            max_comments: Maximum comments to analyze
            model_type: Model type for sentiment analysis
            train_model: Whether to train model first
            export_format: Export format ('csv', 'json', 'html', 'txt')
            
        Returns:
            Analysis results
        """
        logger.info(f"Starting analysis for video: {video_id_or_url}")
        
        # Extract video ID
        if not self.youtube_scraper:
            raise ValueError("YouTube API key not configured")
        
        video_id = self.youtube_scraper.get_video_id_from_url(video_id_or_url)
        
        # Get video info
        logger.info("Fetching video information...")
        video_info = self.youtube_scraper.get_video_info(video_id)
        
        # Get comments
        logger.info(f"Fetching up to {max_comments} comments...")
        comments = self.youtube_scraper.get_comments(video_id, max_results=max_comments)
        
        # Filter spam
        logger.info("Filtering spam comments...")
        comments = self.youtube_scraper.filter_spam_comments(comments)
        
        # Predict sentiment
        logger.info(f"Predicting sentiment for {len(comments)} comments...")
        results = self.sentiment_classifier.batch_predict(comments)
        
        # Generate report
        logger.info("Generating reports...")
        report_paths = self._generate_reports(
            results, video_info, video_id, export_format
        )
        
        logger.info("Analysis complete!")
        
        return {
            'video_id': video_id,
            'video_info': video_info,
            'comments_analyzed': len(results),
            'results': results,
            'report_paths': report_paths
        }

    def batch_analyze_videos(
        self,
        video_ids: List[str],
        max_comments_per_video: int = 100
    ) -> Dict:
        """
        Analyze comments from multiple videos.
        
        Args:
            video_ids: List of video IDs
            max_comments_per_video: Max comments per video
            
        Returns:
            Combined analysis results
        """
        combined_results = []
        
        for video_id in video_ids:
            logger.info(f"Analyzing video {video_id}...")
            try:
                results = self.analyze_video(
                    video_id,
                    max_comments=max_comments_per_video
                )
                combined_results.append(results)
            except Exception as e:
                logger.error(f"Error analyzing video {video_id}: {str(e)}")
        
        return {'videos': combined_results}

    def train_model_on_custom_data(
        self,
        comments_texts: List[str],
        sentiments: List[str],
        save_model: bool = True,
        model_path: str = None
    ):
        """
        Train sentiment classifier on custom data.
        
        Args:
            comments_texts: List of comment texts
            sentiments: List of sentiment labels
            save_model: Whether to save trained model
            model_path: Path to save model
        """
        logger.info(f"Training model on {len(comments_texts)} comments...")
        
        metrics = self.sentiment_classifier.train(
            comments_texts,
            sentiments,
            epochs=20,
            batch_size=32,
            apply_ga_optimization=True
        )
        
        logger.info(f"Training metrics: {metrics}")
        
        if save_model and model_path:
            self.sentiment_classifier.save_model(model_path)
            logger.info(f"Model saved to {model_path}")

    def predict_sentiment(self, text: str) -> Dict:
        """
        Predict sentiment for a single comment.
        
        Args:
            text: Comment text
            
        Returns:
            Sentiment prediction with confidence
        """
        return self.sentiment_classifier.predict_single(text)

    def _generate_reports(
        self,
        results: List[Dict],
        video_info: Dict,
        video_id: str,
        export_format: str
    ) -> Dict[str, str]:
        """
        Generate various report formats.
        
        Args:
            results: Analysis results
            video_info: Video information
            video_id: Video ID
            export_format: Format to export
            
        Returns:
            Dictionary of report paths
        """
        report_paths = {}
        
        try:
            # CSV Report
            if export_format in ['csv', 'all']:
                csv_path = self.report_generator.generate_csv_report(
                    results,
                    filename=f"video_{video_id}_analysis.csv"
                )
                report_paths['csv'] = csv_path
            
            # JSON Report
            if export_format in ['json', 'all']:
                json_path = self.report_generator.generate_json_report(
                    results,
                    metadata={'video_info': video_info},
                    filename=f"video_{video_id}_analysis.json"
                )
                report_paths['json'] = json_path
            
            # HTML Report
            if export_format in ['html', 'all']:
                html_path = self.report_generator.generate_html_report(
                    results,
                    title=f"Analysis: {video_info.get('title', 'Unknown Video')}",
                    filename=f"video_{video_id}_analysis.html"
                )
                report_paths['html'] = html_path
            
            # Text Summary
            if export_format in ['txt', 'all']:
                txt_path = self.report_generator.generate_summary_report(
                    results,
                    filename=f"video_{video_id}_summary.txt"
                )
                report_paths['txt'] = txt_path
        
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
        
        return report_paths


def main():
    """
    Example usage of the sentiment analysis system.
    """
    logger.info("YouTube Sentiment Analysis System")
    logger.info("=" * 50)
    
    # Initialize analyzer
    analyzer = YouTubeSentimentAnalyzer(
        youtube_api_key=None,  # Set your API key here
        model_type='neural_network',
        model_architecture='lstm',
        use_ga_optimization=True
    )
    
    # Example: Train on sample data
    sample_comments = [
        "This video is amazing! Love it!",
        "Best content ever!",
        "I really enjoyed watching this.",
        "Not interested in this topic.",
        "Could be better.",
        "Waste of time.",
        "Terrible content!",
        "This is okay, nothing special.",
        "Great tutorial, very helpful!",
        "Completely disagree with the views here."
    ]
    
    sample_sentiments = [
        "positive", "positive", "positive",
        "neutral", "neutral",
        "negative", "negative",
        "neutral",
        "positive",
        "negative"
    ]
    
    # Train model
    logger.info("Training sentiment classifier...")
    analyzer.train_model_on_custom_data(
        sample_comments,
        sample_sentiments,
        save_model=True,
        model_path='models/sentiment_model.h5'
    )
    
    # Test prediction
    test_comment = "This is the best video I've ever seen!"
    result = analyzer.predict_sentiment(test_comment)
    logger.info(f"Test prediction: {result}")
    
    logger.info("System ready for YouTube video analysis!")
    logger.info("To analyze a video, call: analyzer.analyze_video(video_url)")


if __name__ == '__main__':
    main()
