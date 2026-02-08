"""
Visualization module for sentiment analysis results
"""

import logging
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class SentimentVisualizer:
    """
    Create visualizations for sentiment analysis results.
    """

    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer.
        
        Args:
            style: Visualization style
        """
        self.style = style
        plt.style.use(style if style != 'seaborn' else 'default')

    @staticmethod
    def sentiment_distribution(
        sentiments: List[str],
        title: str = "Sentiment Distribution",
        save_path: str = None
    ):
        """
        Visualize sentiment distribution.
        
        Args:
            sentiments: List of sentiment labels
            title: Plot title
            save_path: Path to save figure
        """
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
        bar_colors = [colors.get(label, '#95a5a6') for label in sentiment_counts.index]
        
        sentiment_counts.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # Add count labels on bars
        for i, v in enumerate(sentiment_counts):
            ax.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved sentiment distribution to {save_path}")
        
        plt.show()

    @staticmethod
    def sentiment_pie_chart(
        sentiments: List[str],
        title: str = "Sentiment Distribution",
        save_path: str = None
    ):
        """
        Create pie chart of sentiment distribution.
        
        Args:
            sentiments: List of sentiment labels
            title: Chart title
            save_path: Path to save figure
        """
        sentiment_counts = pd.Series(sentiments).value_counts()
        colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
        pie_colors = [colors.get(label, '#95a5a6') for label in sentiment_counts.index]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved pie chart to {save_path}")
        
        plt.show()

    @staticmethod
    def sentiment_timeline(
        comments_data: List[Dict],
        time_column: str = 'timestamp',
        title: str = "Sentiment Over Time",
        save_path: str = None
    ):
        """
        Visualize sentiment changes over time.
        
        Args:
            comments_data: List of comment dictionaries with 'sentiment' and time column
            time_column: Name of timestamp column
            title: Plot title
            save_path: Path to save figure
        """
        df = pd.DataFrame(comments_data)
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
        
        for sentiment in ['positive', 'neutral', 'negative']:
            data = df[df['sentiment'] == sentiment]
            ax.scatter(
                data[time_column],
                data.index,
                label=sentiment,
                color=colors.get(sentiment, '#95a5a6'),
                s=100,
                alpha=0.6
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Comment Index', fontsize=12)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved timeline to {save_path}")
        
        plt.show()

    @staticmethod
    def confusion_matrix_plot(
        y_true: List[str],
        y_pred: List[str],
        title: str = "Confusion Matrix",
        save_path: str = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Path to save figure
        """
        from sklearn.metrics import confusion_matrix as cm
        
        labels = ['negative', 'neutral', 'positive']
        confusion_mat = cm(y_true, y_pred, labels=labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()

    @staticmethod
    def interactive_sentiment_distribution(
        sentiments: List[str],
        title: str = "Interactive Sentiment Distribution"
    ):
        """
        Create interactive sentiment distribution plot.
        
        Args:
            sentiments: List of sentiment labels
            title: Plot title
        """
        sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        
        colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
        sentiment_counts['color'] = sentiment_counts['sentiment'].map(colors)
        
        fig = go.Figure(data=[
            go.Bar(
                x=sentiment_counts['sentiment'],
                y=sentiment_counts['count'],
                marker=dict(color=sentiment_counts['color']),
                text=sentiment_counts['count'],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Sentiment",
            yaxis_title="Count",
            height=600,
            width=900,
            showlegend=False
        )
        
        fig.show()

    @staticmethod
    def model_comparison(
        models_metrics: Dict[str, Dict],
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
        save_path: str = None
    ):
        """
        Compare performance metrics across models.
        
        Args:
            models_metrics: Dict mapping model names to metrics dicts
            metrics: List of metrics to compare
            save_path: Path to save figure
        """
        df = pd.DataFrame(models_metrics).T
        df = df[metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind='bar', ax=ax, rot=45)
        
        ax.set_title("Model Comparison", fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved model comparison to {save_path}")
        
        plt.show()

    @staticmethod
    def word_frequency_by_sentiment(
        comments_data: List[Dict],
        top_n: int = 20,
        save_path: str = None
    ):
        """
        Plot top words by sentiment.
        
        Args:
            comments_data: List of comments with 'text' and 'sentiment'
            top_n: Number of top words to show
            save_path: Path to save figure
        """
        from sklearn.feature_extraction.text import CountVectorizer
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors_dict = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
        
        for idx, sentiment in enumerate(['positive', 'neutral', 'negative']):
            texts = [c['text'] for c in comments_data if c['sentiment'] == sentiment]
            
            if texts:
                vec = CountVectorizer(max_features=top_n).fit(texts)
                counts = vec.transform(texts).sum(axis=0).A1
                words = vec.get_feature_names_out()
                
                word_freq = pd.Series(counts, index=words).sort_values(ascending=False)[:top_n]
                
                word_freq.plot(
                    kind='barh',
                    ax=axes[idx],
                    color=colors_dict[sentiment]
                )
                axes[idx].set_title(f'{sentiment.capitalize()} - Top Words', fontweight='bold')
                axes[idx].set_xlabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved word frequency plot to {save_path}")
        
        plt.show()
