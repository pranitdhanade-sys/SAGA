"""
Report generation module for sentiment analysis results
"""

import logging
from typing import List, Dict
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate analysis reports in various formats.
    """

    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_csv_report(
        self,
        comments_data: List[Dict],
        filename: str = None
    ) -> str:
        """
        Generate CSV report.
        
        Args:
            comments_data: List of comment dictionaries
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(comments_data)
        filepath = self.output_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"CSV report saved to {filepath}")
        return str(filepath)

    def generate_json_report(
        self,
        comments_data: List[Dict],
        metadata: Dict = None,
        filename: str = None
    ) -> str:
        """
        Generate JSON report.
        
        Args:
            comments_data: List of comment dictionaries
            metadata: Additional metadata
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'comments': comments_data,
            'summary': self._generate_summary(comments_data)
        }
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"JSON report saved to {filepath}")
        return str(filepath)

    def generate_html_report(
        self,
        comments_data: List[Dict],
        title: str = "Sentiment Analysis Report",
        filename: str = None
    ) -> str:
        """
        Generate HTML report.
        
        Args:
            comments_data: List of comment dictionaries
            title: Report title
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        summary = self._generate_summary(comments_data)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 15px;
                    margin-bottom: 30px;
                }}
                .summary-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .summary-value {{
                    font-size: 28px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .summary-label {{
                    color: #7f8c8d;
                    margin-top: 10px;
                }}
                .positive {{ color: #2ecc71; }}
                .neutral {{ color: #f39c12; }}
                .negative {{ color: #e74c3c; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    margin-top: 20px;
                    border-radius: 5px;
                    overflow: hidden;
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                    padding: 15px;
                    text-align: left;
                }}
                td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                tr:hover {{
                    background-color: #f9f9f9;
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                    color: #7f8c8d;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="summary-card">
                    <div class="summary-value positive">{summary['positive_count']}</div>
                    <div class="summary-label">Positive</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value neutral">{summary['neutral_count']}</div>
                    <div class="summary-label">Neutral</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value negative">{summary['negative_count']}</div>
                    <div class="summary-label">Negative</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{summary['total_comments']}</div>
                    <div class="summary-label">Total Comments</div>
                </div>
            </div>
            
            <h2>Comments Detail</h2>
            <table>
                <tr>
                    <th>Author</th>
                    <th>Comment</th>
                    <th>Sentiment</th>
                    <th>Confidence</th>
                    <th>Likes</th>
                </tr>
        """
        
        for comment in comments_data[:100]:  # Limit to first 100 for performance
            sentiment = comment.get('sentiment', 'unknown')
            confidence = comment.get('confidence', 0)
            html_content += f"""
                <tr>
                    <td>{comment.get('author', 'Unknown')}</td>
                    <td>{comment.get('text', '')[:100]}...</td>
                    <td class="{sentiment}">{sentiment}</td>
                    <td>{confidence:.2%}</td>
                    <td>{comment.get('likes', 0)}</td>
                </tr>
            """
        
        html_content += """
            </table>
            <div class="footer">
                <p>Sentiment Analysis System | YouTube Comments Analysis</p>
            </div>
        </body>
        </html>
        """
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {filepath}")
        return str(filepath)

    def generate_summary_report(
        self,
        comments_data: List[Dict],
        model_metrics: Dict = None,
        filename: str = None
    ) -> str:
        """
        Generate comprehensive summary report.
        
        Args:
            comments_data: List of comment dictionaries
            model_metrics: Model performance metrics
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        summary = self._generate_summary(comments_data)
        
        report_text = f"""
{'='*70}
SENTIMENT ANALYSIS REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS
{'-'*70}
Total Comments Analyzed: {summary['total_comments']}
Positive Comments: {summary['positive_count']} ({summary['positive_pct']:.1f}%)
Neutral Comments: {summary['neutral_count']} ({summary['neutral_pct']:.1f}%)
Negative Comments: {summary['negative_count']} ({summary['negative_pct']:.1f}%)
Average Confidence: {summary['avg_confidence']:.2%}

"""
        
        if model_metrics:
            report_text += f"""
MODEL PERFORMANCE
{'-'*70}
"""
            for metric, value in model_metrics.items():
                if isinstance(value, float):
                    report_text += f"{metric}: {value:.4f}\n"
                else:
                    report_text += f"{metric}: {value}\n"
        
        report_text += f"""

TOP POSITIVE COMMENTS
{'-'*70}
"""
        positive_comments = [c for c in comments_data if c.get('sentiment') == 'positive']
        for i, comment in enumerate(positive_comments[:5], 1):
            report_text += f"{i}. {comment.get('text', '')[:100]}...\n"
        
        report_text += f"""

TOP NEGATIVE COMMENTS
{'-'*70}
"""
        negative_comments = [c for c in comments_data if c.get('sentiment') == 'negative']
        for i, comment in enumerate(negative_comments[:5], 1):
            report_text += f"{i}. {comment.get('text', '')[:100]}...\n"
        
        report_text += f"""

{'='*70}
END OF REPORT
{'='*70}
"""
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Summary report saved to {filepath}")
        return str(filepath)

    @staticmethod
    def _generate_summary(comments_data: List[Dict]) -> Dict:
        """
        Generate summary statistics.
        
        Args:
            comments_data: List of comments
            
        Returns:
            Summary dictionary
        """
        sentiments = [c.get('sentiment', 'unknown') for c in comments_data]
        confidences = [c.get('confidence', 0) for c in comments_data]
        
        positive_count = sentiments.count('positive')
        neutral_count = sentiments.count('neutral')
        negative_count = sentiments.count('negative')
        total = len(sentiments)
        
        return {
            'total_comments': total,
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'positive_pct': (positive_count / total * 100) if total > 0 else 0,
            'neutral_pct': (neutral_count / total * 100) if total > 0 else 0,
            'negative_pct': (negative_count / total * 100) if total > 0 else 0,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
        }
