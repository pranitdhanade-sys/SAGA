"""
YouTube Video Comments Scraper using YouTube Data API
"""

import os
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeScraper:
    """
    Scrapes YouTube comments from videos using the YouTube Data API v3.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize YouTube API client.
        
        Args:
            api_key: YouTube API key. If None, reads from .env file.
        """
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "YouTube API key not provided. Set YOUTUBE_API_KEY in .env or pass as argument."
            )
        
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        logger.info("YouTube API client initialized successfully")

    def get_video_id_from_url(self, url: str) -> str:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID
        """
        if 'youtu.be/' in url:
            return url.split('youtu.be/')[-1].split('?')[0]
        elif 'youtube.com/watch' in url:
            return url.split('v=')[-1].split('&')[0]
        else:
            return url

    def get_comments(
        self,
        video_id: str,
        max_results: int = 100,
        text_format: str = 'plainText'
    ) -> List[Dict]:
        """
        Fetch comments from a YouTube video.
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to retrieve
            text_format: 'plainText' or 'html'
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_results:
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat=text_format,
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token,
                    order='relevance'
                )
                
                response = request.execute()
                
                for item in response.get('items', []):
                    comment_data = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'author': comment_data['authorDisplayName'],
                        'text': comment_data['textDisplay'],
                        'likes': comment_data['likeCount'],
                        'timestamp': comment_data['publishedAt'],
                        'reply_count': item['snippet']['totalReplyCount'],
                        'channel_id': comment_data['authorChannelId']['value']
                                     if comment_data.get('authorChannelId') else None,
                    })
                
                next_page_token = response.get('nextPageToken')
                
                if not next_page_token or len(comments) >= max_results:
                    break
                    
            logger.info(f"Retrieved {len(comments)} comments from video {video_id}")
            return comments[:max_results]
            
        except Exception as e:
            logger.error(f"Error fetching comments: {str(e)}")
            return comments

    def get_video_info(self, video_id: str) -> Dict:
        """
        Get metadata about a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video metadata dictionary
        """
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
            
            if response['items']:
                video = response['items'][0]
                return {
                    'title': video['snippet']['title'],
                    'channel': video['snippet']['channelTitle'],
                    'published_at': video['snippet']['publishedAt'],
                    'view_count': int(video['statistics'].get('viewCount', 0)),
                    'like_count': int(video['statistics'].get('likeCount', 0)),
                    'comment_count': int(video['statistics'].get('commentCount', 0)),
                    'duration': video['contentDetails']['duration'] if 'contentDetails' in video else None,
                }
            return None
            
        except Exception as e:
            logger.error(f"Error fetching video info: {str(e)}")
            return None

    def scrape_comments_batch(
        self,
        video_ids: List[str],
        max_comments_per_video: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Scrape comments from multiple videos.
        
        Args:
            video_ids: List of YouTube video IDs
            max_comments_per_video: Maximum comments per video
            
        Returns:
            Dictionary mapping video_id to list of comments
        """
        results = {}
        
        for video_id in video_ids:
            logger.info(f"Scraping comments from video: {video_id}")
            comments = self.get_comments(video_id, max_comments_per_video)
            results[video_id] = comments
            
        return results

    def filter_spam_comments(self, comments: List[Dict]) -> List[Dict]:
        """
        Basic spam filtering for comments.
        
        Args:
            comments: List of comment dictionaries
            
        Returns:
            Filtered comments list
        """
        filtered = []
        spam_indicators = ['http', ':///', 'click here', 'subscribe', 'follow me']
        
        for comment in comments:
            text = comment['text'].lower()
            is_spam = any(indicator in text for indicator in spam_indicators)
            
            if not is_spam and len(comment['text'].strip()) > 3:
                filtered.append(comment)
        
        logger.info(f"Filtered {len(comments) - len(filtered)} spam comments")
        return filtered
