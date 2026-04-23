"""
youtube.py
----------
Handles all YouTube Data API interactions.
Responsibilities:
  - Extract video ID from a YouTube URL
  - Fetch top-level comments via YouTube Data API v3
"""

import os
import re
import googleapiclient.discovery
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def extract_video_id(url: str) -> str | None:
    """
    Extract the YouTube video ID from various URL formats.

    Supported formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID

    Returns:
        str: video ID if found, else None
    """
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",       # Standard watch URL
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})", # Shortened URL
        r"(?:embed/)([a-zA-Z0-9_-]{11})",     # Embed URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_comments(video_id: str, max_comments: int = 200) -> list[str]:
    """
    Fetch top-level comments for a given YouTube video.

    Uses YouTube Data API v3 commentThreads endpoint.
    Paginates through results until max_comments is reached or no more pages exist.

    Args:
        video_id   (str): YouTube video ID (e.g. "dQw4w9WgXcQ")
        max_comments (int): Maximum number of comments to fetch (default: 200)

    Returns:
        list[str]: List of comment text strings

    Raises:
        ValueError: If YOUTUBE_API_KEY is not set
        googleapiclient.errors.HttpError: On API errors (invalid ID, quota exceeded, etc.)
    """
    if not YOUTUBE_API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY is not set. Please add it to your .env file."
        )

    # Build the YouTube API client
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=YOUTUBE_API_KEY
    )

    comments: list[str] = []
    next_page_token = None

    while len(comments) < max_comments:
        # Calculate how many results to request in this page (max 100 per API call)
        results_per_page = min(100, max_comments - len(comments))

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=results_per_page,
            pageToken=next_page_token,
            textFormat="plainText",
            order="relevance",  # Fetch most relevant comments first
        )
        response = request.execute()

        for item in response.get("items", []):
            top_comment = (
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            )
            comments.append(top_comment.strip())

        # Check for next page
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # No more pages available

    return comments
