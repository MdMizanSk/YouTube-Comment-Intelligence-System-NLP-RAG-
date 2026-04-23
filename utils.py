"""
utils.py
--------
Standalone utility functions for the three additional features:
  1. LLM-based Sentiment Analysis  (Pydantic structured output)
  2. Most Common Words             (frequency analysis, stopword removal)
  3. 50-word Summary               (LLM-based)
"""

import os
import re
from collections import Counter
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Shared LLM instance for utilities (gpt-4o-mini, temperature=0)
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )


# ===========================================================================
# FEATURE 1 — Sentiment Analysis
# ===========================================================================

# Step 1: Pydantic model for structured LLM output
class SentimentOutput(BaseModel):
    """Strict sentiment classification result. Only 'positive' or 'negative'."""
    sentiment: Literal["positive", "negative"]


# Step 2: Sentiment prompt
SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a strict sentiment classifier. Rules:\n"
                "- Only output 'positive' or 'negative'\n"
                "- No explanation\n"
                "- If unclear, choose the closest sentiment"
            ),
        ),
        (
            "human",
            "{comment}",
        ),
    ]
)


def analyze_sentiment(comments: list[str]) -> dict:
    """
    Classify each comment as 'positive' or 'negative' using an LLM.

    NOTE: This uses a brute-force approach — one LLM call per comment.
    Future optimization: Use batch processing (e.g., LangChain batch() or
    async calls) to reduce latency and API costs at scale.

    Uses structured output via Pydantic (SentimentOutput) so the LLM is
    constrained to return exactly {"sentiment": "positive"} or {"sentiment": "negative"}.

    Args:
        comments (list[str]): Raw comment strings to classify

    Returns:
        dict: {
            "positive_percent": float,
            "negative_percent": float,
            "total_analyzed": int
        }
    """
    llm = _get_llm()

    # Step 3: Build structured-output chain
    # model.with_structured_output() forces the LLM to return a SentimentOutput object
    structured_llm = llm.with_structured_output(SentimentOutput)
    chain = SENTIMENT_PROMPT | structured_llm

    positive_count = 0
    negative_count = 0
    total = 0

    # Step 4: Brute-force loop — one call per comment
    # NOTE: Batching (chain.batch([...])) could be used here as a future optimization
    for comment in comments:
        if not comment.strip():
            continue  # Skip empty comments

        try:
            result: SentimentOutput = chain.invoke({"comment": comment})
            if result.sentiment == "positive":
                positive_count += 1
            else:
                negative_count += 1
            total += 1
        except Exception:
            # Gracefully skip any comment that causes an API error
            continue

    if total == 0:
        return {"positive_percent": 0.0, "negative_percent": 0.0, "total_analyzed": 0}

    return {
        "positive_percent": round((positive_count / total) * 100, 2),
        "negative_percent": round((negative_count / total) * 100, 2),
        "total_analyzed": total,
    }


# ===========================================================================
# FEATURE 2 — Most Common Words
# ===========================================================================

# Basic English stopwords (no external library dependency)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
    "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
    "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn", "like", "get", "got", "would", "also", "really", "one",
    "even", "know", "think", "time", "go", "see", "make", "look", "come",
    "want", "way", "good", "much", "well", "say", "said", "going", "still",
    "im", "ive", "its", "thats", "its",
}


def get_common_words(comments: list[str], top_n: int = 10) -> list[tuple[str, int]]:
    """
    Find the most frequently occurring meaningful words across all comments.

    Process:
      1. Concatenate all comments into one text blob
      2. Lowercase and tokenize (letters only, min 3 chars)
      3. Remove stopwords
      4. Return top_n words by frequency

    Args:
        comments (list[str]): Raw comment strings
        top_n    (int)       : Number of top words to return (default: 10)

    Returns:
        list[tuple[str, int]]: [(word, count), ...] sorted by count descending
    """
    all_text = " ".join(comments).lower()

    # Extract only alphabetic tokens of at least 3 characters
    tokens = re.findall(r"\b[a-z]{3,}\b", all_text)

    # Remove stopwords
    filtered = [word for word in tokens if word not in STOPWORDS]

    counter = Counter(filtered)
    return counter.most_common(top_n)


# ===========================================================================
# FEATURE 3 — 50-word Summary
# ===========================================================================

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert summarizer. Summarize the following YouTube comments "
                "in EXACTLY 50 words or fewer. Capture the dominant themes, tone, and "
                "viewer sentiment. Be concise and do not exceed 50 words."
            ),
        ),
        (
            "human",
            "Comments:\n{comments_text}",
        ),
    ]
)


def summarize_comments(comments: list[str], sample_size: int = 100) -> str:
    """
    Generate a ≤50-word summary of the YouTube comments using an LLM.

    To keep token usage manageable, we sample up to `sample_size` comments.
    The LLM is instructed to stay within 50 words.

    Args:
        comments    (list[str]): Raw comment strings
        sample_size (int)      : Max comments to send to LLM (default: 100)

    Returns:
        str: ≤50-word summary paragraph
    """
    llm = _get_llm()
    from langchain_core.output_parsers import StrOutputParser

    chain = SUMMARY_PROMPT | llm | StrOutputParser()

    # Use a representative sample to avoid exceeding context limits
    sample = comments[:sample_size]
    comments_text = "\n".join(
        [f"- {c}" for c in sample if c.strip()]
    )

    return chain.invoke({"comments_text": comments_text})
