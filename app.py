"""
app.py
------
Main Streamlit application entry point.

Layout:
  ┌─────────────────────────────────────────────┐
  │         YouTube Comment RAG Analyzer         │
  ├─────────────────────────────────────────────┤
  │  [YouTube URL Input]  [Load Comments]        │
  ├─────────────────────────────────────────────┤
  │  [Sentiment] [Common Words] [Summary]        │
  ├─────────────────────────────────────────────┤
  │  Chat Interface (st.chat_message)            │
  └─────────────────────────────────────────────┘

Session State keys:
  - comments      : list[str]  — fetched YouTube comments
  - retriever     : FAISS retriever
  - llm           : ChatOpenAI instance
  - messages      : list[dict] — chat history [{role, content}, ...]
  - video_loaded  : bool
"""

import streamlit as st
from dotenv import load_dotenv

from youtube import extract_video_id, get_comments
from rag_pipeline import build_rag_pipeline, answer_query
from utils import analyze_sentiment, get_common_words, summarize_comments

load_dotenv()


# Page config
st.set_page_config(
    page_title="YouTube Comment RAG Analyzer",
    page_icon="🎬",
    layout="centered",
)


# Initialize session state

if "comments" not in st.session_state:
    st.session_state.comments = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False


# Header

st.title("🎬 YouTube Comment RAG Analyzer")
st.caption(
    "Paste a YouTube video link, load the comments, then chat with them "
    "or run quick analyses."
)
st.divider()


# Section 1 — URL Input + Load Comments

st.subheader("📥 Load Comments")

url_col, btn_col = st.columns([4, 1], vertical_alignment="bottom")

with url_col:
    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

with btn_col:
    load_btn = st.button("Load Comments", type="primary", use_container_width=True)

if load_btn:
    if not youtube_url.strip():
        st.warning("Please paste a YouTube video URL first.")
    else:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Could not extract a valid video ID from the URL. Please check the link.")
        else:
            with st.spinner("Fetching comments from YouTube..."):
                try:
                    comments = get_comments(video_id, max_comments=200)
                    if not comments:
                        st.warning("No comments found for this video.")
                    else:
                        st.session_state.comments = comments
                        with st.spinner("Building RAG pipeline (embedding + indexing)..."):
                            retriever, llm = build_rag_pipeline(comments)
                            st.session_state.retriever = retriever
                            st.session_state.llm = llm
                        st.session_state.video_loaded = True
                        st.session_state.messages = []  # Reset chat on new video
                        st.success(
                            f"✅ Loaded **{len(comments)}** comments. "
                            "You can now chat or run analyses below."
                        )
                except ValueError as e:
                    st.error(f"Configuration error: {e}")
                except Exception as e:
                    st.error(f"Failed to fetch comments: {e}")


# Section 2 — Feature Buttons (only shown after comments are loaded)
if st.session_state.video_loaded:
    st.divider()
    st.subheader("🔍 Quick Analyses")

    feat_col1, feat_col2, feat_col3 = st.columns(3)

    # ── Button 1: Sentiment Analysis ──────────────────────────────────────
    with feat_col1:
        if st.button("😊 Sentiment Analysis", use_container_width=True):
            with st.spinner("Classifying comments (one LLM call per comment)..."):
                result = analyze_sentiment(st.session_state.comments)

            pos = result["positive_percent"]
            neg = result["negative_percent"]
            total = result["total_analyzed"]

            st.metric("Positive 😊", f"{pos}%")
            st.metric("Negative 😞", f"{neg}%")
            st.caption(f"Based on {total} comments analyzed")

            # Simple visual bar
            st.progress(pos / 100, text=f"Positive: {pos}%")

    # ── Button 2: Most Common Words ────────────────────────────────────────
    with feat_col2:
        if st.button("📊 Common Words", use_container_width=True):
            word_freq = get_common_words(st.session_state.comments, top_n=10)
            st.markdown("**Top 10 Words**")
            for word, count in word_freq:
                # Inline bar using Unicode blocks for a minimal look
                bar_len = int((count / word_freq[0][1]) * 20)
                bar = "█" * bar_len
                st.text(f"{word:<15} {bar} {count}")

    # ── Button 3: Summary ─────────────────────────────────────────────────
    with feat_col3:
        if st.button("📝 Summarize (50 words)", use_container_width=True):
            with st.spinner("Generating summary..."):
                summary = summarize_comments(st.session_state.comments)
            st.info(summary)

    # Section 3 — Chat Interface
    
    st.divider()
    st.subheader("💬 Chat with Comments")
    st.caption(
        "Ask anything about the video's comment section. "
        "Type **exit** to clear the chat."
    )

    # Render existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the comments...")

    if user_input:
        # Handle exit command
        if user_input.strip().lower() == "exit":
            st.session_state.messages = []
            st.rerun()

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate RAG answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = answer_query(
                        query=user_input,
                        retriever=st.session_state.retriever,
                        llm=st.session_state.llm,
                    )
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"Error generating answer: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


# Footer hint when no video is loaded

if not st.session_state.video_loaded:
    st.info(
        "👆 Paste a YouTube URL above and click **Load Comments** to get started."
    )
