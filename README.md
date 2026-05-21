# 🎬 YouTube Comment RAG Analyzer

A  Streamlit app that fetches YouTube comments and lets you
**chat with them** using a RAG (Retrieval-Augmented Generation) pipeline powered
by LangChain, OpenAI, and FAISS.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **RAG Chat** | Ask any question about the comments; context is retrieved from FAISS and injected into the prompt |
| 😊 **Sentiment Analysis** | LLM classifies every comment as positive/negative (Pydantic structured output) |
| 📊 **Common Words** | Top 10 most frequent meaningful words (stopwords removed) |
| 📝 **50-word Summary** | LLM-generated concise summary of all comments |

---

## 📁 Project Structure

```
yt-rag-app/
│
├── app.py            # Streamlit UI — entry point
├── rag_pipeline.py   # Document loading, FAISS, RAG chain
├── youtube.py        # YouTube Data API — comment fetching
├── utils.py          # Sentiment, common words, summary
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── README.md         # This file
```

---




