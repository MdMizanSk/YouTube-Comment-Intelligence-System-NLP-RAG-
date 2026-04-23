# 🎬 YouTube Comment RAG Analyzer

A production-ready Streamlit app that fetches YouTube comments and lets you
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

## ⚙️ Setup Instructions

### 1. Clone / Download the Project

```bash
git clone <repo-url>
cd yt-rag-app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate (macOS / Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
cp .env.example .env
```

Open `.env` and fill in both keys:

```env
OPENAI_API_KEY=sk-...
YOUTUBE_API_KEY=AIza...
```

#### Getting Your Keys

**OpenAI API Key**
1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Paste it as `OPENAI_API_KEY`

**YouTube Data API v3 Key**
1. Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Create or select a project
3. Navigate to **APIs & Services → Library**
4. Enable **YouTube Data API v3**
5. Go to **APIs & Services → Credentials → Create Credentials → API Key**
6. Paste the key as `YOUTUBE_API_KEY`

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 🧪 Example Usage

1. **Paste a YouTube URL** into the input box, e.g.:
   ```
   https://www.youtube.com/watch?v=dQw4w9WgXcQ
   ```

2. Click **Load Comments** — the app fetches up to 200 comments and builds the FAISS vector store.

3. **Chat with comments:**
   - *"What do viewers think about the video quality?"*
   - *"Are people asking for a part 2?"*
   - *"What is the most praised aspect of this video?"*
   - Type `exit` to clear the conversation.

4. **Run quick analyses:**
   - 😊 **Sentiment** — see positive vs negative breakdown as a percentage
   - 📊 **Common Words** — visual frequency chart of top 10 words
   - 📝 **Summary** — one-paragraph ≤50-word digest of viewer sentiment

---

## 🏗️ Architecture

```
YouTube URL
    │
    ▼
extract_video_id()  ──►  get_comments()  [YouTube Data API v3]
                              │
                              ▼
                    build_documents()  →  chunk_documents()
                              │             (chunk=300, overlap=50)
                              ▼
                    OpenAIEmbeddings  →  FAISS vector store
                              │
                    ┌─────────┴──────────────────────┐
                    │                                │
                 RAG Chat                       Utilities
            (retriever k=5                  ┌───────┬───────┐
          + ChatPromptTemplate          Sentiment  Words  Summary
          + gpt-4o-mini                 (Pydantic) (freq)  (LLM)
          + StrOutputParser)
```

---

## ⚠️ Notes

- **Sentiment analysis** makes one LLM call per comment (brute-force for clarity).
  For large videos (200+ comments) this may take 1–3 minutes.
  Batching is a planned future optimization.

- **Model used:** `gpt-4o-mini` (temperature=0) for cost efficiency and determinism.

- **Comment limit:** Capped at 200 comments by default. Change `max_comments` in
  `app.py → get_comments(video_id, max_comments=200)` to adjust.

- **FAISS is in-memory.** The index is rebuilt every time you load a new video.

---

## 📦 Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| LLM | ChatOpenAI (`gpt-4o-mini`) |
| Embeddings | OpenAIEmbeddings |
| Vector Store | FAISS (in-memory) |
| Orchestration | LangChain (chain, prompt, parser) |
| Structured Output | Pydantic `BaseModel` + `with_structured_output()` |
| YouTube Data | `google-api-python-client` |
| Env Management | `python-dotenv` |

---

## 🐛 Troubleshooting

| Error | Fix |
|---|---|
| `YOUTUBE_API_KEY is not set` | Add key to `.env` file |
| `HttpError 403` | Enable YouTube Data API v3 in GCP console |
| `HttpError 400 videoNotFound` | Check the video URL is correct and public |
| `openai.AuthenticationError` | Check `OPENAI_API_KEY` in `.env` |
| Comments load but chat errors | Ensure sufficient OpenAI quota/credits |

---


