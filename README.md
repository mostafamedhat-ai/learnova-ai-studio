<div align="center">

# ⚡ Learnova AI Studio

**Turn any lecture, recording, or document into an interactive AI study workspace.**

Ingest YouTube videos, audio/video files, and documents — then chat with evidence-grounded answers, generate structured summaries, visual study roadmaps, and adaptive quizzes.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-0467DF)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#️-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#️-configuration)
- [Running on Kaggle / Google Colab](#-running-on-kaggle--google-colab)
- [API Reference](#-api-reference)
- [Known Limitations](#️-known-limitations)
- [Security Notes](#-security-notes)
- [Roadmap](#️-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

Learnova AI Studio is a multimodal learning assistant split into two parts:

- **Frontend (`main.py`)** — a Streamlit app that handles ingestion locally (transcripts, speech-to-text, document parsing) and renders the study workspace.
- **Backend (`backend.py`)** — a FastAPI service that indexes content in FAISS using multilingual embeddings and uses Google Gemini to generate grounded answers and structured study material.

This split lets you run the heavy backend on a GPU machine (Kaggle / Colab, exposed through an Ngrok tunnel) while keeping the UI on your local machine.

<!--
## 📸 Screenshots
Add screenshots or a short demo GIF here, e.g.:
![Workspace](docs/workspace.png)
-->

---

## 🌟 Key Features

| Feature | Description |
| --- | --- |
| 📺 **Multimodal Ingestion** | YouTube URLs, audio/video files (`mp3`, `wav`, `m4a`, `mp4`, `mkv`), and documents (`PDF`, `DOCX`, `PPTX`, `TXT`, `MD`). |
| 🎙️ **Local Speech-to-Text Fallback** | Uses `Faster-Whisper` when a YouTube transcript is unavailable, and for uploaded audio/video. |
| 💬 **Evidence-Grounded RAG Chat** | Q&A over `FAISS` vector search with timestamp (`⏱️ 04:15`) and page (`📄 Page 3`) citations. Answers in the language of the question. |
| 🌍 **Multilingual** | Embeddings based on `paraphrase-multilingual-mpnet-base-v2` — supports Arabic, English, and mixed-language content. |
| 📑 **Structured Summaries** | Executive overview, key concept pillars, glossary of definitions, and actionable takeaways — validated with Pydantic schemas. |
| 🗺️ **Visual Study Roadmaps** | Step-by-step curriculum with time estimates, prerequisites, and an interactive `Mermaid.js` flowchart. |
| 🎯 **Adaptive Assessments** | Multiple-choice quizzes (3–10 questions) with distractors, explanations, source references, and automatic grading. |
| 📥 **Exportable Study Packs** | Download summaries and roadmaps as a portable Markdown file. |
| 🛡️ **Resilient LLM Layer** | Automatic model discovery, multi-model fallback, retries, and tolerant JSON parsing. |

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────┐
│            Streamlit Frontend (main.py)      │
│  • YouTube transcript / yt-dlp audio         │
│  • Faster-Whisper transcription              │
│  • PDF / DOCX / PPTX / TXT parsing           │
└───────────────────────┬──────────────────────┘
                        │  HTTPS + Bearer token
                        │  (local or Ngrok tunnel)
                        ▼
┌──────────────────────────────────────────────┐
│            FastAPI Backend (backend.py)      │
│                                              │
│  Chunks ─► Multilingual Embeddings (MPNet)   │
│                    │                         │
│                    ▼                         │
│              FAISS Vector Store              │
│                    │                         │
│                    ▼                         │
│   Gemini LLM ─► Pydantic Output Parsers      │
│                    │                         │
│                    ▼                         │
│     Summary · Roadmap · Quiz · RAG Answer    │
└──────────────────────────────────────────────┘
```

**Request flow**

1. The frontend extracts text into chunks (with `start`/`end` timestamps or `page` numbers).
2. `POST /INGEST_CONTENT` embeds the chunks, builds a FAISS index, and returns an instant topic acknowledgment.
3. Generation endpoints (`/GENERATE_*`) send the full text to Gemini with schema-enforced prompts and return validated JSON.
4. `POST /ASK_RAG` retrieves the top-k chunks and asks Gemini to answer using only those excerpts, citing their source tags.

---

## 🧰 Tech Stack

- **Frontend:** Streamlit, Requests, Mermaid.js (via CDN)
- **Backend:** FastAPI, Uvicorn, Pydantic v2
- **LLM & Orchestration:** LangChain, Google Gemini (`langchain-google-genai`)
- **Retrieval:** FAISS, Sentence-Transformers (`paraphrase-multilingual-mpnet-base-v2`)
- **Ingestion:** `yt-dlp`, `youtube-transcript-api`, `faster-whisper`, `pypdf`, `python-docx`, `python-pptx`
- **Tunneling:** `pyngrok`, `nest-asyncio` (for notebook environments)

---

## 📁 Project Structure

```text
.
├── main.py            # Streamlit frontend (ingestion + study workspace UI)
├── backend.py         # FastAPI backend (embeddings, FAISS, Gemini, endpoints)
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- A **Google Gemini API key** — get one from [Google AI Studio](https://aistudio.google.com/)
- *(Optional)* An [Ngrok](https://ngrok.com/) auth token if you want to expose the backend remotely
- *(Optional)* A CUDA-capable GPU for faster embeddings

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Set environment variables

```bash
# Linux / macOS
export GOOGLE_API_KEY="your-gemini-api-key"
export API_KEY="choose-a-long-random-secret"

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your-gemini-api-key"
$env:API_KEY="choose-a-long-random-secret"
```

> 💡 Generate a strong secret with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

### 4. Start the backend

```bash
python backend.py
```

The first start downloads the embedding model and pre-warms it, so it may take a minute. The API will be available at `http://localhost:8000` (interactive docs at `/docs`).

If `NGROK_AUTH_TOKEN` is set, a public URL is printed in the console.

### 5. Start the frontend

In a second terminal:

```bash
streamlit run main.py
```

Then open **⚙️ System Configuration** in the sidebar and set:

- **Backend Endpoint** → `http://localhost:8000` (or your Ngrok URL)
- **Security Token** → the same value as your `API_KEY`

Click **🔌 Test Connection** to verify, then ingest your first lecture or document.

---

## ⚙️ Configuration

All backend settings are read from environment variables:

| Variable | Required | Default | Description |
| --- | :---: | --- | --- |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | ✅ | — | Google Gemini API key. |
| `API_KEY` | ✅ | — | Bearer token the frontend must send to the backend. Use a strong random value. |
| `GEMINI_MODEL` | ❌ | `gemini-3.8-flash` | Preferred Gemini model. The backend auto-discovers available models and falls back automatically. |
| `EMBEDDING_MODEL` | ❌ | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Sentence-Transformers model used for retrieval. |
| `NGROK_AUTH_TOKEN` | ❌ | — | Enables a public Ngrok tunnel. |
| `HOST` | ❌ | `0.0.0.0` | Server bind address. |
| `PORT` | ❌ | `8000` | Server port. |

> On Kaggle, secrets named `GOOGLE_API_KEY`, `NGROK_AUTH_TOKEN`, and `API_KEY` are picked up automatically via `UserSecretsClient`.

---

## ☁️ Running on Kaggle / Google Colab

Run the backend on a free GPU and keep the Streamlit UI on your own machine.

```python
!pip install -q fastapi uvicorn pydantic langchain-core langchain-community \
    langchain-google-genai sentence-transformers faiss-cpu pyngrok nest_asyncio

import backend

backend.start_server(
    port=8000,
    ngrok_token="YOUR_NGROK_TOKEN",
    google_api_key="YOUR_GEMINI_KEY",
)
```

Copy the printed **Ngrok public URL** into the Streamlit sidebar (**Backend Endpoint**) and use your `API_KEY` as the **Security Token**.

---

## 📡 API Reference

All endpoints except `/health` require the header:

```http
Authorization: Bearer <API_KEY>
```

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status and the currently active content. |
| `POST` | `/INGEST_CONTENT` | Index chunks into FAISS and return a topic acknowledgment. |
| `POST` | `/GENERATE_SUMMARY` | Structured summary of the active content. |
| `POST` | `/GENERATE_ROADMAP` | Study roadmap with a Mermaid diagram. |
| `POST` | `/GENERATE_QUIZ?num_questions=5` | Multiple-choice quiz with explanations. |
| `POST` | `/ASK_RAG` | Grounded question answering with citations. |

**Example: ingest content**

```bash
curl -X POST http://localhost:8000/INGEST_CONTENT \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "demo_lecture",
    "title": "Intro to Neural Networks",
    "source_type": "youtube",
    "chunks": [
      {"text": "A neural network is a function approximator...", "start": 0.0, "end": 12.5},
      {"text": "Backpropagation computes gradients layer by layer...", "start": 12.5, "end": 31.0}
    ]
  }'
```

For documents, send `"page": <number>` instead of `start`/`end`.

**Example: ask a question**

```bash
curl -X POST http://localhost:8000/ASK_RAG \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is backpropagation?", "top_k": 4}'
```

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## ⚠️ Known Limitations

- **Single active content:** the backend keeps one active document/video in memory at a time. Ingesting new content replaces the previous one for all clients. It is designed for a single user or demo.
- **In-memory index:** the FAISS index is not persisted; restarting the backend clears it.
- **Context window cap:** summaries, roadmaps, and quizzes use only the first ~30,000 characters of the content.
- **Chunking:** documents are indexed in the chunks produced by the frontend. Very long pages are truncated by the embedding model's sequence limit, which can reduce retrieval quality.
- **Quiz and structured output:** LLM output is validated with Pydantic; if generation repeatedly fails, the backend returns a generic fallback result.
- **Whisper on CPU:** transcription uses the `base` model on CPU by default, so long recordings can take a while.

---

## 🔐 Security Notes

- **Always set a strong `API_KEY`.** Never deploy or tunnel the backend with a guessable token.
- Never commit API keys, Ngrok tokens, or `.env` files. The included `.gitignore` excludes `.streamlit/secrets.toml`; add `.env` if you use one.
- When exposing the backend through Ngrok, anyone with the URL can reach `/health` — treat the URL as semi-private and rotate the tunnel when you are done.
- Content you ingest is sent to Google Gemini for processing. Do not upload confidential material unless that is acceptable under your data policy.

---

## 🛣️ Roadmap

- [ ] Persistent vector store and multi-user sessions
- [ ] Smarter chunking (token-aware splitting with overlap)
- [ ] Conversation memory in RAG chat
- [ ] Include roadmap diagram and quiz in the exported study pack
- [ ] Clickable timestamp citations that seek the embedded video
- [ ] Docker / `docker-compose` setup
- [ ] Automated tests and CI

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

Built with ❤️ for learners everywhere.

</div>
