"""Learnova AI Studio - Modern Autonomous Learning & Knowledge Workspace

Advanced Streamlit Frontend (APP2.py) with Modern SaaS UI, Glassmorphism,
Subtle CSS Micro-Animations, and Collapsible System Configuration.

Run locally:
    streamlit run app2.py
"""

from __future__ import annotations

import ast
import io
import os
import tempfile
from typing import Any, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------------------------------- #
# Page Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Learnova AI Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Modern Custom CSS & Subtle Micro-Animations
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.35);
        --accent: #06b6d4;
        --bg-glass: rgba(30, 41, 59, 0.65);
        --border-glass: rgba(255, 255, 255, 0.08);
        --card-hover: rgba(51, 65, 85, 0.5);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Keyframe Animations */
    @keyframes fadeInSlideUp {
        0% {
            opacity: 0;
            transform: translateY(12px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
        }
        50% {
            box-shadow: 0 0 25px rgba(6, 182, 212, 0.4);
        }
    }

    @keyframes liveStatusBlink {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.9); }
    }

    /* Header Container Styling */
    .brand-hero {
        animation: fadeInSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }

    .brand-hero::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 40%);
        pointer-events: none;
    }

    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        animation: liveStatusBlink 2s infinite ease-in-out;
    }

    /* Modern Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: -0.01em;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .stButton>button:active {
        transform: translateY(0);
    }

    /* Cards & Containers */
    .glass-card {
        animation: fadeInSlideUp 0.5s ease-out forwards;
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        margin-bottom: 16px;
    }

    .glass-card:hover {
        background: var(--card-hover);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    /* Concept & Definition Cards */
    .definition-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 4px solid #06b6d4;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 16px 18px;
        border-radius: 10px;
        margin-bottom: 14px;
        transition: all 0.25s ease;
    }

    .definition-box:hover {
        transform: translateX(4px);
        border-left-color: #38bdf8;
        box-shadow: 0 6px 18px rgba(6, 182, 212, 0.15);
    }

    /* Chat bubble smooth animation */
    [data-testid="stChatMessage"] {
        animation: fadeInSlideUp 0.35s ease-out forwards;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 12px 16px;
        margin-bottom: 10px;
        backdrop-filter: blur(8px);
    }

    /* Smooth Input Styling */
    .stTextInput input, .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border-bottom: 2px solid #6366f1 !important;
        color: #818cf8 !important;
    }

    /* Subtle scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.2);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(148, 163, 184, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helper Utilities & Extractors (Local Ingestion)
# --------------------------------------------------------------------------- #
def sanitize_display_text(raw_text: Any) -> str:
    """Extract clean human-readable text from any LLM response format.

    Guarantees no raw dictionaries or metadata structures appear in the UI.
    """
    if raw_text is None:
        return ""
    if isinstance(raw_text, dict):
        return str(raw_text.get("text", raw_text.get("answer", str(raw_text))))
    if isinstance(raw_text, list):
        return "\n".join(sanitize_display_text(part) for part in raw_text)

    text = str(raw_text).strip()

    # If the response string looks like a stringified Python dictionary
    if (text.startswith("{'type': 'text'") or text.startswith('{"type": "text"')) and "'text':" in text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and "text" in parsed:
                return str(parsed["text"]).strip()
        except Exception:
            pass

    return text


def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube Video ID from standard, short, or embedded links."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        return parsed.path.lstrip("/").split("/")[0]
    if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        q = parse_qs(parsed.query)
        if q.get("v"):
            return q["v"][0]
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1]
    return None


def download_youtube_audio_local(youtube_url: str) -> tuple[bytes, str]:
    """Download raw audio stream from YouTube without requiring system ffmpeg."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("Please install yt-dlp for audio fallback: pip install yt-dlp")

    ffmpeg_path = None
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_template = os.path.join(tmp_dir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        if ffmpeg_path:
            ydl_opts["ffmpeg_location"] = ffmpeg_path

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        for file in os.listdir(tmp_dir):
            if file.endswith((".m4a", ".webm", ".opus", ".mp3", ".wav")):
                ext = file.split(".")[-1]
                with open(os.path.join(tmp_dir, file), "rb") as f:
                    return f.read(), ext
    raise RuntimeError("Could not extract native audio track from YouTube video.")


def _get_snippet_attr(item: Any, key: str, default: Any = None) -> Any:
    """Safely extract attribute or dictionary key across youtube_transcript_api versions."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def fetch_youtube_transcript_local(video_id: str, full_url: Optional[str] = None) -> List[dict]:
    """Fetch YouTube transcript locally, with automatic Faster-Whisper fallback if unavailable."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            t = transcript_list.find_transcript(["en", "en-US", "en-GB", "ar"])
        except Exception:
            try:
                t = transcript_list.find_generated_transcript(["en", "ar"])
            except Exception:
                t = next(iter(transcript_list))

        snippets = t.fetch()
        chunks = []
        buffer_text = []
        start_time = 0.0

        for s in snippets:
            s_text = str(_get_snippet_attr(s, "text", "")).replace("\n", " ").strip()
            s_start = float(_get_snippet_attr(s, "start", 0.0))
            s_duration = float(_get_snippet_attr(s, "duration", 0.0))

            if not s_text:
                continue

            if not buffer_text:
                start_time = s_start

            buffer_text.append(s_text)

            if len(buffer_text) >= 6 or (len(buffer_text) >= 3 and s_text.endswith((".", "!", "?"))):
                chunks.append({
                    "text": " ".join(buffer_text),
                    "start": start_time,
                    "end": s_start + s_duration,
                })
                buffer_text = []

        if buffer_text:
            last_snippet = snippets[-1]
            last_end = float(_get_snippet_attr(last_snippet, "start", 0.0)) + float(_get_snippet_attr(last_snippet, "duration", 0.0))
            chunks.append({
                "text": " ".join(buffer_text),
                "start": start_time,
                "end": last_end,
            })

        if chunks:
            return chunks
        raise RuntimeError("No transcript text could be extracted.")

    except Exception as exc:
        if full_url:
            st.info("No official transcript found. Initiating automatic Whisper speech-to-text fallback...")
            audio_bytes, ext = download_youtube_audio_local(full_url)
            return transcribe_audio_whisper_local(audio_bytes, ext)
        raise RuntimeError(f"Could not retrieve YouTube transcript and no URL provided for fallback: {exc}")


def transcribe_audio_whisper_local(file_bytes: bytes, file_ext: str = "mp3") -> List[dict]:
    """Transcribe audio/video using local faster-whisper with av compatibility patch."""
    try:
        import av
        if not hasattr(av, "_patched_open_safe"):
            _orig_av_open = av.open
            def _patched_open_safe(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return _orig_av_open(*args, **kwargs)
            av.open = _patched_open_safe
            av._patched_open_safe = True
    except Exception:
        pass

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("Please install faster-whisper: pip install faster-whisper")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp_path, beam_size=5)

        chunks = []
        for segment in segments:
            chunks.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
            })
        return chunks
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_pdf_local(file_bytes: bytes) -> List[dict]:
    """Extract text from PDF page by page."""
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        chunks = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append({
                    "text": text.strip(),
                    "page": page_num,
                })
        return chunks
    except ImportError:
        raise RuntimeError("Please install pypdf: pip install pypdf")


def extract_docx_local(file_bytes: bytes) -> List[dict]:
    """Extract paragraphs from DOCX."""
    try:
        import docx

        doc = docx.Document(io.BytesIO(file_bytes))
        chunks = []
        current_section = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                current_section.append(text)
                if len(current_section) >= 4:
                    chunks.append({"text": "\n".join(current_section), "page": len(chunks) + 1})
                    current_section = []
        if current_section:
            chunks.append({"text": "\n".join(current_section), "page": len(chunks) + 1})
        return chunks
    except ImportError:
        raise RuntimeError("Please install python-docx: pip install python-docx")


def extract_pptx_local(file_bytes: bytes) -> List[dict]:
    """Extract text from PowerPoint presentations."""
    try:
        import pptx

        prs = pptx.Presentation(io.BytesIO(file_bytes))
        chunks = []
        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_texts.append(text)
            if slide_texts:
                chunks.append({
                    "text": f"--- Slide {slide_idx} ---\n" + "\n".join(slide_texts),
                    "page": slide_idx,
                })
        return chunks
    except ImportError:
        raise RuntimeError("Please install python-pptx: pip install python-pptx")


# --------------------------------------------------------------------------- #
# Backend API Caller
# --------------------------------------------------------------------------- #
def call_api(endpoint: str, payload: dict = None, timeout: int = 300) -> dict:
    base_url = st.session_state.get("backend_url", "http://localhost:8000").rstrip("/")
    api_key = st.session_state.get("api_key", "default-secret-key-123").strip()
    headers = {"Authorization": f"Bearer {api_key}"}

    url = f"{base_url}/{endpoint.lstrip('/')}"
    try:
        if payload is not None:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        else:
            resp = requests.post(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Backend connection error: {e}")

    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Backend error ({resp.status_code}): {detail}")

    return resp.json()


# --------------------------------------------------------------------------- #
# State Initialization
# --------------------------------------------------------------------------- #
if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"
if "api_key" not in st.session_state:
    st.session_state.api_key = "default-secret-key-123"
if "backend_connected" not in st.session_state:
    st.session_state.backend_connected = None
if "active_content" not in st.session_state:
    st.session_state.active_content = None
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None
if "roadmap_data" not in st.session_state:
    st.session_state.roadmap_data = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


# --------------------------------------------------------------------------- #
# Sidebar - Minimalist & Hidden System Configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    # Sleek Brand Badge
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
            <div style="background: linear-gradient(135deg, #6366f1, #06b6d4); border-radius: 12px; padding: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(99,102,241,0.4);">
                <span style="font-size: 22px;">⚡</span>
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; color: #f8fafc;">Learnova AI</div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">Studio Edition v2.5</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean Live Status Indicator (Subtle)
    status_color = "#10b981" if st.session_state.backend_connected else "#94a3b8"
    status_label = "Backend Connected" if st.session_state.backend_connected else "Click to Verify"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 8px 12px; margin-bottom: 16px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span class="live-dot" style="background-color: {status_color};"></span>
                <span style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">{status_label}</span>
            </div>
            <span style="font-size: 0.7rem; color: #64748b; font-family: monospace;">port:8000</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hidden / Collapsed System Config (Expander defaults to False)
    with st.expander("⚙️ System Configuration", expanded=False):
        st.caption("Configure your remote Ngrok tunnel or local FastAPI endpoint.")
        backend_url_input = st.text_input(
            "Backend Endpoint",
            value=st.session_state.backend_url,
            placeholder="https://xxxx.ngrok-free.app",
            help="Paste your public Ngrok tunnel URL or localhost address.",
        )
        api_key_input = st.text_input(
            "Security Token",
            value=st.session_state.api_key,
            type="password",
            help="Backend authentication token.",
        )

        st.session_state.backend_url = backend_url_input
        st.session_state.api_key = api_key_input

        if st.button("🔌 Test Connection", use_container_width=True):
            try:
                h = requests.get(f"{backend_url_input.rstrip('/')}/health", timeout=5).json()
                st.session_state.backend_connected = True
                st.success(f"Verified! Active: {h.get('active_title') or 'None'}")
            except Exception as e:
                st.session_state.backend_connected = False
                st.error(f"Failed: {e}")

    st.divider()

    # Workspace Navigation / Shortcuts
    st.markdown("<p style='font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748b; letter-spacing:0.05em;'>Workspace Specs</p>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.8;">
            <div>⚡ <b>Engine:</b> Gemini 3.8 Flash</div>
            <div>🧠 <b>Vector DB:</b> FAISS Multilingual</div>
            <div>🎙️ <b>Transcriber:</b> Faster-Whisper</div>
            <div>📊 <b>Diagrams:</b> Mermaid.js Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.active_content:
        st.divider()
        if st.button("🗑️ Reset Workspace", use_container_width=True):
            st.session_state.active_content = None
            st.session_state.summary_data = None
            st.session_state.roadmap_data = None
            st.session_state.quiz_data = None
            st.session_state.chat_messages = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()


# --------------------------------------------------------------------------- #
# Hero Header with Glassmorphism
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="brand-hero">
        <div class="brand-badge"><span class="live-dot"></span> Next-Gen Knowledge Hub</div>
        <h1 class="brand-title">Learnova AI Studio</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 8px; margin-bottom: 0;">
            Universal multimodal study workspace. Ingest video lectures, recordings, and documents into an interactive AI companion.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Universal Ingestion Zone
# --------------------------------------------------------------------------- #
with st.expander("📥 Ingest Learning Material (YouTube / Audio / Documents)", expanded=(st.session_state.active_content is None)):
    source_choice = st.radio(
        "Select Content Ingestion Source:",
        ["📺 YouTube Lecture URL", "🎙️ Audio / Video Recording", "📄 Academic Document"],
        horizontal=True,
    )

    chunks_extracted = []
    content_id = ""
    content_title = ""
    source_type = ""

    if source_choice == "📺 YouTube Lecture URL":
        youtube_url = st.text_input("YouTube Lecture URL:", placeholder="https://www.youtube.com/watch?v=...")
        if st.button("⚡ Ingest & Analyze Video", type="primary", use_container_width=True):
            if not youtube_url:
                st.warning("Please provide a valid YouTube video URL.")
            else:
                vid_id = extract_youtube_id(youtube_url)
                if not vid_id:
                    st.error("Invalid YouTube URL structure.")
                else:
                    with st.spinner("Extracting transcript & generating smart index..."):
                        try:
                            chunks_extracted = fetch_youtube_transcript_local(vid_id, full_url=youtube_url)
                            content_id = f"yt_{vid_id}"
                            content_title = f"YouTube Lecture ({vid_id})"
                            source_type = "youtube"
                            st.session_state["media_url"] = youtube_url
                        except Exception as e:
                            st.error(f"Extraction error: {e}")

    elif source_choice == "🎙️ Audio / Video Recording":
        uploaded_media = st.file_uploader("Upload Audio or Video:", type=["mp3", "wav", "m4a", "mp4", "mkv"])
        if uploaded_media and st.button("⚡ Transcribe with Faster-Whisper", type="primary", use_container_width=True):
            with st.spinner("Transcribing speech locally with precise timestamps..."):
                ext = uploaded_media.name.split(".")[-1]
                chunks_extracted = transcribe_audio_whisper_local(uploaded_media.read(), ext)
                content_id = uploaded_media.name
                content_title = uploaded_media.name
                source_type = "audio"

    elif source_choice == "📄 Academic Document":
        uploaded_doc = st.file_uploader("Upload Document File:", type=["pdf", "docx", "pptx", "txt", "md"])
        if uploaded_doc and st.button("⚡ Extract & Index Document", type="primary", use_container_width=True):
            with st.spinner("Parsing document structure & sections..."):
                file_bytes = uploaded_doc.read()
                filename = uploaded_doc.name
                content_id = filename
                content_title = filename
                source_type = "document"

                if filename.endswith(".pdf"):
                    chunks_extracted = extract_pdf_local(file_bytes)
                elif filename.endswith(".docx"):
                    chunks_extracted = extract_docx_local(file_bytes)
                elif filename.endswith(".pptx"):
                    chunks_extracted = extract_pptx_local(file_bytes)
                else:
                    chunks_extracted = [{"text": file_bytes.decode("utf-8", errors="ignore"), "page": 1}]

    # Ingestion Pipeline Call
    if chunks_extracted:
        with st.spinner("Indexing vector embeddings and preparing workspace..."):
            try:
                ingest_payload = {
                    "content_id": content_id,
                    "title": content_title,
                    "source_type": source_type,
                    "chunks": chunks_extracted,
                }
                res = call_api("/INGEST_CONTENT", payload=ingest_payload, timeout=300)
                st.session_state.active_content = res
                st.session_state.summary_data = None
                st.session_state.roadmap_data = None
                st.session_state.quiz_data = None
                st.session_state.chat_messages = []
                st.session_state.quiz_submitted = False
                st.rerun()
            except Exception as e:
                st.error(f"Ingest submission failed: {e}")


# --------------------------------------------------------------------------- #
# Active Knowledge Workspace Tabs
# --------------------------------------------------------------------------- #
if st.session_state.active_content:
    act = st.session_state.active_content

    # Clean Acknowledgment Banner
    ack_clean = sanitize_display_text(act.get("acknowledgment", ""))
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid #6366f1; padding: 14px 20px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #818cf8; font-weight: 700; margin-bottom: 4px;">Active Workspace</div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc;">{act.get('title', 'Content')}</div>
            <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 4px;">{ack_clean}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5 High-Impact Tabs
    tab_chat, tab_summary, tab_roadmap, tab_quiz, tab_export = st.tabs([
        "💬 Interactive RAG Chat",
        "📑 Structured Summary & Concepts",
        "🗺️ Progressive Roadmap",
        "🎯 Adaptive Assessment",
        "📥 Export Study Pack",
    ])

    # ----------------------------------------------------------------------- #
    # TAB 1: RAG Chat
    # ----------------------------------------------------------------------- #
    with tab_chat:
        col_view, col_chat = st.columns([1, 1.25], gap="large")

        with col_view:
            st.markdown("#### 📺 Source Reference")
            if st.session_state.get("media_url"):
                st.video(st.session_state["media_url"])
            else:
                st.info(f"Active Document: **{act['title']}** ({act['chunk_count']} indexed chunks)")

        with col_chat:
            st.markdown("#### 💬 Ask Questions with Evidence Citations")

            # Render Chat Messages
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(sanitize_display_text(msg["content"]))

            user_q = st.chat_input("Ask any question about this content...")
            if user_q:
                st.session_state.chat_messages.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Searching vector index and retrieving grounded evidence..."):
                        try:
                            ans = call_api("/ASK_RAG", {"question": user_q, "top_k": 4})
                            reply = sanitize_display_text(ans.get("answer", "No response received."))
                            st.markdown(reply)
                            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.error(str(e))

    # ----------------------------------------------------------------------- #
    # TAB 2: Structured Summary
    # ----------------------------------------------------------------------- #
    with tab_summary:
        st.markdown("#### 📑 Executive Summary & Conceptual Pillars")

        if not st.session_state.summary_data:
            if st.button("✨ Synthesize Structured Summary", type="primary"):
                with st.spinner("Analyzing content across core themes, definitions, and takeaways..."):
                    try:
                        st.session_state.summary_data = call_api("/GENERATE_SUMMARY")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            sd = st.session_state.summary_data

            st.markdown(
                f"""
                <div class="glass-card">
                    <h5 style="color: #818cf8; margin-top:0;">🌟 Executive Overview</h5>
                    <p style="color: #cbd5e1; line-height: 1.6; margin-bottom: 0;">{sd['overview']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_concepts, col_defs = st.columns(2, gap="medium")
            with col_concepts:
                st.markdown("##### 💡 Key Concepts & Core Themes")
                for kc in sd.get("key_concepts", []):
                    with st.container(border=True):
                        st.markdown(f"**📌 {kc['topic']}**")
                        for pt in kc.get("key_points", []):
                            st.markdown(f"- {pt}")

            with col_defs:
                st.markdown("##### 📖 Important Definitions & Terminology")
                for d in sd.get("important_definitions", []):
                    st.markdown(
                        f"""
                        <div class="definition-box">
                            <strong style="color:#06b6d4;">🏷️ {d['term']}</strong><br/>
                            <span style="color:#e2e8f0; font-size:0.9rem;">{d['definition']}</span><br/>
                            <small style="color:#94a3b8;">Context: {d.get('context', '')}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("##### 🚀 High-Value Actionable Takeaways")
            for tk in sd.get("actionable_takeaways", []):
                st.markdown(f"✅ {tk}")

    # ----------------------------------------------------------------------- #
    # TAB 3: Study Roadmap
    # ----------------------------------------------------------------------- #
    with tab_roadmap:
        st.markdown("#### 🗺️ Progressive Learning Curriculum")

        if not st.session_state.roadmap_data:
            if st.button("✨ Generate Study Roadmap", type="primary"):
                with st.spinner("Synthesizing learning path and visual flowchart..."):
                    try:
                        st.session_state.roadmap_data = call_api("/GENERATE_ROADMAP")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            rd = st.session_state.roadmap_data
            st.info(f"🎯 **Learning Goal:** {rd.get('learning_goal', '')}")

            mermaid_code = rd.get("mermaid_diagram", "")
            if mermaid_code:
                st.markdown("##### 📊 Visual Learning Path")
                html_mermaid = f"""
                <div class="mermaid" style="display:flex; justify-content:center;">
                {mermaid_code}
                </div>
                <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
                </script>
                """
                components.html(html_mermaid, height=360, scrolling=True)

            st.markdown("##### 📝 Step-by-Step Curriculum")
            for step in rd.get("steps", []):
                with st.expander(f"📍 Step {step['step_number']}: {step['title']} (Est. Time: {step['estimated_time']})", expanded=True):
                    st.write(step["description"])
                    if step.get("prerequisites"):
                        st.caption(f"Prerequisites: {', '.join(step['prerequisites'])}")

    # ----------------------------------------------------------------------- #
    # TAB 4: Adaptive Assessment (Quiz)
    # ----------------------------------------------------------------------- #
    with tab_quiz:
        st.markdown("#### 🎯 Interactive Assessment & Knowledge Check")

        if not st.session_state.quiz_data:
            num_q = st.slider("Select Assessment Question Count:", min_value=3, max_value=10, value=5)
            if st.button("✨ Generate Assessment Quiz", type="primary"):
                with st.spinner("Formulating questions, logical distractors, and source citations..."):
                    try:
                        st.session_state.quiz_data = call_api(f"/GENERATE_QUIZ?num_questions={num_q}")
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        else:
            qd = st.session_state.quiz_data
            st.markdown(f"### 📋 {qd.get('quiz_title', 'Knowledge Assessment')} (Difficulty: **{qd.get('difficulty', 'Intermediate')}**)")

            with st.form("quiz_form"):
                for q in qd.get("questions", []):
                    st.markdown(f"**Q{q['id']}: {q['question']}**")
                    user_selected = st.radio(
                        "Select your answer:",
                        q["options"],
                        key=f"q_{q['id']}",
                        index=None,
                        label_visibility="collapsed",
                    )
                    st.session_state.quiz_answers[q["id"]] = user_selected
                    st.write("")

                submitted = st.form_submit_button("📊 Submit & Grade Assessment", type="primary", use_container_width=True)
                if submitted:
                    st.session_state.quiz_submitted = True

            if st.session_state.quiz_submitted:
                total_questions = len(qd.get("questions", []))
                correct_count = 0

                st.divider()
                st.markdown("### 🏆 Detailed Performance Analysis:")

                for q in qd.get("questions", []):
                    chosen = st.session_state.quiz_answers.get(q["id"])
                    correct = q["correct_option"]

                    is_correct = (chosen == correct)
                    if is_correct:
                        correct_count += 1
                        st.success(f"**Q{q['id']}: Correct! ✅**\n\nYour Answer: `{chosen}`\n\n💡 **Explanation:** {q['explanation']}\n\n📌 Reference: `{q.get('source_reference', '')}`")
                    else:
                        st.error(f"**Q{q['id']}: Incorrect ❌**\n\nYour Answer: `{chosen or 'Unanswered'}`\n\nCorrect Answer: `{correct}`\n\n💡 **Explanation:** {q['explanation']}\n\n📌 Reference: `{q.get('source_reference', '')}`")

                score_pct = (correct_count / total_questions) * 100
                st.metric("Final Score:", f"{correct_count} / {total_questions} ({score_pct:.1f}%)")

                if st.button("🔄 Retake with Fresh Questions"):
                    st.session_state.quiz_data = None
                    st.rerun()

    # ----------------------------------------------------------------------- #
    # TAB 5: Export Study Pack
    # ----------------------------------------------------------------------- #
    with tab_export:
        st.markdown("#### 📥 Export Portable Study Package")
        st.caption("Generate and download a Markdown study document with your executive summaries, diagrams, and question sets.")

        markdown_content = f"# 🎓 Study Pack: {act['title']}\n\n"
        markdown_content += f"> **Overview:** {sanitize_display_text(act['acknowledgment'])}\n\n---\n\n"

        if st.session_state.summary_data:
            sd = st.session_state.summary_data
            markdown_content += f"## 📑 Executive Summary\n{sd['overview']}\n\n"
            markdown_content += "### 💡 Key Concepts\n"
            for kc in sd.get("key_concepts", []):
                markdown_content += f"#### {kc['topic']}\n"
                for p in kc.get("key_points", []):
                    markdown_content += f"- {p}\n"
            markdown_content += "\n### 📖 Glossary & Definitions\n"
            for d in sd.get("important_definitions", []):
                markdown_content += f"- **{d['term']}**: {d['definition']} (Context: {d.get('context', '')})\n"
            markdown_content += "\n### ✅ Actionable Takeaways\n"
            for tk in sd.get("actionable_takeaways", []):
                markdown_content += f"- {tk}\n"

        if st.session_state.roadmap_data:
            rd = st.session_state.roadmap_data
            markdown_content += f"\n\n---\n\n## 🗺️ Study Roadmap\n**Goal:** {rd.get('learning_goal', '')}\n\n"
            for step in rd.get("steps", []):
                markdown_content += f"### Step {step['step_number']}: {step['title']} (Time: {step['estimated_time']})\n"
                markdown_content += f"{step['description']}\n\n"

        st.download_button(
            label="💾 Download Comprehensive Study Pack (.md)",
            data=markdown_content.encode("utf-8"),
            file_name=f"StudyPack_{act['content_id']}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )

else:
    st.info("👈 Ingest a video lecture, audio file, or document above to launch your interactive study workspace.")
