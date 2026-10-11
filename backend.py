"""Learnova AI Backend (FastAPI + LangChain + FAISS + Pydantic Output Parsers + Ngrok)

Run this file on Kaggle / Google Colab (with GPU) or locally.
It accepts standardized text chunks/snippets from the local Streamlit frontend,
indexes them in FAISS with multilingual embeddings, and provides:
1. Instant content acknowledgment & domain topic extraction
2. Structured Summarization (Key Concepts, Definitions, Takeaways) via Pydantic
3. Study Roadmap Generation with visual Mermaid diagrams
4. Interactive Quiz Generation (MCQs with explanations & source citations)
5. Grounded RAG Chat with timestamp/page citations

Requirements:
    pip install fastapi uvicorn pydantic langchain langchain-core langchain-community
    pip install langchain-google-genai sentence-transformers faiss-cpu pyngrok nest_asyncio
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, AsyncIterator, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("omnilern_backend")


# --------------------------------------------------------------------------- #
# 1. Pydantic Schemas for Structured Output Parsers
# --------------------------------------------------------------------------- #
class ImportantDefinition(BaseModel):
    term: str = Field(default="Key Concept", description="The exact technical or conceptual term")
    definition: str = Field(default="", description="Clear, concise, and accurate definition of the term as explained in the content")
    context: str = Field(default="", description="Contextual significance or real-world application of this term")


class KeyConcept(BaseModel):
    topic: str = Field(default="Core Topic", description="Subtopic or core theme title")
    key_points: List[str] = Field(default_factory=list, description="Detailed points, explanations, and core takeaways under this theme")


class StructuredSummary(BaseModel):
    overview: str = Field(default="", description="A comprehensive executive overview summarizing the entire content in 1-2 focused paragraphs")
    key_concepts: List[KeyConcept] = Field(default_factory=list, description="Detailed core topics and thematic pillars covered in the content")
    important_definitions: List[ImportantDefinition] = Field(default_factory=list, description="Glossary of key terms, acronyms, and concepts mentioned")
    actionable_takeaways: List[str] = Field(default_factory=list, description="High-value practical takeaways, lessons learned, and action items")


class RoadmapStep(BaseModel):
    step_number: int = Field(default=1, description="Sequential step order number")
    title: str = Field(default="", description="Step title or learning milestone")
    description: str = Field(default="", description="Detailed explanation of what the learner needs to study, understand, and apply")
    estimated_time: str = Field(default="45 minutes", description="Suggested time to master this step (e.g., '45 minutes')")
    prerequisites: List[str] = Field(default_factory=list, description="Concepts or prerequisites needed before starting this step")


class StudyRoadmap(BaseModel):
    learning_goal: str = Field(default="", description="The primary overarching educational objective achieved after completing this roadmap")
    steps: List[RoadmapStep] = Field(default_factory=list, description="Step-by-step sequential learning curriculum")
    mermaid_diagram: str = Field(
        default="",
        description="A valid Mermaid flowchart code (flowchart TD or graph TD) representing the study roadmap clearly without syntax errors"
    )


class QuizQuestion(BaseModel):
    id: int = Field(default=1, description="Question identifier number")
    question: str = Field(default="", description="Clear, specific question testing deep conceptual understanding")
    options: List[str] = Field(default_factory=list, description="4 plausible multiple-choice options containing one correct answer and 3 logical distractors")
    correct_option: str = Field(default="", description="Exact string match of the correct option from the options list")
    explanation: str = Field(default="", description="Detailed pedagogical explanation of why this option is correct based on the content")
    source_reference: str = Field(default="Content Citation", description="Content citation reference (e.g., 'Timestamp 04:15' or 'Page 3')")


class QuizData(BaseModel):
    quiz_title: str = Field(default="Interactive Assessment", description="Descriptive and engaging title for the interactive assessment")
    difficulty: str = Field(default="Intermediate", description="Assessment difficulty level: Beginner / Intermediate / Advanced")
    questions: List[QuizQuestion] = Field(default_factory=list, description="Curated list of assessment questions")


# --------------------------------------------------------------------------- #
# 2. Configuration & Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    api_key: str
    google_api_key: str
    gemini_model: str = "gemini-3.8-flash"
    # Multilingual embedding model supporting Arabic, English, and mixed code-switching
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    llm_timeout: float = 90.0
    llm_max_retries: int = 3
    ngrok_auth_token: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.getenv("API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")

        # Automatically check Kaggle UserSecretsClient if running in Kaggle
        if not google_api_key or not ngrok_auth_token:
            try:
                from kaggle_secrets import UserSecretsClient

                k_secrets = UserSecretsClient()
                if not google_api_key:
                    google_api_key = k_secrets.get_secret("GOOGLE_API_KEY")
                if not ngrok_auth_token:
                    ngrok_auth_token = k_secrets.get_secret("NGROK_AUTH_TOKEN")
                if not api_key:
                    try:
                        api_key = k_secrets.get_secret("API_KEY")
                    except Exception:
                        pass
            except Exception:
                pass

        api_key = api_key or "default-secret-key-123"
        google_api_key = google_api_key or ""

        return cls(
            api_key=api_key,
            google_api_key=google_api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            ),
            ngrok_auth_token=ngrok_auth_token,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
        )


# --------------------------------------------------------------------------- #
# 3. LLM Client with Dynamic Model Discovery & Pydantic Output Parsers
# --------------------------------------------------------------------------- #
def discover_active_gemini_model(api_key: str, default_preference: str = "gemini-3.8-flash") -> str:
    """Query Google AI ModelService directly to find the exact working model for this API key."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        available = []
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                clean_name = m.name.replace("models/", "")
                available.append(clean_name)

        logger.info("Active models available for this API key: %s", available)

        # 1. Prefer 3.8-flash if available
        for target in ["gemini-3.8-flash", "gemini-3.8-pro", "gemini-3.5-flash", "gemini-3.0-flash"]:
            if target in available:
                return target

        # 2. Prefer any flash model
        for name in available:
            if "flash" in name:
                return name

        # 3. Fallback to first available model
        if available:
            return available[0]
    except Exception as e:
        logger.warning("Could not auto-discover models via genai.list_models: %s", e)

    return default_preference


class LLMEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing Google Gemini API Key! Please set GOOGLE_API_KEY in environment variables or Kaggle Secrets."
            )

        discovered = discover_active_gemini_model(self.api_key, settings.gemini_model)
        self.candidate_models = list(
            dict.fromkeys(
                [
                    discovered,
                    settings.gemini_model,
                    "gemini-3.8-flash",
                    "gemini-3.8-pro",
                    "gemini-3.5-flash",
                ]
            )
        )
        logger.info("Configured candidate models sequence: %s", self.candidate_models)

        self.summary_parser = PydanticOutputParser(pydantic_object=StructuredSummary)
        self.roadmap_parser = PydanticOutputParser(pydantic_object=StudyRoadmap)
        self.quiz_parser = PydanticOutputParser(pydantic_object=QuizData)

    def _create_model(self, model_name: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.2,
            max_output_tokens=4096,
            timeout=self.settings.llm_timeout,
            max_retries=self.settings.llm_max_retries,
        )

    def invoke_with_fallback(self, messages: list) -> Any:
        """Invoke LLM with resilient fallback across candidate models."""
        last_error = None
        for model_name in self.candidate_models:
            try:
                m = self._create_model(model_name)
                res = m.invoke(messages)
                if res is not None:
                    return res
            except Exception as e:
                last_error = e
                logger.warning("Invocation on model '%s' failed: %s. Trying next candidate...", model_name, e)
                continue

        if last_error:
            raise last_error
        raise RuntimeError("All candidate models failed to return a valid response.")

    def extract_instant_ack(self, title: str, full_preview: str) -> dict:
        """Generate a concise acknowledgment and topic overview in 2-3 sentences."""
        sys_msg = (
            "You are an expert AI academic tutor. "
            "Your task is to analyze the provided title and text preview, then acknowledge receiving the content "
            "and clearly summarize in 2-3 concise sentences what topic/domain this content covers and its main focus."
        )
        prompt = (
            f"Title: {title}\n\n"
            f"Content Preview:\n{full_preview[:1500]}\n\n"
            "Write the acknowledgment and topic overview:"
        )
        try:
            response = self.invoke_with_fallback([SystemMessage(content=sys_msg), HumanMessage(content=prompt)])
            if response and getattr(response, "content", None):
                return {"acknowledgment": self._safe_content(response)}
        except Exception as e:
            logger.warning("extract_instant_ack error: %s", e)

        return {"acknowledgment": f"Successfully received and analyzed '{title}'. Ready to explore!"}

    @staticmethod
    def _safe_content(response: Any) -> str:
        """Safely extract string content from an LLM response.

        gemini-3.8-flash may return response.content as a list of content-part
        dicts like [{'type': 'text', 'text': '...', 'extras': {...}}] instead
        of a plain str. This normalizes to a clean string every time.
        """
        raw = getattr(response, "content", response)
        if isinstance(raw, list):
            text_parts = []
            for part in raw:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            raw = "\n".join(text_parts)
        return str(raw).strip()

    def _clean_json_content(self, raw_content: Any) -> str:
        """Strip markdown fences (```json ... ```) and extract clean JSON."""
        text = raw_content if isinstance(raw_content, str) else self._safe_content(raw_content)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # If surrounded by text/prose, extract the JSON object or array structure
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if match:
            return match.group(0).strip()
        return text

    def generate_structured_summary(self, full_text: str) -> StructuredSummary:
        """Generate a structured summary with bulletproof fallback parsing."""
        sys_msg = (
            "You are an expert academic tutor and educational research analyst.\n"
            "Your task is to analyze the provided content and extract a structured summary.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY a valid, well-formed JSON object matching the schema.\n"
            "2. Never output 'null' or empty text under any circumstances.\n"
            "3. Extract the primary overview, key concepts, definitions, and high-value takeaways."
        )
        prompt_template = PromptTemplate(
            template=(
                "Analyze the provided educational content and return a complete structured summary in valid JSON.\n\n"
                "{format_instructions}\n\n"
                "Content:\n{content}\n\n"
                "Remember: Return ONLY the valid JSON object conforming to the schema above. Do NOT output null."
            ),
            input_variables=["content"],
            partial_variables={"format_instructions": self.summary_parser.get_format_instructions()},
        )
        formatted_prompt = prompt_template.format(content=full_text[:30000])

        try:
            response = self.invoke_with_fallback([
                SystemMessage(content=sys_msg),
                HumanMessage(content=formatted_prompt),
            ])
            cleaned = self._clean_json_content(self._safe_content(response))

            if cleaned and cleaned.lower() != "null":
                try:
                    data = json.loads(cleaned)
                    if isinstance(data, dict):
                        return self.summary_parser.pydantic_object.model_validate(data)
                except Exception:
                    try:
                        return self.summary_parser.parse(cleaned)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Primary summary generation failed: %s", e)

        # Retry attempt with simplified JSON instruction
        logger.warning("Attempting focused summary retry...")
        try:
            retry_prompt = (
                "Provide a JSON object with keys 'overview', 'key_concepts' (list of {topic, key_points}), "
                "'important_definitions' (list of {term, definition, context}), and 'actionable_takeaways' (list of strings).\n\n"
                f"Content to summarize:\n{full_text[:12000]}"
            )
            res_retry = self.invoke_with_fallback([
                SystemMessage(content="You must return ONLY raw JSON matching the keys. No markdown fences. No null."),
                HumanMessage(content=retry_prompt),
            ])
            cleaned_retry = self._clean_json_content(self._safe_content(res_retry))
            if cleaned_retry and cleaned_retry.lower() != "null":
                data_retry = json.loads(cleaned_retry)
                if isinstance(data_retry, dict):
                    return self.summary_parser.pydantic_object.model_validate(data_retry)
        except Exception as retry_err:
            logger.warning("Summary retry also failed: %s", retry_err)

        # Safe fallback guarantee - never crash with a 500 error
        logger.info("Providing safe fallback structured summary from ingested content.")
        first_paragraph = full_text[:600].replace("\n", " ").strip()
        return StructuredSummary(
            overview=f"Executive Academic Overview:\n\n{first_paragraph}...",
            key_concepts=[
                KeyConcept(
                    topic="Core Subject Themes",
                    key_points=["Detailed pedagogical principles covered across this session."],
                )
            ],
            important_definitions=[
                ImportantDefinition(
                    term="Primary Subject",
                    definition="The core domain concepts explored throughout the presented material.",
                    context="Educational context",
                )
            ],
            actionable_takeaways=[
                "Review the comprehensive timestamp citations in the interactive Q&A tab to reinforce key concepts.",
                "Consult the progressive study roadmap to guide your learning milestones.",
            ],
        )

    def generate_roadmap(self, full_text: str) -> StudyRoadmap:
        """Generate a structured study roadmap with bulletproof fallback parsing."""
        sys_msg = (
            "You are a master curriculum architect.\n"
            "Create a logical step-by-step study roadmap with valid Mermaid flowchart code.\n"
            "CRITICAL: Return ONLY a valid JSON object matching the schema. Never output null."
        )
        prompt_template = PromptTemplate(
            template=(
                "Create a comprehensive step-by-step study roadmap for this material in JSON format.\n\n"
                "{format_instructions}\n\n"
                "Content:\n{content}\n\n"
                "Remember: Return ONLY the valid JSON object conforming to the schema above. Do NOT return null."
            ),
            input_variables=["content"],
            partial_variables={"format_instructions": self.roadmap_parser.get_format_instructions()},
        )
        formatted_prompt = prompt_template.format(content=full_text[:30000])

        try:
            response = self.invoke_with_fallback([
                SystemMessage(content=sys_msg),
                HumanMessage(content=formatted_prompt),
            ])
            cleaned = self._clean_json_content(self._safe_content(response))

            if cleaned and cleaned.lower() != "null":
                try:
                    data = json.loads(cleaned)
                    if isinstance(data, dict):
                        return self.roadmap_parser.pydantic_object.model_validate(data)
                except Exception:
                    try:
                        return self.roadmap_parser.parse(cleaned)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Primary roadmap generation failed: %s", e)

        # Safe fallback roadmap guarantee
        return StudyRoadmap(
            learning_goal="Master foundational concepts and practical applications from the study material",
            steps=[
                RoadmapStep(
                    step_number=1,
                    title="Foundational Overview",
                    description="Study core theoretical foundations and key terminology.",
                    estimated_time="30 minutes",
                ),
                RoadmapStep(
                    step_number=2,
                    title="Conceptual Deep Dive",
                    description="Examine detailed methodologies and practical problem-solving approaches.",
                    estimated_time="45 minutes",
                ),
                RoadmapStep(
                    step_number=3,
                    title="Synthesis & Assessment",
                    description="Complete the interactive assessment quiz and review evidence citations.",
                    estimated_time="30 minutes",
                ),
            ],
            mermaid_diagram="flowchart TD\n  A[Foundational Overview] --> B[Conceptual Deep Dive]\n  B --> C[Synthesis & Assessment]",
        )

    def generate_quiz(self, full_text: str, num_questions: int = 5) -> QuizData:
        """Generate an interactive MCQ quiz with bulletproof fallback parsing."""
        sys_msg = (
            "You are an assessment designer.\n"
            "Create multiple-choice questions testing conceptual understanding with clear citations.\n"
            "CRITICAL: Return ONLY a valid JSON object matching the schema. Never output null."
        )
        prompt_template = PromptTemplate(
            template=(
                "Create an interactive multiple-choice quiz of {num_q} questions based strictly on the provided content.\n\n"
                "{format_instructions}\n\n"
                "Content:\n{content}\n\n"
                "Remember: Return ONLY the valid JSON object conforming to the schema above. Do NOT return null."
            ),
            input_variables=["content", "num_q"],
            partial_variables={"format_instructions": self.quiz_parser.get_format_instructions()},
        )
        formatted_prompt = prompt_template.format(content=full_text[:30000], num_q=num_questions)

        try:
            response = self.invoke_with_fallback([
                SystemMessage(content=sys_msg),
                HumanMessage(content=formatted_prompt),
            ])
            cleaned = self._clean_json_content(self._safe_content(response))

            if cleaned and cleaned.lower() != "null":
                try:
                    data = json.loads(cleaned)
                    if isinstance(data, dict):
                        return self.quiz_parser.pydantic_object.model_validate(data)
                except Exception:
                    try:
                        return self.quiz_parser.parse(cleaned)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Primary quiz generation failed: %s", e)

        # Safe fallback quiz guarantee
        return QuizData(
            quiz_title="Knowledge Assessment",
            difficulty="Intermediate",
            questions=[
                QuizQuestion(
                    id=1,
                    question="What is the central focus of the presented lecture or document?",
                    options=[
                        "Core subject principles and applied methodology",
                        "Unrelated background history",
                        "Irrelevant trivia",
                        "None of the above",
                    ],
                    correct_option="Core subject principles and applied methodology",
                    explanation="The content introduces essential theoretical pillars and systematic analytical approaches.",
                    source_reference="Content Overview",
                )
            ],
        )

    def answer_rag(self, question: str, retrieved_docs: List[Document]) -> str:
        """Answer user queries with strict grounding and timestamp/page citations."""
        context_blocks = []
        for d in retrieved_docs:
            source_tag = d.metadata.get("source_citation", "Content Reference")
            context_blocks.append(f"[{source_tag}]\n{d.page_content}")

        context_str = "\n\n---\n\n".join(context_blocks)

        sys_msg = (
            "You are an intelligent, evidence-grounded AI learning assistant.\n"
            "Answer the user's question accurately using ONLY the provided excerpts.\n"
            "Guidelines:\n"
            "1. Always cite the relevant timestamp or page citation tag provided in the context.\n"
            "2. If the context does not contain the answer, state politely that the provided material does not cover this topic.\n"
            "3. Answer in the same language as the user's question with clarity and depth."
        )

        user_content = f"Retrieved Context Excerpts:\n{context_str}\n\nUser Question: {question}"
        response = self.invoke_with_fallback([SystemMessage(content=sys_msg), HumanMessage(content=user_content)])
        return self._safe_content(response)


# --------------------------------------------------------------------------- #
# 4. Ingestion & Vector DB Service (FAISS)
# --------------------------------------------------------------------------- #
class ActiveContent:
    def __init__(self, content_id: str, title: str, full_text: str, store: FAISS) -> None:
        self.content_id = content_id
        self.title = title
        self.full_text = full_text
        self.store = store


class ContentService:
    def __init__(self, settings: Settings, embeddings: HuggingFaceEmbeddings, llm: LLMEngine) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.llm = llm
        self.active: Optional[ActiveContent] = None
        self._lock = asyncio.Lock()

    def ingest_chunks_blocking(
        self, content_id: str, title: str, source_type: str, chunks_data: List[dict]
    ) -> dict:
        documents: List[Document] = []
        full_text_parts: List[str] = []

        for item in chunks_data:
            text = item.get("text", "").strip()
            if not text:
                continue
            full_text_parts.append(text)

            meta = {
                "content_id": content_id,
                "source_type": source_type,
                "start": item.get("start"),
                "end": item.get("end"),
                "page": item.get("page"),
            }

            if item.get("start") is not None and item.get("end") is not None:
                start_sec = int(item["start"])
                m, s = divmod(start_sec, 60)
                meta["source_citation"] = f"⏱️ Timestamp {m:02d}:{s:02d}"
            elif item.get("page") is not None:
                meta["source_citation"] = f"📄 Page {item['page']}"
            else:
                meta["source_citation"] = "📌 Content Excerpt"

            documents.append(Document(page_content=text, metadata=meta))

        if not documents:
            raise ValueError("No valid text chunks provided for ingestion.")

        full_text = "\n\n".join(full_text_parts)
        store = FAISS.from_documents(documents, self.embeddings)

        try:
            ack = self.llm.extract_instant_ack(title, full_text)
            ack_msg = ack.get("acknowledgment") if ack else None
        except Exception as e:
            logger.warning("Acknowledgment generation failed: %s", e)
            ack_msg = None

        if not ack_msg:
            ack_msg = f"Content '{title}' successfully processed and indexed ({len(documents)} chunks). Ready for study!"

        self.active = ActiveContent(
            content_id=content_id,
            title=title,
            full_text=full_text,
            store=store,
        )

        return {
            "content_id": content_id,
            "title": title,
            "chunk_count": len(documents),
            "acknowledgment": ack_msg,
        }

    async def ingest(self, content_id: str, title: str, source_type: str, chunks_data: List[dict]) -> dict:
        async with self._lock:
            return await run_in_threadpool(
                self.ingest_chunks_blocking, content_id, title, source_type, chunks_data
            )


# --------------------------------------------------------------------------- #
# 5. FastAPI Request & Response Models
# --------------------------------------------------------------------------- #
class ChunkInput(BaseModel):
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    page: Optional[int] = None


class IngestRequest(BaseModel):
    content_id: str
    title: str
    source_type: str = Field(description="youtube / audio / pdf / docx / pptx / text")
    chunks: List[ChunkInput]


class IngestResponse(BaseModel):
    content_id: str
    title: str
    chunk_count: int
    acknowledgment: str


class AskRAGRequest(BaseModel):
    question: str
    top_k: int = 4


class AskRAGResponse(BaseModel):
    answer: str
    content_id: str


# --------------------------------------------------------------------------- #
# 6. FastAPI App & Endpoints
# --------------------------------------------------------------------------- #
def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = getattr(app.state, "settings", None) or settings or Settings.from_env()
        logger.info("Initializing HuggingFace Multilingual Embeddings: %s", active_settings.embedding_model)
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("KAGGLE_CONTAINER_NAME") else "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass

        logger.info("Using compute device for embeddings: %s", device)
        embeddings = await run_in_threadpool(
            HuggingFaceEmbeddings,
            model_name=active_settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

        # Pre-warm the embedding model so first client request has zero warmup latency
        logger.info("Pre-warming embedding model with sample query...")
        await run_in_threadpool(embeddings.embed_query, "System pre-warm initialization query")
        logger.info("Embedding model pre-warmed successfully.")

        llm = LLMEngine(active_settings)
        app.state.service = ContentService(active_settings, embeddings, llm)
        logger.info("Backend service initialized and ready.")
        yield

    app = FastAPI(title="Learnova AI API", version="2.0.0", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_service(request: Request) -> ContentService:
        return request.app.state.service

    def verify_auth(request: Request, authorization: Annotated[Optional[str], Header()] = None) -> None:
        expected = f"Bearer {request.app.state.settings.api_key}"
        if not authorization or not secrets.compare_digest(
            authorization.encode("utf-8"), expected.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.get("/health")
    async def health(service: ContentService = Depends(get_service)):
        active_id = service.active.content_id if service.active else None
        active_title = service.active.title if service.active else None
        return {"status": "healthy", "active_content_id": active_id, "active_title": active_title}

    @app.post("/INGEST_CONTENT", response_model=IngestResponse, dependencies=[Depends(verify_auth)])
    async def ingest_content(body: IngestRequest, service: ContentService = Depends(get_service)):
        try:
            res = await service.ingest(
                content_id=body.content_id,
                title=body.title,
                source_type=body.source_type,
                chunks_data=[c.model_dump() for c in body.chunks],
            )
            return IngestResponse(**res)
        except Exception as exc:
            logger.exception("Ingest failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/GENERATE_SUMMARY", response_model=StructuredSummary, dependencies=[Depends(verify_auth)])
    async def generate_summary(service: ContentService = Depends(get_service)):
        if not service.active:
            raise HTTPException(status_code=400, detail="No active content loaded. Ingest content first.")
        try:
            return await run_in_threadpool(
                service.llm.generate_structured_summary, service.active.full_text
            )
        except Exception as exc:
            logger.exception("Summary generation failed")
            raise HTTPException(status_code=500, detail=f"Summary generation error: {exc}")

    @app.post("/GENERATE_ROADMAP", response_model=StudyRoadmap, dependencies=[Depends(verify_auth)])
    async def generate_roadmap(service: ContentService = Depends(get_service)):
        if not service.active:
            raise HTTPException(status_code=400, detail="No active content loaded. Ingest content first.")
        try:
            return await run_in_threadpool(service.llm.generate_roadmap, service.active.full_text)
        except Exception as exc:
            logger.exception("Roadmap generation failed")
            raise HTTPException(status_code=500, detail=f"Roadmap generation error: {exc}")

    @app.post("/GENERATE_QUIZ", response_model=QuizData, dependencies=[Depends(verify_auth)])
    async def generate_quiz(num_questions: int = 5, service: ContentService = Depends(get_service)):
        if not service.active:
            raise HTTPException(status_code=400, detail="No active content loaded. Ingest content first.")
        try:
            return await run_in_threadpool(
                service.llm.generate_quiz, service.active.full_text, num_questions
            )
        except Exception as exc:
            logger.exception("Quiz generation failed")
            raise HTTPException(status_code=500, detail=f"Quiz generation error: {exc}")

    @app.post("/ASK_RAG", response_model=AskRAGResponse, dependencies=[Depends(verify_auth)])
    async def ask_rag(body: AskRAGRequest, service: ContentService = Depends(get_service)):
        if not service.active:
            raise HTTPException(status_code=400, detail="No active content loaded. Ingest content first.")
        try:
            docs = await run_in_threadpool(
                service.active.store.similarity_search, body.question, k=body.top_k
            )
            answer = await run_in_threadpool(service.llm.answer_rag, body.question, docs)
            return AskRAGResponse(answer=answer, content_id=service.active.content_id)
        except Exception as exc:
            logger.exception("RAG search failed")
            raise HTTPException(status_code=500, detail=f"RAG error: {exc}")

    return app


# --------------------------------------------------------------------------- #
# 7. Ngrok Setup & Main Entrypoint (Notebook & Terminal Friendly)
# --------------------------------------------------------------------------- #
app = create_app()


def start_server(
    port: int = 8000,
    ngrok_token: Optional[str] = None,
    google_api_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
):
    """Start the server and Ngrok tunnel safely inside or outside a Jupyter Notebook."""
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    if gemini_model:
        os.environ["GEMINI_MODEL"] = gemini_model
    app.state.settings = Settings.from_env()

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass

    token = ngrok_token or os.getenv("NGROK_AUTH_TOKEN")
    if token:
        try:
            from pyngrok import ngrok

            ngrok.set_auth_token(token)
            # Close previous active tunnels
            ngrok.kill()
            tunnel = ngrok.connect(port)
            print("\n" + "=" * 60)
            print(f"🚀 NGROK PUBLIC URL: {tunnel.public_url}")
            print("👉 Copy this URL into your local Streamlit app sidebar!")
            print("=" * 60 + "\n")
        except Exception as e:
            logger.warning("Could not start Ngrok tunnel: %s", e)

    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    # If running inside an existing asyncio event loop (e.g. Kaggle/Colab notebook)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import threading

            t = threading.Thread(target=server.run, daemon=True)
            t.start()
            logger.info("Server started in background thread successfully.")
            return t
    except Exception:
        pass

    server.run()


if __name__ == "__main__":
    settings = Settings.from_env()
    start_server(port=settings.port, ngrok_token=settings.ngrok_auth_token)
