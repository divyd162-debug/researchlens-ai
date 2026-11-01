# ResearchLens AI 🔍📄

**ResearchLens AI** is an intelligent, open-source research assistant designed to parse scientific papers, perform semantic search across academic literature, and answer complex research questions with verifiable, passage-level citations.

---

## 🚀 Project Vision

Reviewing scientific papers is time-consuming and cognitively demanding. Traditional keyword search often misses conceptually related insights, while generic AI chatbots hallucinate quotes and page numbers.

**ResearchLens AI solves this by coupling dense semantic vector retrieval with strictly grounded AI generation**:
1. **Upload Papers**: Extract clean text and page metadata from academic PDFs.
2. **Semantic Search**: Search across your collection using concept-level queries rather than exact keywords.
3. **Ask Questions**: Ask nuanced research questions across one or multiple papers.
4. **Inspect Source Citations**: Review generated answers with direct links to exact quotes and page numbers in the original papers.

---

## 📊 Feature Status: Implemented vs. Planned

To maintain complete transparency during development, features are categorized by their current status:

| Feature | Status | Description |
|---|---|---|
| **System Architecture & Data Flow** | ✅ Completed | Comprehensive architectural specification and diagrams in `docs/architecture.md`. |
| **Phased Implementation Plan** | ✅ Completed | Step-by-step roadmap with acceptance criteria in `docs/implementation-plan.md`. |
| **Security & Threat Model** | ✅ Completed | Threat model covering PDF safety, prompt injection, and access control in `docs/threat-model.md`. |
| **PDF Extraction Engine (PyMuPDF)** | ⏳ Planned (Phase 1) | High-speed text extraction preserving page boundaries and academic layouts. |
| **Semantic Chunking Pipeline** | ⏳ Planned (Phase 1) | Context-preserving text chunker with overlap and citation metadata. |
| **Local Embedding Generation** | ⏳ Planned (Phase 2) | Offline vector generation using `sentence-transformers` (`all-MiniLM-L6-v2`). |
| **Vector Similarity Index (FAISS)** | ⏳ Planned (Phase 2) | Millisecond-level vector search with disk persistence. |
| **Relational Storage (PostgreSQL)** | ⏳ Planned (Phase 3) | ACID metadata storage for papers, chunks, and chat history using SQLAlchemy & Alembic. |
| **Pluggable LLM Interface & Mock Provider** | ⏳ Planned (Phase 4) | Strategy pattern supporting zero-cost offline mock testing and external LLM providers. |
| **FastAPI REST API & Pydantic Validation** | ⏳ Planned (Phase 5) | Secure, auto-documented API endpoints for uploads, queries, and search. |
| **Interactive Web UI (React + Vite)** | ⏳ Planned (Phase 6) | Modern web interface for PDF drag-and-drop, search, and citation drawer. |
| **Automated Test Suite (pytest + Vitest)** | ⏳ Planned (Phase 7) | Comprehensive unit and integration test coverage across backend and frontend. |

---

## 🛠 Planned Technology Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Backend**: Python 3.11+, FastAPI, Pydantic v2
- **PDF Extraction**: PyMuPDF (`fitz`)
- **Embeddings**: `sentence-transformers` (runs locally, 100% free)
- **Vector Search**: Meta FAISS (CPU)
- **Database**: PostgreSQL with SQLAlchemy ORM and Alembic migrations
- **AI Generation**: Pluggable provider architecture (deterministic Mock provider for tests, extensible to Gemini / Groq / OpenAI)
- **Testing**: `pytest` (backend) and `Vitest` (frontend)

---

## 📂 Proposed Project Structure

```text
ResearchLens AI/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI route handlers (/documents, /search, /query)
│   │   ├── core/            # App settings, config, security helpers
│   │   ├── models/          # SQLAlchemy ORM database models
│   │   ├── schemas/         # Pydantic validation schemas
│   │   ├── services/        # Business logic & AI pipeline
│   │   │   ├── pdf_service.py       # PyMuPDF text & page extractor
│   │   │   ├── chunking_service.py  # Text chunker with citation metadata
│   │   │   ├── embedding_service.py # Vector embedding generator
│   │   │   ├── vector_store.py      # FAISS index manager
│   │   │   ├── llm_service.py       # Pluggable LLM interface & mock
│   │   │   └── rag_service.py       # Retrieval-augmented generation orchestrator
│   │   └── main.py          # FastAPI application entrypoint
│   ├── alembic/             # Database migration scripts
│   ├── tests/               # Backend unit and integration tests (pytest)
│   ├── requirements.txt     # Python dependencies
│   └── .env.example         # Environment variable template
├── frontend/
│   ├── src/
│   │   ├── components/      # UI components (UploadModal, Chat, CitationViewer)
│   │   ├── hooks/           # Custom React hooks (useChat, useDocuments)
│   │   ├── services/        # API client
│   │   ├── types/           # TypeScript data interfaces
│   │   ├── App.tsx          # Main React layout
│   │   └── main.tsx         # Frontend entrypoint
│   ├── package.json         # Frontend dependencies and scripts
│   └── vite.config.ts       # Vite bundler configuration
├── docs/
│   ├── architecture.md      # System architecture, data flow & Mermaid diagram
│   ├── implementation-plan.md # Phased implementation plan with acceptance criteria
│   └── threat-model.md      # Security analysis and risk mitigations
├── data/                    # Local storage for raw PDF uploads and FAISS index (gitignored)
└── README.md                # Project overview and documentation
```

---

## 📖 Documentation Index

- [Architecture & Design Details](docs/architecture.md)
- [Implementation Plan & Acceptance Criteria](docs/implementation-plan.md)
- [Threat Model & Security Guide](docs/threat-model.md)

---

## 🧭 Next Steps

We are adhering to a structured implementation plan. The current phase establishes the project blueprint and design principles. Once reviewed, we will proceed to **Phase 0 (Environment Setup & Project Skeleton)**.
