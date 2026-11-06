# ResearchLens AI 🔍📄

[![Backend Tests](https://img.shields.io/badge/backend%20tests-675%20passed-brightgreen)](#testing-instructions)
[![Frontend Tests](https://img.shields.io/badge/frontend%20tests-114%20passed-brightgreen)](#testing-instructions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](#prerequisites)
[![Node Version](https://img.shields.io/badge/node-%3E%3D18.0.0-green)](#prerequisites)

**ResearchLens AI** is a local-first research platform designed to parse scientific literature, perform dense semantic vector retrieval, and answer complex research questions with verifiable, passage-level citations grounded in original PDFs.

---

## 1. Project Overview & Problem Statement

Reviewing academic literature is cognitively demanding and time-consuming. Standard keyword search systems frequently miss conceptually relevant passages that use different terminology, while public conversational AI models are prone to hallucinating facts, quoting non-existent literature, and fabricating page numbers.

ResearchLens AI addresses these limitations through **strictly grounded retrieval-augmented generation (RAG)**:
- **Exact Citation Verification**: Generated answers cite specific passages with page numbers, character offsets, and verbatim quotes.
- **Corpus-Scoped Retrieval**: Queries and conversations can be scoped globally, to a single paper, to a selected subset of papers, or to a curated research collection.
- **Multi-Tenant Isolation**: Complete user isolation across documents, chunk embeddings, collections, and conversation history.
- **Offline-First Embeddings**: Fast local embedding generation via `sentence-transformers` without external embedding API costs.

### Intended Users
- **Academic Researchers & Postdocs**: Synthesizing literature across related preprint PDFs.
- **Graduate Students**: Conducting thesis literature reviews with strict citation traceability.
- **R&D Scientists & Engineers**: Analyzing dense technical documentation and white papers.

---

## 2. Implemented Features (Batch 2 Baseline)

The following capabilities are implemented, tested, and available:

- **PDF Ingestion & Safe Parsing**:
  - Validates file size (up to 50 MB) and MIME types.
  - Page-bounded text extraction via PyMuPDF (`fitz`) up to a 100-page limit.
  - SHA-256 deduplication per user.
- **Chunking & Vector Indexing**:
  - Context-preserving paragraph and sentence chunking with configurable overlap.
  - Local dense embedding generation via `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions).
  - High-performance vector similarity search using Meta FAISS CPU (`IndexIDMap2` wrapping `IndexFlatIP`).
  - Startup index verification and deterministic index recovery from PostgreSQL chunk tables.
- **Collection Management**:
  - User-curated research collections with unique names per tenant.
  - Atomic document membership: associate single or multiple documents with a collection.
  - Document removal from collections preserving underlying files and chat sessions.
  - Collection-detail view displaying member document titles, page counts, file sizes, and processing statuses.
  - Document view action to assign any uploaded paper to a collection.
- **Scoped Semantic Search**:
  - Dense vector similarity queries across `GLOBAL`, `DOCUMENT`, `DOCUMENTS`, and `COLLECTION` scopes.
  - Authoritative database pre-filtering ensuring candidate chunks strictly belong to ready documents owned by the caller.
  - Normalized similarity scoring with threshold filtering and `top_k` bounds.
- **Grounded Chat Sessions & Verified Citations**:
  - Persistent conversation threads with title management and archiving.
  - RAG prompt orchestration injecting only retrieved source passages.
  - Exact quote verification rejecting ungrounded claims.
  - Session message history with citation snapshots.
- **Security & Authentication**:
  - JWT Bearer authentication (HS256) with request-scoped authorization.
  - Secure password hashing using bcrypt via `pwdlib`.
  - Strict multi-tenant isolation on all database and vector store queries.
  - Secret-free logging and database URL masking.
- **Health & Readiness Architecture**:
  - Lightweight process liveness endpoint: `GET /health`.
  - Comprehensive dependency readiness endpoint: `GET /health/ready` verifying PostgreSQL connection (`SELECT 1`), FAISS index readiness, and upload directory permissions.

---

## 3. System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend (React 18 + Vite + TypeScript)"]
        UI["SPA Interface: Dashboard / Documents / Collections / Search / Chat"]
        API_CLIENT["Typed Fetch Client (api.ts)"]
        UI --> API_CLIENT
    end

    subgraph Backend ["Backend (FastAPI + Pydantic v2)"]
        ROUTER["API Router (/api/v1)"]
        AUTH["Auth & JWT Middleware"]
        DOC_SVC["Document Ingestion Service"]
        COL_SVC["Collection Service"]
        RET_SVC["Retrieval Service"]
        RAG_SVC["RAG & Citation Engine"]

        API_CLIENT --> ROUTER
        ROUTER --> AUTH
        AUTH --> DOC_SVC
        AUTH --> COL_SVC
        AUTH --> RET_SVC
        AUTH --> RAG_SVC
    end

    subgraph DataStorage ["Data & Storage Layer"]
        PG[("PostgreSQL 14+\n(Users, Documents, Chunks, Collections, Chats)")]
        FAISS[("FAISS Vector Store\n(384-dim IndexIDMap2 on Disk)")]
        FS[("Local Filesystem\n(data/uploads)")]
    end

    subgraph AI ["AI & Model Infrastructure"]
        EMBED["sentence-transformers\n(all-MiniLM-L6-v2)"]
        LLM["LLM Provider Gateway\n(Mock Provider or OmniRoute)"]
    end

    DOC_SVC -->|Store Metadata & Chunks| PG
    DOC_SVC -->|Save PDF| FS
    DOC_SVC -->|Generate Embeddings| EMBED
    EMBED -->|Index Vectors| FAISS

    COL_SVC -->|Manage Membership| PG
    RET_SVC -->|Scope Filter IDs| PG
    RET_SVC -->|Cosine Search| FAISS

    RAG_SVC -->|Retrieve Context| RET_SVC
    RAG_SVC -->|Prompt with Citations| LLM
```

---

## 4. Technology Stack

### Backend
- **Framework**: FastAPI `0.115+` with Python `3.11+`
- **Validation & Serialization**: Pydantic v2
- **PDF Extraction**: PyMuPDF (`fitz`) `1.24+`
- **Embeddings**: `sentence-transformers` `3.0+` (`all-MiniLM-L6-v2`)
- **Vector Search**: Meta FAISS (`faiss-cpu` `1.8+`)
- **Relational ORM**: SQLAlchemy 2.0 with PostgreSQL driver `psycopg` (binary 3.1+)
- **Database Migrations**: Alembic `1.13+`
- **Authentication**: `pwdlib[bcrypt]`, `pyjwt`
- **Testing**: `pytest 8+`, `anyio`

### Frontend
- **Framework**: React `18.3+` with TypeScript `5.6+`
- **Build Tool**: Vite `5.4+`
- **Styling**: Vanilla modular CSS design system (`App.css`), no Tailwind CSS
- **Markdown & Citations**: `react-markdown` `10.1+`, `remark-gfm`
- **Testing**: Vitest `2.1+`, `@testing-library/react` `16.0+`, `jsdom`

---

## 5. Ingestion & Retrieval Data Flow

```
1. PDF Upload ──> 2. PDF Extraction ──> 3. Chunking ──> 4. Embedding ──> 5. FAISS Indexing
   (FastAPI)         (PyMuPDF)          (500 tokens)    (all-MiniLM)     (IndexIDMap2)
                                              │
                                              ▼
                                        PostgreSQL Chunks

6. Query ──> 7. Scope Check ──> 8. FAISS Search ──> 9. RAG Assembly ──> 10. Citation Validation
   (User)     (PostgreSQL)      (Top-K Chunks)     (Grounding Prompt)     (Exact Quotes)
```

1. **Upload**: User uploads an academic PDF via `POST /api/v1/documents`. The system verifies MIME type, computes SHA-256 to prevent duplicate uploads, and saves the file to `data/uploads/`.
2. **Extraction**: PyMuPDF extracts raw text page-by-page, enforcing a 100-page limit and preserving page boundary metadata.
3. **Chunking**: Text is split into overlapping passages (~500 tokens, 100-token overlap) respecting sentence boundaries. Exact character start and end offsets are recorded.
4. **Embedding**: `sentence-transformers` generates 384-dimensional dense vectors locally.
5. **Storage & Indexing**: Chunks are inserted into PostgreSQL. Corresponding vectors are added to FAISS mapped directly to the relational chunk IDs.
6. **Query & Retrieval**: A search query is embedded and pre-filtered in PostgreSQL by user ownership and corpus scope (`GLOBAL`, `DOCUMENT`, `DOCUMENTS`, or `COLLECTION`).
7. **Similarity Search**: FAISS retrieves candidate vectors matching the allowed chunk IDs based on inner product similarity.
8. **RAG & Citations**: The retrieved passages are formatted into a grounded prompt. The LLM generates the answer, and ResearchLens validates that every quoted citation matches the source chunk.

---

## 6. Storage Responsibilities: PostgreSQL vs. FAISS

| Responsibility | PostgreSQL | FAISS |
|---|---|---|
| **User Authentication & Passwords** | ✅ Full ownership | ❌ Not involved |
| **Document Metadata & Statuses** | ✅ Primary source of truth | ❌ Not involved |
| **Passage Text & Character Offsets** | ✅ Persisted in `chunks` table | ❌ Not stored (vectors only) |
| **Collection Hierarchy & Membership** | ✅ Foreign keys & unique constraints | ❌ Not involved |
| **Chat Sessions & Messages** | ✅ Full relational history | ❌ Not involved |
| **Dense Vector Embeddings** | ❌ Not stored in relational tables | ✅ In-memory CPU index with disk sync |
| **Similarity Ranking** | ❌ Delegated to FAISS | ✅ Sub-millisecond vector similarity |
| **Crash Recovery** | ✅ Authoritative chunk archive | ✅ Rebuilt deterministically on startup |

---

## 7. OmniRoute Gateway Configuration

ResearchLens AI uses a pluggable LLM architecture via `LLM_PROVIDER`:
- **`mock`** (Default): Deterministic offline mock provider with zero API cost, ideal for testing and development.
- **`omniroute`**: Production gateway supporting OpenAI-compatible endpoints and multi-model fallbacks (e.g., Gemini).

To enable OmniRoute, configure the following variables in your `.env`:

```bash
LLM_PROVIDER=omniroute
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_API_KEY=your_omniroute_api_key_here
OMNIROUTE_MODEL=gemini-fallback-combo
OMNIROUTE_TIMEOUT_SECONDS=60.0
OMNIROUTE_MAX_RETRIES=1
```

> **Security Note**: Never commit API keys or credentials to Git. Always pass secrets via `.env` or system environment variables.

---

## 8. Prerequisites & Platform Requirements

- **Operating System**: macOS (Apple Silicon / Intel), Linux (Ubuntu 20.04+), Windows via WSL2.
- **Python**: Version `3.11`, `3.12`, or `3.13`.
- **Node.js**: Version `>= 18.0.0` (LTS recommended) and `npm`.
- **PostgreSQL**: Version `14+` running locally or accessible via network.

---

## 9. Step-by-Step Installation & Configuration

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/researchlens-ai.git
cd "researchlens-ai"
```

### 2. Configure Backend Environment

Copy the root template into backend `.env`:

```bash
cp .env.example .env
cp .env.example backend/.env
```

Adjust `DATABASE_URL` in `.env` and `backend/.env` to point to your local PostgreSQL instance:

```env
DATABASE_URL=postgresql+psycopg://postgres:your_password@localhost:5432/researchlens
JWT_SECRET_KEY=generate-a-strong-random-32-character-secret-key-here
LLM_PROVIDER=mock
```

### 3. Create the Python Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 4. Create the Database & Run Migrations

Ensure PostgreSQL is running, then create the database:

```bash
createdb researchlens
```

Run Alembic migrations to create the schema (including collections, documents, chunks, and chat tables):

```bash
alembic upgrade head
```

### 5. Install Frontend Dependencies

```bash
cd frontend
cp .env.example .env
npm install
cd ..
```

---

## 10. Running the Application

### Start the Backend Server

From the repository root with virtual environment activated:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI application will start at `http://localhost:8000`.
- API Documentation: `http://localhost:8000/docs`
- Liveness Probe: `http://localhost:8000/health`
- Readiness Probe: `http://localhost:8000/health/ready`

### Start the Frontend Development Server

In a separate terminal window:

```bash
cd frontend
npm run dev
```

The Vite dev server will start at `http://localhost:5173`. Open your browser to begin using ResearchLens AI.

---

## 11. Health & Readiness Probes

ResearchLens AI separates liveness and readiness probes to ensure robust container orchestration:

- **Liveness Probe (`GET /health`)**:
  - Inexpensive in-memory check returning `{"status": "healthy", "environment": "development"}` with HTTP 200.
  - Does not query external dependencies; transient database outages will not trigger container restarts.
- **Readiness Probe (`GET /health/ready`)**:
  - Validates PostgreSQL connection with a lightweight `SELECT 1`.
  - Validates FAISS vector store readiness without reloading models or rebuilding the index.
  - Validates read/write permissions on the configured upload directory without creating temporary files.
  - Returns `{"status": "ready"}` with HTTP 200 when ready.
  - Returns `{"status": "not_ready", "failures": ["database", ...]}` with HTTP 503 if any dependency is unavailable, concealing sensitive database credentials or exception traces.

---

## 12. Testing Instructions

### Backend Test Suite (pytest)

With the virtual environment activated:

```bash
source .venv/bin/activate

# Run all backend tests
pytest

# Run focused collections tests
pytest backend/tests/test_collections_api.py

# Run health & readiness tests
pytest backend/tests/test_health.py
```

### Frontend Test Suite (Vitest)

```bash
cd frontend

# Run all unit and integration tests
npm test -- --run

# Run TypeScript type check and production bundle build
npm run build
```

---

## 13. Environment Variable Reference

| Variable | Default Value | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Deployment environment (`development`, `staging`, `production`) |
| `APP_NAME` | `ResearchLens AI` | Application branding name |
| `HOST` | `0.0.0.0` | Backend bind host |
| `PORT` | `8000` | Backend bind port |
| `DATABASE_URL` | `postgresql+psycopg://...` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | *(Required in prod)* | Secret key for signing JWT tokens (min 32 characters) |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | JWT bearer token lifetime in minutes |
| `LLM_PROVIDER` | `mock` | Active provider: `mock` (offline) or `omniroute` |
| `OMNIROUTE_BASE_URL` | `http://localhost:20128/v1` | Base URL for OmniRoute gateway |
| `OMNIROUTE_API_KEY` | `""` | API key for OmniRoute authentication |
| `OMNIROUTE_MODEL` | `gemini-fallback-combo` | Target LLM model name |
| `MAX_DOCUMENT_PAGES` | `100` | Maximum pages allowed per uploaded PDF |
| `DATA_DIR` | `data` | Root directory for application storage |
| `UPLOAD_DIR` | `data/uploads` | Directory for raw uploaded PDF files |
| `VECTOR_INDEX_DIR` | `data/vector_index` | Directory storing FAISS index and metadata |
| `VITE_API_BASE_URL` | `/api/v1` | Frontend API target base URL |

---

## 14. Known Limitations & Architecture Constraints

- **Single-Worker FAISS Architecture**: The FAISS CPU index resides in memory with disk snapshots. Due to memory-mapped vector synchronization constraints, the backend must run in a single-worker process per node. Startup concurrency is guarded with an advisory file lock.
- **Initial Embedding Model Download**: On first invocation, `sentence-transformers` downloads the `all-MiniLM-L6-v2` model weights (~90 MB) to the local HuggingFace cache. Subsequent runs are fully offline.
- **Grounded Single-Turn RAG**: Dialogue currently focuses on single-turn research prompts with strict citation extraction. Context carryover across multi-turn conversational dialogue is planned for Batch 3.
- **Hallucination Mitigation vs. Prevention**: Grounded retrieval and quote verification substantially minimize hallucinations by validating claims against source text; however, complex numerical derivations should always be reviewed against the primary source.

---

## 15. License

License: No license has currently been assigned to this repository. The code is publicly viewable, but no permission to copy, modify, distribute, or reuse it is granted.
