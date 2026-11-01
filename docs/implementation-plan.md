# ResearchLens AI — Implementation Plan

This implementation plan outlines the step-by-step roadmap for building **ResearchLens AI**. It is designed specifically for a solo developer learning Python, AI/ML concepts, and full-stack engineering.

Each phase is self-contained, testable, and builds upon the preceding phase without overwhelming leaps in complexity.

---

## Roadmap Overview

```
Phase 0: Project Setup & Repository Skeleton
   │
   ▼
Phase 1: PDF Text Extraction & Semantic Chunking (PyMuPDF)
   │
   ▼
Phase 2: Local Vector Embeddings & Similarity Search (sentence-transformers + FAISS)
   │
   ▼
Phase 3: Database Models & Metadata Storage (PostgreSQL + SQLAlchemy + Alembic)
   │
   ▼
Phase 4: Pluggable AI Generation & Citation Grounding (Mock Provider + LLM Interface)
   │
   ▼
Phase 5: Backend API Endpoints & Business Logic (FastAPI + Pydantic)
   │
   ▼
Phase 6: Frontend Development (React + TypeScript + Vite)
   │
   ▼
Phase 7: Automated Testing & Verification (pytest + Vitest)
   │
   ▼
Phase 8: Hardening, Evaluation & Free/Low-Cost Deployment
```

---

## Phase 0: Project Setup & Repository Skeleton

### Objective
Establish a clean workspace structure, configure Python virtual environments, Git version control, linting tools, and verify development dependencies.

### Tasks
- [ ] Initialize Git repository with a `.gitignore` tailored for Python, Node, and virtual environments.
- [ ] Create base directory layout (`backend/`, `frontend/`, `docs/`, `data/`).
- [ ] Set up Python 3.11+ virtual environment (`venv`) and initial `backend/requirements.txt`.
- [ ] Initialize Vite + React + TypeScript in `frontend/`.
- [ ] Verify that Python and Node environments start cleanly.

### Acceptance Criteria
- [x] All directories exist without file collisions.
- [ ] `python -m pytest` executes (with empty or placeholder tests passing).
- [ ] `npm run dev` in `frontend/` launches a local dev server without errors.
- [ ] No API keys, credentials, or generated files are tracked by Git.

---

## Phase 1: PDF Extraction & Semantic Chunking

### Objective
Extract clean, structured text and page metadata from academic PDFs using PyMuPDF (`fitz`), and implement a chunking strategy that preserves page-level citation boundaries.

### Key Concepts to Learn
- **PDF Structure**: Why PDFs are layout-oriented (streams of coordinates) rather than semantic text.
- **Chunk Size & Overlap**: Why splitting text into ~500 tokens with 10% overlap prevents cutting sentences in half and maintains context across chunk boundaries.

### Tasks
- [ ] Implement `PDFService` using PyMuPDF to extract text along with page numbers.
- [ ] Handle two-column academic paper formats cleanly.
- [ ] Implement `ChunkingService` with configurable chunk size and chunk overlap.
- [ ] Associate each chunk with metadata: `document_id`, `page_number`, `chunk_index`, and character span.

### Acceptance Criteria
- [ ] Given a sample research PDF, `PDFService` extracts all pages with 100% correct page numbers.
- [ ] Given a 10-page PDF, `ChunkingService` produces a list of structured chunks.
- [ ] Unit tests pass: verify chunks retain correct `page_number` tags and overlap boundaries.

---

## Phase 2: Local Vector Embeddings & Vector Search

### Objective
Convert text chunks into dense mathematical vector representations using `sentence-transformers` and index them using FAISS for fast similarity search—entirely offline and free.

### Key Concepts to Learn
- **Dense Vectors & Semantic Space**: How high-dimensional vectors capture semantic similarity.
- **FAISS (Facebook AI Similarity Search)**: How inner product / cosine similarity finds the nearest neighbors in milliseconds.

### Tasks
- [ ] Implement `EmbeddingService` loading a compact, high-performance open model (`all-MiniLM-L6-v2`).
- [ ] Implement `VectorStore` wrapping FAISS Index (`IndexFlatIP` with normalized vectors for cosine similarity).
- [ ] Add methods to add vectors, query top-$K$ vectors, and persist/load FAISS index to disk (`data/faiss_index.bin`).

### Acceptance Criteria
- [ ] Successfully embeds a batch of text chunks into 384-dimensional float vectors.
- [ ] Searching the index with a semantic query (e.g., *"neural network evaluation"*) returns the most relevant passage from the sample document in the top 3 results.
- [ ] Index serializes to disk and reloads cleanly without data loss.

---

## Phase 3: Relational Storage & Metadata Management

### Objective
Model and store persistent document records, chunk content, citations, and conversation history in PostgreSQL using SQLAlchemy ORM and Alembic migrations.

### Key Concepts to Learn
- **Relational vs. Vector Database**: Why vectors belong in FAISS for search, but full text and metadata belong in SQL for ACID compliance and relational queries.
- **Database Migrations with Alembic**: Why schema changes must be version-controlled.

### Tasks
- [ ] Configure database connection string via environment variables (`DATABASE_URL`).
- [ ] Define SQLAlchemy models:
  - `Document`: `id`, `filename`, `file_hash`, `title`, `page_count`, `created_at`
  - `Chunk`: `id`, `document_id`, `chunk_index`, `page_number`, `content`, `token_count`
  - `ChatMessage`: `id`, `conversation_id`, `role`, `content`, `created_at`
- [ ] Set up Alembic migration scripts and run initial migration.

### Acceptance Criteria
- [ ] Alembic migrations run forwards and backwards cleanly.
- [ ] Foreign keys ensure cascading delete (deleting a Document cleans up its Chunks).
- [ ] Database stores full chunk text while FAISS stores vector IDs mapped to chunk IDs.

---

## Phase 4: Pluggable AI Generation & Citation Grounding

### Objective
Implement the RAG orchestration pipeline with a pluggable LLM interface. Create a deterministic Mock Provider first (for fast tests and zero API costs), followed by external providers (Gemini, Groq, or OpenAI).

### Key Concepts to Learn
- **Prompt Engineering for Citations**: Structuring prompts to enforce strict factual grounding and prevent hallucinations.
- **Pluggable Architecture / Strategy Pattern**: Writing code against an abstract interface (`BaseLLMProvider`) so providers can be swapped with a single config flag.

### Milestones & Tasks
- **Milestone 1: Pluggable LLM Interface, Provider Factory & Mock Provider (Completed)**
  - [x] Create Pydantic DTOs (`LLMChunkContext`, `LLMRequest`, `LLMResponse`) with bounds and offset validations.
  - [x] Define typed exception hierarchy (`LLMError`, `LLMConfigurationError`, `LLMProviderError`, `LLMQuotaExceededError`, `LLMContextLimitExceededError`).
  - [x] Create `BaseLLMProvider` abstract class with synchronous `generate(request)` and configuration validation.
  - [x] Implement deterministic `MockLLMProvider` using standard `[CHUNK:<id>]` citation marker syntax and empty-context fallback.
  - [x] Implement provider factory/registry (`get_llm_provider`) rejecting unknown providers without silent fallback.
  - [x] Configure `LLM_PROVIDER=mock` in core settings and `.env.example`.
  - [x] Add 24 comprehensive unit tests verifying contracts, determinism, and error simulations.

- **Milestone 2: Citation Verification, RAG Orchestration & Live Gemini Provider**
  - [x] Implement `CitationValidator` / `CitationParser` to parse raw `[CHUNK:<id>]` tags and verify against retrieved candidate chunk IDs.
  - [x] Implement `RAGService` to orchestrate query embedding, FAISS top-$K$ search, PostgreSQL chunk joins, prompt construction, and citation assembling.
  - [x] Implement live Google Gemini LLM provider adapter (`GeminiLLMProvider`) using the official `google-genai` SDK with grounded prompt construction, secret sanitization, and typed exception mapping.

### Acceptance Criteria
- [x] When using `MockLLMProvider`, unit tests verify deterministic generation without internet access or API keys.
- [x] When using `GeminiLLMProvider`, unit tests verify grounded prompt construction, citation marker extraction, exception translation, and factory selection with mocked client.
- [x] The generated response links each factual claim to specific chunk IDs and page numbers via authoritative citation validator.
- [x] If no relevant chunks are found above a similarity threshold, the system gracefully responds that no source evidence was found.

---

## Phase 5: FastAPI Backend Endpoints & Validation

### Objective
Expose the ingestion, search, and RAG pipelines via clean, type-safe RESTful API endpoints using FastAPI and Pydantic.

### Tasks
- [ ] Implement `POST /api/documents`: Upload PDF, trigger extraction, chunking, embedding, and storage.
- [ ] Implement `GET /api/documents`: List all uploaded documents with status and metadata.
- [ ] Implement `DELETE /api/documents/{id}`: Delete a document, its chunks, and remove vectors from the index.
- [ ] Implement `POST /api/search`: Pure semantic search returning relevant chunks with similarity scores.
- [ ] Implement `POST /api/query`: Full RAG endpoint returning grounded answer with structured citations.
- [ ] Add CORS middleware, global error handling, and request logging.

### Acceptance Criteria
- [ ] Interactive OpenAPI docs (`/docs`) render all endpoints with full request/response schemas.
- [ ] Uploading an invalid file (non-PDF or oversized) returns HTTP 400/422 with a clear error message.
- [ ] Integration tests verify each endpoint using FastAPI's `TestClient`.

---

## Phase 6: Frontend Development (React + TypeScript + Vite)

### Objective
Build a clean, responsive, portfolio-grade UI allowing users to upload papers, review document libraries, search excerpts, and chat with citation sidebars.

### Tasks
- [ ] Set up clean component architecture:
  - `Navbar`: App title, status indicator, upload button.
  - `DocumentList`: List of indexed papers with page counts and deletion options.
  - `UploadModal`: Drag-and-drop PDF uploader with progress/status indicator.
  - `ChatInterface`: Message history, user input box, loading skeleton.
  - `CitationViewer`: Interactive drawer or modal showing the exact PDF excerpt and page number when a citation is clicked.
- [ ] Create strongly-typed API client service (`api/client.ts`).
- [ ] Add state management for active conversation and document list.

### Acceptance Criteria
- [ ] User can drag and drop a PDF file and view processing feedback.
- [ ] User can ask questions and see streaming or instant responses.
- [ ] Clicking on any citation badge (e.g., `[Page 4]`) highlights the relevant source passage in the UI.

---

## Phase 7: Automated Testing & Verification

### Objective
Ensure robust test coverage across backend and frontend layers to guarantee reliability and prevent regressions.

### Tasks
- [ ] **Backend Unit Tests**: PDF extraction, chunking math, mock LLM generation.
- [ ] **Backend Integration Tests**: Full API workflow (upload $\rightarrow$ index $\rightarrow$ query $\rightarrow$ delete).
- [ ] **Frontend Tests**: Component rendering, citation click handler, and API error states using Vitest and React Testing Library.

### Acceptance Criteria
- [ ] `pytest` passes with high test coverage on core services.
- [ ] Frontend tests pass via `npm run test`.
- [ ] CI/CD-ready test script runs seamlessly in local environment.

---

## Phase 8: Hardening, Evaluation & Deployment

### Objective
Prepare ResearchLens AI for public showcase: evaluate retrieval quality, apply security hardening, and provide zero-cost/low-cost deployment guides.

### Tasks
- [ ] Document retrieval evaluation metrics (Recall@K, citation precision).
- [ ] Build Dockerfile for the backend service.
- [ ] Provide simple deployment configurations (e.g., Render / Railway / Docker Compose).
- [ ] Finalize user documentation, demo video / screenshots, and portfolio presentation.

### Acceptance Criteria
- [ ] System runs in a single Docker container or via Docker Compose.
- [ ] Production-ready environment variables documented in `.env.example`.
- [ ] Free/low-cost deployment instructions verified.
