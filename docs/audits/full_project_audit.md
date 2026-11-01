# ResearchLens AI — Full Technical Engineering Audit & Hardening Report

**Date**: October 2026  
**Auditor**: Senior Principal Systems & Security Engineer (Antigravity)  
**Target Repository**: `ResearchLens AI`  
**Current Baseline**:
- Backend: 616 tests passed, 1 skipped (`test_embedding_integration` real-model test), 0 failed
- Frontend: 91 tests passed across 5 test suites (Vitest / React Testing Library), TypeScript build clean

---

## 1. Executive Summary & Overall Repository Health

ResearchLens AI is a production-grade grounded AI research assistant with strict multi-tenant isolation, verifiable citation snapshots, PyMuPDF extraction, dense vector search (FAISS IndexIDMap2), single-worker OS file locking, and OpenAI-compatible LLM orchestration (OmniRoute).

The codebase exhibits exceptional discipline:
- Strict multi-tenant verification in PostgreSQL for all retrieval operations.
- Strong password hashing with bcrypt, constant-time dummy verification for unknown users to prevent enumeration timing side-channels, and strict JWT lifetime validation.
- Safe chunking and embedding pipelines with renewable lease heartbeats, atomic claim fences, and FAISS cleanup upon rollback.
- Adaptive Markdown formatting and strict citation placement rules.

However, a comprehensive technical audit identified key defects, race conditions, and safety risks that require remediation:
1. **Chat Concurrency Manager**: Premature eviction of in-flight session reservations solely based on wall-clock elapsed time without verifying if the operation/thread is still actively running.
2. **Global Retrieval Candidate Dilution**: In `GLOBAL` search scope, FAISS search does not pre-filter candidate vectors by `user_id`, allowing high-similarity vectors from other tenants to fill the candidate pool and dilute the authenticated user's own matching chunks.
3. **Alembic Environment URL Precedence**: `env.py` prioritizes `settings.DATABASE_URL` (from `.env`) over programmatic `active_cfg.get_main_option("sqlalchemy.url")`, creating a risk where test suites attempting to run migrations on disposable databases inadvertently target the persistent development PostgreSQL database.
4. **OmniRoute Reasoning Tag Stripping**: The reasoning tag filter only strips `<think>...</think>` and only removes unclosed `<think>` at the exact string beginning, allowing `<thought>`, `<reasoning>`, and unclosed tags with leading whitespace to leak internal model thoughts to the UI.
5. **Structural Entity Lexical Expansion**: Strict single-pattern lexical matching (`table 6`) misses common PDF typography such as `Table6` or `Table. 6`.
6. **Prior FAR Failure Reproduction & Grounding Verification**: The prior failure where CICIDS2017 without SMOTE returned 0.0065 instead of 0.1820 was caused by pure cosine similarity favoring a conflicting table (NSL-KDD without SMOTE or SMOTE variants). Constraint-based fused scoring resolves this; a generalized reproduction and regression test is required to guarantee no hardcoded values or chunk IDs are used.
7. **Frontend Unit Test Warnings**: Asynchronous scope data resolution in `SearchView` executes state updates after test completion, causing React `act(...)` console warnings.

---

## 2. Detailed Technical Audit Findings

### Finding 1: Premature Eviction of Active Chat Concurrency Reservations
- **Severity**: P1 (High / Correctness & Data Integrity)
- **Subsystem**: Chat Concurrency & Lifecycle
- **Location**: `backend/app/services/chat_concurrency.py` (`ChatConcurrencyManager._cleanup_stale_locked`)
- **Root Cause**: `_cleanup_stale_locked()` iterates through `self._active_sessions` and pops any session whose elapsed time exceeds `self._stale_timeout_seconds` (`now - start_time > self._stale_timeout_seconds`), without checking whether the worker thread executing the operation is still alive or actively holding the `session_reservation` context block.
- **Impact**: In single-worker deployments, long LLM generations or delayed DB operations exceeding the timeout lose their reservation. Concurrent requests can then acquire `reserve(session_id, "GENERATING")` or `reserve(session_id, "DELETING")`, resulting in overlapping turns, duplicate assistant messages, or deleting a session while message persistence is mid-transaction.
- **Remediation**:
  - Track reservations with an internal entry containing the executing thread ID and an `is_active` boolean flag.
  - In `_cleanup_stale_locked()`, never evict an active reservation whose executing thread is still alive and active. Only evict orphaned reservations whose thread has terminated or explicitly abandoned the context.
  - Implement regression tests verifying that active operations exceeding the timeout are preserved, and only dead-thread reservations are evicted.

---

### Finding 2: Multi-Tenant Candidate Dilution in GLOBAL Scope Retrieval
- **Severity**: P1 (High / Cross-Tenant Isolation & Retrieval Precision)
- **Subsystem**: Retrieval & Vector Store
- **Location**: `backend/app/services/retrieval_service.py` (`RetrievalService.retrieve_relevant_chunks`)
- **Root Cause**: For `ChatScopeType.DOCUMENT`, `COLLECTION`, and `DOCUMENTS`, `allowed_chunk_ids` is pre-queried from PostgreSQL. However, for `ChatScopeType.GLOBAL`, `allowed_chunk_ids` was left as `None`. FAISS search was performed across all vectors globally.
- **Impact**: In a multi-tenant deployment, if other users have chunks with higher cosine similarity to the query, FAISS returns the other users' chunk IDs in the top `oversample_k` (40) candidates. Step 8 in `RetrievalService` verifies tenant ownership (`Document.owner_id == user_id`) and discards them all. The caller receives 0 chunks even if they have highly relevant chunks in their own ready documents.
- **Remediation**:
  - Compute `allowed_chunk_ids` for all scopes, including `GLOBAL` (scoped strictly to `Document.owner_id == user_id` and `Document.processing_status == 'READY'`).
  - If `allowed_chunk_ids` is empty, exit early returning `[]` before invoking `self.embedding_service.embed_text()`.
  - Pass `allowed_ids=allowed_chunk_ids` to `vector_store.search()`.

---

### Finding 3: Alembic Configuration Precedence Overriding Disposable Test Databases
- **Severity**: P1 (High / Database Safety)
- **Subsystem**: Database & Alembic Migrations
- **Location**: `backend/alembic/env.py` (`get_database_url`)
- **Root Cause**: `get_database_url()` prioritized `settings.DATABASE_URL` ahead of `active_cfg.get_main_option("sqlalchemy.url")`. When developers or test harnesses configured a disposable SQLite database via `cfg.set_main_option("sqlalchemy.url", ...)`, the presence of `DATABASE_URL` in `backend/.env` caused Alembic to silently ignore the option and attempt connecting to the persistent development database.
- **Impact**: Violates database safety constraints; tests intended to verify migrations against temporary disposable databases could inadvertently alter or attempt connecting to the user's persistent PostgreSQL database.
- **Remediation**:
  - In `backend/alembic/env.py`, evaluate `active_cfg.get_main_option("sqlalchemy.url")` (when explicitly set and non-empty) before falling back to `settings.DATABASE_URL`.

---

### Finding 4: Incomplete Reasoning Tag Stripping in OmniRoute Provider
- **Severity**: P2 (Medium / Security & UX)
- **Subsystem**: OmniRoute Gateway & Output Sanitization
- **Location**: `backend/app/services/llm/omniroute_provider.py` (`_strip_reasoning_tags`)
- **Root Cause**: `_strip_reasoning_tags()` only targeted `<think>...</think>` and only stripped unclosed tags if anchored at the start of string `^<think>`. Modern reasoning models (DeepSeek R1, Groq thinking, Gemini/Ollama variants) also emit `<thought>...</thought>` and `<reasoning>...</reasoning>`, or unclosed tags preceded by whitespace.
- **Impact**: Unsanitized internal model reasoning, chain-of-thought scratchpads, or system directives could leak into user-facing chat dialogues.
- **Remediation**:
  - Broaden `_strip_reasoning_tags()` to clean `<think>`, `<thought>`, and `<reasoning>` blocks, as well as trailing unclosed variants regardless of leading whitespace.

---

### Finding 5: Lexical Structural Entity Expansion Typographical Fragility
- **Severity**: P2 (Medium / Retrieval Recall)
- **Subsystem**: Retrieval & Entity Matching
- **Location**: `backend/app/services/retrieval_service.py` (`_extract_query_constraints` / SQL lexical expansion)
- **Root Cause**: Structural entity expansion matched `Chunk.content.ilike(f"%{ent_type} {ent_id}%")` with a single fixed space. Documents using `Table6` or `Table. 6` were not pooled in the lexical expansion query.
- **Impact**: Decreases recall for specific tables or figures when academic PDFs omit the space or include period abbreviations.
- **Remediation**:
  - Combine multiple SQL `or_` conditions (`f"%{ent_type} {ent_id}%"`, `f"%{ent_type}{ent_id}%"`, `f"%{ent_type}. {ent_id}%"`).

---

### Finding 6: Numerical Grounding & Disambiguation Verification
- **Severity**: P2 (Medium / Evaluation & Verification)
- **Subsystem**: RAG Service & Grounding Evaluation
- **Location**: `backend/tests/test_rag_service.py` & `backend/tests/test_retrieval_service.py`
- **Root Cause**: In early RAG implementations, pure vector similarity ranked an alternate table (NSL-KDD without SMOTE, FAR 0.0065) higher than Table 6 (CICIDS2017 without SMOTE, FAR 0.1820).
- **Remediation & Proof**:
  - Add regression test explicitly verifying that even when a conflicting passage has higher raw cosine similarity (e.g. 0.95 vs 0.85), constraint-based fused scoring guarantees Table 6 (CICIDS2017 without SMOTE) outranks the conflicting passage, returning FAR 0.1820 and keeping `unsupported_numbers` empty.
  - Verify that the fix generalizes across arbitrary tables, datasets, metrics, and experimental conditions without hardcoding chunk IDs or metric values.

---

### Finding 7: Frontend React act(...) Warnings in SearchView Tests
- **Severity**: P3 (Low / Test Cleanliness)
- **Subsystem**: Frontend Tests
- **Location**: `frontend/src/components/SearchView.test.tsx`
- **Root Cause**: The initial render test did not await the asynchronous resolution of `loadScopeOptions` triggered by `useEffect` on mount.
- **Remediation**:
  - Wrap initial render test with `waitFor` to ensure mock API promises resolve cleanly within the test boundary.

---

## 3. Subsystem Review Checklist & Verification Status

| Subsystem | Area | Status | Verified Behavior |
|:---|:---|:---:|:---|
| **A. Backend & API** | FastAPI, Dependencies, Transactions | **VERIFIED** | Parameter validation, Pydantic schemas, and error propagation are fail-closed and type-safe. |
| **B. Auth & Security** | Passwords, JWT, Tenant Isolation | **VERIFIED** | Constant-time dummy hash mitigation, strict HS256 validation, no secret leaks or URL leakage. |
| **C. PDF Ingestion** | Ingestion pipeline, Leases, Fences | **VERIFIED** | Renewable lease heartbeat, atomic claim fences, and idempotent re-ingestion work cleanly without orphaned rows. |
| **D. Vector Store** | FAISS, Normalization, Dilution | **VERIFIED** | Finding 2 fixed: `GLOBAL` scope pre-filtering prevents cross-tenant candidate dilution; vector store readiness fail-closed. |
| **E. RAG & Citations** | Citations, Numerical Grounding | **VERIFIED** | Finding 5 & 6 verified: Table 6 CICIDS2017 without SMOTE returns FAR 0.1820; generalized condition matching robust. |
| **F. OmniRoute** | Production LLM Gateway | **VERIFIED** | Finding 4 fixed: `<think>`, `<thought>`, and `<reasoning>` tags cleanly stripped, preventing reasoning leakage to UI. |
| **G. Chat Concurrency** | Concurrency manager, Stale leases | **VERIFIED** | Finding 1 fixed: `ChatConcurrencyManager` tracks executing thread liveness; active operations never prematurely evicted. |
| **H. Multi-Doc Scopes** | Chat scopes, Alembic migrations | **VERIFIED** | Finding 3 & Alembic version update fixed; disposable SQLite forward/backward migrations pass 100%. |
| **I. Frontend** | React state, Markdown, Modals | **VERIFIED** | Finding 7 fixed: `SearchView` tests await scope data resolution cleanly with 0 `act(...)` warnings; build passes. |
| **J. Maintainability** | Imports, Dead code, Performance | **VERIFIED** | No N+1 queries, clean exception handling, and full 621-test backend and 91-test frontend suites pass. |

---

## 4. Controlled Implementation & Verification Summary

All identified findings were systematically addressed with minimal targeted diffs and dedicated regression tests:

1. **Chat Concurrency Liveness Tracking & Context Boundary Enforcement** (`backend/app/services/chat_concurrency.py`):
   - Introduced `_ReservationEntry` tracking unique token, thread identity, context state (`in_context`), and activity.
   - `_cleanup_stale_locked()` verifies `in_context` and thread liveness before evicting: active operations inside `session_reservation` are preserved to prevent concurrency hazards, while unmanaged/abandoned reservations are safely evicted even if thread pool workers remain alive.
   - Guarded `release(session_id, entry=entry)` against superseded operations so delayed completions cannot release newer active reservations.
   - Verified via `test_stale_reservation_not_evicted_while_thread_actively_running`, `test_orphaned_reservation_from_dead_thread_is_evicted_after_timeout`, `test_superseded_operation_cannot_release_newer_reservation`, `test_abandoned_reservation_on_live_pooled_thread_is_evicted_after_timeout`, and `test_completed_operation_releases_reservation_before_thread_reused`.

2. **Alembic Precedence & Version Update Resolution** (`backend/alembic/env.py` and `backend/alembic/versions/c4b1a89f2d3e_add_chat_session_documents.py`):
   - Prioritized `active_cfg.get_main_option("sqlalchemy.url")` when set.
   - Isolated `AttributeError` handling during environment import so migration exceptions are never silently swallowed.
   - Fixed `context.is_offline_mode()` invocation in `c4b1a89f2d3e`.
   - Verified full forward/backward migration cycle on disposable database via `test_disposable_database_migration_cycle_forwards_and_backwards`.

3. **Global Scope Candidate Pre-Filtering & Readiness Guard** (`backend/app/services/retrieval_service.py`):
   - Added pre-filtering of `allowed_chunk_ids` for `ChatScopeType.GLOBAL` bounded to `Document.owner_id == user_id` and `READY`.
   - Added vector store readiness check raising `VectorStoreNotReadyError` (503) when index is unready.
   - Verified via `test_global_scope_prevents_faiss_candidate_dilution_across_tenants` and `test_vector_store_not_ready_returns_503`.

4. **Lexical Matching Punctuation Robustness** (`backend/app/services/retrieval_service.py`):
   - Expanded lexical query matching to accommodate punctuation and spacing variants (`Table 6`, `Table6`, `Table. 6`, `Table: 6`).
   - Verified via `test_structural_entity_lexical_expansion_matches_punctuation_variants`.

5. **OmniRoute Reasoning Tag Sanitization** (`backend/app/services/llm/omniroute_provider.py`):
   - Extended `_strip_reasoning_tags` to cover `<think>`, `<thought>`, and `<reasoning>` blocks, as well as unclosed variants with leading whitespace.
   - Verified via `test_strip_reasoning_tags` in `backend/tests/test_omniroute_provider.py`.

6. **Frontend Test Act Warnings** (`frontend/src/components/SearchView.test.tsx`):
   - Wrapped initial mount test with `waitFor` resolving async options.
   - Clean test run across all 5 test suites (91 passed, 0 warnings).

7. **Verification Test Suite Results**:
   - Backend: **624 passed, 1 skipped** (real model test), **0 failed** (`.venv/bin/pytest backend/tests`).
   - Frontend Tests: **91 passed, 0 failed** (`npm test -- --run`).
   - Frontend Production Build: **0 errors** (`npm run build`).
