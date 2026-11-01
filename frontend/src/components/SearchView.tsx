import React, { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../services/api'
import {
  CollectionResponse,
  DocumentResponse,
  SearchPassageResponse,
  SearchRequest,
  SearchResponse,
  SearchScopeType,
} from '../types'

export const SearchView: React.FC = () => {
  const [query, setQuery] = useState('')
  const [scopeType, setScopeType] = useState<SearchScopeType>('GLOBAL')
  const [selectedDocId, setSelectedDocId] = useState<string>('')
  const [selectedColId, setSelectedColId] = useState<string>('')
  const [topK, setTopK] = useState<number>(5)
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(0.0)
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false)

  // Document & Collection metadata for scoping dropdowns
  const [documents, setDocuments] = useState<DocumentResponse[]>([])
  const [collections, setCollections] = useState<CollectionResponse[]>([])
  const [isLoadingScopeData, setIsLoadingScopeData] = useState<boolean>(false)

  // Search execution state
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null)

  // Race condition & stale response prevention
  const abortControllerRef = useRef<AbortController | null>(null)
  const activeRequestIdRef = useRef<number>(0)

  // Load user's documents and collections for scope selectors
  useEffect(() => {
    let isMounted = true
    const loadScopeOptions = async () => {
      setIsLoadingScopeData(true)
      try {
        const [docsRes, colsRes] = await Promise.all([
          api.documents.list(100, 0),
          api.collections.list(100, 0),
        ])
        if (isMounted) {
          setDocuments(docsRes.items)
          setCollections(colsRes.items)
          if (docsRes.items.length > 0 && !selectedDocId) {
            setSelectedDocId(docsRes.items[0].id)
          }
          if (colsRes.items.length > 0 && !selectedColId) {
            setSelectedColId(colsRes.items[0].id)
          }
        }
      } catch {
        // Scoping data failure is non-fatal for Global search
      } finally {
        if (isMounted) {
          setIsLoadingScopeData(false)
        }
      }
    }

    loadScopeOptions()

    return () => {
      isMounted = false
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  const executeSearch = async (e?: React.FormEvent) => {
    if (e) {
      e.preventDefault()
    }

    const trimmedQuery = query.trim()
    if (!trimmedQuery) {
      setValidationError('Please enter a search query.')
      return
    }
    if (trimmedQuery.length > 1000) {
      setValidationError('Query cannot exceed 1,000 characters.')
      return
    }

    if (scopeType === 'DOCUMENT' && !selectedDocId) {
      setValidationError('Please select a target document for document-scoped search.')
      return
    }
    if (scopeType === 'COLLECTION' && !selectedColId) {
      setValidationError('Please select a target collection for collection-scoped search.')
      return
    }

    setValidationError(null)
    setError(null)
    setIsLoading(true)

    // Abort previous in-flight search request and increment request sequence ID
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    const controller = new AbortController()
    abortControllerRef.current = controller
    const currentRequestId = ++activeRequestIdRef.current

    const payload: SearchRequest = {
      query: trimmedQuery,
      scope_type: scopeType,
      top_k: topK,
      similarity_threshold: similarityThreshold,
    }

    if (scopeType === 'DOCUMENT') {
      payload.scope_id = selectedDocId
    } else if (scopeType === 'COLLECTION') {
      payload.scope_id = selectedColId
    }

    try {
      const response = await api.search.query(payload, controller.signal)
      // Discard stale responses if a newer query was fired while waiting
      if (currentRequestId === activeRequestIdRef.current) {
        setSearchResponse(response)
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        // Ignored: request was superseded by a newer query
        return
      }
      if (currentRequestId === activeRequestIdRef.current) {
        if (err instanceof ApiError) {
          setError(err.detail)
        } else {
          setError('Search failed. Please check backend connection and try again.')
        }
      }
    } finally {
      if (currentRequestId === activeRequestIdRef.current) {
        setIsLoading(false)
      }
    }
  }

  const getScopeDescription = (): string => {
    if (scopeType === 'GLOBAL') {
      return 'All Documents'
    }
    if (scopeType === 'DOCUMENT') {
      const doc = documents.find((d) => d.id === selectedDocId)
      return doc?.title || doc?.filename || 'Selected Document'
    }
    if (scopeType === 'COLLECTION') {
      const col = collections.find((c) => c.id === selectedColId)
      return col?.name || 'Selected Collection'
    }
    return 'Global'
  }

  const getPassageDocTitle = (passage: SearchPassageResponse): string => {
    if (passage.document_title && passage.document_title !== 'Untitled Document') {
      return passage.document_title
    }
    const doc = documents.find((d) => d.id === passage.document_id)
    return doc?.title || doc?.filename || passage.document_title || 'Untitled Document'
  }

  return (
    <div className="view-container search-view-container">
      <div className="view-header">
        <div>
          <h2>Semantic Passage Search</h2>
          <p className="text-muted">
            Search across scientific documents using dense semantic similarity with exact passage citations.
          </p>
        </div>
        <div className="view-header-actions">
          <span className="badge badge-info">
            {searchResponse ? `${searchResponse.total_results} Results` : 'Vector Store Ready'}
          </span>
        </div>
      </div>

      <div className="search-form-card">
        <form onSubmit={executeSearch} className="search-form" role="search" aria-label="Semantic Search Form">
          <div className="form-group">
            <label htmlFor="search-query-input">Search Query</label>
            <div className="search-input-wrapper">
              <input
                id="search-query-input"
                type="text"
                className="form-input search-input"
                placeholder="Ask a question or enter concepts (e.g., mobile app fetching data from RESTful API)..."
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value)
                  if (validationError) setValidationError(null)
                }}
                maxLength={1000}
                autoFocus
              />
              <button
                type="submit"
                className="btn btn-primary search-submit-btn"
                aria-label="Search Passages"
              >
                {isLoading ? (
                  <>
                    <span className="spinner-inline" aria-hidden="true" />
                    Searching...
                  </>
                ) : (
                  'Search Passages'
                )}
              </button>
            </div>
          </div>

          <div className="search-controls-row">
            <div className="scope-selection-group">
              <span className="scope-label">Corpus Scope:</span>
              <div className="scope-radio-buttons" role="radiogroup" aria-label="Search Scope">
                <label className={`scope-radio-btn ${scopeType === 'GLOBAL' ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="scope_type"
                    value="GLOBAL"
                    checked={scopeType === 'GLOBAL'}
                    onChange={() => setScopeType('GLOBAL')}
                  />
                  All Documents (Global)
                </label>
                <label className={`scope-radio-btn ${scopeType === 'DOCUMENT' ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="scope_type"
                    value="DOCUMENT"
                    checked={scopeType === 'DOCUMENT'}
                    onChange={() => setScopeType('DOCUMENT')}
                  />
                  Document Scoped
                </label>
                <label className={`scope-radio-btn ${scopeType === 'COLLECTION' ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="scope_type"
                    value="COLLECTION"
                    checked={scopeType === 'COLLECTION'}
                    onChange={() => setScopeType('COLLECTION')}
                  />
                  Collection Scoped
                </label>
              </div>
            </div>

            {scopeType === 'DOCUMENT' && (
              <div className="scoped-entity-select-group">
                <label htmlFor="target-document-select" className="sr-only">Select Document</label>
                <select
                  id="target-document-select"
                  className="form-input scope-dropdown"
                  value={selectedDocId}
                  onChange={(e) => setSelectedDocId(e.target.value)}
                  disabled={isLoading || isLoadingScopeData}
                  aria-label="Select target document"
                >
                  {documents.length === 0 ? (
                    <option value="">No documents available</option>
                  ) : (
                    documents.map((doc) => (
                      <option key={doc.id} value={doc.id}>
                        {doc.title ? `${doc.title} (${doc.filename})` : doc.filename}
                        {doc.processing_status !== 'READY' ? ` [${doc.processing_status}]` : ''}
                      </option>
                    ))
                  )}
                </select>
              </div>
            )}

            {scopeType === 'COLLECTION' && (
              <div className="scoped-entity-select-group">
                <label htmlFor="target-collection-select" className="sr-only">Select Collection</label>
                <select
                  id="target-collection-select"
                  className="form-input scope-dropdown"
                  value={selectedColId}
                  onChange={(e) => setSelectedColId(e.target.value)}
                  disabled={isLoading || isLoadingScopeData}
                  aria-label="Select target collection"
                >
                  {collections.length === 0 ? (
                    <option value="">No collections available</option>
                  ) : (
                    collections.map((col) => (
                      <option key={col.id} value={col.id}>
                        {col.name} ({col.document_count} papers)
                      </option>
                    ))
                  )}
                </select>
              </div>
            )}

            <button
              type="button"
              className="btn btn-secondary btn-sm toggle-advanced-btn"
              onClick={() => setShowAdvanced(!showAdvanced)}
              aria-expanded={showAdvanced}
            >
              {showAdvanced ? 'Hide Settings ▲' : 'Settings ▼'}
            </button>
          </div>

          {showAdvanced && (
            <div className="search-advanced-panel">
              <div className="advanced-field">
                <label htmlFor="top-k-input">Max Results (top_k: 1–50):</label>
                <div className="slider-input-combo">
                  <input
                    id="top-k-input"
                    type="range"
                    min="1"
                    max="50"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                  />
                  <span className="slider-value">{topK}</span>
                </div>
              </div>

              <div className="advanced-field">
                <label htmlFor="similarity-threshold-input">Min Similarity Threshold (-1.0 to 1.0):</label>
                <div className="slider-input-combo">
                  <input
                    id="similarity-threshold-input"
                    type="range"
                    min="-1"
                    max="1"
                    step="0.05"
                    value={similarityThreshold}
                    onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
                  />
                  <span className="slider-value">{similarityThreshold.toFixed(2)}</span>
                </div>
              </div>
            </div>
          )}
        </form>
      </div>

      {validationError && (
        <div className="alert alert-warning" role="alert">
          <span>{validationError}</span>
        </div>
      )}

      {error && (
        <div className="alert alert-error" role="alert">
          <span>{error}</span>
          <button type="button" className="btn-retry" onClick={() => executeSearch()}>
            Retry
          </button>
        </div>
      )}

      <div className="search-results-section" aria-live="polite">
        {isLoading ? (
          <div className="loading-state">
            <div className="spinner" aria-label="Searching passages" />
            <p>Searching document passages and ranking dense vector similarities...</p>
          </div>
        ) : searchResponse && searchResponse.results.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">🔍</div>
            <h3>No Matching Passages Found</h3>
            <p>
              No passages met the similarity threshold ({similarityThreshold.toFixed(2)}) for query &ldquo;{searchResponse.query}&rdquo;
              within scope <strong>{getScopeDescription()}</strong>.
            </p>
            <p className="text-muted">
              Try rephrasing your search terms, removing specific technical jargon, or lowering the similarity threshold.
            </p>
          </div>
        ) : searchResponse && searchResponse.results.length > 0 ? (
          <div className="search-results-list">
            <div className="search-results-header">
              <span className="results-count-text" data-testid="results-count">
                Found <strong>{searchResponse.total_results}</strong> ranked {searchResponse.total_results === 1 ? 'passage' : 'passages'} for &ldquo;<strong>{searchResponse.query}</strong>&rdquo;
              </span>
              <span className="badge badge-info">Scope: {getScopeDescription()}</span>
            </div>

            <div className="passage-cards-container">
              {searchResponse.results.map((passage: SearchPassageResponse, index: number) => {
                const scorePercent = Math.max(0, Math.min(100, Math.round(passage.similarity_score * 100)))
                return (
                  <article key={passage.chunk_id} className="passage-card" tabIndex={0}>
                    <header className="passage-card-header">
                      <div className="passage-rank-title">
                        <span className="passage-rank-badge">#{index + 1}</span>
                        <h4 className="passage-doc-title">
                          <span aria-hidden="true">📄</span> {getPassageDocTitle(passage)}
                        </h4>
                      </div>
                      <div className="passage-meta-badges">
                        <span className="badge badge-page">Page {passage.page_number}</span>
                        <span className="badge badge-score" title={`Cosine similarity score: ${passage.similarity_score.toFixed(4)}`}>
                          Similarity: {scorePercent}%
                        </span>
                        <span className="badge badge-offsets">
                          Offsets: [{passage.char_start}–{passage.char_end}]
                        </span>
                      </div>
                    </header>

                    <blockquote className="passage-content">
                      &ldquo;{passage.content}&rdquo;
                    </blockquote>

                    <footer className="passage-card-footer">
                      <span className="text-muted text-xs">
                        Chunk ID: {passage.chunk_id} &bull; Doc UUID: {passage.document_id}
                      </span>
                    </footer>
                  </article>
                )
              })}
            </div>
          </div>
        ) : (
          <div className="search-placeholder-state">
            <div className="placeholder-icon">💡</div>
            <h3>Grounded Semantic Search</h3>
            <p className="text-muted">
              Enter any research query above to retrieve relevant passages directly from your ingested PDF papers.
              Unlike keyword search, semantic search locates conceptually related insights even when phrasing differs.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
