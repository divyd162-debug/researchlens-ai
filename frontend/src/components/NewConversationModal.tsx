import React, { useState, useEffect, useMemo } from 'react'
import { api, ApiError } from '../services/api'
import {
  ChatScopeType,
  ChatSessionCreateRequest,
  ChatSessionResponse,
  CollectionResponse,
  DocumentResponse,
} from '../types'

interface NewConversationModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: (session: ChatSessionResponse) => void
}

export const NewConversationModal: React.FC<NewConversationModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [title, setTitle] = useState<string>('New Research Chat')
  const [scopeType, setScopeType] = useState<ChatScopeType>('GLOBAL')
  const [selectedDocId, setSelectedDocId] = useState<string>('')
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const [selectedColId, setSelectedColId] = useState<string>('')
  const [documentSearch, setDocumentSearch] = useState<string>('')

  const [documents, setDocuments] = useState<DocumentResponse[]>([])
  const [collections, setCollections] = useState<CollectionResponse[]>([])
  const [isLoadingData, setIsLoadingData] = useState<boolean>(false)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // Load documents and collections when modal opens
  useEffect(() => {
    if (!isOpen) {
      setTitle('New Research Chat')
      setScopeType('GLOBAL')
      setSelectedDocId('')
      setSelectedDocIds([])
      setSelectedColId('')
      setDocumentSearch('')
      setError(null)
      return
    }

    let isMounted = true
    setIsLoadingData(true)
    setError(null)

    Promise.all([
      api.documents.list(100, 0).catch(() => ({ items: [], total: 0, limit: 100, offset: 0 })),
      api.collections.list(100, 0).catch(() => ({ items: [], total: 0, limit: 100, offset: 0 })),
    ])
      .then(([docRes, colRes]) => {
        if (isMounted) {
          setDocuments(docRes.items)
          setCollections(colRes.items)
          if (docRes.items.length > 0) {
            const firstReady = docRes.items.find((d) => d.processing_status === 'READY')
            if (firstReady) {
              setSelectedDocId(firstReady.id)
            }
          }
          if (colRes.items.length > 0) {
            setSelectedColId(colRes.items[0].id)
          }
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to load options')
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsLoadingData(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [isOpen])

  // Filter documents by search term
  const filteredDocuments = useMemo(() => {
    if (!documentSearch.trim()) return documents
    const q = documentSearch.toLowerCase()
    return documents.filter((doc) => {
      const matchTitle = doc.title && doc.title.toLowerCase().includes(q)
      const matchFile = doc.filename.toLowerCase().includes(q)
      return matchTitle || matchFile
    })
  }, [documents, documentSearch])

  if (!isOpen) return null

  const handleToggleDocument = (docId: string, isReady: boolean) => {
    if (!isReady) return
    setSelectedDocIds((prev) => {
      if (prev.includes(docId)) {
        return prev.filter((id) => id !== docId)
      } else {
        if (prev.length >= 50) return prev
        return [...prev, docId]
      }
    })
  }

  const handleSelectAllReady = () => {
    const readyIds = filteredDocuments
      .filter((d) => d.processing_status === 'READY')
      .map((d) => d.id)
      .slice(0, 50)
    setSelectedDocIds(readyIds)
  }

  const handleDeselectAll = () => {
    setSelectedDocIds([])
  }

  const isFormValid = (): boolean => {
    if (scopeType === 'GLOBAL') return true
    if (scopeType === 'DOCUMENT') return Boolean(selectedDocId)
    if (scopeType === 'DOCUMENTS') return selectedDocIds.length >= 1 && selectedDocIds.length <= 50
    if (scopeType === 'COLLECTION') return Boolean(selectedColId)
    return false
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!isFormValid() || isSubmitting) return

    setIsSubmitting(true)
    setError(null)

    const payload: ChatSessionCreateRequest = {
      title: title.trim() || 'New Research Chat',
      scope_type: scopeType,
    }

    if (scopeType === 'DOCUMENT') {
      payload.target_document_id = selectedDocId
    } else if (scopeType === 'DOCUMENTS') {
      payload.document_ids = selectedDocIds
    } else if (scopeType === 'COLLECTION') {
      payload.target_collection_id = selectedColId
    }

    try {
      const newSession = await api.chat.createSession(payload)
      onSuccess(newSession)
      onClose()
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail)
      } else {
        setError('Failed to create conversation. Please try again.')
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card modal-new-conversation"
        role="dialog"
        aria-modal="true"
        aria-labelledby="new-conversation-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id="new-conversation-title">New Conversation</h3>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close dialog"
            disabled={isSubmitting}
          >
            &times;
          </button>
        </div>

        {error ? (
          <div className="alert alert-error" role="alert" style={{ marginBottom: '1rem' }}>
            <span>{error}</span>
          </div>
        ) : null}

        <form onSubmit={handleSubmit}>
          <div className="form-group" style={{ marginBottom: '1.25rem' }}>
            <label htmlFor="new-chat-title" className="form-label" style={{ fontWeight: 600 }}>
              Conversation Title
            </label>
            <input
              id="new-chat-title"
              type="text"
              className="form-input"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Attention Mechanism Analysis"
              maxLength={255}
              disabled={isSubmitting}
            />
          </div>

          <div className="form-group" style={{ marginBottom: '1.25rem' }}>
            <label className="form-label" style={{ fontWeight: 600, display: 'block', marginBottom: '0.5rem' }}>
              Corpus Scope
            </label>
            <div className="scope-selection-grid" role="radiogroup" aria-label="Select scope mode">
              <label
                className={`scope-option-card ${scopeType === 'GLOBAL' ? 'selected' : ''}`}
                onClick={() => setScopeType('GLOBAL')}
              >
                <input
                  type="radio"
                  name="scope_type"
                  value="GLOBAL"
                  checked={scopeType === 'GLOBAL'}
                  onChange={() => setScopeType('GLOBAL')}
                />
                <div className="scope-card-content">
                  <span className="scope-card-title">All Documents</span>
                  <span className="scope-card-desc">Search across all eligible documents in your library</span>
                </div>
              </label>

              <label
                className={`scope-option-card ${scopeType === 'DOCUMENTS' ? 'selected' : ''}`}
                onClick={() => setScopeType('DOCUMENTS')}
              >
                <input
                  type="radio"
                  name="scope_type"
                  value="DOCUMENTS"
                  checked={scopeType === 'DOCUMENTS'}
                  onChange={() => setScopeType('DOCUMENTS')}
                />
                <div className="scope-card-content">
                  <span className="scope-card-title">Selected Documents</span>
                  <span className="scope-card-desc">Explicitly choose 1 to 50 documents for this chat</span>
                </div>
              </label>

              <label
                className={`scope-option-card ${scopeType === 'DOCUMENT' ? 'selected' : ''}`}
                onClick={() => setScopeType('DOCUMENT')}
              >
                <input
                  type="radio"
                  name="scope_type"
                  value="DOCUMENT"
                  checked={scopeType === 'DOCUMENT'}
                  onChange={() => setScopeType('DOCUMENT')}
                />
                <div className="scope-card-content">
                  <span className="scope-card-title">One Document</span>
                  <span className="scope-card-desc">Focus conversation on a single document</span>
                </div>
              </label>

              <label
                className={`scope-option-card ${scopeType === 'COLLECTION' ? 'selected' : ''}`}
                onClick={() => setScopeType('COLLECTION')}
              >
                <input
                  type="radio"
                  name="scope_type"
                  value="COLLECTION"
                  checked={scopeType === 'COLLECTION'}
                  onChange={() => setScopeType('COLLECTION')}
                />
                <div className="scope-card-content">
                  <span className="scope-card-title">Collection</span>
                  <span className="scope-card-desc">Scope conversation to a curated collection</span>
                </div>
              </label>
            </div>
          </div>

          {/* Conditional target configuration */}
          {isLoadingData ? (
            <div className="text-muted text-sm" style={{ padding: '1rem', textAlign: 'center' }}>
              Loading options...
            </div>
          ) : (
            <>
              {scopeType === 'DOCUMENTS' && (
                <div className="scope-config-panel multi-doc-selector" style={{ marginBottom: '1.25rem' }}>
                  <div className="multi-doc-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span className="multi-doc-counter" style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                      Selected: {selectedDocIds.length} / 50 documents
                    </span>
                    <div className="multi-doc-quick-actions" style={{ display: 'flex', gap: '0.5rem' }}>
                      <button
                        type="button"
                        className="btn-link text-xs"
                        onClick={handleSelectAllReady}
                        disabled={filteredDocuments.filter((d) => d.processing_status === 'READY').length === 0}
                      >
                        Select All Ready
                      </button>
                      <button
                        type="button"
                        className="btn-link text-xs"
                        onClick={handleDeselectAll}
                        disabled={selectedDocIds.length === 0}
                      >
                        Clear
                      </button>
                    </div>
                  </div>

                  <input
                    type="text"
                    className="form-input form-input-sm"
                    placeholder="Search documents by title or filename..."
                    value={documentSearch}
                    onChange={(e) => setDocumentSearch(e.target.value)}
                    style={{ marginBottom: '0.5rem' }}
                    aria-label="Filter documents"
                  />

                  <div className="multi-doc-list" role="listbox" aria-label="Available documents">
                    {filteredDocuments.length === 0 ? (
                      <div className="text-muted text-xs" style={{ padding: '1rem', textAlign: 'center' }}>
                        No matching documents found.
                      </div>
                    ) : (
                      filteredDocuments.map((doc) => {
                        const isReady = doc.processing_status === 'READY'
                        const isSelected = selectedDocIds.includes(doc.id)
                        return (
                          <div
                            key={doc.id}
                            className={`multi-doc-item ${!isReady ? 'disabled' : ''} ${isSelected ? 'checked' : ''}`}
                            onClick={() => handleToggleDocument(doc.id, isReady)}
                            role="option"
                            aria-selected={isSelected}
                          >
                            <input
                              type="checkbox"
                              checked={isSelected}
                              disabled={!isReady}
                              onChange={() => handleToggleDocument(doc.id, isReady)}
                              onClick={(e) => e.stopPropagation()}
                              aria-label={`Select ${doc.title || doc.filename}`}
                            />
                            <div className="multi-doc-item-details">
                              <span className="doc-item-title" title={doc.title || doc.filename}>
                                {doc.title || doc.filename}
                              </span>
                              <div className="doc-item-meta text-muted text-xs">
                                {doc.page_count != null && <span>{doc.page_count} pages</span>}
                                <span className={`badge badge-sm badge-status-${doc.processing_status.toLowerCase()}`}>
                                  {doc.processing_status}
                                </span>
                              </div>
                            </div>
                          </div>
                        )
                      })
                    )}
                  </div>
                  {selectedDocIds.length === 0 && (
                    <p className="text-muted text-xs" style={{ marginTop: '0.4rem', color: 'var(--text-muted)' }}>
                      Select at least 1 document to create conversation.
                    </p>
                  )}
                </div>
              )}

              {scopeType === 'DOCUMENT' && (
                <div className="scope-config-panel" style={{ marginBottom: '1.25rem' }}>
                  <label htmlFor="select-single-doc" className="form-label text-sm" style={{ fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>
                    Target Document
                  </label>
                  <select
                    id="select-single-doc"
                    className="form-input"
                    value={selectedDocId}
                    onChange={(e) => setSelectedDocId(e.target.value)}
                    disabled={isSubmitting}
                  >
                    <option value="" disabled>-- Select a ready document --</option>
                    {documents
                      .filter((d) => d.processing_status === 'READY')
                      .map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.title || d.filename} ({d.page_count ?? '?'} pages)
                        </option>
                      ))}
                  </select>
                  {documents.filter((d) => d.processing_status === 'READY').length === 0 && (
                    <p className="text-muted text-xs" style={{ marginTop: '0.4rem', color: 'var(--error-text)' }}>
                      No ready documents available. Please upload and process a document first.
                    </p>
                  )}
                </div>
              )}

              {scopeType === 'COLLECTION' && (
                <div className="scope-config-panel" style={{ marginBottom: '1.25rem' }}>
                  <label htmlFor="select-collection" className="form-label text-sm" style={{ fontWeight: 600, display: 'block', marginBottom: '0.35rem' }}>
                    Target Collection
                  </label>
                  <select
                    id="select-collection"
                    className="form-input"
                    value={selectedColId}
                    onChange={(e) => setSelectedColId(e.target.value)}
                    disabled={isSubmitting}
                  >
                    <option value="" disabled>-- Select a collection --</option>
                    {collections.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name} ({c.document_count} docs)
                      </option>
                    ))}
                  </select>
                  {collections.length === 0 && (
                    <p className="text-muted text-xs" style={{ marginTop: '0.4rem', color: 'var(--error-text)' }}>
                      No collections available. Create a collection first.
                    </p>
                  )}
                </div>
              )}
            </>
          )}

          <div className="modal-footer" style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={!isFormValid() || isSubmitting}
            >
              {isSubmitting ? 'Creating...' : 'Create Conversation'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
