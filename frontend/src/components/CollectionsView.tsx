import React, { useEffect, useState } from 'react'
import { api, ApiError } from '../services/api'
import { CollectionResponse, DocumentResponse } from '../types'
import { CreateCollectionModal } from './CreateCollectionModal'

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export const CollectionsView: React.FC = () => {
  const [collections, setCollections] = useState<CollectionResponse[]>([])
  const [total, setTotal] = useState<number>(0)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)

  // Collection detail view state
  const [selectedCollection, setSelectedCollection] = useState<CollectionResponse | null>(null)
  const [memberDocuments, setMemberDocuments] = useState<DocumentResponse[]>([])
  const [isLoadingMembers, setIsLoadingMembers] = useState<boolean>(false)
  const [memberError, setMemberError] = useState<string | null>(null)

  // Document removal from collection state
  const [documentToRemove, setDocumentToRemove] = useState<DocumentResponse | null>(null)
  const [isRemoving, setIsRemoving] = useState<boolean>(false)
  const [removeError, setRemoveError] = useState<string | null>(null)

  const loadCollections = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await api.collections.list(50, 0)
      setCollections(data.items)
      setTotal(data.total)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail)
      } else {
        setError('Failed to load collections from server.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  const loadMemberDocuments = async (collectionId: string) => {
    setIsLoadingMembers(true)
    setMemberError(null)
    try {
      const data = await api.collections.listDocuments(collectionId, 50, 0)
      setMemberDocuments(data.items)
    } catch (err) {
      if (err instanceof ApiError) {
        setMemberError(err.detail)
      } else {
        setMemberError('Failed to load collection member documents.')
      }
    } finally {
      setIsLoadingMembers(false)
    }
  }

  useEffect(() => {
    loadCollections()
  }, [])

  const handleSelectCollection = (col: CollectionResponse) => {
    setSelectedCollection(col)
    loadMemberDocuments(col.id)
  }

  const handleBackToList = () => {
    setSelectedCollection(null)
    setMemberDocuments([])
    setMemberError(null)
    setDocumentToRemove(null)
    setRemoveError(null)
  }

  const handleCollectionCreated = () => {
    loadCollections()
  }

  const handleConfirmRemove = async () => {
    if (!selectedCollection || !documentToRemove) return

    setIsRemoving(true)
    setRemoveError(null)

    try {
      const updated = await api.collections.removeDocument(selectedCollection.id, documentToRemove.id)
      // Update member list
      setMemberDocuments((prev) => prev.filter((d) => d.id !== documentToRemove.id))
      // Synchronize document count in selected collection
      const newCount = typeof updated.document_count === 'number'
        ? updated.document_count
        : Math.max(0, selectedCollection.document_count - 1)
      setSelectedCollection((prev) => prev ? { ...prev, document_count: newCount } : null)
      // Synchronize document count in parent collections array
      setCollections((prev) =>
        prev.map((c) => (c.id === selectedCollection.id ? { ...c, document_count: newCount } : c))
      )
      setDocumentToRemove(null)
    } catch (err) {
      if (err instanceof ApiError) {
        setRemoveError(err.detail)
      } else {
        setRemoveError('Failed to remove document from collection.')
      }
    } finally {
      setIsRemoving(false)
    }
  }

  // --- Render Collection Detail Interface ---
  if (selectedCollection) {
    return (
      <div className="view-container">
        <div className="view-header">
          <div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleBackToList}
              style={{ marginBottom: '0.75rem' }}
            >
              ← Back to Collections
            </button>
            <h2 className="collection-detail-title">{selectedCollection.name}</h2>
            <p className="text-muted collection-detail-desc">
              {selectedCollection.description || 'No description provided.'}
            </p>
          </div>
          <div className="view-header-actions">
            <span className="badge badge-info collection-doc-count-badge">
              {selectedCollection.document_count}{' '}
              {selectedCollection.document_count === 1 ? 'Document' : 'Documents'}
            </span>
          </div>
        </div>

        {memberError ? (
          <div className="alert alert-error" role="alert">
            {memberError}
            <button
              type="button"
              className="btn-retry"
              onClick={() => loadMemberDocuments(selectedCollection.id)}
            >
              Retry
            </button>
          </div>
        ) : null}

        {isLoadingMembers ? (
          <div className="loading-state">
            <div className="spinner" aria-label="Loading member documents" />
            <p>Loading collection documents...</p>
          </div>
        ) : memberDocuments.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📄</div>
            <h3>No Documents in this Collection</h3>
            <p>
              This collection has no documents assigned yet. Go to the Documents tab and click
              "+ Add to Collection" to add research papers.
            </p>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ marginTop: '1rem' }}
              onClick={handleBackToList}
            >
              Back to Collections List
            </button>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Pages</th>
                  <th>File Size</th>
                  <th>Status</th>
                  <th>Added</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {memberDocuments.map((doc) => (
                  <tr key={doc.id}>
                    <td>
                      <div className="doc-cell">
                        <span className="doc-title">{doc.title || doc.filename}</span>
                        {doc.title && doc.title !== doc.filename ? (
                          <span className="doc-subtitle text-muted">{doc.filename}</span>
                        ) : null}
                      </div>
                    </td>
                    <td>{doc.page_count !== null && doc.page_count !== undefined ? doc.page_count : '-'}</td>
                    <td>{formatBytes(doc.file_size_bytes)}</td>
                    <td>
                      <span className={`status-pill status-${doc.processing_status.toLowerCase()}`}>
                        {doc.processing_status}
                      </span>
                    </td>
                    <td className="text-muted">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-secondary btn-xs btn-remove-doc"
                        onClick={() => setDocumentToRemove(doc)}
                        aria-label={`Remove ${doc.title || doc.filename} from collection`}
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Document Removal Confirmation Modal */}
        {documentToRemove ? (
          <div className="modal-backdrop" onClick={() => !isRemoving && setDocumentToRemove(null)}>
            <div
              className="modal-card"
              role="dialog"
              aria-modal="true"
              aria-labelledby="remove-doc-modal-title"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="modal-header">
                <h3 id="remove-doc-modal-title">Remove Document from Collection</h3>
                <button
                  type="button"
                  className="modal-close-btn"
                  onClick={() => !isRemoving && setDocumentToRemove(null)}
                  aria-label="Close dialog"
                  disabled={isRemoving}
                >
                  &times;
                </button>
              </div>

              {removeError ? (
                <div className="alert alert-error" role="alert" style={{ margin: '1rem 1.5rem 0 1.5rem' }}>
                  <span>{removeError}</span>
                </div>
              ) : null}

              <div className="modal-body" style={{ padding: '1.25rem 1.5rem' }}>
                <p>
                  Are you sure you want to remove <strong>{documentToRemove.title || documentToRemove.filename}</strong> from <strong>{selectedCollection.name}</strong>?
                </p>
                <p className="text-muted text-sm" style={{ marginTop: '0.75rem', lineHeight: '1.5' }}>
                  This will remove the document association from this collection. The original research paper in your library will <strong>not</strong> be deleted.
                </p>
              </div>

              <div
                className="modal-actions"
                style={{
                  padding: '1rem 1.5rem',
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: '0.75rem',
                  borderTop: '1px solid var(--border-color)',
                }}
              >
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setDocumentToRemove(null)}
                  disabled={isRemoving}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={handleConfirmRemove}
                  disabled={isRemoving}
                >
                  {isRemoving ? 'Removing...' : 'Remove from Collection'}
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    )
  }

  // --- Render Collections Grid Interface ---
  return (
    <div className="view-container">
      <div className="view-header">
        <div>
          <h2>Research Collections</h2>
          <p className="text-muted">
            Organize related research papers and documents into isolated topic spaces. Click a collection to view and manage its member documents.
          </p>
        </div>
        <div className="view-header-actions">
          <span className="badge badge-info">{total} {total === 1 ? 'Collection' : 'Collections'}</span>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => setIsModalOpen(true)}
          >
            + New Collection
          </button>
        </div>
      </div>

      {error ? (
        <div className="alert alert-error" role="alert">
          {error}
          <button type="button" className="btn-retry" onClick={loadCollections}>
            Retry
          </button>
        </div>
      ) : null}

      {isLoading ? (
        <div className="loading-state">
          <div className="spinner" aria-label="Loading collections" />
          <p>Loading research collections...</p>
        </div>
      ) : collections.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📁</div>
          <h3>No Collections Found</h3>
          <p>
            You have not created any research collections yet. Collections help you segment
            your library by project, topic, or domain.
          </p>
          <button
            type="button"
            className="btn btn-primary"
            style={{ marginTop: '1rem' }}
            onClick={() => setIsModalOpen(true)}
          >
            + Create First Collection
          </button>
        </div>
      ) : (
        <div className="grid-list">
          {collections.map((col) => (
            <div
              key={col.id}
              className="card collection-card"
              onClick={() => handleSelectCollection(col)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  handleSelectCollection(col)
                }
              }}
              style={{ cursor: 'pointer' }}
              aria-label={`View collection ${col.name}`}
            >
              <div className="card-header">
                <h3 className="collection-name">{col.name}</h3>
                <span className="badge badge-secondary">
                  {col.document_count} {col.document_count === 1 ? 'doc' : 'docs'}
                </span>
              </div>
              {col.description ? (
                <p className="collection-desc">{col.description}</p>
              ) : (
                <p className="collection-desc text-muted italic">No description provided</p>
              )}
              <div className="card-footer" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted">
                  Created: {new Date(col.created_at).toLocaleDateString()}
                </span>
                <span className="text-primary text-sm font-medium">
                  View Collection →
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      <CreateCollectionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSuccess={handleCollectionCreated}
      />
    </div>
  )
}
