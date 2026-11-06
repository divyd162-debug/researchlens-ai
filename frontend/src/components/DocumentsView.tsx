import React, { useEffect, useState } from 'react'
import { api, ApiError } from '../services/api'
import { DocumentResponse } from '../types'
import { UploadDocumentModal } from './UploadDocumentModal'
import { DeleteDocumentModal } from './DeleteDocumentModal'
import { AddToCollectionModal } from './AddToCollectionModal'

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export const DocumentsView: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentResponse[]>([])
  const [total, setTotal] = useState<number>(0)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)
  const [documentToDelete, setDocumentToDelete] = useState<DocumentResponse | null>(null)
  const [documentForCollection, setDocumentForCollection] = useState<DocumentResponse | null>(null)
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null)

  const loadDocuments = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await api.documents.list(50, 0)
      setDocuments(data.items)
      setTotal(data.total)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail)
      } else {
        setError('Failed to load documents from server.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  const handleDocumentUploaded = () => {
    loadDocuments()
  }

  const handleDocumentDeleted = (deletedId: string) => {
    setDocuments((prev) => prev.filter((d) => d.id !== deletedId))
    setTotal((prev) => Math.max(0, prev - 1))
  }

  const handleRetryDocument = async (docId: string) => {
    try {
      await api.documents.retry(docId)
      setFeedbackMessage('Document ingestion queued for retry.')
      setTimeout(() => setFeedbackMessage(null), 3000)
      loadDocuments()
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail)
      } else {
        setError('Failed to initiate ingestion retry.')
      }
    }
  }

  return (
    <div className="view-container">
      <div className="view-header">
        <div>
          <h2>Research Documents</h2>
          <p className="text-muted">
            Uploaded PDF documents indexed for analysis and verifiable passage retrieval.
          </p>
        </div>
        <div className="view-header-actions">
          <span className="badge badge-info">{total} {total === 1 ? 'Document' : 'Documents'}</span>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => setIsModalOpen(true)}
          >
            + Upload Document
          </button>
        </div>
      </div>

      <div className="alert alert-info" role="status">
        <div>
          <strong>Automated Pipeline:</strong> Uploaded papers are processed asynchronously through PyMuPDF text extraction, 200-token chunking, all-MiniLM-L6-v2 embeddings, and Meta FAISS vector indexing.
        </div>
      </div>

      {error ? (
        <div className="alert alert-error" role="alert">
          {error}
          <button type="button" className="btn-retry" onClick={loadDocuments}>
            Retry
          </button>
        </div>
      ) : null}

      {feedbackMessage ? (
        <div className="alert alert-success" role="status">
          {feedbackMessage}
        </div>
      ) : null}

      {isLoading ? (
        <div className="loading-state">
          <div className="spinner" aria-label="Loading documents" />
          <p>Loading research documents...</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📄</div>
          <h3>No Documents Uploaded</h3>
          <p>
            You have not uploaded any PDF documents yet. Upload a research paper to begin
            building your grounded citation library.
          </p>
          <button
            type="button"
            className="btn btn-primary"
            style={{ marginTop: '1rem' }}
            onClick={() => setIsModalOpen(true)}
          >
            + Upload First Document
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
                <th>Uploaded</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
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
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      {doc.processing_status.toUpperCase() === 'FAILED' && (
                        <button
                          type="button"
                          className="btn btn-secondary btn-xs"
                          onClick={() => handleRetryDocument(doc.id)}
                          aria-label={`Retry ingestion for ${doc.title || doc.filename}`}
                        >
                          Retry
                        </button>
                      )}
                      <button
                        type="button"
                        className="btn btn-secondary btn-xs btn-add-to-col"
                        onClick={() => setDocumentForCollection(doc)}
                        aria-label={`Add ${doc.title || doc.filename} to collection`}
                      >
                        + Add to Collection
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary btn-xs btn-delete-doc"
                        onClick={() => setDocumentToDelete(doc)}
                        aria-label={`Delete ${doc.title || doc.filename}`}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <UploadDocumentModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSuccess={handleDocumentUploaded}
      />

      <DeleteDocumentModal
        isOpen={Boolean(documentToDelete)}
        document={documentToDelete}
        onClose={() => setDocumentToDelete(null)}
        onSuccess={handleDocumentDeleted}
      />

      <AddToCollectionModal
        isOpen={Boolean(documentForCollection)}
        document={documentForCollection}
        onClose={() => setDocumentForCollection(null)}
        onSuccess={() => {
          setFeedbackMessage('Document added to collection successfully!')
          setTimeout(() => setFeedbackMessage(null), 4000)
        }}
      />
    </div>
  )
}
