import React, { useEffect, useState } from 'react'
import { api, ApiError } from '../services/api'
import { DocumentResponse, CollectionResponse } from '../types'

interface AddToCollectionModalProps {
  isOpen: boolean
  document: DocumentResponse | null
  onClose: () => void
  onSuccess: () => void
}

export const AddToCollectionModal: React.FC<AddToCollectionModalProps> = ({
  isOpen,
  document,
  onClose,
  onSuccess,
}) => {
  const [collections, setCollections] = useState<CollectionResponse[]>([])
  const [selectedCollectionId, setSelectedCollectionId] = useState<string>('')
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen) {
      setError(null)
      setSelectedCollectionId('')
      setIsLoading(true)
      api.collections
        .list()
        .then((res) => {
          setCollections(res.items || [])
          if (res.items && res.items.length > 0) {
            setSelectedCollectionId(res.items[0].id)
          }
        })
        .catch((err) => {
          setError(err instanceof ApiError ? err.message : 'Failed to load collections')
        })
        .finally(() => {
          setIsLoading(false)
        })
    }
  }, [isOpen])

  if (!isOpen || !document) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedCollectionId) return

    setIsSubmitting(true)
    setError(null)

    try {
      await api.collections.addDocuments(selectedCollectionId, {
        document_ids: [document.id],
      })
      onSuccess()
      onClose()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to add document to collection')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="add-to-col-title">
      <div className="modal-card">
        <div className="modal-header">
          <h3 id="add-to-col-title">Add to Collection</h3>
          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            aria-label="Close modal"
          >
            &times;
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            <p>
              Add <strong>{document.title || document.filename}</strong> to a research collection:
            </p>

            {error && (
              <div className="alert alert-error" role="alert">
                {error}
              </div>
            )}

            {isLoading ? (
              <div className="loading-state">Loading collections...</div>
            ) : collections.length === 0 ? (
              <p className="empty-text">You have not created any collections yet.</p>
            ) : (
              <div className="form-group">
                <label htmlFor="collection-select">Select Collection</label>
                <select
                  id="collection-select"
                  value={selectedCollectionId}
                  onChange={(e) => setSelectedCollectionId(e.target.value)}
                  className="form-control"
                  disabled={isSubmitting}
                >
                  {collections.map((col) => (
                    <option key={col.id} value={col.id}>
                      {col.name} ({col.document_count} docs)
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="modal-footer">
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
              disabled={isSubmitting || collections.length === 0 || isLoading}
            >
              {isSubmitting ? 'Adding...' : 'Add to Collection'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
