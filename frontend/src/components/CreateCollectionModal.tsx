import React, { useState } from 'react'
import { api, ApiError } from '../services/api'
import { CollectionResponse } from '../types'

interface CreateCollectionModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: (newCollection: CollectionResponse) => void
}

export const CreateCollectionModal: React.FC<CreateCollectionModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  if (!isOpen) return null

  const handleClose = () => {
    if (!isSubmitting) {
      setError(null)
      setName('')
      setDescription('')
      onClose()
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    const cleanName = name.trim()
    if (!cleanName) {
      setError('Collection name is required.')
      return
    }

    if (cleanName.length > 255) {
      setError('Collection name cannot exceed 255 characters.')
      return
    }

    const cleanDescription = description.trim()
    if (cleanDescription.length > 5000) {
      setError('Description cannot exceed 5000 characters.')
      return
    }

    setIsSubmitting(true)
    try {
      const created = await api.collections.create({
        name: cleanName,
        description: cleanDescription || undefined,
      })
      setName('')
      setDescription('')
      setError(null)
      onSuccess(created)
      onClose()
    } catch (err) {
      // Preserve form and display inline error (e.g. 409 conflict or 400 validation)
      if (err instanceof ApiError) {
        setError(err.detail)
      } else {
        setError('Failed to create collection. Please check your network connection.')
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={handleClose}>
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="create-collection-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id="create-collection-title">Create New Collection</h3>
          <button
            type="button"
            className="modal-close-btn"
            onClick={handleClose}
            aria-label="Close dialog"
            disabled={isSubmitting}
          >
            &times;
          </button>
        </div>

        {error ? (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        ) : null}

        <form onSubmit={handleSubmit} className="auth-form" aria-label="Create Collection Form" noValidate>
          <div className="form-group">
            <label htmlFor="collection-name-input">
              Collection Name <span className="required-indicator">*</span>
            </label>
            <input
              id="collection-name-input"
              type="text"
              required
              maxLength={255}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Quantum Computing Papers"
              disabled={isSubmitting}
              className="form-input"
              autoFocus
            />
            <span className="form-hint">Unique title identifying this research topic (1-255 characters).</span>
          </div>

          <div className="form-group">
            <label htmlFor="collection-desc-input">Description (Optional)</label>
            <textarea
              id="collection-desc-input"
              rows={3}
              maxLength={5000}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g., Focused library of preprints and journal articles on qubit coherence."
              disabled={isSubmitting}
              className="form-textarea"
            />
          </div>

          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Creating...' : 'Create Collection'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
