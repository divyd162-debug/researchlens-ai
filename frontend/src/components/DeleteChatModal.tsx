import React, { useState } from 'react'
import { api, ApiError } from '../services/api'
import { ChatSessionResponse } from '../types'

interface DeleteChatModalProps {
  session: ChatSessionResponse | null
  isOpen: boolean
  onClose: () => void
  onSuccess: (deletedId: string) => void
}

export const DeleteChatModal: React.FC<DeleteChatModalProps> = ({
  session,
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [error, setError] = useState<string | null>(null)
  const [isDeleting, setIsDeleting] = useState<boolean>(false)

  if (!isOpen || !session) return null

  const handleClose = () => {
    if (!isDeleting) {
      setError(null)
      onClose()
    }
  }

  const handleConfirmDelete = async () => {
    setIsDeleting(true)
    setError(null)

    try {
      await api.chat.deleteSession(session.id)
      setError(null)
      onSuccess(session.id)
      onClose()
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail)
      } else {
        setError('Failed to delete conversation. Please check your network connection.')
      }
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={handleClose}>
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-chat-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id="delete-chat-modal-title">Delete Conversation</h3>
          <button
            type="button"
            className="modal-close-btn"
            onClick={handleClose}
            aria-label="Close dialog"
            disabled={isDeleting}
          >
            &times;
          </button>
        </div>

        {error ? (
          <div className="alert alert-error" role="alert" style={{ margin: '1rem 1.5rem 0 1.5rem' }}>
            <span>{error}</span>
          </div>
        ) : null}

        <div className="modal-body" style={{ padding: '1.25rem 1.5rem' }}>
          <p>
            Are you sure you want to delete <strong>{session.title}</strong>?
          </p>
          <p className="text-muted text-sm" style={{ marginTop: '0.75rem', lineHeight: '1.5' }}>
            This will permanently remove this conversation thread and its message history.
            Your uploaded research documents, collections, and search indexes will remain intact.
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
            onClick={handleClose}
            disabled={isDeleting}
          >
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-danger"
            onClick={handleConfirmDelete}
            disabled={isDeleting}
          >
            {isDeleting ? 'Deleting...' : 'Delete Conversation'}
          </button>
        </div>
      </div>
    </div>
  )
}
