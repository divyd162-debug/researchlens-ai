import React, { useRef, useState } from 'react'
import { api, ApiError } from '../services/api'
import { DocumentResponse } from '../types'

interface UploadDocumentModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: (newDocument: DocumentResponse) => void
}

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024 // 25 MB

export const UploadDocumentModal: React.FC<UploadDocumentModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  if (!isOpen) return null

  const handleClose = () => {
    if (!isSubmitting) {
      setError(null)
      setSelectedFile(null)
      setTitle('')
      onClose()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    const file = e.target.files?.[0]
    if (!file) {
      setSelectedFile(null)
      return
    }

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setError('Selected file must be a PDF document (.pdf).')
      setSelectedFile(null)
      return
    }

    if (file.size === 0) {
      setError('Selected file is empty (0 bytes).')
      setSelectedFile(null)
      return
    }

    if (file.size > MAX_UPLOAD_BYTES) {
      setError('File size exceeds the 25 MB upload limit.')
      setSelectedFile(null)
      return
    }

    setSelectedFile(file)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!selectedFile) {
      setError('Please select a PDF file to upload.')
      return
    }

    if (selectedFile.size > MAX_UPLOAD_BYTES) {
      setError('File size exceeds the 25 MB upload limit.')
      return
    }

    setIsSubmitting(true)
    try {
      const created = await api.documents.upload(selectedFile, title.trim() || undefined)
      setSelectedFile(null)
      setTitle('')
      setError(null)
      onSuccess(created)
      onClose()
    } catch (err) {
      // Retain form and surface exact API detail (e.g. 400 invalid PDF, 409 duplicate, 413 oversized)
      if (err instanceof ApiError) {
        setError(err.detail)
      } else {
        setError('Failed to upload document. Please check your network connection.')
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
        aria-labelledby="upload-document-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id="upload-document-title">Upload Research PDF</h3>
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

        <div className="alert alert-info">
          <div>
            <strong>Status Notice:</strong> Uploaded documents will be stored with <code>PENDING</code> status.
            Text extraction and semantic indexing will be performed in the upcoming ingestion pipeline milestone.
          </div>
        </div>

        {error ? (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        ) : null}

        <form onSubmit={handleSubmit} className="auth-form" aria-label="Upload Document Form" noValidate>
          <div className="form-group">
            <label htmlFor="document-file-input">
              Select PDF Document <span className="required-indicator">*</span>
            </label>
            <input
              ref={fileInputRef}
              id="document-file-input"
              type="file"
              accept=".pdf,application/pdf"
              required
              onChange={handleFileChange}
              disabled={isSubmitting}
              className="form-input-file"
            />
            {selectedFile ? (
              <span className="selected-file-badge">
                Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
              </span>
            ) : (
              <span className="form-hint">Accepted format: PDF up to 25 MB.</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="document-title-input">Document Title (Optional)</label>
            <input
              id="document-title-input"
              type="text"
              maxLength={255}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Attention Is All You Need (NeurIPS 2017)"
              disabled={isSubmitting}
              className="form-input"
            />
            <span className="form-hint">Defaults to original filename if left blank.</span>
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
              disabled={isSubmitting || !selectedFile}
            >
              {isSubmitting ? 'Uploading...' : 'Upload PDF'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
