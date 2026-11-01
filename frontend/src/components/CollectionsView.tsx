import React, { useEffect, useState } from 'react'
import { api, ApiError } from '../services/api'
import { CollectionResponse } from '../types'
import { CreateCollectionModal } from './CreateCollectionModal'

export const CollectionsView: React.FC = () => {
  const [collections, setCollections] = useState<CollectionResponse[]>([])
  const [total, setTotal] = useState<number>(0)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)

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

  useEffect(() => {
    loadCollections()
  }, [])

  const handleCollectionCreated = () => {
    loadCollections()
  }

  return (
    <div className="view-container">
      <div className="view-header">
        <div>
          <h2>Research Collections</h2>
          <p className="text-muted">
            Organize related research papers and documents into isolated topic spaces.
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
            <div key={col.id} className="card collection-card">
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
              <div className="card-footer text-muted">
                Created: {new Date(col.created_at).toLocaleDateString()}
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
