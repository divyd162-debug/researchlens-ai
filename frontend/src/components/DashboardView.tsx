import React, { useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { api } from '../services/api'
import { ActiveView } from '../types'

interface DashboardViewProps {
  onNavigate: (view: ActiveView) => void
}

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate }) => {
  const { user } = useAuth()
  const [collectionCount, setCollectionCount] = useState<number | null>(null)
  const [documentCount, setDocumentCount] = useState<number | null>(null)
  const [isLoadingCounts, setIsLoadingCounts] = useState(true)
  const [fetchError, setFetchError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    const loadCounts = async () => {
      try {
        const [collectionsData, documentsData] = await Promise.all([
          api.collections.list(1, 0),
          api.documents.list(1, 0),
        ])
        if (isMounted) {
          setCollectionCount(collectionsData.total)
          setDocumentCount(documentsData.total)
        }
      } catch {
        if (isMounted) {
          setFetchError('Could not load overview statistics from server.')
        }
      } finally {
        if (isMounted) {
          setIsLoadingCounts(false)
        }
      }
    }

    loadCounts()

    return () => {
      isMounted = false
    }
  }, [])

  return (
    <div className="dashboard-container">
      <div className="dashboard-welcome">
        <h2>Welcome back{user?.email ? `, ${user.email}` : ''}</h2>
        <p className="text-muted">
          Your workspace for organizing scientific papers, research documents, and grounded citations.
        </p>
      </div>

      {fetchError ? (
        <div className="alert alert-warning" role="alert">
          {fetchError}
        </div>
      ) : null}

      <div className="metrics-grid">
        <div className="metric-card" onClick={() => onNavigate('collections')} role="button" tabIndex={0}>
          <div className="metric-icon">📁</div>
          <div className="metric-info">
            <span className="metric-label">Collections</span>
            <span className="metric-value">
              {isLoadingCounts ? '...' : collectionCount ?? 0}
            </span>
          </div>
          <span className="metric-action">Manage &rarr;</span>
        </div>

        <div className="metric-card" onClick={() => onNavigate('documents')} role="button" tabIndex={0}>
          <div className="metric-icon">📄</div>
          <div className="metric-info">
            <span className="metric-label">Uploaded Documents</span>
            <span className="metric-value">
              {isLoadingCounts ? '...' : documentCount ?? 0}
            </span>
          </div>
          <span className="metric-action">View &rarr;</span>
        </div>

        <div className="metric-card" onClick={() => onNavigate('search')} role="button" tabIndex={0}>
          <div className="metric-icon">🔍</div>
          <div className="metric-info">
            <span className="metric-label">Semantic Search</span>
            <span className="metric-value">Dense AI</span>
          </div>
          <span className="metric-action">Search &rarr;</span>
        </div>

        <div className="metric-card" onClick={() => onNavigate('chat')} role="button" tabIndex={0}>
          <div className="metric-icon">💬</div>
          <div className="metric-info">
            <span className="metric-label">Research Chat</span>
            <span className="metric-value status-badge-active">Online</span>
          </div>
          <span className="metric-action">Chat &rarr;</span>
        </div>
      </div>

      <div className="system-overview-card">
        <h3>ResearchLens AI Platform Status</h3>
        <p>
          The core document management infrastructure is operational. PDFs are strictly validated, metered,
          and stored with SHA-256 deduplication. Collections group documents for targeted research scopes.
        </p>
        <div className="status-timeline">
          <div className="timeline-item completed">
            <span className="timeline-bullet">✓</span>
            <div>
              <strong>Phase 5 Milestone 1: Collection Management</strong>
              <p>CRUD operations, unique per-owner naming, and deterministic pagination.</p>
            </div>
          </div>
          <div className="timeline-item completed">
            <span className="timeline-bullet">✓</span>
            <div>
              <strong>Phase 5 Milestone 2: Document Management</strong>
              <p>Streaming PDF size limiting, DoS protection, and cross-user isolation.</p>
            </div>
          </div>
          <div className="timeline-item completed">
            <span className="timeline-bullet">✓</span>
            <div>
              <strong>Phase 5 Milestone 3: Grounded Conversations</strong>
              <p>Persistent chat workspace, session management, and chronological history.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
