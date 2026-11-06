import React from 'react'
import { useAuth } from '../context/AuthContext'
import { ActiveView } from '../types'
import { LoginView } from './LoginView'

interface ProtectedRouteProps {
  children: React.ReactNode
  onNavigate: (view: ActiveView) => void
  targetView?: ActiveView
}

const VIEW_TITLES: Partial<Record<ActiveView, string>> = {
  documents: 'Research Documents',
  search: 'Semantic Passage Search',
  chat: 'Conversations',
  collections: 'Research Collections',
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  onNavigate,
  targetView,
}) => {
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-16 space-y-3" role="status" aria-live="polite">
        <div className="w-8 h-8 border-2 border-neutral-300 border-t-neutral-950 rounded-full animate-spin" />
        <p className="text-xs text-neutral-500 font-medium">Verifying session...</p>
      </div>
    )
  }

  if (!isAuthenticated) {
    const title = targetView ? VIEW_TITLES[targetView] : undefined
    return (
      <div className="max-w-2xl mx-auto px-4 py-12">
        {title && (
          <div className="mb-6 pb-4 border-b border-neutral-200">
            <h2 className="text-2xl font-bold text-neutral-950">{title}</h2>
            <p className="text-xs text-neutral-500 mt-1">Authentication required to access private tenant data</p>
          </div>
        )}
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-900 mb-6" role="status">
          <strong>Authentication Required:</strong> Please sign in to access your private research workspace, documents, and collections.
        </div>
        <div className="max-w-md mx-auto">
          <LoginView
            onNavigate={onNavigate}
            onSuccess={() => onNavigate(targetView || 'explore')}
          />
        </div>
      </div>
    )
  }

  return <>{children}</>
}
