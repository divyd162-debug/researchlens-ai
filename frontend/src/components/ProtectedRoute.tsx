import React from 'react'
import { useAuth } from '../context/AuthContext'
import { ActiveView } from '../types'
import { LoginView } from './LoginView'

interface ProtectedRouteProps {
  children: React.ReactNode
  onNavigate: (view: ActiveView) => void
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children, onNavigate }) => {
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="auth-loading-screen" role="status" aria-live="polite">
        <div className="spinner" />
        <p>Verifying session...</p>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="protected-prompt-container">
        <div className="alert alert-info" role="status">
          Authentication required. Please sign in to access your research workspace.
        </div>
        <LoginView
          onNavigate={onNavigate}
          onSuccess={() => onNavigate('dashboard')}
        />
      </div>
    )
  }

  return <>{children}</>
}
