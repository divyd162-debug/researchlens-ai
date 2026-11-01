import React from 'react'
import { useAuth } from '../context/AuthContext'
import { ActiveView } from '../types'

interface NavbarProps {
  activeView: ActiveView
  onNavigate: (view: ActiveView) => void
}

export const Navbar: React.FC<NavbarProps> = ({ activeView, onNavigate }) => {
  const { user, isAuthenticated, logout } = useAuth()

  return (
    <header className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand" onClick={() => onNavigate(isAuthenticated ? 'dashboard' : 'login')}>
          <span className="brand-logo" aria-hidden="true">🔍</span>
          <div className="brand-text">
            <span className="brand-title">ResearchLens AI</span>
            <span className="brand-badge">v0.1.0</span>
          </div>
        </div>

        <nav className="navbar-nav" aria-label="Main Navigation">
          {isAuthenticated ? (
            <>
              <button
                type="button"
                className={`nav-link ${activeView === 'dashboard' ? 'active' : ''}`}
                onClick={() => onNavigate('dashboard')}
              >
                Dashboard
              </button>
              <button
                type="button"
                className={`nav-link ${activeView === 'collections' ? 'active' : ''}`}
                onClick={() => onNavigate('collections')}
              >
                Collections
              </button>
              <button
                type="button"
                className={`nav-link ${activeView === 'documents' ? 'active' : ''}`}
                onClick={() => onNavigate('documents')}
              >
                Documents
              </button>
              <button
                type="button"
                className={`nav-link ${activeView === 'search' ? 'active' : ''}`}
                onClick={() => onNavigate('search')}
              >
                Search
              </button>
              <button
                type="button"
                className={`nav-link ${activeView === 'chat' ? 'active' : ''}`}
                onClick={() => onNavigate('chat')}
              >
                Chat
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className={`nav-link ${activeView === 'login' ? 'active' : ''}`}
                onClick={() => onNavigate('login')}
              >
                Sign In
              </button>
              <button
                type="button"
                className={`nav-link ${activeView === 'register' ? 'active' : ''}`}
                onClick={() => onNavigate('register')}
              >
                Register
              </button>
            </>
          )}
        </nav>

        <div className="navbar-user">
          {isAuthenticated && user ? (
            <div className="user-profile">
              <span className="user-email" title={user.email}>
                {user.email}
              </span>
              <button
                type="button"
                className="btn-logout"
                onClick={() => {
                  logout()
                  onNavigate('login')
                }}
              >
                Log Out
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </header>
  )
}
