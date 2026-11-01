import React, { useEffect, useState } from 'react'
import { AuthProvider, useAuth } from './context/AuthContext'
import { Navbar } from './components/Navbar'
import { LoginView } from './components/LoginView'
import { RegisterView } from './components/RegisterView'
import { DashboardView } from './components/DashboardView'
import { CollectionsView } from './components/CollectionsView'
import { DocumentsView } from './components/DocumentsView'
import { SearchView } from './components/SearchView'
import { ChatWorkspaceView } from './components/ChatWorkspaceView'
import { ProtectedRoute } from './components/ProtectedRoute'
import { ActiveView } from './types'
import './App.css'

const getViewFromHash = (): ActiveView => {
  const raw = window.location.hash.replace('#', '').toLowerCase()
  const viewName = raw.split('?')[0].split('/')[0]
  if (['dashboard', 'collections', 'documents', 'search', 'chat', 'login', 'register'].includes(viewName)) {
    return viewName as ActiveView
  }
  return 'dashboard'
}

const AppContent: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth()
  const [activeView, setActiveView] = useState<ActiveView>(() => getViewFromHash())

  useEffect(() => {
    const handleHashChange = () => {
      setActiveView(getViewFromHash())
    }
    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  const handleNavigate = (view: ActiveView) => {
    setActiveView(view)
    window.location.hash = `#${view}`
  }

  // Redirect unauthenticated user from protected view to login once initial auth check finishes
  useEffect(() => {
    if (!isLoading && !isAuthenticated && ['dashboard', 'collections', 'documents', 'search', 'chat'].includes(activeView)) {
      handleNavigate('login')
    } else if (!isLoading && isAuthenticated && ['login', 'register'].includes(activeView)) {
      handleNavigate('dashboard')
    }
  }, [isLoading, isAuthenticated, activeView])

  return (
    <div className="app-layout">
      <Navbar activeView={activeView} onNavigate={handleNavigate} />

      <main className="app-main">
        {activeView === 'login' && (
          <div className="auth-view-wrapper">
            <LoginView
              onNavigate={handleNavigate}
              onSuccess={() => handleNavigate('dashboard')}
            />
          </div>
        )}

        {activeView === 'register' && (
          <div className="auth-view-wrapper">
            <RegisterView
              onNavigate={handleNavigate}
              onSuccess={() => handleNavigate('dashboard')}
            />
          </div>
        )}

        {activeView === 'dashboard' && (
          <ProtectedRoute onNavigate={handleNavigate}>
            <DashboardView onNavigate={handleNavigate} />
          </ProtectedRoute>
        )}

        {activeView === 'collections' && (
          <ProtectedRoute onNavigate={handleNavigate}>
            <CollectionsView />
          </ProtectedRoute>
        )}

        {activeView === 'documents' && (
          <ProtectedRoute onNavigate={handleNavigate}>
            <DocumentsView />
          </ProtectedRoute>
        )}

        {activeView === 'search' && (
          <ProtectedRoute onNavigate={handleNavigate}>
            <SearchView />
          </ProtectedRoute>
        )}

        {activeView === 'chat' && (
          <ProtectedRoute onNavigate={handleNavigate}>
            <ChatWorkspaceView />
          </ProtectedRoute>
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>ResearchLens AI &bull; Grounded research assistant with verifiable passage citations</p>
          <span className="footer-status">Phase 5: Document &amp; Collection Management</span>
        </div>
      </footer>
    </div>
  )
}

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App
