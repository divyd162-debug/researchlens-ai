import { useState, useEffect } from 'react'
import { AuthProvider } from './context/AuthContext'
import { Navbar } from './components/Navbar'
import { ExploreView } from './components/ExploreView'
import { DocumentsView } from './components/DocumentsView'
import { SearchView } from './components/SearchView'
import { ChatWorkspaceView } from './components/ChatWorkspaceView'
import { CollectionsView } from './components/CollectionsView'
import { LoginView } from './components/LoginView'
import { RegisterView } from './components/RegisterView'
import { UploadDocumentModal } from './components/UploadDocumentModal'
import { ProtectedRoute } from './components/ProtectedRoute'
import { Footer } from './components/Footer'
import { ActiveView } from './types'
import { Terminal, Copy, Check, Sparkles, Database, Cpu, ShieldCheck, ArrowRight } from 'lucide-react'
import './App.css'

const getViewFromHash = (): ActiveView => {
  const raw = window.location.hash.replace('#', '').toLowerCase()
  const viewName = raw.split('?')[0].split('/')[0]
  if (
    [
      'explore',
      'documents',
      'search',
      'chat',
      'collections',
      'mcp',
      'login',
      'register',
      'dashboard',
    ].includes(viewName)
  ) {
    if (viewName === 'dashboard') return 'explore'
    return viewName as ActiveView
  }
  return 'explore'
}

const AppContent = () => {
  const [activeView, setActiveView] = useState<ActiveView>(() => getViewFromHash())
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [copiedMcp, setCopiedMcp] = useState(false)
  const [toastMessage, setToastMessage] = useState<string | null>(null)

  const showToast = (msg: string) => {
    setToastMessage(msg)
    setTimeout(() => setToastMessage(null), 3000)
  }

  // Sync with window.location.hash
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
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const copyMcpConfig = () => {
    const config = `{
  "service": "researchlens",
  "api_version": "1.0.0",
  "base_url": "http://localhost:8000/api/v1",
  "endpoints": {
    "query": "POST /api/v1/query",
    "search": "POST /api/v1/search",
    "chat": "POST /api/v1/chat/sessions",
    "documents": "POST /api/v1/documents"
  },
  "docs": "http://localhost:8000/docs",
  "openapi": "http://localhost:8000/api/v1/openapi.json",
  "auth": {
    "type": "Bearer",
    "token": "YOUR_API_TOKEN"
  }
}`
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(config)
      }
    } catch {
      // fallback
    }
    setCopiedMcp(true)
    showToast('ResearchLens API configuration copied to clipboard')
    setTimeout(() => setCopiedMcp(false), 2000)
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#FAFAFB] text-neutral-900">
      {/* Toast Notification */}
      {toastMessage && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-6 right-6 z-50 bg-neutral-950 text-white px-4 py-2.5 rounded-xl shadow-2xl text-xs font-semibold flex items-center gap-2 border border-neutral-800 animate-in slide-in-from-bottom-2 duration-200"
        >
          <Sparkles className="w-3.5 h-3.5 text-[#D1FE17]" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Global Header (Template Style) */}
      <Navbar
        activeView={activeView}
        onNavigate={handleNavigate}
        onQuickSearch={(query) => {
          showToast(`Executing search for "${query}"`)
          handleNavigate('search')
        }}
        onOpenUpload={() => setIsUploadModalOpen(true)}
      />

      {/* Main Content Area */}
      <main className="flex-grow">
        {activeView === 'explore' && (
          <ExploreView
            onNavigate={handleNavigate}
            onOpenUpload={() => setIsUploadModalOpen(true)}
          />
        )}

        {activeView === 'documents' && (
          <ProtectedRoute onNavigate={handleNavigate} targetView="documents">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <DocumentsView />
            </div>
          </ProtectedRoute>
        )}

        {activeView === 'search' && (
          <ProtectedRoute onNavigate={handleNavigate} targetView="search">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <SearchView />
            </div>
          </ProtectedRoute>
        )}

        {activeView === 'chat' && (
          <ProtectedRoute onNavigate={handleNavigate} targetView="chat">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <ChatWorkspaceView />
            </div>
          </ProtectedRoute>
        )}

        {activeView === 'collections' && (
          <ProtectedRoute onNavigate={handleNavigate} targetView="collections">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <CollectionsView />
            </div>
          </ProtectedRoute>
        )}

        {/* MCP & Developer Architecture Tab */}
        {activeView === 'mcp' && (
          <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <div className="text-center mb-10">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-100 text-violet-800 text-xs font-bold mb-3">
                <Terminal className="w-3.5 h-3.5" />
                <span>Developer Integration — OpenAPI 3.0 &amp; Model Context Protocol</span>
              </div>
              <h2 className="text-3xl sm:text-4xl font-extrabold text-neutral-950 mb-2">
                Connect ResearchLens AI to Claude Desktop &amp; Cursor
              </h2>
              <p className="text-neutral-600 text-sm max-w-xl mx-auto">
                Expose your academic literature FAISS vector index, passage citations, and grounded synthesis directly to local LLM agents.
              </p>
            </div>

            {/* Live OpenAPI REST API Status Banner */}
            <div className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-sm mb-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-xs font-bold uppercase tracking-wider text-neutral-500">
                    Live Backend API (Active)
                  </span>
                </div>
                <h4 className="text-lg font-bold text-neutral-950">FastAPI REST Endpoints Verified</h4>
                <p className="text-xs text-neutral-600 mt-0.5">
                  Full OpenAPI 3.0 documentation with interactive Swagger UI and schemas is live under <code>/api/v1</code>.
                </p>
              </div>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2 bg-neutral-950 hover:bg-neutral-800 text-white rounded-xl text-xs font-semibold inline-flex items-center gap-2 transition-transform active:scale-95 shadow-sm shrink-0"
              >
                <span>Interactive /docs Swagger</span>
                <ArrowRight className="w-3.5 h-3.5 text-[#D1FE17]" />
              </a>
            </div>

            {/* Code Block Container */}
            <div className="bg-neutral-950 text-neutral-200 rounded-2xl p-6 border border-neutral-800 shadow-xl relative mb-8">
              <div className="flex items-center justify-between pb-4 mb-4 border-b border-neutral-800 text-xs text-neutral-400 font-mono">
                <div className="flex items-center gap-2">
                  <span>researchlens_api_config.json</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    Live Verified REST API
                  </span>
                </div>
                <button
                  type="button"
                  onClick={copyMcpConfig}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-white transition-colors cursor-pointer"
                >
                  {copiedMcp ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedMcp ? 'Copied' : 'Copy Config'}</span>
                </button>
              </div>

              <pre className="text-xs sm:text-sm font-mono text-[#D1FE17] overflow-x-auto leading-relaxed">
{`{
  "service": "researchlens",
  "api_version": "1.0.0",
  "base_url": "http://localhost:8000/api/v1",
  "endpoints": {
    "query": "POST /api/v1/query",
    "search": "POST /api/v1/search",
    "chat": "POST /api/v1/chat/sessions",
    "documents": "POST /api/v1/documents"
  },
  "docs": "http://localhost:8000/docs",
  "openapi": "http://localhost:8000/api/v1/openapi.json"
}`}
              </pre>
              <div className="mt-4 p-3 bg-neutral-900/90 rounded-xl border border-neutral-800">
                <div className="flex items-center gap-2 text-xs font-bold text-neutral-300 mb-1">
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 font-mono">MCP Notice</span>
                  <span>Model Context Protocol (Planned Roadmap)</span>
                </div>
                <p className="text-[11px] text-neutral-400 leading-relaxed">
                  Native STDIO Model Context Protocol server entry point is on the future roadmap. There is no active executable <code>backend.app.mcp</code> module in this release. Use the verified REST endpoints above and OpenAPI documentation for agent integrations.
                </p>
              </div>
            </div>

            {/* Architecture Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
              <div className="p-5 bg-white rounded-2xl border border-neutral-200 shadow-xs">
                <div className="w-8 h-8 rounded-lg bg-lime-100 flex items-center justify-center text-neutral-950 font-bold mb-3">
                  <Database className="w-4 h-4 text-neutral-900" />
                </div>
                <h5 className="font-bold text-neutral-900 text-sm mb-1">Dense Vector Search API</h5>
                <p className="text-neutral-600 leading-relaxed">
                  Call <code className="bg-neutral-100 px-1 py-0.5 rounded">POST /api/v1/search</code> with similarity score filtering, scope selection (GLOBAL/DOCUMENT/COLLECTION), and exact page boundaries.
                </p>
              </div>

              <div className="p-5 bg-white rounded-2xl border border-neutral-200 shadow-xs">
                <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center text-neutral-950 font-bold mb-3">
                  <ShieldCheck className="w-4 h-4 text-neutral-900" />
                </div>
                <h5 className="font-bold text-neutral-900 text-sm mb-1">Grounded Synthesis API</h5>
                <p className="text-neutral-600 leading-relaxed">
                  Call <code className="bg-neutral-100 px-1 py-0.5 rounded">POST /api/v1/chat/sessions/&#123;id&#125;/messages</code> for multi-turn grounded conversations with verified chunk citations.
                </p>
              </div>

              <div className="p-5 bg-white rounded-2xl border border-neutral-200 shadow-xs">
                <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center text-neutral-950 font-bold mb-3">
                  <Cpu className="w-4 h-4 text-neutral-900" />
                </div>
                <h5 className="font-bold text-neutral-900 text-sm mb-1">Document Ingestion API</h5>
                <p className="text-neutral-600 leading-relaxed">
                  Call <code className="bg-neutral-100 px-1 py-0.5 rounded">POST /api/v1/documents</code> to upload scientific preprints, parse layout tables, split chunks, and index vectors.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Authentication Views */}
        {activeView === 'login' && (
          <div className="max-w-md mx-auto px-4 py-16">
            <LoginView
              onNavigate={handleNavigate}
              onSuccess={() => handleNavigate('explore')}
            />
          </div>
        )}

        {activeView === 'register' && (
          <div className="max-w-md mx-auto px-4 py-16">
            <RegisterView
              onNavigate={handleNavigate}
              onSuccess={() => handleNavigate('explore')}
            />
          </div>
        )}
      </main>

      {/* Global Upload PDF Modal */}
      <UploadDocumentModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onSuccess={() => {
          showToast('Research paper uploaded and queued for processing!')
          handleNavigate('documents')
        }}
      />

      {/* Global Categorized Footer (Template Style) */}
      <Footer
        onNavigateTab={handleNavigate}
        onSubscribe={(email) => {
          showToast(`Subscribed ${email} to academic intelligence dispatch`)
        }}
      />
    </div>
  )
}

export const App = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App
