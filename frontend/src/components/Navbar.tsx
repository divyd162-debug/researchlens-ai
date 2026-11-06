import { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { ActiveView } from '../types'
import {
  Sparkles,
  FolderGit2,
  FileText,
  Search,
  MessageSquare,
  LogOut,
  ArrowUpRight,
  Menu,
  X,
  Upload,
  Terminal,
  LogIn,
} from 'lucide-react'

interface NavbarProps {
  activeView: ActiveView
  onNavigate: (view: ActiveView) => void
  onQuickSearch?: (query: string) => void
  onOpenUpload?: () => void
}

export const Navbar = ({
  activeView,
  onNavigate,
  onQuickSearch,
  onOpenUpload,
}: NavbarProps) => {
  const { user, isAuthenticated, logout } = useAuth()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [headerSearchQuery, setHeaderSearchQuery] = useState('')
  const [searchFocused, setSearchFocused] = useState(false)

  const handleHeaderSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (headerSearchQuery.trim()) {
      if (onQuickSearch) {
        onQuickSearch(headerSearchQuery)
      }
      onNavigate('search')
    }
  }

  const navItems = [
    { id: 'explore' as ActiveView, label: 'Explore', icon: Sparkles },
    { id: 'documents' as ActiveView, label: 'Documents', icon: FileText },
    { id: 'search' as ActiveView, label: 'Search', icon: Search },
    { id: 'chat' as ActiveView, label: 'Research Chat', icon: MessageSquare, badge: 'RAG' },
    { id: 'collections' as ActiveView, label: 'Collections', icon: FolderGit2 },
    { id: 'mcp' as ActiveView, label: 'API & MCP', icon: Terminal, badge: 'NEW' },
  ]

  return (
    <header className="sticky top-0 z-40 w-full bg-white/90 backdrop-blur-md border-b border-neutral-200/80 transition-all shadow-xs">
      {/* Top promotional announcement strip (Template Style) */}
      <div className="bg-neutral-950 text-white text-xs font-medium py-1.5 px-4 text-center flex items-center justify-center gap-2 border-b border-neutral-800">
        <span className="inline-flex items-center gap-1.5 bg-[#C8FA00] text-neutral-950 px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase">
          GROUNDED RAG
        </span>
        <span className="text-neutral-300 hidden sm:inline">
          Meta FAISS Index &amp; all-MiniLM-L6-v2 Embeddings for Dense Vector Retrieval &amp; Verifiable Citations.
        </span>
        <button
          type="button"
          onClick={() => onNavigate('explore')}
          className="inline-flex items-center gap-1 font-semibold text-[#C8FA00] hover:underline cursor-pointer bg-transparent border-0 p-0 text-xs"
        >
          Open Research Studio <ArrowUpRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Main navigation row */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-3">

          {/* Brand Logo */}
          <div className="flex items-center gap-3 shrink-0">
            <button
              type="button"
              onClick={() => onNavigate('explore')}
              className="flex items-center gap-2.5 group cursor-pointer text-left bg-transparent border-0 p-0"
              aria-label="Go to ResearchLens AI homepage"
            >
              <div className="w-8 h-8 rounded-lg bg-neutral-950 flex items-center justify-center text-[#C8FA00] font-black text-lg shadow-sm transition-transform group-hover:scale-105">
                <svg className="w-4.5 h-4.5 fill-current" viewBox="0 0 24 24">
                  <path d="M12 2L3 7v10l9 5 9-5V7l-9-5zm0 3.3L18 8.7v6.6L12 18.7 6 15.3V8.7L12 5.3z" />
                </svg>
              </div>
              <div className="flex flex-col">
                <span className="font-heading font-extrabold text-xl tracking-tight text-neutral-950">
                  researchlens<span className="text-neutral-400 font-light text-sm ml-0.5">ai</span>
                </span>
              </div>
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1" aria-label="Main Navigation">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = activeView === item.id
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onNavigate(item.id)}
                  className={`relative flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all cursor-pointer ${
                    isActive
                      ? 'text-neutral-950 bg-neutral-100 font-semibold shadow-xs'
                      : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-neutral-950' : 'text-neutral-400'}`} />
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-neutral-900 text-[#C8FA00]">
                      {item.badge}
                    </span>
                  )}
                </button>
              )
            })}
          </nav>

          {/* Search bar & Action triggers */}
          <div className="flex items-center gap-2.5 sm:gap-3">
            {/* Quick Header Search input */}
            <form
              onSubmit={handleHeaderSearchSubmit}
              className={`relative hidden md:flex items-center transition-all ${
                searchFocused ? 'w-64' : 'w-48'
              }`}
            >
              <Search className="w-4 h-4 text-neutral-400 absolute left-3 pointer-events-none" />
              <input
                type="text"
                placeholder="Search literature, topics..."
                value={headerSearchQuery}
                onChange={(e) => setHeaderSearchQuery(e.target.value)}
                onFocus={() => setSearchFocused(true)}
                onBlur={() => setSearchFocused(false)}
                className="w-full pl-9 pr-3 py-1.5 bg-neutral-100 border border-neutral-200 rounded-lg text-xs text-neutral-900 placeholder:text-neutral-400 focus:outline-none focus:ring-2 focus:ring-neutral-900/10 focus:border-neutral-400 focus:bg-white transition-all"
              />
            </form>

            {/* Auth / Account */}
            {isAuthenticated ? (
              <div className="hidden sm:flex items-center gap-2">
                <span className="text-xs font-semibold text-neutral-700 bg-neutral-100 border border-neutral-200 px-2.5 py-1 rounded-lg truncate max-w-[120px]">
                  {user?.email?.split('@')[0] || 'Researcher'}
                </span>
                <button
                  type="button"
                  onClick={logout}
                  className="p-1.5 rounded-lg text-neutral-500 hover:text-neutral-900 hover:bg-neutral-100 transition-colors cursor-pointer bg-transparent border-0"
                  title="Sign Out"
                  aria-label="Sign Out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => onNavigate('login')}
                className="hidden sm:inline-flex items-center gap-1.5 text-xs font-medium text-neutral-700 hover:text-neutral-950 px-2.5 py-1.5 rounded-md hover:bg-neutral-100 cursor-pointer bg-transparent border-0"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Sign In</span>
              </button>
            )}

            {/* Primary Action Button (+ Upload Paper) */}
            <button
              type="button"
              onClick={() => {
                if (onOpenUpload) onOpenUpload()
                else onNavigate('documents')
              }}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-[#D1FE17] text-neutral-950 border border-neutral-300 hover:bg-[#c2ed14] transition-all shadow-xs active:scale-95 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Upload Paper</span>
            </button>

            {/* Mobile menu toggle */}
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-1.5 text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100 rounded-lg cursor-pointer bg-transparent border-0"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t border-neutral-200 bg-white px-4 pt-3 pb-6 space-y-3 shadow-lg animate-in slide-in-from-top-2 duration-200">
          <form onSubmit={handleHeaderSearchSubmit} className="relative mb-3">
            <Search className="w-4 h-4 text-neutral-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search literature, topics..."
              value={headerSearchQuery}
              onChange={(e) => setHeaderSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 bg-neutral-100 border border-neutral-200 rounded-lg text-sm text-neutral-900"
            />
          </form>
          <div className="grid grid-cols-2 gap-2">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = activeView === item.id
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    onNavigate(item.id)
                    setMobileMenuOpen(false)
                  }}
                  className={`flex items-center gap-2 p-2.5 rounded-lg text-sm font-medium cursor-pointer ${
                    isActive ? 'bg-neutral-900 text-white' : 'bg-neutral-50 text-neutral-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </div>
          <div className="pt-2 flex items-center justify-between border-t border-neutral-100">
            {isAuthenticated ? (
              <div className="flex items-center justify-between w-full">
                <span className="text-xs text-neutral-600">
                  Signed in as <strong>{user?.email}</strong>
                </span>
                <button
                  type="button"
                  onClick={() => {
                    logout()
                    setMobileMenuOpen(false)
                  }}
                  className="text-xs font-semibold text-red-600 hover:underline bg-transparent border-0 cursor-pointer"
                >
                  Sign Out
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => {
                    onNavigate('login')
                    setMobileMenuOpen(false)
                  }}
                  className="text-xs font-semibold text-neutral-800 hover:underline bg-transparent border-0 cursor-pointer"
                >
                  Sign In
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onNavigate('register')
                    setMobileMenuOpen(false)
                  }}
                  className="text-xs font-semibold text-neutral-500 hover:underline bg-transparent border-0 cursor-pointer"
                >
                  Create Account
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </header>
  )
}
export default Navbar
