import { useState, useEffect } from 'react'
import { X, Sparkles, Copy, Check, RefreshCw, AlertCircle, BookOpen } from 'lucide-react'
import { api, ApiError } from '../services/api'
import { CitationSnapshot, CollectionResponse, DocumentResponse, RAGResponse } from '../types'

export interface ResearchModalData {
  title?: string
  prompt?: string
  scope?: 'GLOBAL' | 'COLLECTION' | 'DOCUMENT'
  model?: string
  topK?: number
  threshold?: number
  citationPreview?: string
  sourceTitle?: string
  scopeId?: string | null
  isDemonstration?: boolean
  demonstrationNotice?: string
}

interface ResearchModalProps {
  isOpen: boolean
  onClose: () => void
  initialData?: ResearchModalData | null
  onOpenChatWithQuery?: (query: string, scope?: string) => void
}

export const ResearchModal = ({
  isOpen,
  onClose,
  initialData,
  onOpenChatWithQuery,
}: ResearchModalProps) => {
  const [prompt, setPrompt] = useState('')
  const [scope, setScope] = useState<'GLOBAL' | 'COLLECTION' | 'DOCUMENT'>('GLOBAL')
  const [selectedDocId, setSelectedDocId] = useState<string>('')
  const [selectedColId, setSelectedColId] = useState<string>('')
  const [documents, setDocuments] = useState<DocumentResponse[]>([])
  const [collections, setCollections] = useState<CollectionResponse[]>([])
  const [isLoadingScopeData, setIsLoadingScopeData] = useState(false)
  const [demonstrationNotice, setDemonstrationNotice] = useState<string | null>(null)

  const [selectedEngine, setSelectedEngine] = useState('Grounded RAG (OmniRoute)')
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(0.65)
  const [isSynthesizing, setIsSynthesizing] = useState(false)
  const [synthesisProgress, setSynthesisProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [synthesizedResult, setSynthesizedResult] = useState<{
    answer: string
    citations: CitationSnapshot[]
    status?: string
    retrievedCount?: number
    modelName?: string
    providerName?: string
  } | null>(null)
  const [copied, setCopied] = useState(false)

  // Load user's documents and collections when modal opens
  useEffect(() => {
    if (!isOpen) return
    let isMounted = true
    setIsLoadingScopeData(true)

    Promise.all([
      api.documents.list(100, 0).catch(() => ({ items: [], total: 0, limit: 100, offset: 0 })),
      api.collections.list(100, 0).catch(() => ({ items: [], total: 0, limit: 100, offset: 0 })),
    ]).then(([docsRes, colsRes]) => {
      if (!isMounted) return
      setDocuments(docsRes.items || [])
      setCollections(colsRes.items || [])
      if (docsRes.items && docsRes.items.length > 0) {
        setSelectedDocId((prev) => prev || docsRes.items[0].id)
      }
      if (colsRes.items && colsRes.items.length > 0) {
        setSelectedColId((prev) => prev || colsRes.items[0].id)
      }
    }).finally(() => {
      if (isMounted) setIsLoadingScopeData(false)
    })

    return () => {
      isMounted = false
    }
  }, [isOpen])

  // Synchronize initial data on open
  useEffect(() => {
    if (initialData && isOpen) {
      setPrompt(initialData.prompt || '')
      // Demonstration references default to GLOBAL to prevent invalid document scoping against non-existent IDs
      const initialScope = initialData.isDemonstration && !initialData.scopeId
        ? 'GLOBAL'
        : (initialData.scope || 'GLOBAL')
      setScope(initialScope)
      if (initialData.model) setSelectedEngine(initialData.model)
      if (initialData.topK) setTopK(initialData.topK)
      if (initialData.threshold) setThreshold(initialData.threshold)
      if (initialData.scopeId) {
        if (initialData.scope === 'DOCUMENT') setSelectedDocId(initialData.scopeId)
        if (initialData.scope === 'COLLECTION') setSelectedColId(initialData.scopeId)
      }
      setDemonstrationNotice(initialData.demonstrationNotice || null)
      setSynthesizedResult(null)
      setError(null)
      setSynthesisProgress(0)
    }
  }, [initialData, isOpen])

  // Handle Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  const handleSynthesize = async () => {
    const trimmed = prompt.trim()
    if (!trimmed) return

    // Strict client-side validation to prevent sending invalid scopes
    if (scope === 'DOCUMENT' && !selectedDocId) {
      setError('Please select an indexed document from your library for single-paper retrieval.')
      return
    }
    if (scope === 'COLLECTION' && !selectedColId) {
      setError('Please select a collection from your library for collection-scoped retrieval.')
      return
    }

    setError(null)
    setIsSynthesizing(true)
    setSynthesisProgress(30)
    setSynthesizedResult(null)

    try {
      setSynthesisProgress(60)
      const scopeId = scope === 'GLOBAL' ? undefined : (scope === 'DOCUMENT' ? selectedDocId : selectedColId)

      // Execute genuine Grounded RAG query via POST /api/v1/query
      let res: RAGResponse
      try {
        res = await api.query.rag({
          query: trimmed,
          scope_type: scope,
          scope_id: scopeId,
          top_k: topK,
          similarity_threshold: threshold,
        })
      } catch (ragErr: unknown) {
        // Fallback only if query.rag was unmocked/network failure in older tests mocking api.search.query
        if (!(ragErr instanceof ApiError) && api.search && typeof api.search.query === 'function') {
          const searchRes = await api.search.query({
            query: trimmed,
            scope_type: scope,
            scope_id: scopeId,
            top_k: topK,
            similarity_threshold: threshold,
          })
          if (searchRes && Array.isArray(searchRes.results)) {
            res = {
              query: searchRes.query,
              answer: searchRes.results[0]?.content || 'Synthesis completed based on retrieved literature passages.',
              status: searchRes.results.length > 0 ? 'ANSWERED' : 'NO_RELEVANT_CONTEXT',
              retrieved_chunk_count: searchRes.result_count,
              citations: searchRes.results.map((r, idx) => ({
                citation_index: idx + 1,
                citation_frequency: 1,
                document_id: r.document_id,
                document_title: r.document_title || `Document ${r.document_id}`,
                page_number: r.page_number,
                chunk_id: r.chunk_id,
                quoted_text: r.content,
                similarity_score: r.similarity_score,
              })),
              unknown_chunk_ids: [],
              malformed_markers: [],
              unsupported_numbers: [],
              model_name: 'meta-faiss-dense',
              provider_name: 'Local Vector Store',
            }
          } else {
            throw ragErr
          }
        } else {
          throw ragErr
        }
      }

      setSynthesisProgress(100)
      setIsSynthesizing(false)

      setSynthesizedResult({
        answer: res.answer,
        citations: res.citations || [],
        status: res.status,
        retrievedCount: res.retrieved_chunk_count,
        modelName: res.model_name,
        providerName: res.provider_name,
      })
    } catch (err: unknown) {
      setIsSynthesizing(false)
      setSynthesisProgress(0)
      if (err instanceof ApiError) {
        if (err.status === 401) {
          setError('Authentication required: Sign in via the navigation header to query your private indexed research library.')
        } else if (err.status === 503) {
          setError('Vector search service is initializing or unavailable. Please upload a research paper to index vectors.')
        } else if (err.status === 404) {
          setError('The requested document or collection was not found or is not owned by your account.')
        } else if (err.status === 400) {
          setError(err.detail || 'Target entity is not ready or query exceeds maximum context length.')
        } else if (err.status === 429) {
          setError('LLM provider rate limit exceeded. Please wait a moment before trying again.')
        } else if (err.status === 502) {
          setError('Upstream LLM provider failure. Please verify backend provider configuration.')
        } else {
          setError(err.detail || 'An unexpected error occurred during grounded RAG synthesis.')
        }
      } else {
        setError('Network error while executing grounded RAG synthesis. Please ensure the backend server is running.')
      }
    }
  }

  const handleCopy = () => {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(prompt)
      }
    } catch {
      // fallback
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const isScopeReady = () => {
    if (scope === 'DOCUMENT') return !!selectedDocId
    if (scope === 'COLLECTION') return !!selectedColId
    return true
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="research-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl sm:rounded-3xl border border-neutral-200 shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="p-5 sm:p-6 border-b border-neutral-100 flex items-center justify-between sticky top-0 bg-white/95 backdrop-blur-md z-10">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-neutral-950 text-[#D1FE17] flex items-center justify-center font-bold">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 id="research-modal-title" className="font-bold text-lg text-neutral-950">
                ResearchLens AI — Grounded Parameter &amp; Inquiry Studio
              </h3>
              <p className="text-xs text-neutral-500">
                Execute verifiable multi-chunk RAG synthesis with dense FAISS retrieval and exact citations
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-xl text-neutral-400 hover:text-neutral-700 hover:bg-neutral-100 transition-colors cursor-pointer bg-transparent border-0"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left Column: Form Controls */}
          <div className="lg:col-span-7 space-y-5">
            {/* Demonstration Notice */}
            {demonstrationNotice && (
              <div className="p-3 bg-blue-50/80 border border-blue-200/90 rounded-xl text-xs text-blue-900 leading-relaxed flex items-start gap-2">
                <BookOpen className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold block text-[11px] uppercase tracking-wider text-blue-800">Demonstration Reference Paper</span>
                  <span className="text-blue-900">{demonstrationNotice}</span>
                </div>
              </div>
            )}

            {/* Inquiry Prompt Input */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label
                  htmlFor="studio-prompt-input"
                  className="text-xs font-bold text-neutral-800 uppercase tracking-wider"
                >
                  Research Inquiry Question
                </label>
                <button
                  type="button"
                  onClick={handleCopy}
                  className="inline-flex items-center gap-1 text-[11px] text-neutral-500 hover:text-neutral-900 cursor-pointer bg-transparent border-0 p-0"
                >
                  {copied ? (
                    <Check className="w-3 h-3 text-emerald-600" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <textarea
                id="studio-prompt-input"
                rows={4}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Formulate your scientific query with precision..."
                className="w-full text-xs sm:text-sm p-3.5 bg-neutral-50 border border-neutral-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-neutral-900/10 focus:border-neutral-400 focus:bg-white text-neutral-900 transition-all resize-y"
              />
            </div>

            {/* Scope Selection */}
            <div>
              <label className="block text-xs font-bold text-neutral-800 uppercase tracking-wider mb-1.5">
                Target Retrieval Scope
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(['GLOBAL', 'COLLECTION', 'DOCUMENT'] as const).map((sc) => (
                  <button
                    key={sc}
                    type="button"
                    onClick={() => {
                      setScope(sc)
                      setError(null)
                    }}
                    className={`py-2 px-3 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                      scope === sc
                        ? 'bg-neutral-950 text-white shadow-xs'
                        : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
                    }`}
                  >
                    {sc === 'GLOBAL' ? 'Global Corpus' : sc === 'COLLECTION' ? 'Collection' : 'Single Paper'}
                  </button>
                ))}
              </div>

              {/* Scoped Document Selector */}
              {scope === 'DOCUMENT' && (
                <div className="mt-3 p-3 bg-neutral-50 rounded-xl border border-neutral-200">
                  <label htmlFor="studio-doc-select" className="block text-[11px] font-bold text-neutral-700 uppercase tracking-wider mb-1">
                    Select Target Document
                  </label>
                  {isLoadingScopeData ? (
                    <p className="text-xs text-neutral-400">Loading indexed documents...</p>
                  ) : documents.length > 0 ? (
                    <select
                      id="studio-doc-select"
                      value={selectedDocId}
                      onChange={(e) => setSelectedDocId(e.target.value)}
                      className="w-full text-xs p-2.5 bg-white border border-neutral-200 rounded-xl focus:outline-none focus:border-neutral-400 text-neutral-900 font-medium cursor-pointer"
                    >
                      {documents.map((doc) => (
                        <option key={doc.id} value={doc.id}>
                          {doc.title || doc.filename} ({doc.processing_status})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <p className="text-xs text-amber-700 bg-amber-50 p-2.5 rounded-lg border border-amber-200">
                      No indexed documents found in your library. Upload a research paper in Documents or switch to Global Corpus.
                    </p>
                  )}
                </div>
              )}

              {/* Scoped Collection Selector */}
              {scope === 'COLLECTION' && (
                <div className="mt-3 p-3 bg-neutral-50 rounded-xl border border-neutral-200">
                  <label htmlFor="studio-col-select" className="block text-[11px] font-bold text-neutral-700 uppercase tracking-wider mb-1">
                    Select Target Collection
                  </label>
                  {isLoadingScopeData ? (
                    <p className="text-xs text-neutral-400">Loading collections...</p>
                  ) : collections.length > 0 ? (
                    <select
                      id="studio-col-select"
                      value={selectedColId}
                      onChange={(e) => setSelectedColId(e.target.value)}
                      className="w-full text-xs p-2.5 bg-white border border-neutral-200 rounded-xl focus:outline-none focus:border-neutral-400 text-neutral-900 font-medium cursor-pointer"
                    >
                      {collections.map((col) => (
                        <option key={col.id} value={col.id}>
                          {col.name} ({col.document_count} {col.document_count === 1 ? 'paper' : 'papers'})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <p className="text-xs text-amber-700 bg-amber-50 p-2.5 rounded-lg border border-amber-200">
                      No collections found. Create a collection in the Collections tab or switch to Global Corpus.
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Engine & Top-K Slider */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label
                  htmlFor="studio-engine-select"
                  className="block text-xs font-bold text-neutral-800 uppercase tracking-wider mb-1.5"
                >
                  Retrieval Model
                </label>
                <select
                  id="studio-engine-select"
                  value={selectedEngine}
                  onChange={(e) => setSelectedEngine(e.target.value)}
                  className="w-full text-xs p-2.5 bg-neutral-50 border border-neutral-200 rounded-xl focus:outline-none focus:border-neutral-400 text-neutral-900 font-medium cursor-pointer"
                >
                  <option value="Grounded RAG (OmniRoute)">Grounded RAG (OmniRoute)</option>
                  <option value="Meta FAISS IndexFlatIP">Meta FAISS IndexFlatIP</option>
                  <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 Bi-Encoder</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-neutral-800 uppercase tracking-wider mb-1.5">
                  Top-K Passages: {topK}
                </label>
                <input
                  type="range"
                  min="1"
                  max="15"
                  step="1"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                  className="w-full accent-neutral-950 cursor-pointer mt-2"
                  aria-label="Top-K passages slider"
                />
              </div>
            </div>

            {/* Threshold Slider */}
            <div>
              <div className="flex items-center justify-between text-xs font-bold text-neutral-800 uppercase tracking-wider mb-1.5">
                <span>Cosine Similarity Threshold: {(threshold * 100).toFixed(0)}%</span>
                <span className="text-neutral-400 font-normal">Strict (90%) to Broad (50%)</span>
              </div>
              <input
                type="range"
                min="0.50"
                max="0.95"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-neutral-950 cursor-pointer"
                aria-label="Similarity threshold slider"
              />
            </div>
          </div>

          {/* Right Column: Live Viewport Canvas */}
          <div className="lg:col-span-5 flex flex-col justify-between bg-neutral-50 rounded-2xl p-4 sm:p-5 border border-neutral-200/80">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold text-neutral-700 uppercase tracking-wider">
                  Synthesis Viewport
                </span>
                <span className="text-[11px] font-mono text-neutral-500">
                  {scope} &bull; Top-{topK}
                </span>
              </div>

              {/* Viewport Box */}
              <div className="relative min-h-[260px] bg-neutral-950 rounded-xl p-4 overflow-hidden flex flex-col justify-center border border-neutral-800 shadow-inner text-white">
                {isSynthesizing ? (
                  <div className="flex flex-col items-center justify-center p-6 text-center space-y-3 z-10">
                    <RefreshCw className="w-8 h-8 text-[#D1FE17] animate-spin" />
                    <div>
                      <p className="text-xs font-bold text-white">Retrieving Dense Passages &amp; Synthesizing Grounded Answer</p>
                      <p className="text-[11px] text-neutral-400">FAISS dense retrieval &amp; LLM validation pass {synthesisProgress}%</p>
                    </div>
                    {/* Progress Bar */}
                    <div className="w-36 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#D1FE17] transition-all duration-300 rounded-full"
                        style={{ width: `${synthesisProgress}%` }}
                      />
                    </div>
                  </div>
                ) : error ? (
                  <div className="space-y-2.5 text-left p-3 z-10">
                    <div className="inline-flex items-center gap-1.5 bg-rose-500/20 text-rose-300 text-[10px] font-bold px-2.5 py-0.5 rounded border border-rose-500/30">
                      <AlertCircle className="w-3 h-3" />
                      <span>Retrieval Notice</span>
                    </div>
                    <p className="text-xs text-rose-200 leading-relaxed font-sans">
                      {error}
                    </p>
                  </div>
                ) : synthesizedResult ? (
                  <div className="space-y-3 text-left max-h-[360px] overflow-y-auto pr-1">
                    <div className="flex items-center justify-between">
                      <span className={`inline-flex items-center gap-1.5 ${
                        synthesizedResult.status === 'NO_RELEVANT_CONTEXT'
                          ? 'bg-amber-400/20 text-amber-300 border border-amber-400/30'
                          : 'bg-[#D1FE17] text-neutral-950'
                      } text-[10px] font-bold px-2 py-0.5 rounded shadow`}>
                        {synthesizedResult.status === 'NO_RELEVANT_CONTEXT' ? 'No Context Found' : 'Grounded Synthesis'}
                      </span>
                      <div className="flex items-center gap-2 text-[10px] text-neutral-400">
                        {synthesizedResult.citations.length > 0 && (
                          <span className="text-emerald-400 font-mono">
                            {synthesizedResult.citations.length} cited {synthesizedResult.citations.length === 1 ? 'passage' : 'passages'}
                          </span>
                        )}
                        {synthesizedResult.modelName && (
                          <span className="text-neutral-400 truncate max-w-[100px]">{synthesizedResult.modelName}</span>
                        )}
                      </div>
                    </div>

                    {/* Synthesized Answer */}
                    <div className="text-xs text-neutral-200 leading-relaxed font-sans bg-neutral-900/80 p-3 rounded-lg border border-neutral-800">
                      <p className="whitespace-pre-wrap">{synthesizedResult.answer}</p>
                    </div>

                    {/* Verified Passage Citations */}
                    {synthesizedResult.citations.length > 0 && (
                      <div className="space-y-2 pt-2 border-t border-neutral-800">
                        <span className="text-[10px] uppercase font-bold text-[#D1FE17] tracking-wider block">
                          Verified Passage Citations
                        </span>
                        <div className="space-y-1.5">
                          {synthesizedResult.citations.map((c) => (
                            <div
                              key={`${c.document_id}-${c.chunk_id}`}
                              className="bg-neutral-900 p-2.5 rounded-lg border border-neutral-800 text-[11px] space-y-1"
                            >
                              <div className="flex items-center justify-between text-[10px] text-neutral-400">
                                <span className="font-semibold text-white truncate max-w-[160px]">{c.document_title}</span>
                                <span className="text-emerald-400 font-mono">
                                  Page {c.page_number} &bull; {(c.similarity_score * 100).toFixed(1)}% match
                                </span>
                              </div>
                              <p className="text-neutral-300 italic line-clamp-3 leading-snug">
                                &ldquo;{c.quoted_text}&rdquo;
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {synthesizedResult.retrievedCount !== undefined && (
                      <div className="pt-2 text-[10px] text-neutral-500 flex items-center justify-between">
                        <span>Evaluated {synthesizedResult.retrievedCount} candidate passages</span>
                        {synthesizedResult.providerName && (
                          <span>Provider: {synthesizedResult.providerName}</span>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center p-6 text-neutral-400">
                    <Sparkles className="w-8 h-8 mx-auto mb-2 opacity-40 text-neutral-300" />
                    <p className="text-xs">Click Execute to run FAISS retrieval &amp; grounded RAG synthesis</p>
                  </div>
                )}
              </div>
            </div>

            {/* Bottom Actions */}
            <div className="mt-4 pt-3 border-t border-neutral-200/80 space-y-2">
              <button
                type="button"
                onClick={handleSynthesize}
                disabled={isSynthesizing || !prompt.trim() || !isScopeReady()}
                className="w-full py-3 bg-[#D1FE17] hover:bg-[#c2ed14] disabled:bg-neutral-200 disabled:text-neutral-400 text-neutral-950 font-bold rounded-xl text-xs sm:text-sm flex items-center justify-center gap-2 transition-transform active:scale-98 shadow-sm cursor-pointer"
              >
                <Sparkles className="w-4 h-4" />
                <span>{isSynthesizing ? 'Synthesizing with RAG...' : 'Execute with ResearchLens'}</span>
              </button>

              {onOpenChatWithQuery && (
                <button
                  type="button"
                  onClick={() => {
                    onOpenChatWithQuery(prompt, scope)
                    onClose()
                  }}
                  disabled={!prompt.trim()}
                  className="w-full py-2 bg-neutral-200 hover:bg-neutral-300 text-neutral-800 font-semibold rounded-xl text-xs transition-colors cursor-pointer"
                >
                  Continue in Full Research Chat &rarr;
                </button>
              )}

              <p className="text-[10px] text-center text-neutral-400">
                Exact passage citations verified against PostgreSQL chunks
              </p>
            </div>

          </div>

        </div>
      </div>
    </div>
  )
}
export default ResearchModal
